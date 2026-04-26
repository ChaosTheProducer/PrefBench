"""Evaluate a trained DreamerV3 pricing agent on the negotiation environment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scripts.common.dreamer_config import (
    DEFAULT_DREAMER_CONFIG_PATH,
    get_optional,
    get_required,
    load_dreamer_config,
    resolve_repo_path,
)
from pricing_agent.world_model import (
    DreamerDiscretePricingEnv,
    DreamerEnvBuildSpec,
    DreamerTTAConfig,
    build_dreamer_config,
    load_dreamer_policy_actor,
    require_dreamerv3_dependencies,
)


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments.

    Returns:
        Parsed command-line namespace.
    """

    parser = argparse.ArgumentParser(description="Evaluate trained DreamerV3 model.")
    parser.add_argument(
        "--config-path",
        type=Path,
        default=DEFAULT_DREAMER_CONFIG_PATH,
        help="Unified Dreamer config YAML path.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional run name override.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=None,
        help="Optional explicit Dreamer checkpoint path.",
    )
    parser.add_argument("--episodes", type=int, default=None, help="Optional evaluation episode override.")
    parser.add_argument("--seed", type=int, default=123, help="Evaluation seed.")
    parser.add_argument(
        "--persona-bank-path",
        type=Path,
        default=None,
        help="Optional persona-bank JSONL path override.",
    )
    parser.add_argument(
        "--persona-bank-split",
        type=str,
        default=None,
        choices=["train", "val", "test"],
        help="Optional evaluation split override.",
    )
    parser.add_argument(
        "--metrics-out",
        type=Path,
        default=None,
        help="Optional explicit output JSON path for evaluation metrics.",
    )
    return parser.parse_args()


def _parse_presets(value: Any) -> list[str]:
    """Parses Dreamer presets from config.

    Args:
        value: Raw config value.

    Returns:
        Ordered preset list.
    """

    if not isinstance(value, list):
        raise ValueError("`dreamer.presets` must be a list.")
    presets = [str(token).strip() for token in value if str(token).strip()]
    if not presets:
        raise ValueError("`dreamer.presets` cannot be empty.")
    return presets


def main() -> None:
    """Runs Dreamer evaluation episodes and prints aggregate metrics."""

    args = parse_args()
    config_path = resolve_repo_path(args.config_path)
    raw_config = load_dreamer_config(config_path)

    run_name = (
        str(args.run_name).strip()
        if args.run_name is not None and str(args.run_name).strip()
        else str(get_required(raw_config, "defaults.run_name"))
    )
    output_root = resolve_repo_path(get_required(raw_config, "defaults.output_root"))
    catalog_path = resolve_repo_path(get_required(raw_config, "paths.catalog_path"))
    persona_config_path = resolve_repo_path(get_required(raw_config, "paths.persona_config_path"))
    persona_bank_path = (
        resolve_repo_path(args.persona_bank_path)
        if args.persona_bank_path is not None
        else resolve_repo_path(get_required(raw_config, "paths.persona_bank_path"))
    )
    persona_bank_split = (
        str(args.persona_bank_split)
        if args.persona_bank_split is not None
        else str(get_required(raw_config, "eval.persona_bank_split"))
    )
    clip_enabled = bool(get_optional(raw_config, "clip.enabled", False))
    clip_semantic_path = (
        resolve_repo_path(get_required(raw_config, "clip.semantic_path")) if clip_enabled else None
    )
    clip_legacy_proxy_enabled = bool(get_optional(raw_config, "clip.legacy_proxy_enabled", True))
    episodes = int(args.episodes) if args.episodes is not None else int(get_required(raw_config, "eval.episodes"))
    if episodes <= 0:
        raise ValueError("`episodes` must be positive.")

    config_checkpoint = get_optional(raw_config, "paths.checkpoint_path", None)
    checkpoint_path = (
        Path(args.checkpoint_path)
        if args.checkpoint_path is not None
        else (
            resolve_repo_path(config_checkpoint)
            if config_checkpoint is not None
            else output_root / "checkpoints" / f"{run_name}_final.ckpt"
        )
    )
    metrics_out = (
        Path(args.metrics_out)
        if args.metrics_out is not None
        else output_root / "reports" / f"{run_name}_eval_{persona_bank_split}.json"
    )

    embodied, dreamer_agent_module, dreamer_train_module, _wrappers = require_dreamerv3_dependencies()
    presets = _parse_presets(get_required(raw_config, "dreamer.presets"))
    dreamer_cfg = build_dreamer_config(
        embodied=embodied,
        dreamer_agent_module=dreamer_agent_module,
        dreamer_train_module=dreamer_train_module,
        presets=presets,
        logdir=output_root / "logs" / "dreamer" / run_name,
        seed=int(args.seed),
        steps=int(get_required(raw_config, "train.steps")),
        jax_platform=str(get_required(raw_config, "dreamer.jax_platform")),
        jax_precision=str(get_required(raw_config, "dreamer.jax_precision")),
        envs_amount=1,
        envs_parallel="none",
        replay_size=int(get_required(raw_config, "dreamer.replay_size")),
        batch_size=int(get_required(raw_config, "dreamer.batch_size")),
        batch_length=int(get_required(raw_config, "dreamer.batch_length")),
        train_ratio=float(get_required(raw_config, "dreamer.train_ratio")),
        train_fill=int(get_required(raw_config, "dreamer.train_fill")),
        log_every=int(get_required(raw_config, "dreamer.log_every")),
        save_every=int(get_required(raw_config, "dreamer.save_every")),
    )
    build_spec = DreamerEnvBuildSpec(
        catalog_path=catalog_path,
        persona_config_path=persona_config_path,
        persona_bank_path=persona_bank_path,
        persona_bank_split=persona_bank_split,
        price_bin_count=int(get_required(raw_config, "environment.price_bin_count")),
        price_step_usd=float(get_required(raw_config, "environment.price_step_usd")),
        clip_enabled=bool(clip_enabled),
        clip_semantic_path=(
            Path(clip_semantic_path)
            if clip_semantic_path is not None
            else (ROOT / "datasets" / "clip_semantics" / "e350_clip_text_v1.json")
        ),
        clip_legacy_proxy_enabled=bool(clip_legacy_proxy_enabled),
        reward_scale_usd=float(get_required(raw_config, "environment.reward_scale_usd")),
        no_deal_penalty=float(get_required(raw_config, "environment.no_deal_penalty")),
        step_no_deal_penalty=float(get_optional(raw_config, "dreamer_reward.step_no_deal_penalty", 0.0)),
        step_no_deal_penalty_start_round=int(
            get_optional(raw_config, "dreamer_reward.step_no_deal_penalty_start_round", 3)
        ),
        profit_target_usd=float(get_required(raw_config, "environment.profit_target_usd")),
        low_profit_penalty=float(get_required(raw_config, "environment.low_profit_penalty")),
        soft_shortfall_penalty_coeff=float(
            get_optional(raw_config, "dreamer_reward.soft_shortfall_penalty_coeff", 0.0)
        ),
        no_deal_requires_positive_margin=bool(
            get_required(raw_config, "environment.no_deal_requires_positive_margin")
        ),
        invalid_accept_penalty=float(get_required(raw_config, "environment.invalid_accept_penalty")),
        initial_offer_markup=float(get_required(raw_config, "environment.initial_offer_markup")),
        reward_mode=str(get_optional(raw_config, "dreamer_reward.mode", "ppo_compatible")),
        grace_rounds_no_deal_penalty=int(
            get_optional(raw_config, "dreamer_reward.grace_rounds_no_deal_penalty", 3)
        ),
        early_deal_round_cutoff=int(get_optional(raw_config, "dreamer_reward.early_deal_round_cutoff", 3)),
        early_deal_bonus=float(get_optional(raw_config, "dreamer_reward.early_deal_bonus", 0.0)),
        delay_penalty_start_round=int(get_optional(raw_config, "dreamer_reward.delay_penalty_start_round", 4)),
        delay_penalty_per_round=float(get_optional(raw_config, "dreamer_reward.delay_penalty_per_round", 0.0)),
        probe_bonus_round_cutoff=int(get_optional(raw_config, "dreamer_reward.probe_bonus_round_cutoff", 2)),
        probe_bonus=float(get_optional(raw_config, "dreamer_reward.probe_bonus", 0.0)),
    )
    tta_config = DreamerTTAConfig(
        enabled=bool(get_optional(raw_config, "tta.enabled", False)),
        mode=str(get_optional(raw_config, "tta.mode", "belief_shift_v1")),
        logit_bias_scale=float(get_optional(raw_config, "tta.logit_bias_scale", 1.0)),
        alpha_wtp=float(get_optional(raw_config, "tta.alpha_wtp", 0.45)),
        alpha_counter=float(get_optional(raw_config, "tta.alpha_counter", 0.35)),
        alpha_risk=float(get_optional(raw_config, "tta.alpha_risk", 0.40)),
        max_bias_abs=float(get_optional(raw_config, "tta.max_bias_abs", 0.45)),
        max_price_adjust_usd_per_round=float(
            get_optional(raw_config, "tta.max_price_adjust_usd_per_round", 200.0)
        ),
        max_candidates=int(get_optional(raw_config, "tta.max_candidates", 8)),
        offer_neighbor_bins=int(get_optional(raw_config, "tta.offer_neighbor_bins", 2)),
        max_price_adjust_usd=float(get_optional(raw_config, "tta.max_price_adjust_usd", 200.0)),
        imagination_horizon=int(get_optional(raw_config, "tta.imagination_horizon", 1)),
        w_policy=float(get_optional(raw_config, "tta.w_policy", 0.20)),
        w_value=float(get_optional(raw_config, "tta.w_value", 0.35)),
        w_margin=float(get_optional(raw_config, "tta.w_margin", 0.20)),
        w_feasibility=float(get_optional(raw_config, "tta.w_feasibility", 0.15)),
        w_risk=float(get_optional(raw_config, "tta.w_risk", 0.10)),
    )
    actor, _codec = load_dreamer_policy_actor(
        embodied=embodied,
        dreamer_agent_module=dreamer_agent_module,
        dreamer_train_module=dreamer_train_module,
        config=dreamer_cfg,
        build_spec=build_spec,
        checkpoint_path=checkpoint_path,
        seed=int(args.seed),
        tta_config=tta_config,
    )

    env = DreamerDiscretePricingEnv(
        catalog_path=str(catalog_path),
        persona_config_path=str(persona_config_path),
        persona_bank_path=str(persona_bank_path),
        persona_bank_split=persona_bank_split,
        price_bin_count=int(build_spec.price_bin_count),
        price_step_usd=float(build_spec.price_step_usd),
        clip_enabled=bool(build_spec.clip_enabled),
        clip_semantic_path=str(build_spec.clip_semantic_path),
        clip_legacy_proxy_enabled=bool(build_spec.clip_legacy_proxy_enabled),
        reward_scale_usd=float(build_spec.reward_scale_usd),
        no_deal_penalty=float(build_spec.no_deal_penalty),
        step_no_deal_penalty=float(build_spec.step_no_deal_penalty),
        step_no_deal_penalty_start_round=int(build_spec.step_no_deal_penalty_start_round),
        profit_target_usd=float(build_spec.profit_target_usd),
        low_profit_penalty=float(build_spec.low_profit_penalty),
        soft_shortfall_penalty_coeff=float(build_spec.soft_shortfall_penalty_coeff),
        no_deal_requires_positive_margin=bool(build_spec.no_deal_requires_positive_margin),
        invalid_accept_penalty=float(build_spec.invalid_accept_penalty),
        initial_offer_markup=float(build_spec.initial_offer_markup),
        reward_mode=str(build_spec.reward_mode),
        grace_rounds_no_deal_penalty=int(build_spec.grace_rounds_no_deal_penalty),
        early_deal_round_cutoff=int(build_spec.early_deal_round_cutoff),
        early_deal_bonus=float(build_spec.early_deal_bonus),
        delay_penalty_start_round=int(build_spec.delay_penalty_start_round),
        delay_penalty_per_round=float(build_spec.delay_penalty_per_round),
        probe_bonus_round_cutoff=int(build_spec.probe_bonus_round_cutoff),
        probe_bonus=float(build_spec.probe_bonus),
    )

    deals = 0
    total_profit = 0.0
    total_rounds = 0
    walkaways = 0
    terminated_count = 0
    truncated_count = 0
    total_env_reward = 0.0
    total_policy_reward = 0.0
    trace_len_total = 0

    for episode_idx in range(episodes):
        obs, reset_info = env.reset(seed=int(args.seed) + episode_idx)
        actor.reset(reset_info=reset_info)
        done = False
        is_first = True
        last_info: Dict[str, Any] = {}
        terminated = False
        truncated = False
        while not done:
            action_idx = actor.predict_action_index(obs, is_first=is_first)
            obs, reward, terminated, truncated, step_info = env.step(action_idx)
            is_first = False
            done = bool(terminated or truncated)
            actor.set_prev_reward(float(reward))
            actor.observe_step_info(step_info)

            total_policy_reward += float(reward)
            total_env_reward += float(step_info.get("ppo_reward_raw_env", 0.0))
            terminated_count += int(terminated)
            truncated_count += int(truncated)
            last_info = step_info

        metrics = last_info.get("episode_metrics", {})
        deals += int(metrics.get("deal_reached", False))
        total_profit += float(metrics.get("profit_usd", 0.0))
        total_rounds += int(metrics.get("rounds_used", 0))
        walkaways += int(metrics.get("walkaway", False))
        trace_len_total += int(last_info.get("trace_len", 0))

    env.close()

    denom = float(max(1, episodes))
    result = {
        "run_name": run_name,
        "config_path": str(config_path),
        "output_root": str(output_root),
        "checkpoint_path": str(checkpoint_path),
        "dreamer_reward_mode": str(build_spec.reward_mode),
        "dreamer_step_no_deal_penalty": float(build_spec.step_no_deal_penalty),
        "dreamer_step_no_deal_penalty_start_round": int(build_spec.step_no_deal_penalty_start_round),
        "dreamer_soft_shortfall_penalty_coeff": float(build_spec.soft_shortfall_penalty_coeff),
        "episodes": episodes,
        "seed": int(args.seed),
        "persona_bank_split": persona_bank_split,
        "clip_enabled": bool(build_spec.clip_enabled),
        "clip_semantic_path": str(build_spec.clip_semantic_path),
        "deal_rate": deals / denom,
        "avg_profit_usd": total_profit / denom,
        "avg_rounds": total_rounds / denom,
        "walkaway_rate": walkaways / denom,
        "terminated_rate": terminated_count / denom,
        "truncated_rate": truncated_count / denom,
        "avg_trace_len": trace_len_total / denom,
        "avg_env_reward": total_env_reward / denom,
        "avg_policy_reward": total_policy_reward / denom,
        "tta_config": {
            "enabled": bool(tta_config.enabled),
            "mode": str(tta_config.mode),
            "logit_bias_scale": float(tta_config.logit_bias_scale),
            "alpha_wtp": float(tta_config.alpha_wtp),
            "alpha_counter": float(tta_config.alpha_counter),
            "alpha_risk": float(tta_config.alpha_risk),
            "max_bias_abs": float(tta_config.max_bias_abs),
            "max_price_adjust_usd_per_round": float(tta_config.max_price_adjust_usd_per_round),
            "max_candidates": int(tta_config.max_candidates),
            "offer_neighbor_bins": int(tta_config.offer_neighbor_bins),
            "max_price_adjust_usd": float(tta_config.max_price_adjust_usd),
            "imagination_horizon": int(tta_config.imagination_horizon),
            "w_policy": float(tta_config.w_policy),
            "w_value": float(tta_config.w_value),
            "w_margin": float(tta_config.w_margin),
            "w_feasibility": float(tta_config.w_feasibility),
            "w_risk": float(tta_config.w_risk),
        },
        "tta": actor.tta_report(),
    }
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    metrics_out.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
