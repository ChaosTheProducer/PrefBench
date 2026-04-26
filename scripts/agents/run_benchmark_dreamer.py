"""Run benchmark for heuristics and DreamerV3 on shared consumer episodes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
from typing import Any, Dict, List, Mapping, Sequence

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
from pricing_agent.baselines import ConcessionPolicy, RandomPolicy
from pricing_agent.world_model import (
    DreamerDiscretePricingEnv,
    DreamerEnvBuildSpec,
    DreamerTTAConfig,
    build_dreamer_config,
    load_dreamer_policy_actor,
    require_dreamerv3_dependencies,
)
from pricing_env.gym_wrapper import GYMNASIUM_AVAILABLE, PricingNegotiationGymWrapper


def parse_args() -> argparse.Namespace:
    """Parses benchmark command-line arguments.

    Returns:
        Parsed argument namespace.
    """

    parser = argparse.ArgumentParser(description="Benchmark heuristics and Dreamer on same persona episodes.")
    parser.add_argument(
        "--config-path",
        type=Path,
        default=DEFAULT_DREAMER_CONFIG_PATH,
        help="Unified Dreamer config YAML path.",
    )
    parser.add_argument("--episodes", type=int, default=None, help="Optional evaluation episodes override.")
    parser.add_argument("--seed", type=int, default=123, help="Global benchmark seed.")
    parser.add_argument(
        "--policies",
        type=str,
        default=None,
        help="Optional comma-separated policy list override.",
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
    parser.add_argument(
        "--report-out",
        type=Path,
        default=None,
        help="Optional explicit benchmark report output path.",
    )
    parser.add_argument(
        "--persona-bank-path",
        type=Path,
        default=None,
        help="Optional persona bank JSONL path override.",
    )
    parser.add_argument(
        "--persona-bank-split",
        type=str,
        default=None,
        choices=["train", "val", "test"],
        help="Optional evaluation split override.",
    )
    parser.add_argument(
        "--allow-consumer-mismatch",
        action="store_true",
        help="Disable strict persona-id equality checks across policies.",
    )
    return parser.parse_args()


def _parse_policy_list(raw: str) -> List[str]:
    """Parses and validates policy names.

    Args:
        raw: Comma-separated policy names.

    Returns:
        Ordered policy list.
    """

    tokens = [token.strip().lower() for token in raw.split(",") if token.strip()]
    allowed = {"random", "concession", "dreamer"}
    invalid = [token for token in tokens if token not in allowed]
    if invalid:
        raise ValueError(f"Unsupported policies: {invalid}. Allowed: {sorted(allowed)}")
    if not tokens:
        raise ValueError("At least one policy must be selected.")
    return tokens


def _new_metric() -> Dict[str, float]:
    """Creates one zero-initialized metric accumulator."""

    return {
        "episodes": 0.0,
        "deal_count": 0.0,
        "total_profit_usd": 0.0,
        "total_rounds": 0.0,
        "walkaway_count": 0.0,
        "terminated_count": 0.0,
        "truncated_count": 0.0,
        "total_trace_len": 0.0,
        "total_env_reward": 0.0,
        "total_policy_reward": 0.0,
    }


def _finalize_metric(acc: Dict[str, float]) -> Dict[str, float | int]:
    """Converts metric accumulator into report-friendly metrics.

    Args:
        acc: Mutable metric accumulator.

    Returns:
        Finalized metric dictionary.
    """

    episodes = int(acc["episodes"])
    denom = float(max(1, episodes))
    return {
        "episodes": episodes,
        "deal_rate": acc["deal_count"] / denom,
        "avg_profit_usd": acc["total_profit_usd"] / denom,
        "avg_rounds": acc["total_rounds"] / denom,
        "walkaway_rate": acc["walkaway_count"] / denom,
        "terminated_rate": acc["terminated_count"] / denom,
        "truncated_rate": acc["truncated_count"] / denom,
        "avg_trace_len": acc["total_trace_len"] / denom,
        "avg_env_reward": acc["total_env_reward"] / denom,
        "avg_policy_reward": acc["total_policy_reward"] / denom,
    }


def _run_heuristic_policy(
    *,
    policy_name: str,
    episode_seeds: Sequence[int],
    policy_seed: int,
    catalog_path: Path,
    persona_config_path: Path,
    persona_bank_path: Path,
    persona_bank_split: str,
    clip_enabled: bool,
    clip_semantic_path: Path | None,
    clip_legacy_proxy_enabled: bool,
) -> Dict[str, Any]:
    """Runs heuristic policy benchmark with shared episode seeds.

    Args:
        policy_name: `random` or `concession`.
        episode_seeds: Shared episode seeds.
        policy_seed: Seed for policy stochasticity.
        catalog_path: Catalog path.
        persona_config_path: Persona config path.
        persona_bank_path: Persona bank path.
        persona_bank_split: Persona split.
        clip_enabled: Whether to append CLIP semantics to observation.
        clip_semantic_path: Path to CLIP semantics artifact when enabled.
        clip_legacy_proxy_enabled: Whether proxy aesthetic scalar is preserved.

    Returns:
        Benchmark payload with `metrics` and `persona_ids`.
    """

    env = PricingNegotiationGymWrapper(
        catalog_path=catalog_path,
        persona_config_path=persona_config_path,
        persona_bank_path=persona_bank_path,
        persona_bank_split=persona_bank_split,
        clip_enabled=bool(clip_enabled),
        clip_semantic_path=clip_semantic_path,
        clip_legacy_proxy_enabled=bool(clip_legacy_proxy_enabled),
    )
    offer_min = float(env.action_space["price"].low[0])
    offer_max = float(env.action_space["price"].high[0])
    policy_rng = random.Random(policy_seed)
    if policy_name == "random":
        policy = RandomPolicy(rng=policy_rng, offer_min=offer_min, offer_max=offer_max)
    elif policy_name == "concession":
        policy = ConcessionPolicy(rng=policy_rng, offer_min=offer_min, offer_max=offer_max)
    else:  # pragma: no cover - guarded by caller validation
        raise ValueError(f"Unsupported heuristic policy: {policy_name}")

    acc = _new_metric()
    persona_ids: List[str] = []
    for ep_seed in episode_seeds:
        _obs, info = env.reset(seed=int(ep_seed))
        persona_ids.append(str(info.get("persona_id", "")))
        terminated = False
        truncated = False
        total_reward = 0.0
        while not (terminated or truncated):
            action = policy.act(info)
            _obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)

        metrics = info.get("episode_metrics", {})
        acc["episodes"] += 1.0
        acc["deal_count"] += float(bool(metrics.get("deal_reached", False)))
        acc["total_profit_usd"] += float(metrics.get("profit_usd", 0.0))
        acc["total_rounds"] += float(metrics.get("rounds_used", 0))
        acc["walkaway_count"] += float(bool(metrics.get("walkaway", False)))
        acc["terminated_count"] += float(bool(terminated))
        acc["truncated_count"] += float(bool(truncated))
        acc["total_trace_len"] += float(info.get("trace_len", 0))
        acc["total_env_reward"] += total_reward
        acc["total_policy_reward"] += total_reward

    env.close()
    return {
        "policy": policy_name,
        "metrics": _finalize_metric(acc),
        "persona_ids": persona_ids,
    }


def _run_dreamer_policy(
    *,
    episode_seeds: Sequence[int],
    seed: int,
    raw_config: Mapping[str, Any],
    checkpoint_path: Path,
    catalog_path: Path,
    persona_config_path: Path,
    persona_bank_path: Path,
    persona_bank_split: str,
    clip_enabled: bool,
    clip_semantic_path: Path | None,
    clip_legacy_proxy_enabled: bool,
    run_name: str,
    output_root: Path,
) -> Dict[str, Any]:
    """Runs Dreamer policy benchmark with shared episode seeds.

    Args:
        episode_seeds: Shared episode seeds.
        seed: Global seed.
        raw_config: Raw Dreamer config mapping.
        checkpoint_path: Dreamer checkpoint path.
        catalog_path: Catalog path.
        persona_config_path: Persona config path.
        persona_bank_path: Persona bank path.
        persona_bank_split: Persona split.
        clip_enabled: Whether to append CLIP semantics to observation.
        clip_semantic_path: Path to CLIP semantics artifact when enabled.
        clip_legacy_proxy_enabled: Whether proxy aesthetic scalar is preserved.
        run_name: Run name.
        output_root: Output root for logs.

    Returns:
        Benchmark payload with `metrics` and `persona_ids`.
    """

    embodied, dreamer_agent_module, dreamer_train_module, _wrappers = require_dreamerv3_dependencies()
    presets = get_required(raw_config, "dreamer.presets")
    if not isinstance(presets, list) or not presets:
        raise ValueError("`dreamer.presets` must be a non-empty list.")

    dreamer_cfg = build_dreamer_config(
        embodied=embodied,
        dreamer_agent_module=dreamer_agent_module,
        dreamer_train_module=dreamer_train_module,
        presets=[str(item) for item in presets],
        logdir=output_root / "logs" / "dreamer" / run_name,
        seed=int(seed),
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
        seed=int(seed),
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

    acc = _new_metric()
    persona_ids: List[str] = []
    for ep_seed in episode_seeds:
        obs, info = env.reset(seed=int(ep_seed))
        persona_ids.append(str(info.get("persona_id", "")))
        actor.reset(reset_info=info)
        is_first = True
        terminated = False
        truncated = False
        done = False
        total_env_reward = 0.0
        total_policy_reward = 0.0
        last_info: Dict[str, Any] = info

        while not done:
            action_idx = actor.predict_action_index(obs, is_first=is_first)
            obs, reward, terminated, truncated, step_info = env.step(action_idx)
            is_first = False
            done = bool(terminated or truncated)
            actor.set_prev_reward(float(reward))
            actor.observe_step_info(step_info)

            total_policy_reward += float(reward)
            total_env_reward += float(step_info.get("ppo_reward_raw_env", 0.0))
            last_info = step_info

        metrics = last_info.get("episode_metrics", {})
        acc["episodes"] += 1.0
        acc["deal_count"] += float(bool(metrics.get("deal_reached", False)))
        acc["total_profit_usd"] += float(metrics.get("profit_usd", 0.0))
        acc["total_rounds"] += float(metrics.get("rounds_used", 0))
        acc["walkaway_count"] += float(bool(metrics.get("walkaway", False)))
        acc["terminated_count"] += float(bool(terminated))
        acc["truncated_count"] += float(bool(truncated))
        acc["total_trace_len"] += float(last_info.get("trace_len", 0))
        acc["total_env_reward"] += total_env_reward
        acc["total_policy_reward"] += total_policy_reward

    env.close()
    return {
        "policy": "dreamer",
        "metrics": _finalize_metric(acc),
        "persona_ids": persona_ids,
        "tta": actor.tta_report(),
    }


def _assert_same_consumers(policy_results: Dict[str, Dict[str, Any]]) -> None:
    """Checks that persona-id sequence is identical across evaluated policies.

    Args:
        policy_results: Per-policy benchmark payloads.
    """

    if not policy_results:
        return
    ordered_names = list(policy_results.keys())
    base_name = ordered_names[0]
    base_ids = policy_results[base_name]["persona_ids"]
    for name in ordered_names[1:]:
        ids = policy_results[name]["persona_ids"]
        if ids != base_ids:
            mismatch_idx = next((i for i, (a, b) in enumerate(zip(base_ids, ids)) if a != b), None)
            raise RuntimeError(
                f"Consumer mismatch between `{base_name}` and `{name}` at episode index {mismatch_idx}."
            )


def main() -> None:
    """Runs benchmark and writes JSON report."""

    if not GYMNASIUM_AVAILABLE:
        raise RuntimeError("Gymnasium is required for benchmark execution.")

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
        else str(get_required(raw_config, "benchmark.persona_bank_split"))
    )
    clip_enabled = bool(get_optional(raw_config, "clip.enabled", False))
    clip_semantic_path = (
        resolve_repo_path(get_required(raw_config, "clip.semantic_path")) if clip_enabled else None
    )
    clip_legacy_proxy_enabled = bool(get_optional(raw_config, "clip.legacy_proxy_enabled", True))
    episodes = int(args.episodes) if args.episodes is not None else int(get_required(raw_config, "benchmark.episodes"))
    if episodes <= 0:
        raise ValueError("`episodes` must be positive.")
    config_policies = get_required(raw_config, "benchmark.policies")
    if not isinstance(config_policies, list):
        raise ValueError("`benchmark.policies` must be a list.")
    policies = _parse_policy_list(
        str(args.policies) if args.policies is not None else ",".join(str(item) for item in config_policies)
    )
    allow_consumer_mismatch = bool(args.allow_consumer_mismatch) or bool(
        get_required(raw_config, "benchmark.allow_consumer_mismatch")
    )
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
    report_out = (
        Path(args.report_out)
        if args.report_out is not None
        else output_root / "reports" / f"{run_name}_benchmark_{persona_bank_split}.json"
    )

    episode_seeds = [int(args.seed) + i for i in range(episodes)]
    policy_results: Dict[str, Dict[str, Any]] = {}

    for idx, policy_name in enumerate(policies):
        if policy_name in {"random", "concession"}:
            payload = _run_heuristic_policy(
                policy_name=policy_name,
                episode_seeds=episode_seeds,
                policy_seed=int(args.seed) + 10_000 + idx,
                catalog_path=catalog_path,
                persona_config_path=persona_config_path,
                persona_bank_path=persona_bank_path,
                persona_bank_split=persona_bank_split,
                clip_enabled=bool(clip_enabled),
                clip_semantic_path=clip_semantic_path,
                clip_legacy_proxy_enabled=bool(clip_legacy_proxy_enabled),
            )
        elif policy_name == "dreamer":
            payload = _run_dreamer_policy(
                episode_seeds=episode_seeds,
                seed=int(args.seed),
                raw_config=raw_config,
                checkpoint_path=checkpoint_path,
                catalog_path=catalog_path,
                persona_config_path=persona_config_path,
                persona_bank_path=persona_bank_path,
                persona_bank_split=persona_bank_split,
                clip_enabled=bool(clip_enabled),
                clip_semantic_path=clip_semantic_path,
                clip_legacy_proxy_enabled=bool(clip_legacy_proxy_enabled),
                run_name=run_name,
                output_root=output_root,
            )
        else:  # pragma: no cover - already validated
            raise ValueError(f"Unsupported policy name: {policy_name}")
        policy_results[policy_name] = payload

    same_consumers = True
    if len(policy_results) >= 2 and not allow_consumer_mismatch:
        _assert_same_consumers(policy_results)
    elif len(policy_results) >= 2:
        try:
            _assert_same_consumers(policy_results)
        except RuntimeError:
            same_consumers = False

    report = {
        "run_name": run_name,
        "config_path": str(config_path),
        "output_root": str(output_root),
        "checkpoint_path": str(checkpoint_path),
        "seed": int(args.seed),
        "episodes": episodes,
        "episode_seed_range": [episode_seeds[0], episode_seeds[-1]],
        "persona_bank_path": str(persona_bank_path),
        "persona_bank_split": persona_bank_split,
        "clip_enabled": bool(clip_enabled),
        "clip_semantic_path": None if clip_semantic_path is None else str(clip_semantic_path),
        "dreamer_reward_mode": str(get_optional(raw_config, "dreamer_reward.mode", "ppo_compatible")),
        "dreamer_step_no_deal_penalty": float(get_optional(raw_config, "dreamer_reward.step_no_deal_penalty", 0.0)),
        "dreamer_step_no_deal_penalty_start_round": int(
            get_optional(raw_config, "dreamer_reward.step_no_deal_penalty_start_round", 3)
        ),
        "dreamer_soft_shortfall_penalty_coeff": float(
            get_optional(raw_config, "dreamer_reward.soft_shortfall_penalty_coeff", 0.0)
        ),
        "tta_config": {
            "enabled": bool(get_optional(raw_config, "tta.enabled", False)),
            "mode": str(get_optional(raw_config, "tta.mode", "belief_shift_v1")),
            "logit_bias_scale": float(get_optional(raw_config, "tta.logit_bias_scale", 1.0)),
            "alpha_wtp": float(get_optional(raw_config, "tta.alpha_wtp", 0.45)),
            "alpha_counter": float(get_optional(raw_config, "tta.alpha_counter", 0.35)),
            "alpha_risk": float(get_optional(raw_config, "tta.alpha_risk", 0.40)),
            "max_bias_abs": float(get_optional(raw_config, "tta.max_bias_abs", 0.45)),
            "max_price_adjust_usd_per_round": float(
                get_optional(raw_config, "tta.max_price_adjust_usd_per_round", 200.0)
            ),
            "max_candidates": int(get_optional(raw_config, "tta.max_candidates", 8)),
            "offer_neighbor_bins": int(get_optional(raw_config, "tta.offer_neighbor_bins", 2)),
            "max_price_adjust_usd": float(get_optional(raw_config, "tta.max_price_adjust_usd", 200.0)),
            "imagination_horizon": int(get_optional(raw_config, "tta.imagination_horizon", 1)),
            "w_policy": float(get_optional(raw_config, "tta.w_policy", 0.20)),
            "w_value": float(get_optional(raw_config, "tta.w_value", 0.35)),
            "w_margin": float(get_optional(raw_config, "tta.w_margin", 0.20)),
            "w_feasibility": float(get_optional(raw_config, "tta.w_feasibility", 0.15)),
            "w_risk": float(get_optional(raw_config, "tta.w_risk", 0.10)),
        },
        "policies": policies,
        "same_consumers_verified": same_consumers,
        "results": {name: payload["metrics"] for name, payload in policy_results.items()},
        "tta": policy_results.get("dreamer", {}).get("tta", {"tta_enabled": False}),
    }
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
