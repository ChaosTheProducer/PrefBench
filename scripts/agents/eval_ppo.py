"""Evaluate a trained PPO pricing agent on the negotiation environment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pricing_agent.ppo_env import PPOPricingEnv
from scripts.common.ppo_config import DEFAULT_PPO_CONFIG_PATH, get_optional, get_required, load_ppo_config, resolve_repo_path

def parse_args() -> argparse.Namespace:
    """Parses command-line arguments.

    Returns:
        Parsed argument namespace.
    """

    parser = argparse.ArgumentParser(description="Evaluate trained Recurrent PPO model.")
    parser.add_argument(
        "--config-path",
        type=Path,
        default=DEFAULT_PPO_CONFIG_PATH,
        help="Unified PPO config YAML path.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional run name override.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Optional explicit path to trained model zip.",
    )
    parser.add_argument("--episodes", type=int, default=None, help="Optional evaluation episode override.")
    parser.add_argument("--seed", type=int, default=123, help="Evaluation seed.")
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
        "--deterministic",
        action="store_true",
        help="Force deterministic policy actions (overrides config).",
    )
    parser.add_argument(
        "--metrics-out",
        type=Path,
        default=None,
        help="Optional explicit output JSON path for evaluation metrics.",
    )
    return parser.parse_args()


def main() -> None:
    """Runs evaluation episodes and prints aggregate metrics as JSON."""

    args = parse_args()
    config_path = resolve_repo_path(args.config_path)
    config = load_ppo_config(config_path)

    run_name = (
        str(args.run_name).strip()
        if args.run_name is not None and str(args.run_name).strip()
        else str(get_required(config, "defaults.run_name"))
    )
    output_root = resolve_repo_path(get_required(config, "defaults.output_root"))
    catalog_path = resolve_repo_path(get_required(config, "paths.catalog_path"))
    persona_config_path = resolve_repo_path(get_required(config, "paths.persona_config_path"))
    persona_bank_path = (
        resolve_repo_path(args.persona_bank_path)
        if args.persona_bank_path is not None
        else resolve_repo_path(get_required(config, "paths.persona_bank_path"))
    )
    persona_bank_split = (
        str(args.persona_bank_split)
        if args.persona_bank_split is not None
        else str(get_required(config, "eval.persona_bank_split"))
    )
    episodes = int(args.episodes) if args.episodes is not None else int(get_required(config, "eval.episodes"))
    deterministic = bool(args.deterministic) or bool(get_required(config, "eval.deterministic"))

    price_bin_count = int(get_required(config, "environment.price_bin_count"))
    price_step_usd = float(get_required(config, "environment.price_step_usd"))
    clip_enabled = bool(get_optional(config, "clip.enabled", False))
    clip_semantic_path = (
        resolve_repo_path(get_required(config, "clip.semantic_path")) if clip_enabled else None
    )
    clip_legacy_proxy_enabled = bool(get_optional(config, "clip.legacy_proxy_enabled", True))
    reward_scale_usd = float(get_required(config, "environment.reward_scale_usd"))
    no_deal_penalty = float(get_required(config, "environment.no_deal_penalty"))
    step_no_deal_penalty = float(get_optional(config, "environment.step_no_deal_penalty", 0.0))
    step_no_deal_penalty_start_round = int(
        get_optional(config, "environment.step_no_deal_penalty_start_round", 3)
    )
    profit_target_usd = float(get_required(config, "environment.profit_target_usd"))
    low_profit_penalty = float(get_required(config, "environment.low_profit_penalty"))
    no_deal_requires_positive_margin = bool(get_required(config, "environment.no_deal_requires_positive_margin"))
    invalid_accept_penalty = float(get_required(config, "environment.invalid_accept_penalty"))
    initial_offer_markup = float(get_required(config, "environment.initial_offer_markup"))

    if episodes <= 0:
        raise ValueError("`episodes` must be positive.")
    model_path = (
        Path(args.model_path)
        if args.model_path is not None
        else output_root / "checkpoints" / f"{run_name}_final.zip"
    )
    metrics_out = (
        Path(args.metrics_out)
        if args.metrics_out is not None
        else output_root / "reports" / f"{run_name}_eval_{persona_bank_split}.json"
    )
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    try:
        from sb3_contrib import RecurrentPPO
    except Exception as exc:  # pragma: no cover - dependency-dependent path
        raise RuntimeError(
            "Missing PPO dependencies. Install `stable-baselines3` and `sb3-contrib`."
        ) from exc

    env = PPOPricingEnv(
        catalog_path=str(catalog_path),
        persona_config_path=str(persona_config_path),
        persona_bank_path=str(persona_bank_path),
        persona_bank_split=persona_bank_split,
        price_bin_count=price_bin_count,
        price_step_usd=price_step_usd,
        clip_enabled=clip_enabled,
        clip_semantic_path=clip_semantic_path,
        clip_legacy_proxy_enabled=clip_legacy_proxy_enabled,
        reward_scale_usd=reward_scale_usd,
        no_deal_penalty=no_deal_penalty,
        step_no_deal_penalty=step_no_deal_penalty,
        step_no_deal_penalty_start_round=step_no_deal_penalty_start_round,
        profit_target_usd=profit_target_usd,
        low_profit_penalty=low_profit_penalty,
        no_deal_requires_positive_margin=no_deal_requires_positive_margin,
        invalid_accept_penalty=invalid_accept_penalty,
        initial_offer_markup=initial_offer_markup,
    )
    model = RecurrentPPO.load(path=str(model_path))

    deals = 0
    total_profit = 0.0
    total_rounds = 0
    walkaways = 0
    terminated_count = 0
    truncated_count = 0
    total_env_reward = 0.0
    total_shaped_reward = 0.0
    trace_len_total = 0

    for episode_idx in range(episodes):
        obs, _info = env.reset(seed=int(args.seed) + episode_idx)
        done = False
        lstm_states = None
        episode_start = True
        last_info: Dict[str, Any] = {}

        while not done:
            action, lstm_states = model.predict(
                observation=obs,
                state=lstm_states,
                episode_start=np.array([episode_start], dtype=bool),
                deterministic=deterministic,
            )
            obs, reward, terminated, truncated, step_info = env.step(action)
            episode_start = False
            done = bool(terminated or truncated)
            last_info = step_info
            total_shaped_reward += float(reward)
            total_env_reward += float(step_info.get("ppo_reward_raw_env", 0.0))
            terminated_count += int(terminated)
            truncated_count += int(truncated)

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
        "model_path": str(model_path),
        "episodes": episodes,
        "seed": int(args.seed),
        "persona_bank_split": persona_bank_split,
        "deal_rate": deals / denom,
        "avg_profit_usd": total_profit / denom,
        "avg_rounds": total_rounds / denom,
        "clip_enabled": bool(clip_enabled),
        "clip_semantic_path": None if clip_semantic_path is None else str(clip_semantic_path),
        "step_no_deal_penalty": float(step_no_deal_penalty),
        "step_no_deal_penalty_start_round": int(step_no_deal_penalty_start_round),
        "walkaway_rate": walkaways / denom,
        "terminated_rate": terminated_count / denom,
        "truncated_rate": truncated_count / denom,
        "avg_trace_len": trace_len_total / denom,
        "avg_env_reward": total_env_reward / denom,
        "avg_shaped_reward": total_shaped_reward / denom,
    }
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    metrics_out.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
