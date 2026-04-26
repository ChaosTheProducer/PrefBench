"""Run unified policy benchmark on the same persona-bank consumer episodes.

This script evaluates heuristic policies and PPO on identical episode seeds so
that each policy negotiates with the same sampled consumers/configurations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
from typing import Any, Dict, List, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pricing_agent.baselines import ConcessionPolicy, RandomPolicy
from pricing_agent.ppo_env import PPOPricingEnv
from pricing_env.gym_wrapper import GYMNASIUM_AVAILABLE, PricingNegotiationGymWrapper
from scripts.common.ppo_config import DEFAULT_PPO_CONFIG_PATH, get_optional, get_required, load_ppo_config, resolve_repo_path


def parse_args() -> argparse.Namespace:
    """Parses benchmark command-line arguments.

    Returns:
        Parsed argument namespace.
    """

    parser = argparse.ArgumentParser(description="Benchmark heuristics and PPO on same persona episodes.")
    parser.add_argument(
        "--config-path",
        type=Path,
        default=DEFAULT_PPO_CONFIG_PATH,
        help="Unified PPO config YAML path.",
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
        "--model-path",
        type=Path,
        default=None,
        help="Optional explicit PPO checkpoint path.",
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
    parser.add_argument("--deterministic", action="store_true", help="Force deterministic PPO action inference.")
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
    allowed = {"random", "concession", "ppo"}
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
    args: argparse.Namespace,
    policy_seed: int,
) -> Dict[str, Any]:
    """Runs one heuristic policy benchmark.

    Args:
        policy_name: `random` or `concession`.
        episode_seeds: Shared episode seeds.
        args: Parsed arguments.
        policy_seed: Seed for policy stochasticity.

    Returns:
        Benchmark payload containing metrics and persona ids.
    """

    env = PricingNegotiationGymWrapper(
        catalog_path=args.catalog_path,
        persona_config_path=args.persona_config_path,
        persona_bank_path=args.persona_bank_path,
        persona_bank_split=args.persona_bank_split,
        clip_enabled=bool(args.clip_enabled),
        clip_semantic_path=args.clip_semantic_path,
        clip_legacy_proxy_enabled=bool(args.clip_legacy_proxy_enabled),
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


def _run_ppo_policy(
    *,
    episode_seeds: Sequence[int],
    args: argparse.Namespace,
    model_path: Path,
) -> Dict[str, Any]:
    """Runs PPO benchmark on shared episode seeds.

    Args:
        episode_seeds: Shared episode seeds.
        args: Parsed arguments.
        model_path: PPO model path.

    Returns:
        Benchmark payload containing metrics and persona ids.
    """

    try:
        from sb3_contrib import RecurrentPPO
    except Exception as exc:  # pragma: no cover - dependency-dependent path
        raise RuntimeError(
            "Missing PPO dependencies. Install `stable-baselines3` and `sb3-contrib`."
        ) from exc

    env = PPOPricingEnv(
        catalog_path=args.catalog_path,
        persona_config_path=args.persona_config_path,
        persona_bank_path=args.persona_bank_path,
        persona_bank_split=args.persona_bank_split,
        price_bin_count=int(args.price_bin_count),
        price_step_usd=float(args.price_step_usd),
        clip_enabled=bool(args.clip_enabled),
        clip_semantic_path=args.clip_semantic_path,
        clip_legacy_proxy_enabled=bool(args.clip_legacy_proxy_enabled),
        reward_scale_usd=float(args.reward_scale_usd),
        no_deal_penalty=float(args.no_deal_penalty),
        step_no_deal_penalty=float(args.step_no_deal_penalty),
        step_no_deal_penalty_start_round=int(args.step_no_deal_penalty_start_round),
        profit_target_usd=float(args.profit_target_usd),
        low_profit_penalty=float(args.low_profit_penalty),
        no_deal_requires_positive_margin=bool(args.no_deal_requires_positive_margin),
        invalid_accept_penalty=float(args.invalid_accept_penalty),
        initial_offer_markup=float(args.initial_offer_markup),
    )
    model = RecurrentPPO.load(path=str(model_path))

    acc = _new_metric()
    persona_ids: List[str] = []
    for ep_seed in episode_seeds:
        obs, info = env.reset(seed=int(ep_seed))
        persona_ids.append(str(info.get("persona_id", "")))

        done = False
        lstm_states = None
        episode_start = True
        total_shaped_reward = 0.0
        total_env_reward = 0.0
        last_info: Dict[str, Any] = info
        terminated = False
        truncated = False
        while not done:
            action, lstm_states = model.predict(
                observation=obs,
                state=lstm_states,
                episode_start=np.array([episode_start], dtype=bool),
                deterministic=bool(args.deterministic),
            )
            obs, reward, terminated, truncated, step_info = env.step(action)
            episode_start = False
            done = bool(terminated or truncated)
            total_shaped_reward += float(reward)
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
        acc["total_policy_reward"] += total_shaped_reward

    env.close()
    return {
        "policy": "ppo",
        "metrics": _finalize_metric(acc),
        "persona_ids": persona_ids,
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
    """Runs the unified benchmark and writes a report file."""

    if not GYMNASIUM_AVAILABLE:
        raise RuntimeError("Gymnasium is required for benchmark execution.")

    args = parse_args()
    config_path = resolve_repo_path(args.config_path)
    config = load_ppo_config(config_path)

    args.catalog_path = resolve_repo_path(get_required(config, "paths.catalog_path"))
    args.persona_config_path = resolve_repo_path(get_required(config, "paths.persona_config_path"))
    args.persona_bank_path = (
        resolve_repo_path(args.persona_bank_path)
        if args.persona_bank_path is not None
        else resolve_repo_path(get_required(config, "paths.persona_bank_path"))
    )
    args.persona_bank_split = (
        str(args.persona_bank_split)
        if args.persona_bank_split is not None
        else str(get_required(config, "benchmark.persona_bank_split"))
    )
    args.episodes = int(args.episodes) if args.episodes is not None else int(get_required(config, "benchmark.episodes"))
    config_policies = get_required(config, "benchmark.policies")
    if not isinstance(config_policies, list):
        raise ValueError("`benchmark.policies` must be a list in PPO config.")
    args.policies = (
        str(args.policies)
        if args.policies is not None
        else ",".join(str(token) for token in config_policies)
    )
    args.allow_consumer_mismatch = bool(args.allow_consumer_mismatch) or bool(
        get_required(config, "benchmark.allow_consumer_mismatch")
    )
    args.deterministic = bool(args.deterministic) or bool(get_required(config, "benchmark.deterministic"))

    args.price_bin_count = int(get_required(config, "environment.price_bin_count"))
    args.price_step_usd = float(get_required(config, "environment.price_step_usd"))
    args.clip_enabled = bool(get_optional(config, "clip.enabled", False))
    args.clip_semantic_path = (
        resolve_repo_path(get_required(config, "clip.semantic_path")) if args.clip_enabled else None
    )
    args.clip_legacy_proxy_enabled = bool(get_optional(config, "clip.legacy_proxy_enabled", True))
    args.reward_scale_usd = float(get_required(config, "environment.reward_scale_usd"))
    args.no_deal_penalty = float(get_required(config, "environment.no_deal_penalty"))
    args.step_no_deal_penalty = float(get_optional(config, "environment.step_no_deal_penalty", 0.0))
    args.step_no_deal_penalty_start_round = int(
        get_optional(config, "environment.step_no_deal_penalty_start_round", 3)
    )
    args.profit_target_usd = float(get_required(config, "environment.profit_target_usd"))
    args.low_profit_penalty = float(get_required(config, "environment.low_profit_penalty"))
    args.no_deal_requires_positive_margin = bool(
        get_required(config, "environment.no_deal_requires_positive_margin")
    )
    args.invalid_accept_penalty = float(get_required(config, "environment.invalid_accept_penalty"))
    args.initial_offer_markup = float(get_required(config, "environment.initial_offer_markup"))

    if int(args.episodes) <= 0:
        raise ValueError("`episodes` must be positive.")
    policies = _parse_policy_list(args.policies)

    run_name = (
        str(args.run_name).strip()
        if args.run_name is not None and str(args.run_name).strip()
        else str(get_required(config, "defaults.run_name"))
    )
    output_root = resolve_repo_path(get_required(config, "defaults.output_root"))
    model_path = (
        Path(args.model_path)
        if args.model_path is not None
        else output_root / "checkpoints" / f"{run_name}_final.zip"
    )
    report_out = (
        Path(args.report_out)
        if args.report_out is not None
        else output_root / "reports" / f"{run_name}_benchmark_{args.persona_bank_split}.json"
    )
    episode_seeds = [int(args.seed) + i for i in range(int(args.episodes))]

    policy_results: Dict[str, Dict[str, Any]] = {}
    for idx, policy_name in enumerate(policies):
        if policy_name in {"random", "concession"}:
            payload = _run_heuristic_policy(
                policy_name=policy_name,
                episode_seeds=episode_seeds,
                args=args,
                policy_seed=int(args.seed) + 10_000 + idx,
            )
        else:
            if not model_path.exists():
                raise FileNotFoundError(
                    f"PPO model not found at `{model_path}`. Train PPO first or pass `--model-path`."
                )
            payload = _run_ppo_policy(
                episode_seeds=episode_seeds,
                args=args,
                model_path=model_path,
            )
        policy_results[policy_name] = payload

    same_consumers = True
    if len(policy_results) >= 2 and not bool(args.allow_consumer_mismatch):
        _assert_same_consumers(policy_results)
    elif len(policy_results) >= 2:
        try:
            _assert_same_consumers(policy_results)
        except RuntimeError:
            same_consumers = False

    report = {
        "run_name": run_name,
        "config_path": str(config_path),
        "seed": int(args.seed),
        "episodes": int(args.episodes),
        "episode_seed_range": [episode_seeds[0], episode_seeds[-1]],
        "persona_bank_path": str(args.persona_bank_path),
        "persona_bank_split": str(args.persona_bank_split),
        "clip_enabled": bool(args.clip_enabled),
        "clip_semantic_path": None if args.clip_semantic_path is None else str(args.clip_semantic_path),
        "step_no_deal_penalty": float(args.step_no_deal_penalty),
        "step_no_deal_penalty_start_round": int(args.step_no_deal_penalty_start_round),
        "policies": policies,
        "same_consumers_verified": same_consumers,
        "results": {name: payload["metrics"] for name, payload in policy_results.items()},
    }
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
