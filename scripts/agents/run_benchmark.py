"""Run checkpoint-free heuristic baselines on shared benchmark episodes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
from typing import Any, Dict, List, Mapping, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pricing_agent.baselines import ConcessionPolicy, RandomPolicy
from pricing_env.negotiation_env import NegotiationEnv


DEFAULT_CATALOG_PATH = ROOT / "catalog" / "e350_core_catalog.yaml"
DEFAULT_PERSONA_CONFIG_PATH = ROOT / "configs" / "personas_v2.yaml"
DEFAULT_PERSONA_BANK_PATH = ROOT / "datasets" / "persona_bank" / "bank50k_s123" / "test.jsonl"
DEFAULT_REPORT_OUT = ROOT / "artifacts" / "heuristic" / "benchmark_test.json"


def parse_args() -> argparse.Namespace:
    """Parses benchmark command-line arguments."""

    parser = argparse.ArgumentParser(description="Benchmark checkpoint-free heuristic pricing policies.")
    parser.add_argument("--catalog-path", type=Path, default=DEFAULT_CATALOG_PATH)
    parser.add_argument("--persona-config-path", type=Path, default=DEFAULT_PERSONA_CONFIG_PATH)
    parser.add_argument("--persona-bank-path", type=Path, default=DEFAULT_PERSONA_BANK_PATH)
    parser.add_argument("--persona-bank-split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--policies", type=str, default="random,concession")
    parser.add_argument("--run-name", type=str, default="heuristic_test")
    parser.add_argument("--report-out", type=Path, default=DEFAULT_REPORT_OUT)
    parser.add_argument("--episodes-out", type=Path, default=None)
    return parser.parse_args()


def _resolve_repo_path(path: Path) -> Path:
    """Resolves a path relative to the repository root."""

    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def _parse_policy_list(raw: str) -> List[str]:
    """Parses and validates comma-separated policy names."""

    tokens = [token.strip().lower() for token in raw.split(",") if token.strip()]
    allowed = {"random", "concession"}
    invalid = [token for token in tokens if token not in allowed]
    if invalid:
        raise ValueError(f"Unsupported policies: {invalid}. Allowed: {sorted(allowed)}")
    if not tokens:
        raise ValueError("At least one policy must be selected.")
    return tokens


def _offer_bounds(env: NegotiationEnv) -> Tuple[float, float]:
    """Computes stable offer bounds from catalog option prices."""

    grouped = env.catalog.by_dimension()
    min_msrp = sum(min(option.price_delta_usd for option in options) for options in grouped.values())
    max_msrp = sum(max(option.price_delta_usd for option in options) for options in grouped.values())
    offer_min = max(100.0, float(min_msrp) * 0.4)
    offer_max = max(offer_min + 500.0, float(max_msrp) * 3.0, 60000.0)
    return float(offer_min), float(offer_max)


def _new_metric() -> Dict[str, float]:
    """Creates one zero-initialized metric accumulator."""

    return {
        "episodes": 0.0,
        "deal_count": 0.0,
        "total_profit_usd": 0.0,
        "total_rounds": 0.0,
        "walkaway_count": 0.0,
        "total_trace_len": 0.0,
        "total_env_reward": 0.0,
    }


def _finalize_metric(acc: Mapping[str, float]) -> Dict[str, float | int]:
    """Converts metric accumulator into report-friendly metrics."""

    episodes = int(acc["episodes"])
    denom = float(max(1, episodes))
    return {
        "episodes": episodes,
        "deal_rate": float(acc["deal_count"]) / denom,
        "avg_profit_usd": float(acc["total_profit_usd"]) / denom,
        "avg_rounds": float(acc["total_rounds"]) / denom,
        "walkaway_rate": float(acc["walkaway_count"]) / denom,
        "avg_trace_len": float(acc["total_trace_len"]) / denom,
        "avg_env_reward": float(acc["total_env_reward"]) / denom,
    }


def _episode_key(info: Mapping[str, Any]) -> Dict[str, Any]:
    """Extracts the fixed episode identity used for cross-policy checks."""

    return {
        "persona_id": str(info.get("persona_id", "")),
        "selected_option_keys": list(info.get("initial_observation", {}).get("selected_option_keys", [])),
    }


def _build_policy(policy_name: str, *, seed: int, offer_min: float, offer_max: float) -> Any:
    """Creates one heuristic policy."""

    rng = random.Random(seed)
    if policy_name == "random":
        return RandomPolicy(rng=rng, offer_min=offer_min, offer_max=offer_max)
    if policy_name == "concession":
        return ConcessionPolicy(rng=rng, offer_min=offer_min, offer_max=offer_max)
    raise ValueError(f"Unsupported policy: {policy_name}")


def _run_policy(
    *,
    policy_name: str,
    episode_seeds: Sequence[int],
    args: argparse.Namespace,
    policy_seed: int,
) -> Dict[str, Any]:
    """Runs one heuristic policy on the shared episode stream."""

    env = NegotiationEnv(
        catalog_path=args.catalog_path,
        persona_config_path=args.persona_config_path,
        persona_bank_path=args.persona_bank_path,
        persona_bank_split=args.persona_bank_split,
    )
    offer_min, offer_max = _offer_bounds(env)
    policy = _build_policy(policy_name, seed=policy_seed, offer_min=offer_min, offer_max=offer_max)

    acc = _new_metric()
    episode_keys: List[Dict[str, Any]] = []
    episode_results: List[Dict[str, Any]] = []
    for ep_seed in episode_seeds:
        env.rng.seed(int(ep_seed))
        observation = env.reset()
        initial_info = dict(env.current_persona_metadata())
        initial_info["initial_observation"] = dict(observation)
        episode_keys.append(_episode_key(initial_info))

        done = False
        total_reward = 0.0
        info: Dict[str, Any] = {}
        while not done:
            action = policy.act(observation)
            observation, reward, done, info = env.step(action)
            total_reward += float(reward)

        metrics = info.get("episode_metrics", {})
        acc["episodes"] += 1.0
        acc["deal_count"] += float(bool(metrics.get("deal_reached", False)))
        acc["total_profit_usd"] += float(metrics.get("profit_usd", 0.0))
        acc["total_rounds"] += float(metrics.get("rounds_used", 0))
        acc["walkaway_count"] += float(bool(metrics.get("walkaway", False)))
        acc["total_trace_len"] += float(info.get("trace_len", 0))
        acc["total_env_reward"] += total_reward
        episode_results.append(
            {
                "policy": policy_name,
                "episode_idx": len(episode_results),
                "episode_seed": int(ep_seed),
                "persona": {
                    "persona_id": str(initial_info.get("persona_id", "")),
                    "persona_source": str(initial_info.get("persona_source", "")),
                    "persona_split": str(initial_info.get("persona_split", "")),
                },
                "selected_option_keys": list(initial_info["initial_observation"].get("selected_option_keys", [])),
                "metrics": metrics,
                "total_env_reward": float(total_reward),
                "trace_len": int(info.get("trace_len", 0)),
            }
        )

    return {
        "policy": policy_name,
        "metrics": _finalize_metric(acc),
        "episode_keys": episode_keys,
        "episodes": episode_results,
    }


def _assert_same_episodes(policy_results: Mapping[str, Dict[str, Any]]) -> bool:
    """Checks that policies saw the same persona/configuration sequence."""

    ordered_names = list(policy_results.keys())
    if len(ordered_names) < 2:
        return True
    base_name = ordered_names[0]
    base_keys = policy_results[base_name]["episode_keys"]
    for name in ordered_names[1:]:
        keys = policy_results[name]["episode_keys"]
        if keys != base_keys:
            mismatch_idx = next((idx for idx, (left, right) in enumerate(zip(base_keys, keys)) if left != right), None)
            raise RuntimeError(
                f"Episode mismatch between `{base_name}` and `{name}` at episode index {mismatch_idx}."
            )
    return True


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Writes records as JSONL."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def main() -> None:
    """Runs the benchmark and writes a JSON report."""

    args = parse_args()
    args.catalog_path = _resolve_repo_path(args.catalog_path)
    args.persona_config_path = _resolve_repo_path(args.persona_config_path)
    args.persona_bank_path = _resolve_repo_path(args.persona_bank_path)
    args.report_out = _resolve_repo_path(args.report_out)
    args.episodes_out = _resolve_repo_path(args.episodes_out) if args.episodes_out is not None else None

    if int(args.episodes) <= 0:
        raise ValueError("`episodes` must be positive.")
    policies = _parse_policy_list(str(args.policies))
    episode_seeds = [int(args.seed) + idx for idx in range(int(args.episodes))]

    policy_results: Dict[str, Dict[str, Any]] = {}
    for idx, policy_name in enumerate(policies):
        policy_results[policy_name] = _run_policy(
            policy_name=policy_name,
            episode_seeds=episode_seeds,
            args=args,
            policy_seed=int(args.seed) + 10_000 + idx,
        )

    same_episodes = _assert_same_episodes(policy_results)
    report = {
        "schema_version": "heuristic_benchmark_v1",
        "run_name": str(args.run_name),
        "seed": int(args.seed),
        "episodes": int(args.episodes),
        "episode_seed_range": [episode_seeds[0], episode_seeds[-1]],
        "catalog_path": str(args.catalog_path),
        "persona_config_path": str(args.persona_config_path),
        "persona_bank_path": str(args.persona_bank_path),
        "persona_bank_split": str(args.persona_bank_split),
        "policies": policies,
        "same_episodes_verified": bool(same_episodes),
        "episodes_out": str(args.episodes_out) if args.episodes_out is not None else None,
        "results": {name: payload["metrics"] for name, payload in policy_results.items()},
    }
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2, sort_keys=True))
    if args.episodes_out is not None:
        rows = [row for name in policies for row in policy_results[name]["episodes"]]
        _write_jsonl(args.episodes_out, rows)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
