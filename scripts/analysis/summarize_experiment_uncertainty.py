"""Summarize exact metrics and bootstrap uncertainty for paper tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[2]


MAIN_METHODS = [
    {
        "key": "random",
        "display_name": "Random",
        "path": ROOT / "artifacts" / "heuristic" / "full_test7500" / "episodes.jsonl",
        "policy": "random",
    },
    {
        "key": "concession",
        "display_name": "Concession",
        "path": ROOT / "artifacts" / "heuristic" / "full_test7500" / "episodes.jsonl",
        "policy": "concession",
    },
    {
        "key": "deepseek_v4_flash",
        "display_name": "DeepSeek V4 Flash",
        "path": ROOT
        / "artifacts"
        / "llm"
        / "deepseek_v4_flash"
        / "prompt_v1"
        / "full_test7500"
        / "episodes.jsonl",
    },
    {
        "key": "kimi_k2_6",
        "display_name": "Kimi K2.6",
        "path": ROOT / "artifacts" / "llm" / "kimi_k2_6" / "prompt_v1" / "full_test7500" / "episodes.jsonl",
    },
    {
        "key": "qwen3_6_plus",
        "display_name": "Qwen3.6 Plus",
        "path": ROOT
        / "artifacts"
        / "llm"
        / "qwen3_6_plus"
        / "prompt_v1"
        / "full_test7500"
        / "episodes.jsonl",
    },
]


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260511)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "artifacts" / "analysis" / "main_results_uncertainty.json",
    )
    return parser.parse_args()


def load_rows(path: Path, *, policy: str | None = None) -> list[Mapping[str, Any]]:
    """Loads episode rows, optionally filtering a shared heuristic sidecar by policy."""

    rows: list[Mapping[str, Any]] = []
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            if policy is not None and row.get("policy") != policy:
                continue
            rows.append(row)
    return rows


def values_from_rows(rows: Iterable[Mapping[str, Any]]) -> Dict[str, np.ndarray]:
    """Extracts per-episode metric arrays."""

    deal: list[float] = []
    profit: list[float] = []
    rounds: list[float] = []
    invalid: list[float] = []
    for row in rows:
        metrics = row.get("metrics", {})
        deal.append(float(bool(metrics.get("deal_reached", False))))
        profit.append(float(metrics.get("profit_usd", 0.0)))
        rounds.append(float(metrics.get("rounds_used", 0)))
        invalid.append(float(bool(metrics.get("invalid_terminated", False))))
    return {
        "deal": np.asarray(deal, dtype=float),
        "profit": np.asarray(profit, dtype=float),
        "rounds": np.asarray(rounds, dtype=float),
        "invalid": np.asarray(invalid, dtype=float),
    }


def bootstrap_mean_ci(
    values: np.ndarray,
    *,
    samples: int,
    rng: np.random.Generator,
    batch_size: int = 500,
) -> list[float]:
    """Computes a percentile bootstrap CI for the sample mean."""

    if values.ndim != 1 or values.size == 0:
        raise ValueError("Bootstrap input must be a non-empty one-dimensional array.")
    n = int(values.size)
    estimates = np.empty(int(samples), dtype=float)
    start = 0
    while start < int(samples):
        stop = min(start + int(batch_size), int(samples))
        idx = rng.integers(0, n, size=(stop - start, n))
        estimates[start:stop] = values[idx].mean(axis=1)
        start = stop
    low, high = np.percentile(estimates, [2.5, 97.5])
    return [float(low), float(high)]


def summarize_method(
    spec: Mapping[str, Any],
    *,
    bootstrap_samples: int,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    """Summarizes one method from per-episode records."""

    rows = load_rows(Path(spec["path"]), policy=spec.get("policy"))
    arrays = values_from_rows(rows)
    n = int(arrays["profit"].size)
    if n == 0:
        raise ValueError(f"No rows loaded for method {spec['key']}.")
    deal_count = int(arrays["deal"].sum())
    total_profit = float(arrays["profit"].sum())
    return {
        "key": spec["key"],
        "display_name": spec["display_name"],
        "episodes": n,
        "deal_count": deal_count,
        "total_profit_usd": total_profit,
        "deal_rate": float(arrays["deal"].mean()),
        "deal_rate_ci95": bootstrap_mean_ci(arrays["deal"], samples=bootstrap_samples, rng=rng),
        "avg_profit_usd": float(arrays["profit"].mean()),
        "avg_profit_usd_ci95": bootstrap_mean_ci(arrays["profit"], samples=bootstrap_samples, rng=rng),
        "profit_per_deal_usd": float(total_profit / deal_count) if deal_count else 0.0,
        "avg_rounds": float(arrays["rounds"].mean()),
        "invalid_rate": float(arrays["invalid"].mean()),
    }


def main() -> None:
    """Writes uncertainty summaries for the paper."""

    args = parse_args()
    if int(args.bootstrap_samples) <= 0:
        raise ValueError("`--bootstrap-samples` must be positive.")
    rng = np.random.default_rng(int(args.seed))
    summaries = [
        summarize_method(spec, bootstrap_samples=int(args.bootstrap_samples), rng=rng) for spec in MAIN_METHODS
    ]
    payload = {
        "schema_version": "paper_main_results_uncertainty_v1",
        "bootstrap": {
            "samples": int(args.bootstrap_samples),
            "seed": int(args.seed),
            "interval": "percentile",
            "confidence": 0.95,
            "resampling_unit": "episode",
        },
        "methods": summaries,
    }
    args.out = args.out if args.out.is_absolute() else ROOT / args.out
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True))
    json.dump(payload, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
