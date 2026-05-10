"""Summarize the frozen PrefBench persona bank for simulator sanity checks."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from math import sqrt
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PERSONA_BANK_PATH = ROOT / "datasets" / "persona_bank" / "bank50k_s123" / "persona_bank.jsonl"
DEFAULT_OUTPUT_PATH = ROOT / "artifacts" / "simulator" / "sanity_summary.json"

OBSERVABLE_FIELDS = (
    "age_band",
    "income_band",
    "household_stage",
    "ownership_stage",
    "primary_use_case",
)
CATEGORICAL_HIDDEN_FIELDS = (
    "decision_style",
    "tech_affinity_band",
)
NUMERIC_HIDDEN_FIELDS = (
    "reservation_price_base",
    "price_sensitivity",
    "aesthetic_sensitivity",
    "patience",
    "counter_strength",
    "walkaway_threshold",
    "belief_obscurity",
    "brand_loyalty",
    "impulsivity",
)


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments."""

    parser = argparse.ArgumentParser(description="Summarize the PrefBench persona bank.")
    parser.add_argument("--persona-bank-path", type=Path, default=DEFAULT_PERSONA_BANK_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Loads persona-bank records from JSONL."""

    if not path.exists():
        raise FileNotFoundError(f"Persona bank not found: {path}")
    records: List[Dict[str, Any]] = []
    for line_idx, line in enumerate(path.read_text().splitlines(), start=1):
        raw = line.strip()
        if not raw:
            continue
        record = json.loads(raw)
        if not isinstance(record, dict):
            raise ValueError(f"Line {line_idx} is not a JSON object.")
        records.append(record)
    if not records:
        raise ValueError(f"Persona bank is empty: {path}")
    return records


def _counter_summary(counter: Counter[str], *, total: int) -> Dict[str, Dict[str, float | int]]:
    """Converts counts into count/proportion records."""

    return {
        key: {
            "count": int(count),
            "proportion": float(count) / float(total),
        }
        for key, count in sorted(counter.items())
    }


def _numeric_summary(values: Iterable[float]) -> Dict[str, float | int]:
    """Computes compact numeric summary statistics."""

    series = [float(value) for value in values]
    count = len(series)
    if count == 0:
        raise ValueError("Cannot summarize an empty numeric series.")
    mean = sum(series) / float(count)
    variance = sum((value - mean) ** 2 for value in series) / float(count)
    ordered = sorted(series)
    return {
        "count": count,
        "min": ordered[0],
        "mean": mean,
        "std": sqrt(variance),
        "max": ordered[-1],
    }


def _price_sensitivity_bucket(value: float) -> str:
    """Maps price sensitivity into coarse sanity-check buckets."""

    scalar = float(value)
    if scalar < 0.85:
        return "low"
    if scalar <= 1.15:
        return "medium"
    return "high"


def _split_counts(records: List[Mapping[str, Any]]) -> Dict[str, int]:
    """Counts records by split."""

    counter = Counter(str(record["split"]) for record in records)
    return {key: int(counter[key]) for key in sorted(counter)}


def _observable_distributions(records: List[Mapping[str, Any]]) -> Dict[str, Any]:
    """Summarizes observable persona fields."""

    total = len(records)
    output: Dict[str, Any] = {}
    for field in OBSERVABLE_FIELDS:
        counter = Counter(str(record["observable"][field]) for record in records)
        output[field] = _counter_summary(counter, total=total)
    return output


def _hidden_distributions(records: List[Mapping[str, Any]]) -> Dict[str, Any]:
    """Summarizes hidden persona fields."""

    total = len(records)
    categorical: Dict[str, Any] = {}
    for field in CATEGORICAL_HIDDEN_FIELDS:
        counter = Counter(str(record["hidden"][field]) for record in records)
        categorical[field] = _counter_summary(counter, total=total)

    priority_counter = Counter("|".join(str(item) for item in record["hidden"]["stated_priority_top2"]) for record in records)
    categorical["stated_priority_top2"] = _counter_summary(priority_counter, total=total)

    numeric = {
        field: _numeric_summary(float(record["hidden"][field]) for record in records)
        for field in NUMERIC_HIDDEN_FIELDS
    }
    return {
        "categorical": categorical,
        "numeric": numeric,
    }


def _reservation_price_by_income(records: List[Mapping[str, Any]]) -> Dict[str, Any]:
    """Summarizes reservation price conditional on observable income band."""

    by_income: Dict[str, List[float]] = defaultdict(list)
    for record in records:
        income = str(record["observable"]["income_band"])
        by_income[income].append(float(record["hidden"]["reservation_price_base"]))
    return {
        income: _numeric_summary(values)
        for income, values in sorted(by_income.items())
    }


def _price_sensitivity_buckets(records: List[Mapping[str, Any]]) -> Dict[str, Any]:
    """Summarizes coarse price-sensitivity buckets."""

    total = len(records)
    counter = Counter(_price_sensitivity_bucket(float(record["hidden"]["price_sensitivity"])) for record in records)
    return {
        "bucket_definitions": {
            "low": "price_sensitivity < 0.85",
            "medium": "0.85 <= price_sensitivity <= 1.15",
            "high": "price_sensitivity > 1.15",
        },
        "distribution": _counter_summary(counter, total=total),
    }


def build_summary(*, records: List[Dict[str, Any]], persona_bank_path: Path) -> Dict[str, Any]:
    """Builds the simulator sanity summary payload."""

    return {
        "schema_version": "simulator_sanity_summary_v1",
        "persona_bank_path": str(persona_bank_path),
        "count_total": len(records),
        "counts_by_split": _split_counts(records),
        "observable_distributions": _observable_distributions(records),
        "hidden_distributions": _hidden_distributions(records),
        "reservation_price_by_income": _reservation_price_by_income(records),
        "price_sensitivity_buckets": _price_sensitivity_buckets(records),
    }


def main() -> None:
    """Writes the persona-bank sanity summary."""

    args = parse_args()
    persona_bank_path = Path(args.persona_bank_path)
    output_path = Path(args.output)
    records = _load_jsonl(persona_bank_path)
    summary = build_summary(records=records, persona_bank_path=persona_bank_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
