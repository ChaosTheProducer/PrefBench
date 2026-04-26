"""Builds deterministic persona-bank JSONL files from `persona_v2` config."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
from typing import Any, Dict, List, Sequence, Tuple

import yaml

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pricing_env.persona import PersonaSampler


DEFAULT_OUTPUT_DIR = ROOT / "datasets" / "persona_bank"
DEFAULT_PERSONA_CONFIG = ROOT / "configs" / "personas_v2.yaml"


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments.

    Returns:
        Parsed argument namespace.
    """

    parser = argparse.ArgumentParser(description="Build deterministic persona-bank JSONL files.")
    parser.add_argument(
        "--persona-config-path",
        type=Path,
        default=DEFAULT_PERSONA_CONFIG,
        help="Path to `persona_v2` config YAML.",
    )
    parser.add_argument("--count", type=int, required=True, help="Total persona records to generate.")
    parser.add_argument("--seed", type=int, default=None, help="Generation seed. Defaults to config seed.")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Train split ratio.")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation split ratio.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for persona-bank files.",
    )
    parser.add_argument(
        "--source-tag",
        type=str,
        default="distribution_hidden_mapping_v1",
        help="Value for record `source` field.",
    )
    parser.add_argument(
        "--split-method",
        type=str,
        default="stratified",
        choices=["stratified", "random"],
        help=(
            "Split strategy. `stratified` keeps split distributions aligned by "
            "observable strata; `random` keeps legacy shuffle slicing."
        ),
    )
    parser.add_argument(
        "--stratify-fields",
        type=str,
        default="age_band,income_band",
        help=(
            "Comma-separated observable fields used for stratified splitting. "
            "Only used when `--split-method stratified`."
        ),
    )
    parser.add_argument(
        "--preview-count",
        type=int,
        default=50,
        help="Number of records to export into a readable preview JSON file.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files.")
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    """Validates build arguments.

    Args:
        args: Parsed argument namespace.
    """

    if int(args.count) <= 0:
        raise ValueError("`count` must be positive.")
    if not (0.0 < float(args.train_ratio) < 1.0):
        raise ValueError("`train_ratio` must be in (0, 1).")
    if not (0.0 < float(args.val_ratio) < 1.0):
        raise ValueError("`val_ratio` must be in (0, 1).")
    if float(args.train_ratio) + float(args.val_ratio) >= 1.0:
        raise ValueError("`train_ratio + val_ratio` must be < 1.0.")
    if int(args.preview_count) <= 0:
        raise ValueError("`preview_count` must be positive.")


def _load_persona_config(path: Path) -> Dict[str, Any]:
    """Loads and validates persona config.

    Args:
        path: Persona config path.

    Returns:
        Parsed config dictionary.
    """

    if not path.exists():
        raise FileNotFoundError(f"Persona config not found: {path}")
    payload = yaml.safe_load(path.read_text())
    schema = str(payload.get("schema_version", "")).strip().lower()
    if schema not in {"persona_v2", "v2"}:
        raise ValueError("`build_persona_bank.py` requires `schema_version: persona_v2`.")
    return payload


def _stable_record(*, idx: int, split: str, seed: int, source: str, profile: Any) -> Dict[str, Any]:
    """Builds one stable persona-bank record.

    Args:
        idx: Deterministic record index.
        split: Dataset split label.
        seed: Global generation seed.
        source: Source tag for traceability.
        profile: Sampled `PersonaProfile`.

    Returns:
        JSON-serializable persona record.
    """

    persona_id = f"pb_s{seed}_{idx:06d}"
    return {
        "schema_version": "persona_v2",
        "persona_id": persona_id,
        "split": split,
        "source": source,
        "seed": seed,
        "observable": {
            "age_band": str(profile.age_band),
            "income_band": str(profile.income_band),
            "household_stage": str(profile.household_stage),
            "ownership_stage": str(profile.ownership_stage),
            "primary_use_case": str(profile.primary_use_case),
        },
        "hidden": {
            "decision_style": str(profile.decision_style),
            "tech_affinity_band": str(profile.tech_affinity_band),
            "stated_priority_top2": [str(profile.stated_priority_top2[0]), str(profile.stated_priority_top2[1])],
            "reservation_price_base": float(profile.reservation_price_base),
            "price_sensitivity": float(profile.price_sensitivity),
            "aesthetic_sensitivity": float(profile.aesthetic_sensitivity),
            "patience": int(profile.patience),
            "counter_strength": float(profile.counter_strength),
            "walkaway_threshold": float(profile.walkaway_threshold),
            "belief_obscurity": float(profile.belief_obscurity),
            "brand_loyalty": float(profile.brand_loyalty),
            "impulsivity": float(profile.impulsivity),
            "feature_weight_vector": {
                "safety": float(profile.feature_weight_vector["safety"]),
                "comfort": float(profile.feature_weight_vector["comfort"]),
                "performance": float(profile.feature_weight_vector["performance"]),
                "tech": float(profile.feature_weight_vector["tech"]),
                "aesthetics": float(profile.feature_weight_vector["aesthetics"]),
            },
        },
    }


def _split_indices(*, count: int, seed: int, train_ratio: float, val_ratio: float) -> Dict[int, str]:
    """Creates deterministic split assignments.

    Args:
        count: Total record count.
        seed: Global seed.
        train_ratio: Train split ratio.
        val_ratio: Validation split ratio.

    Returns:
        Mapping from record index to split label.
    """

    indices = list(range(count))
    random.Random(seed + 1).shuffle(indices)

    n_train = int(count * train_ratio)
    n_val = int(count * val_ratio)
    n_test = count - n_train - n_val
    if n_train <= 0 or n_val <= 0 or n_test <= 0:
        raise ValueError(
            "Invalid split sizes; increase count or adjust ratios so train/val/test are all non-empty."
        )

    split_by_index: Dict[int, str] = {}
    for index in indices[:n_train]:
        split_by_index[index] = "train"
    for index in indices[n_train : n_train + n_val]:
        split_by_index[index] = "val"
    for index in indices[n_train + n_val :]:
        split_by_index[index] = "test"
    return split_by_index


def _split_sizes(*, count: int, train_ratio: float, val_ratio: float) -> Tuple[int, int, int]:
    """Computes train/val/test target sizes.

    Args:
        count: Total number of records.
        train_ratio: Train split ratio.
        val_ratio: Validation split ratio.

    Returns:
        A tuple of `(n_train, n_val, n_test)`.
    """

    n_train = int(count * train_ratio)
    n_val = int(count * val_ratio)
    n_test = count - n_train - n_val
    if n_train <= 0 or n_val <= 0 or n_test <= 0:
        raise ValueError(
            "Invalid split sizes; increase count or adjust ratios so train/val/test are all non-empty."
        )
    return n_train, n_val, n_test


def _parse_stratify_fields(raw: str) -> List[str]:
    """Parses and validates stratification fields.

    Args:
        raw: Comma-separated field names.

    Returns:
        Normalized list of field names.
    """

    fields = [item.strip() for item in str(raw).split(",") if item.strip()]
    if not fields:
        raise ValueError("`stratify_fields` cannot be empty for stratified splitting.")
    return fields


def _stable_key_seed(*, seed: int, key: Tuple[str, ...]) -> int:
    """Creates a deterministic integer seed from a key tuple.

    Args:
        seed: Global seed.
        key: Stratum key tuple.

    Returns:
        Deterministic seed integer.
    """

    payload = f"{seed}::{'|'.join(key)}".encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:16], 16)


def _compute_split_allocation_for_groups(
    *,
    group_sizes: Dict[Tuple[str, ...], int],
    n_target: int,
    ratio: float,
    capacity_by_group: Dict[Tuple[str, ...], int] | None = None,
) -> Dict[Tuple[str, ...], int]:
    """Allocates an exact split count per stratum with largest-remainder rounding.

    Args:
        group_sizes: Number of records in each stratum.
        n_target: Global target size for this split.
        ratio: Ratio used to compute per-stratum soft targets.
        capacity_by_group: Optional per-stratum maximum allocatable count.

    Returns:
        Per-stratum allocation counts that sum to `n_target`.
    """

    keys = sorted(group_sizes.keys())
    allocations: Dict[Tuple[str, ...], int] = {}
    remainders: Dict[Tuple[str, ...], float] = {}
    capacities: Dict[Tuple[str, ...], int] = {}

    for key in keys:
        size = int(group_sizes[key])
        cap = int(capacity_by_group[key]) if capacity_by_group is not None else size
        if cap < 0:
            raise ValueError("Invalid negative per-group capacity during stratified allocation.")
        capacities[key] = cap
        raw_target = float(size) * float(ratio)
        base = int(raw_target)
        base = min(base, cap)
        allocations[key] = base
        remainders[key] = raw_target - float(base)

    current = int(sum(allocations.values()))
    need = int(n_target - current)
    if need < 0:
        raise ValueError("Base allocation exceeded target size during stratified split.")

    if need > 0:
        ranked_keys = sorted(keys, key=lambda key: (remainders[key], str(key)), reverse=True)
        for key in ranked_keys:
            if need <= 0:
                break
            available = int(capacities[key] - allocations[key])
            if available <= 0:
                continue
            add = min(available, need)
            allocations[key] += add
            need -= add

    if need != 0:
        raise ValueError("Failed to satisfy exact target size during stratified split allocation.")
    return allocations


def _observable_value(*, profile: Any, field: str) -> str:
    """Reads one observable field value from a sampled profile.

    Args:
        profile: Sampled `PersonaProfile`.
        field: Observable field name.

    Returns:
        Stringified observable value.
    """

    if not hasattr(profile, field):
        raise ValueError(f"Unknown stratify field: `{field}`")
    value = getattr(profile, field)
    return str(value)


def _split_indices_stratified(
    *,
    profiles: Sequence[Any],
    seed: int,
    train_ratio: float,
    val_ratio: float,
    stratify_fields: Sequence[str],
) -> Dict[int, str]:
    """Creates deterministic stratified split assignments from sampled profiles.

    Args:
        profiles: Sequence of sampled persona profiles.
        seed: Global seed.
        train_ratio: Train split ratio.
        val_ratio: Validation split ratio.
        stratify_fields: Observable fields used to define strata.

    Returns:
        Mapping from record index to split label.
    """

    count = int(len(profiles))
    n_train, n_val, _n_test = _split_sizes(count=count, train_ratio=train_ratio, val_ratio=val_ratio)

    groups: Dict[Tuple[str, ...], List[int]] = {}
    for idx, profile in enumerate(profiles):
        key = tuple(_observable_value(profile=profile, field=field) for field in stratify_fields)
        groups.setdefault(key, []).append(idx)

    # Deterministic within-stratum shuffle.
    for key in groups:
        local_rng = random.Random(_stable_key_seed(seed=seed, key=key))
        local_rng.shuffle(groups[key])

    group_sizes = {key: len(indices) for key, indices in groups.items()}
    train_alloc = _compute_split_allocation_for_groups(
        group_sizes=group_sizes,
        n_target=n_train,
        ratio=train_ratio,
    )
    val_capacity = {key: group_sizes[key] - train_alloc[key] for key in groups}
    val_alloc = _compute_split_allocation_for_groups(
        group_sizes=group_sizes,
        n_target=n_val,
        ratio=val_ratio,
        capacity_by_group=val_capacity,
    )

    split_by_index: Dict[int, str] = {}
    for key, indices in groups.items():
        n_t = int(train_alloc[key])
        n_v = int(val_alloc[key])
        for idx in indices[:n_t]:
            split_by_index[idx] = "train"
        for idx in indices[n_t : n_t + n_v]:
            split_by_index[idx] = "val"
        for idx in indices[n_t + n_v :]:
            split_by_index[idx] = "test"

    if len(split_by_index) != count:
        raise ValueError("Stratified split assignment did not cover all records.")

    counts = {"train": 0, "val": 0, "test": 0}
    for split in split_by_index.values():
        counts[split] += 1
    if counts["train"] != n_train or counts["val"] != n_val or counts["test"] != count - n_train - n_val:
        raise ValueError("Stratified split assignment size mismatch.")

    return split_by_index


def _write_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    """Writes JSONL records with stable serialization.

    Args:
        path: Output file path.
        records: JSON-serializable record list.
    """

    lines = [json.dumps(record, separators=(",", ":")) for record in records]
    path.write_text("\n".join(lines) + "\n")


def _write_pretty_json(path: Path, records: List[Dict[str, Any]]) -> None:
    """Writes an indented JSON file for human inspection.

    Args:
        path: Output file path.
        records: JSON-serializable record list.
    """

    path.write_text(json.dumps(records, indent=2))


def _sha256(path: Path) -> str:
    """Computes SHA-256 digest for a file.

    Args:
        path: File path.

    Returns:
        Hex-encoded digest.
    """

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    """Builds deterministic persona-bank files and prints build summary."""

    args = parse_args()
    _validate_args(args)

    persona_config_path = Path(args.persona_config_path)
    config = _load_persona_config(persona_config_path)
    seed = int(args.seed) if args.seed is not None else int(config["seed"])

    sampler = PersonaSampler(config=config, config_path=persona_config_path)
    rng = random.Random(seed)

    count = int(args.count)
    if int(args.preview_count) > count:
        raise ValueError("`preview_count` cannot be larger than `count`.")

    profiles = [sampler.sample(rng) for _ in range(count)]
    split_method = str(args.split_method).strip().lower()
    if split_method == "stratified":
        stratify_fields = _parse_stratify_fields(str(args.stratify_fields))
        split_by_index = _split_indices_stratified(
            profiles=profiles,
            seed=seed,
            train_ratio=float(args.train_ratio),
            val_ratio=float(args.val_ratio),
            stratify_fields=stratify_fields,
        )
    else:
        stratify_fields = []
        split_by_index = _split_indices(
            count=count,
            seed=seed,
            train_ratio=float(args.train_ratio),
            val_ratio=float(args.val_ratio),
        )

    records: List[Dict[str, Any]] = []
    for idx, profile in enumerate(profiles):
        split = split_by_index[idx]
        records.append(
            _stable_record(
                idx=idx,
                split=split,
                seed=seed,
                source=str(args.source_tag).strip() or "distribution_hidden_mapping_v1",
                profile=profile,
            )
        )

    by_split: Dict[str, List[Dict[str, Any]]] = {"train": [], "val": [], "test": []}
    for record in records:
        by_split[str(record["split"])].append(record)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    combined_path = output_dir / "persona_bank.jsonl"
    split_paths = {
        "train": output_dir / "train.jsonl",
        "val": output_dir / "val.jsonl",
        "test": output_dir / "test.jsonl",
    }
    manifest_path = output_dir / "manifest.json"
    preview_path = output_dir / "persona_bank_preview.json"

    targets = [
        combined_path,
        split_paths["train"],
        split_paths["val"],
        split_paths["test"],
        preview_path,
        manifest_path,
    ]
    if not bool(args.overwrite):
        existing = [path for path in targets if path.exists()]
        if existing:
            raise FileExistsError(
                "Output files already exist. Use `--overwrite` to replace: "
                + ", ".join(str(path) for path in existing)
            )

    _write_jsonl(combined_path, records)
    _write_jsonl(split_paths["train"], by_split["train"])
    _write_jsonl(split_paths["val"], by_split["val"])
    _write_jsonl(split_paths["test"], by_split["test"])
    _write_pretty_json(preview_path, records[: int(args.preview_count)])

    manifest = {
        "status": "ok",
        "schema_version": "persona_v2",
        "seed": seed,
        "count_total": count,
        "counts_by_split": {key: len(value) for key, value in by_split.items()},
        "train_ratio": float(args.train_ratio),
        "val_ratio": float(args.val_ratio),
        "test_ratio": float(1.0 - float(args.train_ratio) - float(args.val_ratio)),
        "persona_config_path": str(persona_config_path.resolve()),
        "source_tag": str(args.source_tag),
        "split_method": split_method,
        "stratify_fields": stratify_fields,
        "output_dir": str(output_dir.resolve()),
        "files": {
            "combined": {
                "path": str(combined_path.resolve()),
                "sha256": _sha256(combined_path),
            },
            "train": {
                "path": str(split_paths["train"].resolve()),
                "sha256": _sha256(split_paths["train"]),
            },
            "val": {
                "path": str(split_paths["val"].resolve()),
                "sha256": _sha256(split_paths["val"]),
            },
            "test": {
                "path": str(split_paths["test"].resolve()),
                "sha256": _sha256(split_paths["test"]),
            },
            "preview": {
                "path": str(preview_path.resolve()),
                "sha256": _sha256(preview_path),
                "count": int(args.preview_count),
            },
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
