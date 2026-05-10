"""Build fixed evaluation subsets from persona-bank split files."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_PATH = ROOT / "datasets" / "persona_bank" / "bank50k_s123" / "test.jsonl"
DEFAULT_OUTPUT_PATH = ROOT / "datasets" / "persona_bank" / "bank50k_s123" / "llm_test_500.jsonl"
DEFAULT_METADATA_PATH = ROOT / "datasets" / "persona_bank" / "bank50k_s123" / "llm_test_500_metadata.json"


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments."""

    parser = argparse.ArgumentParser(description="Build a fixed persona-bank evaluation subset.")
    parser.add_argument("--source-path", type=Path, default=DEFAULT_SOURCE_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--metadata-path", type=Path, default=DEFAULT_METADATA_PATH)
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260501)
    parser.add_argument("--subset-name", type=str, default="llm_test_500")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Loads JSONL records."""

    if not path.exists():
        raise FileNotFoundError(f"Source split not found: {path}")
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
        raise ValueError(f"Source split is empty: {path}")
    return records


def _write_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    """Writes JSONL records with stable serialization."""

    lines = [json.dumps(record, separators=(",", ":")) for record in records]
    path.write_text("\n".join(lines) + "\n")


def _sha256(path: Path) -> str:
    """Computes SHA-256 digest for a file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_args(args: argparse.Namespace) -> None:
    """Validates command arguments."""

    if int(args.count) <= 0:
        raise ValueError("`count` must be positive.")
    if args.output_path.exists() and not bool(args.overwrite):
        raise FileExistsError(f"Output exists: {args.output_path}. Use `--overwrite` to replace it.")
    if args.metadata_path.exists() and not bool(args.overwrite):
        raise FileExistsError(f"Metadata exists: {args.metadata_path}. Use `--overwrite` to replace it.")


def main() -> None:
    """Builds the fixed subset and metadata."""

    args = parse_args()
    _validate_args(args)

    source_path = Path(args.source_path)
    output_path = Path(args.output_path)
    metadata_path = Path(args.metadata_path)
    records = _load_jsonl(source_path)
    count = int(args.count)
    if count > len(records):
        raise ValueError(f"`count` ({count}) cannot exceed source size ({len(records)}).")

    indices = list(range(len(records)))
    random.Random(int(args.seed)).shuffle(indices)
    selected_indices = sorted(indices[:count])
    subset = [records[idx] for idx in selected_indices]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_path, subset)
    metadata = {
        "schema_version": "eval_subset_v1",
        "subset_name": str(args.subset_name),
        "source_path": str(source_path),
        "source_sha256": _sha256(source_path),
        "source_count": len(records),
        "output_path": str(output_path),
        "output_sha256": _sha256(output_path),
        "count": count,
        "seed": int(args.seed),
        "selection": "shuffle_without_replacement_then_sort_indices",
        "selected_source_indices": selected_indices,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
