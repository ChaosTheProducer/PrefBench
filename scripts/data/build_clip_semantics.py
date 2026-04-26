"""Builds offline text-only CLIP semantics for E350 customization options."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import open_clip
import torch
import yaml


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CATALOG_PATH = ROOT / "catalog" / "e350_core_catalog.yaml"
DEFAULT_OUTPUT_PATH = ROOT / "datasets" / "clip_semantics" / "e350_clip_text_v1.json"

AXIS_PROMPTS: Dict[str, str] = {
    "aesthetic_luxury": "a luxury premium elegant high-status car design",
    "aesthetic_sporty": "a sporty aggressive dynamic performance-oriented car design",
    "aesthetic_modern_tech": "a modern high-tech futuristic digital car design",
    "aesthetic_premium_material": "a premium material-rich handcrafted interior design",
    "aesthetic_visual_impact": "a visually striking bold attention-grabbing car design",
}

DEFAULT_AESTHETIC_PROJECTION: List[float] = [0.24, 0.22, 0.20, 0.18, 0.16]

DEFAULT_OPTION_PROMPTS: Dict[str, str] = {
    "paint_color.paint_standard": "standard solid exterior paint, conservative mainstream sedan look",
    "paint_color.paint_metallic": "metallic exterior paint, refined premium finish",
    "paint_color.paint_manufaktur": "exclusive MANUFAKTUR paint, ultra-premium luxury appearance",
    "wheels.wheel_18_standard": "18-inch standard wheels, practical balanced styling",
    "wheels.wheel_19_upgrade": "19-inch upgraded alloy wheels, sportier premium stance",
    "wheels.wheel_amg_high": "large AMG performance wheels, aggressive sporty luxury styling",
    "exterior_style.styling_upgrade": "night package and illuminated grille styling, bold premium exterior",
    "upholstery.mb_tex": "MB-Tex interior upholstery, durable entry premium cabin",
    "upholstery.leather": "genuine leather seats, premium comfort interior",
    "upholstery.nappa_leather": "Nappa leather interior, handcrafted ultra-luxury comfort",
    "trim.standard_trim": "standard interior trim, clean understated cabin style",
    "trim.premium_trim": "premium wood or metallic trim, upscale interior details",
    "comfort.multicontour_package": "multicontour seats with massage comfort package, executive luxury seating",
    "comfort.seat_comfort_upgrade": "ventilated or heated seat comfort upgrade, premium daily comfort",
    "comfort.soft_close_doors": "soft-close doors, luxury convenience refinement",
    "audio.burmester_4d": "Burmester 4D surround audio system, high-end immersive cabin experience",
    "technology.mbux_superscreen": "MBUX superscreen cockpit, futuristic high-tech digital interior",
    "safety.driver_assistance_package": "advanced driver assistance package, confidence-inspiring premium safety technology",
    "performance.airmatic_package": "AIRMATIC suspension package, dynamic premium ride and handling",
    "lighting.digital_light": "DIGITAL LIGHT adaptive headlamps, cutting-edge premium lighting design",
}


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments."""

    parser = argparse.ArgumentParser(description="Build offline text-only CLIP semantics artifact.")
    parser.add_argument(
        "--catalog-path",
        type=Path,
        default=DEFAULT_CATALOG_PATH,
        help="Path to customization catalog YAML.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Target JSON artifact path.",
    )
    parser.add_argument(
        "--prompt-map-path",
        type=Path,
        default=None,
        help="Optional YAML/JSON map of `option_key -> clip_prompt_en`.",
    )
    parser.add_argument("--model-name", type=str, default="ViT-B-32", help="OpenCLIP model name.")
    parser.add_argument("--pretrained", type=str, default="openai", help="OpenCLIP pretrained tag.")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Embedding device. `auto` prefers CUDA.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Determinism seed stored into metadata.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output artifact if it exists.")
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    """Validates command arguments.

    Args:
        args: Parsed command arguments.
    """

    if not args.catalog_path.exists():
        raise FileNotFoundError(f"Catalog file not found: {args.catalog_path}")
    if args.output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {args.output_path}. Use `--overwrite` to replace it.")


def _sha256_text(payload: str) -> str:
    """Returns SHA256 hash of a UTF-8 string.

    Args:
        payload: Input text.

    Returns:
        Hex digest string.
    """

    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    """Returns SHA256 hash of a file.

    Args:
        path: File path.

    Returns:
        Hex digest string.
    """

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_catalog(path: Path) -> Dict[str, Any]:
    """Loads and validates catalog YAML.

    Args:
        path: Catalog path.

    Returns:
        Parsed catalog payload.
    """

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    options = payload.get("options")
    if not isinstance(options, list) or not options:
        raise ValueError("Catalog must contain a non-empty `options` list.")
    keys = [str(item.get("key", "")).strip() for item in options]
    if any(not key for key in keys):
        raise ValueError("Every catalog option must include a non-empty `key`.")
    if len(keys) != len(set(keys)):
        raise ValueError("Catalog option keys must be unique.")
    return payload


def _load_prompt_map(path: Path) -> Dict[str, str]:
    """Loads prompt map from YAML or JSON.

    Args:
        path: Prompt map path.

    Returns:
        Option prompt mapping.
    """

    if not path.exists():
        raise FileNotFoundError(f"Prompt map file not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".json":
        raw = json.loads(path.read_text(encoding="utf-8"))
    elif suffix in {".yaml", ".yml"}:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    else:
        raise ValueError("Prompt map path must be `.json`, `.yaml`, or `.yml`.")
    if not isinstance(raw, dict):
        raise ValueError("Prompt map must be a dict of `option_key -> prompt`.")
    mapping = {str(k).strip(): str(v).strip() for k, v in raw.items()}
    if any(not key or not prompt for key, prompt in mapping.items()):
        raise ValueError("Prompt map must not contain empty keys/prompts.")
    return mapping


def _resolve_option_prompts(*, option_keys: Sequence[str], prompt_map_path: Path | None) -> Dict[str, str]:
    """Resolves option prompts with strict key coverage.

    Args:
        option_keys: Catalog option keys.
        prompt_map_path: Optional external prompt map path.

    Returns:
        Final mapping for all option keys.
    """

    prompt_map = _load_prompt_map(prompt_map_path) if prompt_map_path else dict(DEFAULT_OPTION_PROMPTS)
    missing = [key for key in option_keys if key not in prompt_map]
    extra = [key for key in prompt_map.keys() if key not in set(option_keys)]
    if missing:
        raise ValueError(f"Prompt map missing option keys: {missing}")
    if extra:
        raise ValueError(f"Prompt map contains unknown option keys: {extra}")
    return {key: prompt_map[key] for key in option_keys}


def _pick_device(device_arg: str) -> str:
    """Resolves target device string for Torch.

    Args:
        device_arg: CLI device argument.

    Returns:
        Resolved device string.
    """

    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("`--device cuda` was requested but CUDA is not available.")
    return device_arg


def _normalize_rows(matrix: torch.Tensor) -> torch.Tensor:
    """L2-normalizes embedding rows.

    Args:
        matrix: Embedding tensor with shape `[N, D]`.

    Returns:
        Row-normalized tensor with shape `[N, D]`.
    """

    denom = matrix.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return matrix / denom


def _encode_texts(
    *,
    model: Any,
    tokenizer: Any,
    texts: Sequence[str],
    device: str,
) -> torch.Tensor:
    """Encodes text strings to normalized embeddings.

    Args:
        model: OpenCLIP model.
        tokenizer: OpenCLIP tokenizer callable.
        texts: Input text list.
        device: Torch device string.

    Returns:
        CPU float tensor of shape `[N, D]`, normalized.
    """

    tokens = tokenizer(list(texts)).to(device)
    with torch.no_grad():
        embeddings = model.encode_text(tokens)
    embeddings = _normalize_rows(embeddings.float())
    return embeddings.cpu()


def _build_artifact(
    *,
    catalog: Mapping[str, Any],
    catalog_path: Path,
    option_prompts: Mapping[str, str],
    model_name: str,
    pretrained: str,
    device: str,
    seed: int,
) -> Dict[str, Any]:
    """Builds CLIP semantics artifact payload.

    Args:
        catalog: Parsed catalog payload.
        catalog_path: Catalog path.
        option_prompts: Option prompt mapping.
        model_name: OpenCLIP model name.
        pretrained: OpenCLIP pretrained tag.
        device: Target device.
        seed: Determinism seed.

    Returns:
        JSON-serializable artifact payload.
    """

    torch.manual_seed(seed)
    np.random.seed(seed)

    model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=device)
    model.eval()
    tokenizer = open_clip.get_tokenizer(model_name)

    axis_labels = list(AXIS_PROMPTS.keys())
    axis_texts = [AXIS_PROMPTS[label] for label in axis_labels]
    axis_embeddings = _encode_texts(model=model, tokenizer=tokenizer, texts=axis_texts, device=device)

    catalog_options = sorted(catalog["options"], key=lambda item: str(item["key"]))
    option_keys = [str(item["key"]) for item in catalog_options]
    option_texts = [option_prompts[key] for key in option_keys]
    option_embeddings = _encode_texts(model=model, tokenizer=tokenizer, texts=option_texts, device=device)

    similarity = torch.matmul(option_embeddings, axis_embeddings.T).numpy()
    scores = np.clip((similarity + 1.0) / 2.0, 0.0, 1.0)

    records: List[Dict[str, Any]] = []
    for row_idx, item in enumerate(catalog_options):
        option_key = str(item["key"])
        row_scores = scores[row_idx]
        vector = [round(float(value), 6) for value in row_scores.tolist()]
        axis_scores = {label: vector[idx] for idx, label in enumerate(axis_labels)}
        records.append(
            {
                "option_key": option_key,
                "dimension": str(item["dimension"]),
                "price_delta_usd": float(item["price_delta_usd"]),
                "clip_prompt_en": option_prompts[option_key],
                "axis_scores": axis_scores,
                "semantic_vector": vector,
            }
        )

    projection = np.asarray(DEFAULT_AESTHETIC_PROJECTION, dtype=np.float32)
    projection = projection / max(float(projection.sum()), 1e-8)
    projection_list = [round(float(value), 6) for value in projection.tolist()]

    content_hash_payload = {
        "axis_prompts": AXIS_PROMPTS,
        "aesthetic_projection_vector": projection_list,
        "records": records,
    }
    content_hash = _sha256_text(json.dumps(content_hash_payload, sort_keys=True, separators=(",", ":")))

    return {
        "schema_version": "clip_text_semantics_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": int(seed),
        "catalog_path": str(catalog_path),
        "catalog_sha256": _sha256_file(catalog_path),
        "semantic_dim": len(axis_labels),
        "axis_labels": axis_labels,
        "axis_prompts": AXIS_PROMPTS,
        "aesthetic_projection_vector": projection_list,
        "clip_backend": {
            "library": "open-clip-torch",
            "open_clip_version": str(getattr(open_clip, "__version__", "unknown")),
            "torch_version": str(torch.__version__),
            "model_name": model_name,
            "pretrained": pretrained,
            "tokenizer": model_name,
            "device": device,
        },
        "records": records,
        "record_count": len(records),
        "content_sha256": content_hash,
    }


def _validate_artifact(*, artifact: Mapping[str, Any], option_keys: Sequence[str]) -> None:
    """Validates artifact shape and score bounds.

    Args:
        artifact: Built artifact payload.
        option_keys: Expected catalog option keys.
    """

    records = artifact["records"]
    if len(records) != len(option_keys):
        raise ValueError("Record count mismatch between catalog and artifact.")
    keys_in_artifact = [str(record["option_key"]) for record in records]
    if sorted(keys_in_artifact) != sorted(option_keys):
        raise ValueError("Artifact option keys mismatch catalog option keys.")
    semantic_dim = int(artifact["semantic_dim"])
    for record in records:
        vector = record["semantic_vector"]
        if len(vector) != semantic_dim:
            raise ValueError(f"Semantic dim mismatch for option `{record['option_key']}`.")
        for value in vector:
            numeric = float(value)
            if numeric < 0.0 or numeric > 1.0:
                raise ValueError(f"Semantic score out of range [0,1]: {numeric}")


def main() -> None:
    """Runs offline CLIP text semantics generation."""

    args = parse_args()
    _validate_args(args)

    catalog = _load_catalog(args.catalog_path)
    option_keys = [str(item["key"]) for item in catalog["options"]]
    option_prompts = _resolve_option_prompts(option_keys=option_keys, prompt_map_path=args.prompt_map_path)

    device = _pick_device(args.device)
    artifact = _build_artifact(
        catalog=catalog,
        catalog_path=args.catalog_path,
        option_prompts=option_prompts,
        model_name=str(args.model_name),
        pretrained=str(args.pretrained),
        device=device,
        seed=int(args.seed),
    )
    _validate_artifact(artifact=artifact, option_keys=option_keys)

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")

    summary = {
        "status": "ok",
        "output_path": str(args.output_path),
        "record_count": int(artifact["record_count"]),
        "semantic_dim": int(artifact["semantic_dim"]),
        "model_name": str(args.model_name),
        "pretrained": str(args.pretrained),
        "device": device,
        "catalog_path": str(args.catalog_path),
        "content_sha256": str(artifact["content_sha256"]),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
