"""Shared PPO config loader used by train/eval/benchmark scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PPO_CONFIG_PATH = ROOT / "configs" / "ppo" / "default.yaml"


def load_ppo_config(config_path: Path) -> dict[str, Any]:
    """Load and validate PPO YAML config.

    Args:
        config_path: YAML config path.

    Returns:
        Parsed config dictionary.
    """

    if not config_path.exists():
        raise FileNotFoundError(f"PPO config not found: {config_path}")
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("PPO config must be a mapping.")
    schema = str(payload.get("schema_version", "")).strip()
    if schema != "ppo_config_v1":
        raise ValueError("PPO config must declare `schema_version: ppo_config_v1`.")
    return payload


def get_required(config: Mapping[str, Any], dotted_key: str) -> Any:
    """Get a required value from nested mapping using dotted key syntax.

    Args:
        config: Source mapping.
        dotted_key: Nested key path, e.g. `train.n_steps`.

    Returns:
        The selected value.
    """

    current: Any = config
    for token in dotted_key.split("."):
        if not isinstance(current, Mapping) or token not in current:
            raise ValueError(f"Missing required config key: `{dotted_key}`")
        current = current[token]
    return current


def get_optional(config: Mapping[str, Any], dotted_key: str, default: Any) -> Any:
    """Get an optional value from nested mapping using dotted key syntax.

    Args:
        config: Source mapping.
        dotted_key: Nested key path.
        default: Fallback value when key path does not exist.

    Returns:
        Existing value or `default` when key path is missing.
    """

    current: Any = config
    for token in dotted_key.split("."):
        if not isinstance(current, Mapping) or token not in current:
            return default
        current = current[token]
    return current


def resolve_repo_path(value: str | Path) -> Path:
    """Resolve path values relative to repository root.

    Args:
        value: Raw string or path from config/CLI.

    Returns:
        Absolute path.
    """

    path = value if isinstance(value, Path) else Path(str(value))
    if path.is_absolute():
        return path
    return (ROOT / path).resolve()
