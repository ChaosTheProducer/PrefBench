"""Train DreamerV3 baseline on the pricing negotiation environment."""

from __future__ import annotations

import atexit
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, List

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
from pricing_agent.world_model import (
    DreamerEnvBuildSpec,
    build_dreamer_config,
    require_dreamerv3_dependencies,
    run_dreamer_training,
)


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments.

    Returns:
        Parsed command-line namespace.
    """

    parser = argparse.ArgumentParser(description="Train DreamerV3 baseline.")
    parser.add_argument(
        "--config-path",
        type=Path,
        default=DEFAULT_DREAMER_CONFIG_PATH,
        help="Unified Dreamer config YAML path.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional run name override.",
    )
    parser.add_argument("--seed", type=int, default=123, help="Global random seed.")
    parser.add_argument("--steps", type=int, default=None, help="Optional training step override.")
    parser.add_argument(
        "--persona-bank-path",
        type=Path,
        default=None,
        help="Optional persona-bank JSONL path override.",
    )
    parser.add_argument(
        "--persona-bank-split",
        type=str,
        default=None,
        choices=["train", "val", "test"],
        help="Optional training split override.",
    )
    parser.add_argument(
        "--checkpoint-out",
        type=Path,
        default=None,
        help="Optional explicit output path for Dreamer checkpoint file.",
    )
    parser.add_argument(
        "--metadata-out",
        type=Path,
        default=None,
        help="Optional explicit output path for training metadata JSON.",
    )
    parser.add_argument(
        "--start-tensorboard",
        action="store_true",
        help="Whether to launch TensorBoard automatically for this run logdir.",
    )
    parser.add_argument(
        "--tensorboard-port",
        type=int,
        default=6007,
        help="TensorBoard port used when --start-tensorboard is set.",
    )
    return parser.parse_args()


def _parse_presets(value: Any) -> List[str]:
    """Normalizes Dreamer preset list.

    Args:
        value: Raw config value.

    Returns:
        Ordered preset list.
    """

    if not isinstance(value, list):
        raise ValueError("`dreamer.presets` must be a list.")
    presets = [str(token).strip() for token in value if str(token).strip()]
    if not presets:
        raise ValueError("`dreamer.presets` must contain at least one preset.")
    return presets


def _maybe_launch_tensorboard(*, enabled: bool, logdir: Path, port: int) -> dict[str, Any] | None:
    """Starts TensorBoard subprocess if requested.

    Args:
        enabled: Whether launch is requested by CLI.
        logdir: Run logdir passed to TensorBoard.
        port: TCP port for TensorBoard HTTP server.

    Returns:
        Metadata dictionary when launched; otherwise None.
    """

    if not enabled:
        return None
    if importlib.util.find_spec("tensorboard.main") is None:
        print("[dreamer] TensorBoard not installed; skip auto-launch.")
        return None
    command = [
        sys.executable,
        "-m",
        "tensorboard.main",
        "--logdir",
        str(logdir),
        "--port",
        str(int(port)),
    ]
    process = subprocess.Popen(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    atexit.register(process.terminate)
    url = f"http://localhost:{int(port)}"
    print(f"[dreamer] TensorBoard started: {url} (pid={process.pid})")
    return {
        "enabled": True,
        "pid": int(process.pid),
        "port": int(port),
        "url": url,
        "logdir": str(logdir),
    }


def main() -> None:
    """Runs DreamerV3 training and persists metadata/checkpoint artifacts."""

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
        else str(get_required(raw_config, "train.persona_bank_split"))
    )
    clip_enabled = bool(get_optional(raw_config, "clip.enabled", False))
    clip_semantic_path = (
        resolve_repo_path(get_required(raw_config, "clip.semantic_path")) if clip_enabled else None
    )
    clip_legacy_proxy_enabled = bool(get_optional(raw_config, "clip.legacy_proxy_enabled", True))
    steps = int(args.steps) if args.steps is not None else int(get_required(raw_config, "train.steps"))

    checkpoint_out = (
        Path(args.checkpoint_out)
        if args.checkpoint_out is not None
        else output_root / "checkpoints" / f"{run_name}_final.ckpt"
    )
    metadata_out = (
        Path(args.metadata_out)
        if args.metadata_out is not None
        else output_root / "metadata" / f"{run_name}_train.json"
    )
    logdir = output_root / "logs" / "dreamer" / run_name

    embodied, dreamer_agent_module, dreamer_train_module, _wrappers = require_dreamerv3_dependencies()
    presets = _parse_presets(get_required(raw_config, "dreamer.presets"))
    dreamer_cfg = build_dreamer_config(
        embodied=embodied,
        dreamer_agent_module=dreamer_agent_module,
        dreamer_train_module=dreamer_train_module,
        presets=presets,
        logdir=logdir,
        seed=int(args.seed),
        steps=steps,
        jax_platform=str(get_required(raw_config, "dreamer.jax_platform")),
        jax_precision=str(get_required(raw_config, "dreamer.jax_precision")),
        envs_amount=int(get_required(raw_config, "dreamer.envs_amount")),
        envs_parallel=str(get_required(raw_config, "dreamer.envs_parallel")),
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
    tensorboard_info = _maybe_launch_tensorboard(
        enabled=bool(args.start_tensorboard),
        logdir=logdir,
        port=int(args.tensorboard_port),
    )

    run_dreamer_training(
        config=dreamer_cfg,
        embodied=embodied,
        dreamer_agent_module=dreamer_agent_module,
        dreamer_train_module=dreamer_train_module,
        build_spec=build_spec,
        seed=int(args.seed),
        logdir=logdir,
        checkpoint_out=checkpoint_out,
    )

    metadata = {
        "run_name": run_name,
        "config_path": str(config_path),
        "output_root": str(output_root),
        "seed": int(args.seed),
        "steps": steps,
        "persona_bank_path": str(persona_bank_path),
        "persona_bank_split": persona_bank_split,
        "catalog_path": str(catalog_path),
        "persona_config_path": str(persona_config_path),
        "dreamer_presets": presets,
        "dreamer_reward": {
            "mode": str(build_spec.reward_mode),
            "profit_target_usd": float(build_spec.profit_target_usd),
            "low_profit_penalty": float(build_spec.low_profit_penalty),
            "soft_shortfall_penalty_coeff": float(build_spec.soft_shortfall_penalty_coeff),
            "step_no_deal_penalty": float(build_spec.step_no_deal_penalty),
            "step_no_deal_penalty_start_round": int(build_spec.step_no_deal_penalty_start_round),
            "no_deal_requires_positive_margin": bool(build_spec.no_deal_requires_positive_margin),
            "grace_rounds_no_deal_penalty": int(build_spec.grace_rounds_no_deal_penalty),
            "early_deal_round_cutoff": int(build_spec.early_deal_round_cutoff),
            "early_deal_bonus": float(build_spec.early_deal_bonus),
            "delay_penalty_start_round": int(build_spec.delay_penalty_start_round),
            "delay_penalty_per_round": float(build_spec.delay_penalty_per_round),
            "probe_bonus_round_cutoff": int(build_spec.probe_bonus_round_cutoff),
            "probe_bonus": float(build_spec.probe_bonus),
        },
        "clip": {
            "enabled": bool(build_spec.clip_enabled),
            "semantic_path": str(build_spec.clip_semantic_path),
            "legacy_proxy_enabled": bool(build_spec.clip_legacy_proxy_enabled),
        },
        "tta": {
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
        "logdir": str(logdir),
        "checkpoint_path": str(checkpoint_out),
        "metadata_path": str(metadata_out),
        "tensorboard": tensorboard_info,
    }
    metadata_out.parent.mkdir(parents=True, exist_ok=True)
    metadata_out.write_text(json.dumps(metadata, indent=2, sort_keys=True))
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
