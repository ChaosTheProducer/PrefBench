"""Train a PPO baseline on the pricing negotiation environment.

This script uses a PPO-compatible environment adapter with:
- discrete move actions (`offer/accept/walkaway`)
- discrete price delta bins
- profit-dominant reward shaping

Default model is Recurrent PPO to better handle partial observability.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Callable
import time

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pricing_agent.ppo_env import PPOPricingEnv
from scripts.common.ppo_config import DEFAULT_PPO_CONFIG_PATH, get_optional, get_required, load_ppo_config, resolve_repo_path


class CompactConsoleMetricsCallback:  # pragma: no cover - training-loop utility
    """Prints compact line-by-line training metrics focused on business outcomes.

    The callback intentionally reports only project-relevant performance signals
    and avoids optimizer-level details in stdout. Each refresh prints one new
    line instead of updating the same terminal line.
    """

    def __init__(
        self,
        *,
        total_timesteps: int,
        print_freq_steps: int = 2000,
        metrics_sink: list[dict[str, Any]] | None = None,
        enable_tqdm: bool = True,
    ):
        """Initializes compact console callback.

        Args:
            total_timesteps: Planned total training timesteps.
            print_freq_steps: Minimum timestep interval between console updates.
            metrics_sink: Optional list used to collect per-refresh metric snapshots.
        """

        from stable_baselines3.common.callbacks import BaseCallback

        class _Impl(BaseCallback):
            def __init__(self, parent: "CompactConsoleMetricsCallback"):
                super().__init__()
                self._parent = parent

            def _on_training_start(self) -> None:
                self._parent._start_time = time.time()
                self._parent._last_print_step = 0
                self._parent._last_tqdm_step = 0
                self._parent._open_progress_bar()

            def _on_step(self) -> bool:
                self._parent._collect_episode_metrics(self.locals.get("infos", []))
                self._parent._update_progress_bar(current_step=self.num_timesteps)
                snapshot = self._parent._maybe_print(current_step=self.num_timesteps)
                if snapshot is not None:
                    self._parent._log_tensorboard(
                        logger=self.logger,
                        snapshot=snapshot,
                        step=self.num_timesteps,
                    )
                return True

            def _on_training_end(self) -> None:
                snapshot = self._parent._print_final(self.num_timesteps)
                if snapshot is not None:
                    self._parent._log_tensorboard(
                        logger=self.logger,
                        snapshot=snapshot,
                        step=self.num_timesteps,
                    )
                self._parent._close_progress_bar()

        self._callback = _Impl(self)
        self._total_timesteps = int(total_timesteps)
        self._print_freq_steps = int(max(1, print_freq_steps))
        self._episode_count = 0
        self._deal_count = 0
        self._profit_sum = 0.0
        self._rounds_sum = 0.0
        self._walkaway_count = 0
        self._start_time = 0.0
        self._last_print_step = 0
        self._last_tqdm_step = 0
        self._metrics_sink = metrics_sink
        self._enable_tqdm = bool(enable_tqdm)
        self._progress_bar: Any | None = None

    @property
    def callback(self):
        """Returns underlying SB3 callback instance."""

        return self._callback

    def _collect_episode_metrics(self, infos: object) -> None:
        """Collects episode metrics from environment info payloads.

        Args:
            infos: Raw `infos` object from vectorized env step.
        """

        if not isinstance(infos, list):
            return
        for item in infos:
            if not isinstance(item, dict):
                continue
            metrics = item.get("episode_metrics")
            if not isinstance(metrics, dict):
                continue
            self._episode_count += 1
            self._deal_count += int(bool(metrics.get("deal_reached", False)))
            self._profit_sum += float(metrics.get("profit_usd", 0.0))
            self._rounds_sum += float(metrics.get("rounds_used", 0))
            self._walkaway_count += int(bool(metrics.get("walkaway", False)))

    def _maybe_print(self, *, current_step: int) -> dict[str, Any] | None:
        """Prints one-line summary when step interval is reached.

        Args:
            current_step: Current global timestep.
        """

        if current_step - self._last_print_step < self._print_freq_steps:
            return None
        self._last_print_step = int(current_step)
        return self._print_line(current_step=current_step)

    def _print_final(self, current_step: int) -> dict[str, Any] | None:
        """Prints final line and line break on training end.

        Args:
            current_step: Final timestep.
        """

        if current_step != self._last_print_step:
            return self._print_line(current_step=current_step)
        return None

    def _print_line(self, *, current_step: int) -> dict[str, Any]:
        """Renders one compact metric line.

        Args:
            current_step: Current timestep.
        """

        elapsed = max(1e-6, time.time() - self._start_time)
        fps = int(current_step / elapsed)
        progress = min(1.0, float(current_step) / float(max(1, self._total_timesteps)))
        episodes = max(1, self._episode_count)
        deal_rate = float(self._deal_count) / float(episodes)
        avg_profit = float(self._profit_sum) / float(episodes)
        avg_rounds = float(self._rounds_sum) / float(episodes)
        walkaway_rate = float(self._walkaway_count) / float(episodes)
        snapshot = {
            "step": int(current_step),
            "total_timesteps": int(self._total_timesteps),
            "progress": float(progress),
            "episodes": int(self._episode_count),
            "deal_rate": float(deal_rate),
            "avg_profit_usd": float(avg_profit),
            "avg_rounds": float(avg_rounds),
            "walkaway_rate": float(walkaway_rate),
            "fps": int(fps),
            "elapsed_seconds": float(elapsed),
        }
        if self._metrics_sink is not None:
            self._metrics_sink.append(snapshot)

        if self._progress_bar is not None:
            self._progress_bar.set_postfix(
                {
                    "deal_rate": f"{deal_rate:.3f}",
                    "profit": f"{avg_profit:.1f}",
                    "walkaway": f"{walkaway_rate:.3f}",
                },
                refresh=False,
            )
        line = (
            f"[train] step={current_step}/{self._total_timesteps} "
            f"progress={progress:.1%} episodes={self._episode_count} "
            f"deal_rate={deal_rate:.3f} avg_profit_usd={avg_profit:.1f} "
            f"avg_rounds={avg_rounds:.2f} walkaway_rate={walkaway_rate:.3f} fps={fps}"
        )
        if self._progress_bar is not None:
            from tqdm.auto import tqdm

            tqdm.write(line)
        else:
            print(line, flush=True)
        return snapshot

    def _open_progress_bar(self) -> None:
        """Opens a tqdm progress bar when enabled."""

        if not self._enable_tqdm:
            return
        from tqdm.auto import tqdm

        self._progress_bar = tqdm(
            total=self._total_timesteps,
            desc="train",
            unit="step",
            dynamic_ncols=True,
            leave=True,
        )

    def _update_progress_bar(self, *, current_step: int) -> None:
        """Updates tqdm progress bar to the current global step."""

        if self._progress_bar is None:
            return
        target = int(max(0, min(current_step, self._total_timesteps)))
        delta = target - int(self._last_tqdm_step)
        if delta > 0:
            self._progress_bar.update(delta)
            self._last_tqdm_step = target

    def _close_progress_bar(self) -> None:
        """Closes tqdm progress bar and ensures final completion."""

        if self._progress_bar is None:
            return
        if self._last_tqdm_step < self._total_timesteps:
            self._progress_bar.update(self._total_timesteps - self._last_tqdm_step)
        self._progress_bar.close()
        self._progress_bar = None

    @staticmethod
    def _log_tensorboard(*, logger: Any, snapshot: dict[str, Any], step: int) -> None:
        """Writes compact business metrics to TensorBoard."""

        logger.record("business/deal_rate", float(snapshot["deal_rate"]))
        logger.record("business/avg_profit_usd", float(snapshot["avg_profit_usd"]))
        logger.record("business/avg_rounds", float(snapshot["avg_rounds"]))
        logger.record("business/walkaway_rate", float(snapshot["walkaway_rate"]))
        logger.record("business/progress", float(snapshot["progress"]))
        logger.record("business/fps", float(snapshot["fps"]))
        logger.dump(step=int(step))


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments.

    Returns:
        Parsed command-line namespace.
    """
    parser = argparse.ArgumentParser(description="Train Recurrent PPO baseline.")
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
    parser.add_argument("--seed", type=int, default=123, help="Global random seed.")
    parser.add_argument("--timesteps", type=int, default=None, help="Optional timesteps override.")
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
        help="Optional persona split override.",
    )
    parser.add_argument(
        "--model-out",
        type=Path,
        default=None,
        help="Optional explicit output path for trained model zip.",
    )
    parser.add_argument(
        "--metadata-out",
        type=Path,
        default=None,
        help="Optional explicit output path for training metadata JSON.",
    )
    parser.add_argument(
        "--tensorboard-log-dir",
        type=Path,
        default=None,
        help="Optional explicit TensorBoard log directory.",
    )
    parser.add_argument(
        "--metrics-out",
        type=Path,
        default=None,
        help="Optional explicit output path for compact business metrics JSON.",
    )
    parser.add_argument(
        "--disable-tqdm",
        action="store_true",
        help="Disable tqdm progress bar.",
    )
    return parser.parse_args()


def _build_env_factory(args: argparse.Namespace, rank: int) -> Callable[[], PPOPricingEnv]:
    """Builds one vectorized env factory.

    Args:
        args: Parsed training arguments.
        rank: Worker rank used for seed offsets.

    Returns:
        Callable that creates one PPO environment instance.
    """

    def _make_env() -> PPOPricingEnv:
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
        env.reset(seed=int(args.seed) + rank)
        return env

    return _make_env


def main() -> None:
    """Runs PPO training and persists model + metadata."""
    # Most hyperparameters are loaded from the unified YAML config.
    # High-frequency CLI flags are applied as explicit overrides.

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
        else str(get_required(config, "train.persona_bank_split"))
    )
    args.timesteps = int(args.timesteps) if args.timesteps is not None else int(get_required(config, "train.timesteps"))

    args.n_envs = int(get_required(config, "train.n_envs"))
    args.n_steps = int(get_required(config, "train.n_steps"))
    args.console_log_freq = int(get_required(config, "train.console_log_freq"))
    args.batch_size = int(get_required(config, "train.batch_size"))
    args.learning_rate = float(get_required(config, "train.learning_rate"))
    args.gamma = float(get_required(config, "train.gamma"))
    args.gae_lambda = float(get_required(config, "train.gae_lambda"))
    args.ent_coef = float(get_required(config, "train.ent_coef"))
    args.clip_range = float(get_required(config, "train.clip_range"))

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

    try:
        from sb3_contrib import RecurrentPPO
        from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
    except Exception as exc:  # pragma: no cover - dependency-dependent path
        raise RuntimeError(
            "Missing PPO dependencies. Install `stable-baselines3` and `sb3-contrib`."
        ) from exc
    if not bool(args.disable_tqdm):
        try:
            from tqdm.auto import tqdm as _tqdm  # noqa: F401
        except Exception as exc:
            raise RuntimeError("Missing dependency `tqdm`. Install it or pass `--disable-tqdm`.") from exc

    if int(args.batch_size) <= 0 or int(args.n_steps) <= 0:
        raise ValueError("`batch-size` and `n-steps` must be positive.")
    if int(args.n_envs) <= 0:
        raise ValueError("`n-envs` must be positive.")

    output_root = resolve_repo_path(get_required(config, "defaults.output_root"))
    run_name = (
        str(args.run_name).strip()
        if args.run_name is not None and str(args.run_name).strip()
        else str(get_required(config, "defaults.run_name"))
    )
    model_out = Path(args.model_out) if args.model_out is not None else output_root / "checkpoints" / f"{run_name}_final.zip"
    metadata_out = (
        Path(args.metadata_out)
        if args.metadata_out is not None
        else output_root / "metadata" / f"{run_name}_train.json"
    )
    tensorboard_log_dir = (
        Path(args.tensorboard_log_dir)
        if args.tensorboard_log_dir is not None
        else output_root / "logs" / "tensorboard" / run_name
    )
    metrics_out = (
        Path(args.metrics_out)
        if args.metrics_out is not None
        else output_root / "logs" / "metrics" / f"{run_name}_train_metrics.json"
    )

    env_fns = [_build_env_factory(args, rank=i) for i in range(int(args.n_envs))]
    vec_env = DummyVecEnv(env_fns)
    vec_env = VecMonitor(venv=vec_env)

    model = RecurrentPPO(
        policy="MlpLstmPolicy",
        env=vec_env,
        seed=int(args.seed),
        verbose=0,
        n_steps=int(args.n_steps),
        batch_size=int(args.batch_size),
        learning_rate=float(args.learning_rate),
        gamma=float(args.gamma),
        gae_lambda=float(args.gae_lambda),
        ent_coef=float(args.ent_coef),
        clip_range=float(args.clip_range),
        tensorboard_log=str(tensorboard_log_dir),
    )
    metrics_history: list[dict[str, Any]] = []
    compact_callback = CompactConsoleMetricsCallback(
        total_timesteps=int(args.timesteps),
        print_freq_steps=int(args.console_log_freq),
        metrics_sink=metrics_history,
        enable_tqdm=not bool(args.disable_tqdm),
    )
    model.learn(
        total_timesteps=int(args.timesteps),
        progress_bar=False,
        callback=compact_callback.callback,
    )

    model_out.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(model_out))

    metadata = {
        "run_name": run_name,
        "config_path": str(config_path),
        "output_root": str(output_root),
        "timesteps": int(args.timesteps),
        "seed": int(args.seed),
        "n_envs": int(args.n_envs),
        "n_steps": int(args.n_steps),
        "console_log_freq": int(args.console_log_freq),
        "batch_size": int(args.batch_size),
        "learning_rate": float(args.learning_rate),
        "gamma": float(args.gamma),
        "gae_lambda": float(args.gae_lambda),
        "ent_coef": float(args.ent_coef),
        "clip_range": float(args.clip_range),
        "price_bin_count": int(args.price_bin_count),
        "price_step_usd": float(args.price_step_usd),
        "clip_enabled": bool(args.clip_enabled),
        "clip_semantic_path": None if args.clip_semantic_path is None else str(args.clip_semantic_path),
        "clip_legacy_proxy_enabled": bool(args.clip_legacy_proxy_enabled),
        "reward_scale_usd": float(args.reward_scale_usd),
        "no_deal_penalty": float(args.no_deal_penalty),
        "step_no_deal_penalty": float(args.step_no_deal_penalty),
        "step_no_deal_penalty_start_round": int(args.step_no_deal_penalty_start_round),
        "profit_target_usd": float(args.profit_target_usd),
        "low_profit_penalty": float(args.low_profit_penalty),
        "no_deal_requires_positive_margin": bool(args.no_deal_requires_positive_margin),
        "invalid_accept_penalty": float(args.invalid_accept_penalty),
        "initial_offer_markup": float(args.initial_offer_markup),
        "persona_bank_path": str(args.persona_bank_path),
        "persona_bank_split": str(args.persona_bank_split),
        "model_path": str(model_out),
        "metadata_path": str(metadata_out),
        "tensorboard_log_dir": str(tensorboard_log_dir),
        "metrics_path": str(metrics_out),
    }
    metadata_out.parent.mkdir(parents=True, exist_ok=True)
    metadata_out.write_text(json.dumps(metadata, indent=2, sort_keys=True))

    metrics_payload = {
        "run_name": run_name,
        "total_timesteps": int(args.timesteps),
        "console_log_freq": int(args.console_log_freq),
        "series": metrics_history,
    }
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    metrics_out.write_text(json.dumps(metrics_payload, indent=2, sort_keys=True))
    vec_env.close()

    print(
        json.dumps(
            {
                "status": "ok",
                "run_name": run_name,
                "model_path": str(model_out),
                "metadata_path": str(metadata_out),
                "tensorboard_log_dir": str(tensorboard_log_dir),
                "metrics_path": str(metrics_out),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
