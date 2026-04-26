"""Runtime utilities for DreamerV3 integration on pricing environment."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
import importlib.util
import io
from pathlib import Path
import re
import shutil
import sys
from typing import Any, Dict, Iterable, List, Mapping, Tuple

import numpy as np
import ruamel.yaml as yaml

from .adapter import DreamerActionCodec, DreamerDiscretePricingEnv
from .tta import DreamerTTAAdapter, DreamerTTAConfig

# Optional Dreamer imports are resolved at module load.
_EMBODIED_MODULE = None
_DREAMER_AGENT_MODULE = None
_DREAMER_TRAIN_MODULE = None
_WRAPPERS_MODULE = None
_DREAMER_IMPORT_ERROR: Exception | None = None

try:
    import elements
except Exception:  # pragma: no cover - dependency-dependent path
    elements = None

try:  # pragma: no cover - dependency-dependent path
    import dreamerv3  # noqa: F401
    import embodied as _embodied
    from dreamerv3 import agent as _dreamer_agent
    try:
        from dreamerv3 import train as _dreamer_train
    except Exception:  # pragma: no cover - package-version compatibility
        from dreamerv3 import main as _dreamer_train
    from embodied import wrappers as _wrappers
except Exception as _exc:  # pragma: no cover - dependency-dependent path
    _DREAMER_IMPORT_ERROR = _exc
else:
    _EMBODIED_MODULE = _embodied
    _DREAMER_AGENT_MODULE = _dreamer_agent
    _DREAMER_TRAIN_MODULE = _dreamer_train
    _WRAPPERS_MODULE = _wrappers


ROOT = Path(__file__).resolve().parents[3]
DREAMER_PRESETS_FALLBACK_PATH = ROOT / "configs" / "dreamer" / "upstream_configs.yaml"
NOISY_DREAMER_STDOUT_PATTERN = re.compile(r"^[A-Za-z0-9_]+(?:/[A-Za-z0-9_.-]+)+$")
NOISY_DREAMER_STDOUT_TOKENS = {
    ".inner_state",
    ".notfinite_count",
    ".last_finite",
    ".total_notfinite",
    ".count",
}


def _is_noisy_dreamer_line(line: str) -> bool:
    """Returns whether one stdout line is low-value Dreamer internals."""

    stripped = line.strip()
    if not stripped:
        return False
    if stripped in NOISY_DREAMER_STDOUT_TOKENS:
        return True
    if re.fullmatch(r"\[\d+\]", stripped):
        return True
    return bool(NOISY_DREAMER_STDOUT_PATTERN.fullmatch(stripped))


class _DreamerStdoutFilter(io.TextIOBase):
    """Filters noisy Dreamer internal stdout lines while keeping key metrics."""

    def __init__(self, wrapped: io.TextIOBase) -> None:
        self._wrapped = wrapped
        self._buffer = ""

    def writable(self) -> bool:
        return True

    def write(self, s: str) -> int:
        chunk = str(s)
        self._buffer += chunk
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if not _is_noisy_dreamer_line(line):
                self._wrapped.write(f"{line}\n")
        return len(chunk)

    def flush(self) -> None:
        if self._buffer and not _is_noisy_dreamer_line(self._buffer):
            self._wrapped.write(self._buffer)
        self._buffer = ""
        self._wrapped.flush()


@contextlib.contextmanager
def _filtered_dreamer_stdout() -> Iterable[None]:
    """Temporarily filters low-value Dreamer stdout spam."""

    filtered = _DreamerStdoutFilter(sys.stdout)
    with contextlib.redirect_stdout(filtered):
        try:
            yield
        finally:
            filtered.flush()


def _require_elements_module() -> Any:
    """Returns imported `elements` module or raises a clear dependency error."""

    if elements is None:
        raise RuntimeError(
            "DreamerV3 dependency `elements` is missing. Install Dreamer stack first "
            "(see `require_dreamerv3_dependencies`)."
        )
    return elements


def require_dreamerv3_dependencies() -> Tuple[Any, Any, Any, Any]:
    """Imports DreamerV3 runtime dependencies.

    Returns:
        Tuple `(embodied, dreamer_agent_module, dreamer_train_module, wrappers_module)`.

    Raises:
        RuntimeError: If DreamerV3/JAX dependencies are missing.
    """

    if (
        _EMBODIED_MODULE is None
        or _DREAMER_AGENT_MODULE is None
        or _DREAMER_TRAIN_MODULE is None
        or _WRAPPERS_MODULE is None
    ):
        raise RuntimeError(
            "DreamerV3 dependencies are missing. Install with:\n"
            "1) pip install dreamerv3\n"
            "2) install JAX/JAXLIB compatible with your CUDA stack.\n"
            "Then re-run this script."
        ) from _DREAMER_IMPORT_ERROR
    return (
        _EMBODIED_MODULE,
        _DREAMER_AGENT_MODULE,
        _DREAMER_TRAIN_MODULE,
        _WRAPPERS_MODULE,
    )


def _resolve_dreamer_presets_path(*, dreamer_train_module: Any) -> Path:
    """Resolves Dreamer preset YAML path.

    Args:
        dreamer_train_module: Imported Dreamer runtime module (`train` or `main`).

    Returns:
        Existing `configs.yaml` path.

    Raises:
        FileNotFoundError: If no preset YAML file can be found.
    """

    if DREAMER_PRESETS_FALLBACK_PATH.exists():
        return DREAMER_PRESETS_FALLBACK_PATH
    module_path = Path(getattr(dreamer_train_module, "__file__", "")).resolve()
    if module_path.is_file():
        candidate = module_path.with_name("configs.yaml")
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Dreamer preset file `configs.yaml` not found. "
        "Expected at "
        f"`{DREAMER_PRESETS_FALLBACK_PATH}` or next to the dreamerv3 runtime module."
    )


def _load_dreamer_presets(*, dreamer_train_module: Any) -> Dict[str, Any]:
    """Loads Dreamer preset table from YAML."""

    config_path = _resolve_dreamer_presets_path(dreamer_train_module=dreamer_train_module)
    parser = yaml.YAML(typ="safe")
    payload = parser.load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Dreamer preset file must be a mapping: {config_path}")
    if "defaults" not in payload:
        raise ValueError(f"Dreamer preset file missing `defaults`: {config_path}")
    return payload


def build_dreamer_config(
    *,
    embodied: Any,
    dreamer_agent_module: Any,
    dreamer_train_module: Any,
    presets: Iterable[str],
    logdir: Path,
    seed: int,
    steps: int,
    jax_platform: str,
    jax_precision: str,
    envs_amount: int,
    envs_parallel: str,
    replay_size: int,
    batch_size: int,
    batch_length: int,
    train_ratio: float,
    train_fill: int,
    log_every: int,
    save_every: int,
) -> Any:
    """Builds DreamerV3 config object for pricing task.

    Args:
        embodied: Imported `embodied` module.
        dreamer_agent_module: Imported `dreamerv3.agent` module.
        dreamer_train_module: Imported `dreamerv3.train`/`dreamerv3.main` module.
        presets: Dreamer preset names to apply.
        logdir: Training log directory.
        seed: Global random seed.
        steps: Total environment steps.
        jax_platform: JAX platform (`cpu/gpu/tpu`).
        jax_precision: JAX compute precision (for example, `float16`).
        envs_amount: Number of parallel env instances.
        envs_parallel: Dreamer env parallel mode (currently `none` only here).
        replay_size: Replay buffer capacity.
        batch_size: Training batch size.
        batch_length: Sequence length for world-model training.
        train_ratio: Gradient steps per environment step ratio.
        train_fill: Replay prefill steps.
        log_every: Logging interval in env steps.
        save_every: Checkpoint save interval in env steps.

    Returns:
        Dreamer `elements.Config`.
    """

    elements_module = _require_elements_module()

    preset_table = _load_dreamer_presets(dreamer_train_module=dreamer_train_module)
    config = elements_module.Config(preset_table["defaults"])
    for name in presets:
        preset = str(name).strip()
        if not preset:
            continue
        if preset not in preset_table:
            raise ValueError(f"Unknown Dreamer preset: `{preset}`")
        config = config.update(preset_table[preset])

    if str(envs_parallel).strip().lower() != "none":
        raise ValueError("Current integration supports only `envs_parallel: none`.")

    platform = str(jax_platform).strip().lower()
    if platform == "gpu":
        platform = "cuda"

    has_tensorflow = importlib.util.find_spec("tensorflow") is not None
    logger_outputs = ["jsonl", "tensorboard"] if has_tensorflow else ["jsonl"]

    config = config.update(
        {
            # Task string is required by upstream config logic; env is injected by our factory.
            "task": "dummy_disc",
            "logdir": str(logdir),
            "seed": int(seed),
            "replica": 0,
            "replicas": 1,
            "batch_size": int(batch_size),
            "batch_length": int(batch_length),
            "replay.size": int(replay_size),
            "run.steps": int(steps),
            "run.envs": int(envs_amount),
            "run.train_ratio": float(train_ratio),
            "run.log_every": int(log_every),
            "run.save_every": int(save_every),
            "run.debug": True,
            "jax.platform": platform,
            "jax.compute_dtype": str(jax_precision),
            # TensorBoard output in upstream logger requires tensorflow.
            "logger.outputs": logger_outputs,
        }
    )
    # `train_fill` is kept for experiment traceability even if upstream runtime
    # does not consume it directly in this code path.
    _ = int(train_fill)
    _ = dreamer_agent_module
    _ = embodied
    return config


@dataclass
class DreamerEnvBuildSpec:
    """Immutable inputs for creating Dreamer pricing environments."""

    catalog_path: Path
    persona_config_path: Path
    persona_bank_path: Path
    persona_bank_split: str
    price_bin_count: int
    price_step_usd: float
    clip_enabled: bool
    clip_semantic_path: Path
    clip_legacy_proxy_enabled: bool
    reward_scale_usd: float
    no_deal_penalty: float
    step_no_deal_penalty: float
    step_no_deal_penalty_start_round: int
    profit_target_usd: float
    low_profit_penalty: float
    soft_shortfall_penalty_coeff: float
    no_deal_requires_positive_margin: bool
    invalid_accept_penalty: float
    initial_offer_markup: float
    reward_mode: str
    grace_rounds_no_deal_penalty: int
    early_deal_round_cutoff: int
    early_deal_bonus: float
    delay_penalty_start_round: int
    delay_penalty_per_round: float
    probe_bonus_round_cutoff: int
    probe_bonus: float


class DreamerEmbodiedPricingEnv:
    """Embodied-compatible unbatched environment for DreamerV3.

    The environment wraps `DreamerDiscretePricingEnv` and exposes observations
    with Dreamer-required keys (`reward/is_first/is_last/is_terminal`) plus the
    numeric observation vector.
    """

    def __init__(
        self,
        *,
        space_cls: Any,
        build_spec: DreamerEnvBuildSpec,
        reset_seed_base: int,
    ) -> None:
        """Initializes unbatched Dreamer environment.

        Args:
            space_cls: `embodied.Space` class.
            build_spec: Immutable environment build parameters.
            reset_seed_base: Base seed used for deterministic episode resets.
        """

        self._space_cls = space_cls
        self._env = DreamerDiscretePricingEnv(
            catalog_path=str(build_spec.catalog_path),
            persona_config_path=str(build_spec.persona_config_path),
            persona_bank_path=str(build_spec.persona_bank_path),
            persona_bank_split=str(build_spec.persona_bank_split),
            price_bin_count=int(build_spec.price_bin_count),
            price_step_usd=float(build_spec.price_step_usd),
            clip_enabled=bool(build_spec.clip_enabled),
            clip_semantic_path=str(build_spec.clip_semantic_path),
            clip_legacy_proxy_enabled=bool(build_spec.clip_legacy_proxy_enabled),
            reward_scale_usd=float(build_spec.reward_scale_usd),
            no_deal_penalty=float(build_spec.no_deal_penalty),
            step_no_deal_penalty=float(build_spec.step_no_deal_penalty),
            step_no_deal_penalty_start_round=int(build_spec.step_no_deal_penalty_start_round),
            profit_target_usd=float(build_spec.profit_target_usd),
            low_profit_penalty=float(build_spec.low_profit_penalty),
            soft_shortfall_penalty_coeff=float(build_spec.soft_shortfall_penalty_coeff),
            no_deal_requires_positive_margin=bool(build_spec.no_deal_requires_positive_margin),
            invalid_accept_penalty=float(build_spec.invalid_accept_penalty),
            initial_offer_markup=float(build_spec.initial_offer_markup),
            reward_mode=str(build_spec.reward_mode),
            grace_rounds_no_deal_penalty=int(build_spec.grace_rounds_no_deal_penalty),
            early_deal_round_cutoff=int(build_spec.early_deal_round_cutoff),
            early_deal_bonus=float(build_spec.early_deal_bonus),
            delay_penalty_start_round=int(build_spec.delay_penalty_start_round),
            delay_penalty_per_round=float(build_spec.delay_penalty_per_round),
            probe_bonus_round_cutoff=int(build_spec.probe_bonus_round_cutoff),
            probe_bonus=float(build_spec.probe_bonus),
        )
        self._codec = self._env.codec
        self._needs_reset = True
        self._reset_seed_base = int(reset_seed_base)
        self._episode_idx = 0
        self._last_reset_info: Dict[str, Any] = {}

    def __len__(self) -> int:
        """Returns zero for unbatched env interface."""

        return 0

    @property
    def obs_space(self) -> Dict[str, Any]:
        """Returns Dreamer observation-space mapping."""

        obs_dim = int(self._env.observation_space.shape[0])
        return {
            "vector": self._space_cls(np.float32, (obs_dim,), -np.inf, np.inf),
            "reward": self._space_cls(np.float32),
            "is_first": self._space_cls(bool),
            "is_last": self._space_cls(bool),
            "is_terminal": self._space_cls(bool),
        }

    @property
    def act_space(self) -> Dict[str, Any]:
        """Returns Dreamer action-space mapping."""

        return {
            "action": self._space_cls(np.int32, (), 0, int(self._codec.action_size)),
            "reset": self._space_cls(bool),
        }

    @property
    def last_reset_info(self) -> Dict[str, Any]:
        """Returns reset info from latest episode start."""

        return dict(self._last_reset_info)

    def step(self, action: Mapping[str, Any]) -> Dict[str, Any]:
        """Applies one Dreamer action.

        Args:
            action: Action dictionary with `reset` and `action` keys.

        Returns:
            Observation dictionary following Dreamer conventions.
        """

        if bool(action.get("reset", False)) or self._needs_reset:
            seed = int(self._reset_seed_base + self._episode_idx)
            obs, info = self._env.reset(seed=seed)
            self._episode_idx += 1
            self._needs_reset = False
            self._last_reset_info = dict(info)
            return self._build_obs(
                obs_vector=obs,
                reward=0.0,
                is_first=True,
                is_last=False,
                is_terminal=False,
                episode_metrics=None,
            )

        action_idx = int(np.asarray(action["action"]).reshape(()))
        obs, reward, terminated, truncated, info = self._env.step(action_idx)
        done = bool(terminated or truncated)
        self._needs_reset = done
        metrics = info.get("episode_metrics") if isinstance(info, dict) else None
        return self._build_obs(
            obs_vector=obs,
            reward=float(reward),
            is_first=False,
            is_last=done,
            is_terminal=bool(terminated),
            episode_metrics=metrics if isinstance(metrics, dict) else None,
        )

    def close(self) -> None:
        """Closes wrapped env resources."""

        self._env.close()

    def _build_obs(
        self,
        *,
        obs_vector: np.ndarray,
        reward: float,
        is_first: bool,
        is_last: bool,
        is_terminal: bool,
        episode_metrics: Mapping[str, Any] | None,
    ) -> Dict[str, Any]:
        """Builds Dreamer observation payload.

        Args:
            obs_vector: Numeric observation vector.
            reward: Step reward.
            is_first: Episode-first flag.
            is_last: Episode-end flag.
            is_terminal: True terminal flag (excluding truncation).
            episode_metrics: Optional terminal metrics payload.

        Returns:
            Dreamer-compatible observation dictionary.
        """

        _ = episode_metrics

        return {
            "vector": np.asarray(obs_vector, dtype=np.float32),
            "reward": np.float32(reward),
            "is_first": bool(is_first),
            "is_last": bool(is_last),
            "is_terminal": bool(is_terminal),
        }


def _make_wrapped_dreamer_env(
    *,
    dreamer_train_module: Any,
    config: Any,
    build_spec: DreamerEnvBuildSpec,
    seed: int,
    env_index: int,
) -> Any:
    """Builds one wrapped Dreamer-compatible environment instance."""

    elements_module = _require_elements_module()

    env_seed_base = int(seed) + int(env_index) * 1_000_000
    raw_env = DreamerEmbodiedPricingEnv(
        space_cls=elements_module.Space,
        build_spec=build_spec,
        reset_seed_base=env_seed_base,
    )
    return dreamer_train_module.wrap_env(raw_env, config)


def _extract_agent_spaces(env: Any) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Extracts Dreamer agent spaces from wrapped env.

    Drops runtime-only keys:
    - observation keys prefixed with `log/`
    - action key `reset`
    """

    obs_space = {k: v for k, v in env.obs_space.items() if not str(k).startswith("log/")}
    act_space = {k: v for k, v in env.act_space.items() if str(k) != "reset"}
    return obs_space, act_space


def _build_agent_runtime_config(*, config: Any) -> Any:
    """Builds agent-specific runtime config payload."""

    elements_module = _require_elements_module()

    return elements_module.Config(
        **config.agent,
        logdir=config.logdir,
        seed=config.seed,
        jax=config.jax,
        batch_size=config.batch_size,
        batch_length=config.batch_length,
        replay_context=config.replay_context,
        report_length=config.report_length,
        replica=config.replica,
        replicas=config.replicas,
    )


def run_dreamer_training(
    *,
    config: Any,
    embodied: Any,
    dreamer_agent_module: Any,
    dreamer_train_module: Any,
    build_spec: DreamerEnvBuildSpec,
    seed: int,
    logdir: Path,
    checkpoint_out: Path,
) -> None:
    """Runs Dreamer training and exports checkpoint artifact.

    Args:
        config: Dreamer runtime config.
        embodied: Imported `embodied` module.
        dreamer_agent_module: Imported `dreamerv3.agent` module.
        dreamer_train_module: Imported `dreamerv3.train` module.
        build_spec: Environment build spec.
        seed: Global seed.
        logdir: Dreamer log directory.
        checkpoint_out: Export path for final checkpoint artifact.
    """

    elements_module = _require_elements_module()

    logdir.mkdir(parents=True, exist_ok=True)
    config.save(logdir / "config.yaml")

    def make_env(env_index: int) -> Any:
        return _make_wrapped_dreamer_env(
            dreamer_train_module=dreamer_train_module,
            config=config,
            build_spec=build_spec,
            seed=seed,
            env_index=env_index,
        )

    def make_agent() -> Any:
        env = make_env(0)
        try:
            obs_space, act_space = _extract_agent_spaces(env)
        finally:
            env.close()
        agent_config = _build_agent_runtime_config(config=config)
        return dreamer_agent_module.Agent(obs_space, act_space, agent_config)

    def make_replay() -> Any:
        return dreamer_train_module.make_replay(config, "replay")

    def make_stream(replay: Any, mode: str) -> Any:
        return dreamer_train_module.make_stream(config, replay, mode)

    def make_logger() -> Any:
        return dreamer_train_module.make_logger(config)

    run_args = elements_module.Config(
        **config.run,
        replica=int(config.replica),
        replicas=int(config.replicas),
        logdir=str(logdir),
        batch_size=int(config.batch_size),
        batch_length=int(config.batch_length),
        report_length=int(config.report_length),
        consec_train=int(config.consec_train),
        consec_report=int(config.consec_report),
        replay_context=int(config.replay_context),
    )
    with _filtered_dreamer_stdout():
        embodied.run.train(
            make_agent,
            make_replay,
            make_env,
            make_stream,
            make_logger,
            run_args,
        )

    checkpoint_candidates = [logdir / "checkpoint.pkl", logdir / "ckpt"]
    checkpoint_src = next((path for path in checkpoint_candidates if path.exists()), None)
    if checkpoint_src is None:
        candidate_text = ", ".join(str(path) for path in checkpoint_candidates)
        raise FileNotFoundError(
            "Dreamer checkpoint directory not found at expected paths: "
            f"{candidate_text}"
        )
    checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
    if checkpoint_out.exists():
        if checkpoint_out.is_dir():
            shutil.rmtree(checkpoint_out)
        else:
            checkpoint_out.unlink()
    shutil.copytree(checkpoint_src, checkpoint_out)


class DreamerPolicyActor:
    """Inference-time Dreamer policy actor for evaluation and benchmark."""

    _ACTION_KEY = "action"

    def __init__(
        self,
        *,
        agent: Any,
        codec: DreamerActionCodec,
        price_step_usd: float,
        initial_offer_markup: float,
        tta_config: DreamerTTAConfig | None = None,
    ):
        """Initializes policy actor.

        Args:
            agent: Loaded Dreamer agent instance.
            codec: Action codec for flattened discrete actions.
            price_step_usd: USD represented by one delta-bin step.
            initial_offer_markup: Initial offer anchor ratio from env config.
            tta_config: Optional TTA config. Disabled when omitted or `enabled=false`.
        """

        self._agent = agent
        self._codec = codec
        self._state = None
        self._prev_reward = 0.0
        self._price_step_usd = float(price_step_usd)
        cfg = tta_config if tta_config is not None else DreamerTTAConfig(enabled=False)
        self._tta_config = cfg
        self._validate_tta_runtime_contract()
        self._tta = (
            DreamerTTAAdapter(
                codec=codec,
                tta_config=cfg,
                price_step_usd=float(price_step_usd),
                initial_offer_markup=float(initial_offer_markup),
            )
            if bool(cfg.enabled)
            else None
        )

    def reset(self, *, reset_info: Mapping[str, Any] | None = None) -> None:
        """Resets recurrent state at episode boundary."""

        self._state = self._agent.init_policy(1)
        self._prev_reward = 0.0
        if self._tta is not None:
            self._tta.reset(reset_info=reset_info)

    def set_prev_reward(self, reward: float) -> None:
        """Updates reward fed into next policy step.

        Args:
            reward: Previous step reward.
        """

        self._prev_reward = float(reward)

    def observe_step_info(self, step_info: Mapping[str, Any] | None) -> None:
        """Updates optional TTA state from latest env step information."""

        if self._tta is not None:
            self._tta.observe_step_info(step_info)

    def tta_report(self) -> Dict[str, float | bool | str]:
        """Returns TTA metadata and aggregate adaptation statistics."""

        report: Dict[str, float | bool | str] = {
            "tta_enabled": bool(self._tta is not None),
            "tta_mode": str(self._tta_config.mode),
        }
        if self._tta is None:
            report.update(
                {
                    "prediction_count": 0.0,
                    "action_shift_count": 0.0,
                    "action_shift_rate": 0.0,
                    "avg_abs_price_adjust_usd": 0.0,
                    "candidate_count_avg": 0.0,
                    "avg_selected_score": 0.0,
                    "offer_override_count": 0.0,
                    "accept_override_count": 0.0,
                    "walkaway_override_count": 0.0,
                }
            )
            return report
        report.update(self._tta.metrics())
        return report

    def predict_action_index(self, obs_vector: np.ndarray, *, is_first: bool) -> int:
        """Predicts flattened discrete action index.

        Args:
            obs_vector: Environment observation vector.
            is_first: Whether this is the first step in episode.

        Returns:
            Flattened action index.
        """

        obs_batch = {
            "vector": np.asarray(obs_vector, dtype=np.float32)[None, :],
            "reward": np.asarray([self._prev_reward], dtype=np.float32),
            "is_first": np.asarray([bool(is_first)], dtype=bool),
            "is_last": np.asarray([False], dtype=bool),
            "is_terminal": np.asarray([False], dtype=bool),
        }
        if self._state is None:
            self._state = self._agent.init_policy(1)
        self._state, action_dict, _outs = self._agent.policy(self._state, obs_batch, mode="eval")
        raw_action_idx = self._decode_action_index(action_dict[self._ACTION_KEY])
        if self._tta is None:
            return int(raw_action_idx)
        if self._tta.mode == "belief_shift_v1":
            return int(self._tta.adapt_action_index(raw_action_idx))
        if self._tta.mode != "candidate_rerank_v2":
            raise RuntimeError(f"Unsupported TTA mode: {self._tta.mode}")
        return int(self._predict_reranked_action_index(raw_action_idx=int(raw_action_idx)))

    def _validate_tta_runtime_contract(self) -> None:
        """Checks Dreamer agent interfaces required by the chosen TTA mode."""

        if not bool(self._tta_config.enabled):
            return
        if str(self._tta_config.mode) != "candidate_rerank_v2":
            return
        if not hasattr(self._agent, "act_space"):
            raise ValueError("Dreamer `candidate_rerank_v2` requires `agent.act_space`.")
        act_space = getattr(self._agent, "act_space")
        if not isinstance(act_space, dict) or set(act_space.keys()) != {self._ACTION_KEY}:
            raise ValueError(
                "Dreamer `candidate_rerank_v2` requires one discrete action head named `action`."
            )

    @staticmethod
    def _decode_action_index(action: np.ndarray | Any) -> int:
        """Decodes Dreamer policy action payload into flattened action index."""

        action_arr = np.asarray(action)
        if action_arr.ndim == 1 and action_arr.shape[0] == 1:
            return int(action_arr[0])
        if action_arr.ndim == 2 and action_arr.shape == (1, 1):
            return int(action_arr[0, 0])
        if action_arr.ndim == 2 and action_arr.shape[0] == 1:
            return int(np.argmax(action_arr[0]))
        raise ValueError(f"Unexpected Dreamer action shape: {action_arr.shape}")

    def _predict_reranked_action_index(
        self,
        *,
        raw_action_idx: int,
    ) -> int:
        """Re-ranks candidate actions using runtime-compatible heuristic scoring."""

        if self._tta is None:
            return int(raw_action_idx)
        candidates = self._tta.build_candidate_action_indices(int(raw_action_idx))
        component_rows: List[Tuple[int, float, float, float, float, float]] = []
        for candidate_idx in candidates:
            policy_score = self._policy_locality_score(
                raw_action_idx=int(raw_action_idx),
                candidate_idx=int(candidate_idx),
            )
            world_score = self._surrogate_world_score(int(candidate_idx))
            margin_score = self._tta.candidate_margin_signal(int(candidate_idx))
            feasibility_score = self._tta.candidate_feasibility_signal(int(candidate_idx))
            risk_score = self._tta.candidate_risk_signal(int(candidate_idx))
            component_rows.append(
                (
                    int(candidate_idx),
                    float(policy_score),
                    float(world_score),
                    float(margin_score),
                    float(feasibility_score),
                    float(risk_score),
                )
            )

        policy_norm = self._minmax_normalize([row[1] for row in component_rows])
        world_norm = self._minmax_normalize([row[2] for row in component_rows])
        margin_norm = self._minmax_normalize([row[3] for row in component_rows])
        feasibility_norm = self._minmax_normalize([row[4] for row in component_rows])
        risk_norm = self._minmax_normalize([row[5] for row in component_rows])

        best_idx = int(raw_action_idx)
        best_score = -np.inf
        for idx, row in enumerate(component_rows):
            candidate_idx = int(row[0])
            final_score = (
                float(self._tta_config.w_policy) * float(policy_norm[idx])
                + float(self._tta_config.w_value) * float(world_norm[idx])
                + float(self._tta_config.w_margin) * float(margin_norm[idx])
                + float(self._tta_config.w_feasibility) * float(feasibility_norm[idx])
                - float(self._tta_config.w_risk) * float(risk_norm[idx])
            )
            if final_score > best_score + 1e-8:
                best_idx = candidate_idx
                best_score = float(final_score)
            elif abs(final_score - best_score) <= 1e-8 and candidate_idx == int(raw_action_idx):
                best_idx = candidate_idx

        self._tta.finalize_rerank_selection(
            raw_action_idx=int(raw_action_idx),
            selected_action_idx=int(best_idx),
            candidate_count=len(candidates),
            selected_score=float(best_score),
        )
        return int(best_idx)

    def _policy_locality_score(self, *, raw_action_idx: int, candidate_idx: int) -> float:
        """Returns a raw-policy-locality prior for one candidate."""

        raw_move, raw_delta = self._codec.unflatten(int(raw_action_idx))
        candidate_move, candidate_delta = self._codec.unflatten(int(candidate_idx))
        if int(candidate_idx) == int(raw_action_idx):
            return 1.0
        if raw_move == 0 and candidate_move == 0:
            max_delta = max(
                1,
                int(round(float(self._tta_config.max_price_adjust_usd) / float(self._price_step_usd))),
            )
            distance = abs(int(candidate_delta) - int(raw_delta))
            return float(max(0.0, 1.0 - distance / max_delta))
        if candidate_move == 1:
            return 0.35
        if candidate_move == 2:
            return 0.10
        return 0.0

    def _surrogate_world_score(self, action_idx: int) -> float:
        """Returns a runtime-compatible surrogate world score for one candidate."""

        if self._tta is None:
            return 0.0
        feasibility = float(self._tta.candidate_feasibility_signal(int(action_idx)))
        risk = float(self._tta.candidate_risk_signal(int(action_idx)))
        return float(feasibility - risk)

    @staticmethod
    def _minmax_normalize(values: List[float]) -> List[float]:
        """Normalizes one score list into `[0, 1]` with stable degenerate handling."""

        if not values:
            return []
        lo = float(min(values))
        hi = float(max(values))
        if abs(hi - lo) <= 1e-8:
            return [0.0 for _ in values]
        return [float((value - lo) / (hi - lo)) for value in values]


def load_dreamer_policy_actor(
    *,
    embodied: Any,
    dreamer_agent_module: Any,
    dreamer_train_module: Any,
    config: Any,
    build_spec: DreamerEnvBuildSpec,
    checkpoint_path: Path,
    seed: int,
    tta_config: DreamerTTAConfig | None = None,
) -> Tuple[DreamerPolicyActor, DreamerActionCodec]:
    """Loads Dreamer checkpoint and returns policy actor + action codec.

    Args:
        embodied: Imported `embodied` module.
        dreamer_agent_module: Imported `dreamerv3.agent` module.
        dreamer_train_module: Imported `dreamerv3.train` module.
        config: Dreamer runtime config.
        build_spec: Environment build spec.
        checkpoint_path: Dreamer checkpoint path.
        seed: Global seed.
        tta_config: Optional inference-time TTA config.

    Returns:
        Tuple of `(DreamerPolicyActor, DreamerActionCodec)`.
    """

    elements_module = _require_elements_module()

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Dreamer checkpoint not found: {checkpoint_path}")
    if checkpoint_path.is_file():
        raise ValueError(
            f"Dreamer checkpoint must be a directory path, got file: {checkpoint_path}"
        )

    env = _make_wrapped_dreamer_env(
        dreamer_train_module=dreamer_train_module,
        config=config,
        build_spec=build_spec,
        seed=seed,
        env_index=0,
    )
    try:
        obs_space, act_space = _extract_agent_spaces(env)
        agent_config = _build_agent_runtime_config(config=config)
        agent = dreamer_agent_module.Agent(obs_space, act_space, agent_config)
        checkpoint = elements_module.Checkpoint(directory=str(checkpoint_path), write=False)
        checkpoint.agent = agent
        with _filtered_dreamer_stdout():
            checkpoint.load()
    finally:
        env.close()

    codec = DreamerActionCodec(
        move_count=3,
        delta_bin_count=int(build_spec.price_bin_count),
    )
    return (
        DreamerPolicyActor(
            agent=agent,
            codec=codec,
            price_step_usd=float(build_spec.price_step_usd),
            initial_offer_markup=float(build_spec.initial_offer_markup),
            tta_config=tta_config,
        ),
        codec,
    )
