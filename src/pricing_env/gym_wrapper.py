"""Gymnasium wrapper for the pricing negotiation environment."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces

    GYMNASIUM_AVAILABLE = True
except Exception:  # pragma: no cover - environment-dependent optional dependency
    gym = None  # type: ignore[assignment]
    spaces = None  # type: ignore[assignment]
    GYMNASIUM_AVAILABLE = False

from .negotiation_env import NegotiationEnv
from .types import EnvAction, ObservationDict, PersonaProfile


MOVE_TOKENS = ("offer", "accept", "walkaway")
RESPONSE_TOKENS = (
    "none",
    "counter",
    "reject",
    "accept",
    "walkaway",
    "timeout",
    "agent_walkaway",
    "invalid_accept",
    "closed",
)
AGE_BANDS = ("18-25", "26-35", "36-50", "50+", "unknown")
INCOME_BANDS = ("<60k", "60-100k", "100-180k", "180k+", "unknown")
HOUSEHOLD_STAGES = ("single", "couple", "family", "unknown")
OWNERSHIP_STAGES = ("first_time", "replacement", "additional", "unknown")
PRIMARY_USE_CASES = ("commute", "family", "luxury", "performance", "mixed", "unknown")


class PricingNegotiationGymWrapper(gym.Env if GYMNASIUM_AVAILABLE else object):
    """Wraps `NegotiationEnv` with Gymnasium-compatible APIs.

    The wrapper keeps all business logic in `NegotiationEnv` and only adapts I/O:
    - actions: `Dict(move, price)`
    - observations: fixed-size normalized float vector
    - step signature: `(obs, reward, terminated, truncated, info)`
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        catalog_path: str | Path,
        persona_config_path: str | Path,
        persona_bank_path: str | Path | None = None,
        persona_bank_split: str = "train",
        offer_price_min_usd: Optional[float] = None,
        offer_price_max_usd: Optional[float] = None,
        clip_enabled: bool = False,
        clip_semantic_path: str | Path | None = None,
        clip_legacy_proxy_enabled: bool = True,
    ) -> None:
        """Initializes the Gymnasium wrapper.

        Args:
            catalog_path: YAML path for canonical customization catalog.
            persona_config_path: YAML path for persona sampling config.
            persona_bank_path: Optional JSONL persona-bank path.
            persona_bank_split: Split label used when persona-bank sampling is enabled.
            offer_price_min_usd: Optional minimum offer price for action space.
            offer_price_max_usd: Optional maximum offer price for action space.
            clip_enabled: Whether to append offline CLIP semantics into observation.
            clip_semantic_path: JSON path of precomputed CLIP semantics artifact.
            clip_legacy_proxy_enabled: Whether to keep proxy aesthetic scalar in base
                observation slot when CLIP is enabled.
        """

        if not GYMNASIUM_AVAILABLE:
            raise RuntimeError(
                "Gymnasium is not installed. Install `gymnasium` to use PricingNegotiationGymWrapper."
            )

        super().__init__()
        self._env = NegotiationEnv(
            catalog_path=catalog_path,
            persona_config_path=persona_config_path,
            persona_bank_path=persona_bank_path,
            persona_bank_split=persona_bank_split,
        )
        self.max_rounds = int(self._env.max_rounds)
        self._response_to_idx = {token: idx for idx, token in enumerate(RESPONSE_TOKENS)}
        self._age_to_idx = {token: idx for idx, token in enumerate(AGE_BANDS)}
        self._income_to_idx = {token: idx for idx, token in enumerate(INCOME_BANDS)}
        self._household_to_idx = {token: idx for idx, token in enumerate(HOUSEHOLD_STAGES)}
        self._ownership_to_idx = {token: idx for idx, token in enumerate(OWNERSHIP_STAGES)}
        self._use_case_to_idx = {token: idx for idx, token in enumerate(PRIMARY_USE_CASES)}

        option_keys = sorted(option.key for option in self._env.catalog.options)
        self._option_keys = option_keys
        self._option_to_idx = {key: idx for idx, key in enumerate(option_keys)}
        self._option_by_key = {option.key: option for option in self._env.catalog.options}

        self._clip_enabled = bool(clip_enabled)
        self._clip_legacy_proxy_enabled = bool(clip_legacy_proxy_enabled)
        self._clip_semantic_dim = 0
        self._clip_semantics_version = "disabled"
        self._clip_axis_labels: List[str] = []
        self._clip_semantic_by_option: Dict[str, np.ndarray] = {}
        self._clip_projection_weights = np.zeros((0,), dtype=np.float32)
        if self._clip_enabled:
            if clip_semantic_path is None:
                raise ValueError("`clip_semantic_path` is required when `clip_enabled` is true.")
            (
                self._clip_axis_labels,
                self._clip_semantic_by_option,
                self._clip_projection_weights,
                self._clip_semantics_version,
            ) = self._load_clip_semantics_artifact(
                path=Path(clip_semantic_path),
                option_keys=self._option_keys,
            )
            self._clip_semantic_dim = int(len(self._clip_axis_labels))

        min_total_cost, max_total_cost = self._estimate_total_cost_bounds()
        self._min_total_cost = float(min_total_cost)
        self._max_total_cost = float(max_total_cost)
        default_min = max(100.0, self._min_total_cost * 0.4)
        default_max = max(default_min + 500.0, self._max_total_cost * 3.0, 60000.0)
        self._offer_price_min = float(default_min if offer_price_min_usd is None else offer_price_min_usd)
        self._offer_price_max = float(default_max if offer_price_max_usd is None else offer_price_max_usd)

        obs_dim = 13 + len(self._option_keys) + int(self._clip_semantic_dim)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(obs_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.Dict(
            {
                "move": spaces.Discrete(len(MOVE_TOKENS)),
                "price": spaces.Box(
                    low=np.array([self._offer_price_min], dtype=np.float32),
                    high=np.array([self._offer_price_max], dtype=np.float32),
                    shape=(1,),
                    dtype=np.float32,
                ),
            }
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: Dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, Dict[str, Any]]:
        """Resets wrapped environment and returns encoded observation.

        Args:
            seed: Optional seed used to reseed wrapped RNG.
            options: Optional reset options.
                Supported keys: `selected_option_keys`, `persona`.

        Returns:
            Encoded observation vector and auxiliary info.
        """

        super().reset(seed=seed)
        if seed is not None:
            self._env.rng.seed(int(seed))
        selected_option_keys = None
        persona = None
        if options is not None:
            selected_option_keys = options.get("selected_option_keys")
            persona = options.get("persona")
            if persona is not None and not isinstance(persona, PersonaProfile):
                raise ValueError("`options['persona']` must be a PersonaProfile instance.")

        obs_dict_raw = self._env.reset(selected_option_keys=selected_option_keys, persona=persona)
        obs_dict = self._augment_observation_with_clip(obs_dict_raw)
        obs = self._encode_observation(obs_dict)
        info: Dict[str, Any] = {"raw_observation": obs_dict}
        info.update(self._env.current_persona_metadata())
        return obs, info

    def step(self, action: Mapping[str, Any]) -> tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Takes one step in wrapped environment.

        Args:
            action: Action mapping with:
                - `move`: discrete move id (`0=offer, 1=accept, 2=walkaway`)
                - `price`: one-element array or scalar offer price in USD

        Returns:
            Encoded observation, reward, terminated, truncated, and info.
        """

        move_idx = int(action["move"])
        if move_idx < 0 or move_idx >= len(MOVE_TOKENS):
            raise ValueError(f"Invalid move index {move_idx}.")
        move_token = MOVE_TOKENS[move_idx]

        raw_price = action["price"]
        price_scalar = float(raw_price[0] if isinstance(raw_price, (list, tuple, np.ndarray)) else raw_price)
        clipped_price = float(np.clip(price_scalar, self._offer_price_min, self._offer_price_max))
        env_action = EnvAction(
            move=move_token,
            price_offer_usd=clipped_price if move_token == "offer" else 0.0,
        )

        obs_dict_raw, reward, done, info = self._env.step(env_action)
        obs_dict = self._augment_observation_with_clip(obs_dict_raw)
        obs = self._encode_observation(obs_dict)
        terminated, truncated = self._split_done(done=done, obs_dict=obs_dict, info=info)
        info["raw_observation"] = obs_dict
        return obs, float(reward), bool(terminated), bool(truncated), info

    def close(self) -> None:
        """Closes wrapper resources."""

        return None

    def _encode_observation(self, obs: ObservationDict) -> np.ndarray:
        """Encodes raw dict observation into fixed-size numeric vector.

        Args:
            obs: Raw observation dictionary from wrapped environment.

        Returns:
            Normalized float32 observation vector.
        """

        response_idx = self._safe_index(self._response_to_idx, str(obs.get("last_consumer_response", "none")), "none")
        age_idx = self._safe_index(self._age_to_idx, str(obs.get("persona_age_band", "unknown")), "unknown")
        income_idx = self._safe_index(self._income_to_idx, str(obs.get("persona_income_band", "unknown")), "unknown")
        household_idx = self._safe_index(
            self._household_to_idx,
            str(obs.get("persona_household_stage", "unknown")),
            "unknown",
        )
        ownership_idx = self._safe_index(
            self._ownership_to_idx,
            str(obs.get("persona_ownership_stage", "unknown")),
            "unknown",
        )
        use_case_idx = self._safe_index(
            self._use_case_to_idx,
            str(obs.get("persona_primary_use_case", "unknown")),
            "unknown",
        )

        round_idx = float(obs.get("round_idx", 1))
        remaining_rounds = float(obs.get("remaining_rounds", self.max_rounds))
        last_offer = float(obs.get("last_agent_offer_usd") or 0.0)
        last_counter = float(obs.get("last_consumer_offer_usd") or 0.0)
        history_len = float(obs.get("history_len", 0))
        total_cost = float(obs.get("total_customization_cost_usd", 0.0))
        if self._clip_enabled and not self._clip_legacy_proxy_enabled:
            aesthetic = float(obs.get("aesthetic_clip_score", obs.get("aesthetic_proxy_score", 0.0)))
        elif self._clip_enabled:
            aesthetic = float(obs.get("aesthetic_proxy_score", 0.0))
        else:
            aesthetic = float(obs.get("aesthetic_proxy_score", 0.0))
        selected_option_keys = obs.get("selected_option_keys") or []

        option_multi_hot = np.zeros((len(self._option_keys),), dtype=np.float32)
        for key in selected_option_keys:
            idx = self._option_to_idx.get(str(key))
            if idx is not None:
                option_multi_hot[idx] = 1.0
        clip_semantic_vector = self._extract_clip_semantic_vector(obs)

        base = np.array(
            [
                self._normalize(round_idx, 1.0, float(max(self.max_rounds, 1))),
                self._normalize(remaining_rounds, 0.0, float(max(self.max_rounds, 1))),
                self._normalize(last_offer, self._offer_price_min, self._offer_price_max),
                self._normalize(last_counter, self._offer_price_min, self._offer_price_max),
                self._normalize(history_len, 0.0, float(max(1, self.max_rounds * 2))),
                self._normalize(total_cost, self._min_total_cost, self._max_total_cost),
                self._normalize(aesthetic, 0.0, 1.0),
                self._normalize(float(response_idx), 0.0, float(max(1, len(RESPONSE_TOKENS) - 1))),
                self._normalize(float(age_idx), 0.0, float(max(1, len(AGE_BANDS) - 1))),
                self._normalize(float(income_idx), 0.0, float(max(1, len(INCOME_BANDS) - 1))),
                self._normalize(float(household_idx), 0.0, float(max(1, len(HOUSEHOLD_STAGES) - 1))),
                self._normalize(float(ownership_idx), 0.0, float(max(1, len(OWNERSHIP_STAGES) - 1))),
                self._normalize(float(use_case_idx), 0.0, float(max(1, len(PRIMARY_USE_CASES) - 1))),
            ],
            dtype=np.float32,
        )
        return np.concatenate([base, option_multi_hot, clip_semantic_vector], axis=0, dtype=np.float32)

    def _estimate_total_cost_bounds(self) -> tuple[float, float]:
        """Estimates min/max customization totals using dimension-wise extremes.

        Returns:
            Lower and upper bounds of total customization cost.
        """

        min_by_dim: Dict[str, float] = defaultdict(lambda: float("inf"))
        max_by_dim: Dict[str, float] = defaultdict(lambda: float("-inf"))
        ratio = float(self._env.catalog.implementation_cost_ratio)
        for option in self._env.catalog.options:
            dim = option.dimension
            cost = float(option.price_delta_usd) * ratio
            min_by_dim[dim] = min(min_by_dim[dim], cost)
            max_by_dim[dim] = max(max_by_dim[dim], cost)
        min_total = sum(min_by_dim.values()) if min_by_dim else 0.0
        max_total = sum(max_by_dim.values()) if max_by_dim else 1.0
        return float(min_total), float(max(max_total, min_total + 1.0))

    def _split_done(self, *, done: bool, obs_dict: ObservationDict, info: Dict[str, Any]) -> tuple[bool, bool]:
        """Splits `done` into Gymnasium `terminated` and `truncated`.

        Args:
            done: Original done flag from wrapped environment.
            obs_dict: Current raw observation.
            info: Current info dictionary.

        Returns:
            `(terminated, truncated)` tuple.
        """

        if not done:
            return False, False
        response = str(obs_dict.get("last_consumer_response", ""))
        termination_cause = str(info.get("termination_cause", ""))
        is_timeout = response == "timeout" or termination_cause in {"env_round_timeout", "mechanism_timeout"}
        return (not is_timeout), is_timeout

    @staticmethod
    def _normalize(value: float, low: float, high: float) -> float:
        """Normalizes a scalar value to [0, 1].

        Args:
            value: Input scalar.
            low: Lower bound.
            high: Upper bound.

        Returns:
            Clipped normalized value.
        """

        denom = float(max(high - low, 1e-9))
        return float(np.clip((value - low) / denom, 0.0, 1.0))

    @staticmethod
    def _safe_index(mapping: Dict[str, int], value: str, default_key: str) -> int:
        """Fetches category index with default fallback.

        Args:
            mapping: Token-to-index mapping.
            value: Input token.
            default_key: Fallback token if input token is unknown.

        Returns:
            Category index.
        """

        return int(mapping.get(value, mapping[default_key]))

    def _extract_clip_semantic_vector(self, obs: ObservationDict) -> np.ndarray:
        """Extracts normalized CLIP semantic vector from observation payload.

        Args:
            obs: Raw observation dictionary.

        Returns:
            Semantic vector in `[0, 1]` with fixed length.
        """

        if self._clip_semantic_dim <= 0:
            return np.zeros((0,), dtype=np.float32)
        raw = obs.get("config_semantic_vector", [])
        if not isinstance(raw, list) or len(raw) != self._clip_semantic_dim:
            return np.zeros((self._clip_semantic_dim,), dtype=np.float32)
        values = [float(np.clip(float(value), 0.0, 1.0)) for value in raw]
        return np.asarray(values, dtype=np.float32)

    def _augment_observation_with_clip(self, obs_dict: ObservationDict) -> ObservationDict:
        """Augments raw observation with CLIP-derived configuration semantics.

        Args:
            obs_dict: Original observation from `NegotiationEnv`.

        Returns:
            Augmented observation dictionary.
        """

        payload: ObservationDict = dict(obs_dict)
        if not self._clip_enabled:
            return payload
        selected_keys = [str(key) for key in (payload.get("selected_option_keys") or [])]
        semantic_vector = self._compute_config_semantic_vector(selected_keys=selected_keys)
        clip_score = float(np.dot(semantic_vector, self._clip_projection_weights))
        payload["config_semantic_vector"] = [float(value) for value in semantic_vector.tolist()]
        payload["aesthetic_clip_score"] = float(np.clip(clip_score, 0.0, 1.0))
        payload["clip_semantics_version"] = str(self._clip_semantics_version)
        return payload

    def _compute_config_semantic_vector(self, *, selected_keys: List[str]) -> np.ndarray:
        """Computes weighted CLIP semantic mean for selected options.

        Args:
            selected_keys: Selected option keys for current configuration.

        Returns:
            Weighted semantic vector.
        """

        if self._clip_semantic_dim <= 0:
            return np.zeros((0,), dtype=np.float32)
        weighted_sum = np.zeros((self._clip_semantic_dim,), dtype=np.float32)
        total_weight = 0.0
        for key in selected_keys:
            vector = self._clip_semantic_by_option.get(key)
            option = self._option_by_key.get(key)
            if vector is None or option is None:
                continue
            weight = max(300.0, float(option.price_delta_usd))
            weighted_sum += vector * float(weight)
            total_weight += float(weight)
        if total_weight <= 1e-9:
            return np.zeros((self._clip_semantic_dim,), dtype=np.float32)
        return (weighted_sum / float(total_weight)).astype(np.float32)

    def _load_clip_semantics_artifact(
        self,
        *,
        path: Path,
        option_keys: List[str],
    ) -> tuple[List[str], Dict[str, np.ndarray], np.ndarray, str]:
        """Loads and validates offline CLIP semantics artifact.

        Args:
            path: Artifact JSON path.
            option_keys: Catalog option keys expected by runtime.

        Returns:
            Tuple of axis labels, per-option vectors, projection vector, and schema version.
        """

        if not path.exists():
            raise FileNotFoundError(f"CLIP semantics artifact not found: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("CLIP semantics artifact must be a JSON object.")
        axis_labels = payload.get("axis_labels", [])
        records = payload.get("records", [])
        if not isinstance(axis_labels, list) or not axis_labels:
            raise ValueError("`axis_labels` must be a non-empty list in CLIP artifact.")
        if not isinstance(records, list) or not records:
            raise ValueError("`records` must be a non-empty list in CLIP artifact.")

        semantic_dim = int(len(axis_labels))
        by_option: Dict[str, np.ndarray] = {}
        for record in records:
            if not isinstance(record, dict):
                raise ValueError("Each CLIP record must be a JSON object.")
            option_key = str(record.get("option_key", "")).strip()
            vector = record.get("semantic_vector", [])
            if not option_key:
                raise ValueError("Each CLIP record must contain `option_key`.")
            if not isinstance(vector, list) or len(vector) != semantic_dim:
                raise ValueError(
                    f"CLIP record `{option_key}` has invalid `semantic_vector` length; expected {semantic_dim}."
                )
            by_option[option_key] = np.asarray(
                [float(np.clip(float(value), 0.0, 1.0)) for value in vector],
                dtype=np.float32,
            )

        catalog_key_set = set(option_keys)
        artifact_key_set = set(by_option.keys())
        missing = sorted(catalog_key_set - artifact_key_set)
        extra = sorted(artifact_key_set - catalog_key_set)
        if missing or extra:
            raise ValueError(
                "CLIP artifact key coverage mismatch. "
                f"Missing keys: {missing}. Extra keys: {extra}."
            )

        projection = np.full((semantic_dim,), 1.0 / float(semantic_dim), dtype=np.float32)
        schema_version = str(payload.get("schema_version", "unknown"))
        return [str(label) for label in axis_labels], by_option, projection, schema_version
