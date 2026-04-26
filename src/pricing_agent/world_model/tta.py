"""Inference-time TTA utilities for Dreamer pricing policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping

import numpy as np

from .adapter import DreamerActionCodec


@dataclass(frozen=True)
class DreamerTTAConfig:
    """Configuration for inference-time test-time adaptation (TTA).

    Attributes:
        enabled: Whether TTA adaptation is applied.
        mode: TTA mode token. Supported modes are `belief_shift_v1` and
            `candidate_rerank_v2`.
        logit_bias_scale: Global scaling factor of `belief_shift_v1`.
        alpha_wtp: EMA update strength for estimated acceptable price ceiling.
        alpha_counter: EMA update strength for counter aggressiveness.
        alpha_risk: EMA update strength for walkaway risk.
        max_bias_abs: Maximum absolute normalized adaptation strength in
            `belief_shift_v1`.
        max_price_adjust_usd_per_round: Maximum USD adjustment magnitude for
            `belief_shift_v1`.
        max_candidates: Maximum candidate count for `candidate_rerank_v2`.
        offer_neighbor_bins: Neighboring offer bins explored around the raw
            offer bin for `candidate_rerank_v2`.
        max_price_adjust_usd: Maximum absolute offer adjustment magnitude for
            `candidate_rerank_v2`.
        imagination_horizon: Candidate imagination horizon for
            `candidate_rerank_v2`. Thesis-stage implementation supports `1`.
        w_policy: Weight of normalized policy score in candidate reranking.
        w_value: Weight of one-step imagined world-model score.
        w_margin: Weight of seller-margin score.
        w_feasibility: Weight of observable deal-feasibility score.
        w_risk: Weight of rejection-risk score.
    """

    enabled: bool = False
    mode: str = "belief_shift_v1"
    logit_bias_scale: float = 1.0
    alpha_wtp: float = 0.45
    alpha_counter: float = 0.35
    alpha_risk: float = 0.40
    max_bias_abs: float = 0.45
    max_price_adjust_usd_per_round: float = 200.0
    max_candidates: int = 8
    offer_neighbor_bins: int = 2
    max_price_adjust_usd: float = 200.0
    imagination_horizon: int = 1
    w_policy: float = 0.20
    w_value: float = 0.35
    w_margin: float = 0.20
    w_feasibility: float = 0.15
    w_risk: float = 0.10

    def __post_init__(self) -> None:
        """Validates TTA hyperparameter ranges."""

        mode = str(self.mode)
        if mode not in {"belief_shift_v1", "candidate_rerank_v2"}:
            raise ValueError(
                "`tta.mode` must be one of: `belief_shift_v1`, `candidate_rerank_v2`."
            )
        if float(self.logit_bias_scale) < 0.0:
            raise ValueError("`tta.logit_bias_scale` must be >= 0.")
        if not (0.0 <= float(self.alpha_wtp) <= 1.0):
            raise ValueError("`tta.alpha_wtp` must be in [0, 1].")
        if not (0.0 <= float(self.alpha_counter) <= 1.0):
            raise ValueError("`tta.alpha_counter` must be in [0, 1].")
        if not (0.0 <= float(self.alpha_risk) <= 1.0):
            raise ValueError("`tta.alpha_risk` must be in [0, 1].")
        if float(self.max_bias_abs) < 0.0:
            raise ValueError("`tta.max_bias_abs` must be >= 0.")
        if float(self.max_price_adjust_usd_per_round) < 0.0:
            raise ValueError("`tta.max_price_adjust_usd_per_round` must be >= 0.")
        if int(self.max_candidates) <= 0:
            raise ValueError("`tta.max_candidates` must be positive.")
        if int(self.offer_neighbor_bins) < 0:
            raise ValueError("`tta.offer_neighbor_bins` must be >= 0.")
        if float(self.max_price_adjust_usd) < 0.0:
            raise ValueError("`tta.max_price_adjust_usd` must be >= 0.")
        if int(self.imagination_horizon) != 1:
            raise ValueError("`tta.imagination_horizon` must be `1` in thesis-stage TTA v2.")
        for key, value in {
            "w_policy": self.w_policy,
            "w_value": self.w_value,
            "w_margin": self.w_margin,
            "w_feasibility": self.w_feasibility,
            "w_risk": self.w_risk,
        }.items():
            if float(value) < 0.0:
                raise ValueError(f"`tta.{key}` must be >= 0.")


class DreamerTTAAdapter:
    """Inference-only TTA adapter using observable negotiation history.

    The adapter maintains a compact belief state, provides candidate generation
    and heuristic scores for `candidate_rerank_v2`, and preserves the original
    bounded price-shift behavior for `belief_shift_v1`.
    """

    _MOVE_OFFER = 0
    _MOVE_ACCEPT = 1
    _MOVE_WALKAWAY = 2

    def __init__(
        self,
        *,
        codec: DreamerActionCodec,
        tta_config: DreamerTTAConfig,
        price_step_usd: float,
        initial_offer_markup: float,
    ) -> None:
        """Initializes the TTA adapter.

        Args:
            codec: Dreamer action codec for flatten/unflatten.
            tta_config: TTA hyperparameters.
            price_step_usd: USD represented by one delta bin step.
            initial_offer_markup: Initial anchor ratio used by environment.
        """

        self._codec = codec
        self._config = tta_config
        self._price_step_usd = float(price_step_usd)
        self._delta_center = int(codec.delta_bin_count // 2)
        self._initial_offer_markup = float(initial_offer_markup)

        self._wtp_ceiling_est: float | None = None
        self._counter_aggressiveness_est = 0.0
        self._walkaway_risk_est = 0.0
        self._reference_price_usd: float | None = None
        self._latest_raw_observation: Dict[str, Any] = {}

        self._prediction_count = 0
        self._shift_count = 0
        self._total_abs_price_adjust_usd = 0.0
        self._last_shift_steps = 0
        self._last_bias_value = 0.0

        self._candidate_count_total = 0.0
        self._selected_score_total = 0.0
        self._offer_override_count = 0.0
        self._accept_override_count = 0.0
        self._walkaway_override_count = 0.0

    @property
    def mode(self) -> str:
        """Returns current TTA mode."""

        return str(self._config.mode)

    def reset(self, *, reset_info: Mapping[str, Any] | None = None) -> None:
        """Resets per-episode belief state and counters."""

        self._wtp_ceiling_est = None
        self._counter_aggressiveness_est = 0.0
        self._walkaway_risk_est = 0.0
        self._reference_price_usd = None
        self._latest_raw_observation = {}
        self._last_shift_steps = 0
        self._last_bias_value = 0.0

        raw = self._extract_raw_observation(reset_info)
        if raw:
            self._latest_raw_observation = dict(raw)
            self._reference_price_usd = self._infer_reference_price(raw)
            if self._reference_price_usd is None:
                total_msrp = self._to_float(raw.get("total_msrp_delta_usd"))
                if total_msrp is not None:
                    self._reference_price_usd = float(total_msrp * self._initial_offer_markup)

    def observe_step_info(self, step_info: Mapping[str, Any] | None) -> None:
        """Updates belief state using one environment step info payload."""

        raw = self._extract_raw_observation(step_info)
        if not raw:
            return

        self._latest_raw_observation = dict(raw)
        self._reference_price_usd = self._infer_reference_price(raw)
        response = str(raw.get("last_consumer_response", "")).strip().lower()
        last_offer = self._to_float(raw.get("last_agent_offer_usd"))
        counter_offer = self._to_float(raw.get("last_consumer_offer_usd"))
        round_progress = self._round_progress(raw)

        if counter_offer is not None and last_offer is not None and last_offer > 0.0:
            gap_ratio = float(np.clip((last_offer - counter_offer) / max(last_offer, 1.0), 0.0, 1.0))
            self._counter_aggressiveness_est = self._ema(
                prev=self._counter_aggressiveness_est,
                new_value=gap_ratio,
                alpha=float(self._config.alpha_counter),
            )
            self._wtp_ceiling_est = self._ema_optional(
                prev=self._wtp_ceiling_est,
                new_value=counter_offer,
                alpha=float(self._config.alpha_wtp),
            )
        elif response in {"reject", "walkaway", "timeout"} and last_offer is not None:
            implied_ceiling = max(0.0, last_offer - self._price_step_usd)
            self._wtp_ceiling_est = self._ema_optional(
                prev=self._wtp_ceiling_est,
                new_value=implied_ceiling,
                alpha=float(self._config.alpha_wtp),
            )
            self._counter_aggressiveness_est = self._ema(
                prev=self._counter_aggressiveness_est,
                new_value=0.8,
                alpha=float(self._config.alpha_counter),
            )

        if response in {"walkaway", "timeout"}:
            risk_signal = 1.0
        elif response in {"counter", "reject"}:
            risk_signal = float(np.clip(0.5 + 0.5 * round_progress, 0.0, 1.0))
        elif response == "accept":
            risk_signal = 0.0
        else:
            risk_signal = float(np.clip(0.1 + 0.6 * round_progress, 0.0, 1.0))
        self._walkaway_risk_est = self._ema(
            prev=self._walkaway_risk_est,
            new_value=risk_signal,
            alpha=float(self._config.alpha_risk),
        )

    def adapt_action_index(self, action_idx: int) -> int:
        """Applies `belief_shift_v1` price shift to one flattened action.

        Args:
            action_idx: Raw predicted flattened action index.

        Returns:
            Adapted flattened action index.
        """

        if self.mode != "belief_shift_v1":
            raise RuntimeError("`adapt_action_index()` is only valid for `belief_shift_v1`.")

        self._prediction_count += 1
        move_idx, delta_idx = self._codec.unflatten(int(action_idx))
        if move_idx != self._MOVE_OFFER:
            self._last_shift_steps = 0
            self._last_bias_value = 0.0
            return int(action_idx)

        reference = float(self._current_reference_price())
        proposed_price = reference + float(delta_idx - self._delta_center) * self._price_step_usd
        shift_steps = self._compute_shift_steps(proposed_price=proposed_price)
        if shift_steps == 0:
            self._last_shift_steps = 0
            self._last_bias_value = 0.0
            return int(action_idx)

        new_delta_idx = int(np.clip(delta_idx + shift_steps, 0, self._codec.delta_bin_count - 1))
        applied_steps = int(new_delta_idx - delta_idx)
        if applied_steps != 0:
            self._shift_count += 1
            self._total_abs_price_adjust_usd += abs(applied_steps) * self._price_step_usd

        self._last_shift_steps = int(applied_steps)
        max_steps = max(1, int(round(self._config.max_price_adjust_usd_per_round / self._price_step_usd)))
        self._last_bias_value = float(np.clip(applied_steps / max_steps, -1.0, 1.0))
        return self._codec.flatten(move_idx=move_idx, delta_idx=new_delta_idx)

    def build_candidate_action_indices(self, raw_action_idx: int) -> List[int]:
        """Builds a bounded candidate set for `candidate_rerank_v2`.

        Args:
            raw_action_idx: Raw action sampled from the Dreamer policy.

        Returns:
            Ordered candidate list with no duplicates.
        """

        if self.mode != "candidate_rerank_v2":
            raise RuntimeError("`build_candidate_action_indices()` is only valid for `candidate_rerank_v2`.")

        raw_move, raw_delta = self._codec.unflatten(int(raw_action_idx))
        candidates: List[int] = []
        seen: set[int] = set()

        def add(action_idx: int) -> None:
            idx = int(action_idx)
            if idx not in seen and len(candidates) < int(self._config.max_candidates):
                seen.add(idx)
                candidates.append(idx)

        add(int(raw_action_idx))

        max_steps = int(round(float(self._config.max_price_adjust_usd) / self._price_step_usd))
        max_steps = max(0, max_steps)
        base_delta = int(raw_delta) if raw_move == self._MOVE_OFFER else int(self._delta_center)

        counter_offer = self._to_float(self._latest_raw_observation.get("last_consumer_offer_usd"))
        if counter_offer is not None:
            add(self._codec.flatten(move_idx=self._MOVE_ACCEPT, delta_idx=self._delta_center))

        if self._should_consider_walkaway():
            add(self._codec.flatten(move_idx=self._MOVE_WALKAWAY, delta_idx=self._delta_center))

        for delta_idx in self._offer_delta_candidates(anchor_delta=base_delta, base_delta=base_delta, max_steps=max_steps):
            add(self._codec.flatten(move_idx=self._MOVE_OFFER, delta_idx=delta_idx))

        if counter_offer is not None:
            counter_delta = self._price_to_delta_idx(counter_offer)
            for delta_idx in self._offer_delta_candidates(
                anchor_delta=counter_delta,
                base_delta=base_delta,
                max_steps=max_steps,
            ):
                add(self._codec.flatten(move_idx=self._MOVE_OFFER, delta_idx=delta_idx))

        return candidates

    def candidate_margin_signal(self, action_idx: int) -> float:
        """Returns unnormalized margin score for one candidate action."""

        move_idx, _ = self._codec.unflatten(int(action_idx))
        total_cost = self._required_float(self._latest_raw_observation, "total_customization_cost_usd")
        if move_idx == self._MOVE_WALKAWAY:
            return 0.0
        if move_idx == self._MOVE_ACCEPT:
            accept_price = self._to_float(self._latest_raw_observation.get("last_consumer_offer_usd"))
            if accept_price is None:
                # Invalid accept without a counter should stay a valid candidate
                # but receive a clearly unattractive margin score.
                return float(-total_cost)
            return float(accept_price - total_cost)
        price = self._candidate_price_usd(action_idx)
        if price is None:
            raise RuntimeError("Candidate price is unavailable for non-walkaway action.")
        return float(price - total_cost)

    def candidate_feasibility_signal(self, action_idx: int) -> float:
        """Returns deal-feasibility score in `[0, 1]` for one candidate."""

        move_idx, _ = self._codec.unflatten(int(action_idx))
        counter_offer = self._to_float(self._latest_raw_observation.get("last_consumer_offer_usd"))
        total_cost = self._required_float(self._latest_raw_observation, "total_customization_cost_usd")
        round_progress = self._round_progress(self._latest_raw_observation)

        if move_idx == self._MOVE_WALKAWAY:
            best_known_margin = None if counter_offer is None else float(counter_offer - total_cost)
            if best_known_margin is None or best_known_margin <= 0.0:
                return float(np.clip(0.7 + 0.3 * self._walkaway_risk_est, 0.0, 1.0))
            return float(np.clip(0.1 + 0.4 * self._walkaway_risk_est, 0.0, 1.0))

        if move_idx == self._MOVE_ACCEPT:
            if counter_offer is None:
                return 0.0
            counter_margin = float(counter_offer - total_cost)
            if counter_margin <= 0.0:
                return 0.05
            return float(np.clip(0.8 + 0.2 * (1.0 - self._walkaway_risk_est), 0.0, 1.0))

        price = self._candidate_price_usd(action_idx)
        if price is None:
            raise RuntimeError("Offer candidate price is unavailable.")
        ceiling = self._wtp_ceiling_est
        if ceiling is None and counter_offer is not None:
            ceiling = float(counter_offer)
        if ceiling is None:
            reference = self._current_reference_price()
            gap_ratio = max(0.0, float(price - reference) / max(reference, 1.0))
            base = 0.65 - 0.35 * gap_ratio
        else:
            gap_ratio = max(0.0, float(price - ceiling) / max(abs(ceiling), 1.0))
            base = 1.0 - min(1.0, 3.5 * gap_ratio)
        base -= 0.25 * self._counter_aggressiveness_est
        base -= 0.20 * self._walkaway_risk_est * round_progress
        return float(np.clip(base, 0.0, 1.0))

    def candidate_risk_signal(self, action_idx: int) -> float:
        """Returns rejection-risk score in `[0, 1]` for one candidate."""

        move_idx, _ = self._codec.unflatten(int(action_idx))
        counter_offer = self._to_float(self._latest_raw_observation.get("last_consumer_offer_usd"))
        round_progress = self._round_progress(self._latest_raw_observation)

        if move_idx == self._MOVE_WALKAWAY:
            return 0.0
        if move_idx == self._MOVE_ACCEPT:
            return 0.05 if counter_offer is not None else 1.0

        price = self._candidate_price_usd(action_idx)
        if price is None:
            raise RuntimeError("Offer candidate price is unavailable.")
        reference = counter_offer if counter_offer is not None else self._wtp_ceiling_est
        if reference is None:
            reference = self._current_reference_price()
        gap_ratio = max(0.0, float(price - reference) / max(abs(reference), 1.0))
        risk = (
            0.15
            + 0.35 * self._walkaway_risk_est
            + 0.20 * self._counter_aggressiveness_est
            + 0.20 * round_progress
            + 1.20 * gap_ratio
        )
        return float(np.clip(risk, 0.0, 1.0))

    def finalize_rerank_selection(
        self,
        *,
        raw_action_idx: int,
        selected_action_idx: int,
        candidate_count: int,
        selected_score: float,
    ) -> None:
        """Updates TTA v2 aggregate metrics after one selection.

        Args:
            raw_action_idx: Raw action sampled from the policy.
            selected_action_idx: Final action after reranking.
            candidate_count: Candidate count used in this step.
            selected_score: Final selected reranking score.
        """

        self._prediction_count += 1
        self._candidate_count_total += float(candidate_count)
        self._selected_score_total += float(selected_score)

        raw_move, _ = self._codec.unflatten(int(raw_action_idx))
        selected_move, _ = self._codec.unflatten(int(selected_action_idx))
        raw_price = self._candidate_price_usd(int(raw_action_idx))
        selected_price = self._candidate_price_usd(int(selected_action_idx))
        price_adjust = 0.0 if raw_price is None or selected_price is None else abs(selected_price - raw_price)

        self._last_shift_steps = 0
        self._last_bias_value = 0.0
        if int(selected_action_idx) != int(raw_action_idx):
            self._shift_count += 1
            self._total_abs_price_adjust_usd += float(price_adjust)
            if selected_move == self._MOVE_OFFER:
                self._offer_override_count += 1.0
            elif selected_move == self._MOVE_ACCEPT:
                self._accept_override_count += 1.0
            elif selected_move == self._MOVE_WALKAWAY:
                self._walkaway_override_count += 1.0

            if raw_move == self._MOVE_OFFER and selected_move == self._MOVE_OFFER:
                raw_delta = self._codec.unflatten(int(raw_action_idx))[1]
                selected_delta = self._codec.unflatten(int(selected_action_idx))[1]
                delta_steps = int(selected_delta - raw_delta)
                self._last_shift_steps = delta_steps
                max_steps = max(1, int(round(self._config.max_price_adjust_usd / self._price_step_usd)))
                self._last_bias_value = float(np.clip(delta_steps / max_steps, -1.0, 1.0))

    def metrics(self) -> Dict[str, float]:
        """Returns aggregate TTA metrics for reporting."""

        denom = float(max(1, self._prediction_count))
        return {
            "prediction_count": float(self._prediction_count),
            "action_shift_count": float(self._shift_count),
            "action_shift_rate": float(self._shift_count / denom),
            "avg_abs_price_adjust_usd": float(self._total_abs_price_adjust_usd / denom),
            "last_shift_steps": float(self._last_shift_steps),
            "last_bias_value": float(self._last_bias_value),
            "belief_wtp_ceiling_est": float(self._wtp_ceiling_est) if self._wtp_ceiling_est is not None else -1.0,
            "belief_counter_aggressiveness_est": float(self._counter_aggressiveness_est),
            "belief_walkaway_risk_est": float(self._walkaway_risk_est),
            "candidate_count_avg": float(self._candidate_count_total / denom),
            "avg_selected_score": float(self._selected_score_total / denom),
            "offer_override_count": float(self._offer_override_count),
            "accept_override_count": float(self._accept_override_count),
            "walkaway_override_count": float(self._walkaway_override_count),
        }

    def _compute_shift_steps(self, *, proposed_price: float) -> int:
        """Computes bounded delta-index shift from current belief state."""

        max_steps = int(round(float(self._config.max_price_adjust_usd_per_round) / self._price_step_usd))
        if max_steps <= 0:
            return 0

        bias_core = (
            float(self._config.alpha_counter) * self._counter_aggressiveness_est
            + float(self._config.alpha_risk) * self._walkaway_risk_est
        )
        bias_core *= float(self._config.logit_bias_scale)
        bias_core = float(np.clip(bias_core, -float(self._config.max_bias_abs), float(self._config.max_bias_abs)))
        shift_steps = -int(round(bias_core * max_steps))

        if self._wtp_ceiling_est is not None:
            gap_usd = proposed_price - float(self._wtp_ceiling_est)
            if gap_usd > 0.0:
                shift_steps -= int(np.ceil(gap_usd / max(self._price_step_usd, 1e-6)))

        return int(np.clip(shift_steps, -max_steps, max_steps))

    def _offer_delta_candidates(self, *, anchor_delta: int, base_delta: int, max_steps: int) -> List[int]:
        """Builds ordered offer delta candidates around one anchor bin."""

        lower = max(0, int(base_delta) - int(max_steps))
        upper = min(int(self._codec.delta_bin_count) - 1, int(base_delta) + int(max_steps))
        ordered = [int(anchor_delta)]
        for offset in range(1, int(self._config.offer_neighbor_bins) + 1):
            ordered.append(int(anchor_delta) - offset)
            ordered.append(int(anchor_delta) + offset)
        result: List[int] = []
        seen: set[int] = set()
        for delta_idx in ordered:
            if lower <= int(delta_idx) <= upper and int(delta_idx) not in seen:
                seen.add(int(delta_idx))
                result.append(int(delta_idx))
        return result

    def _candidate_price_usd(self, action_idx: int) -> float | None:
        """Returns concrete candidate price in USD when defined."""

        move_idx, delta_idx = self._codec.unflatten(int(action_idx))
        if move_idx == self._MOVE_OFFER:
            reference = self._current_reference_price()
            return float(reference + (int(delta_idx) - self._delta_center) * self._price_step_usd)
        if move_idx == self._MOVE_ACCEPT:
            return self._to_float(self._latest_raw_observation.get("last_consumer_offer_usd"))
        return None

    def _current_reference_price(self) -> float:
        """Returns current reference price for offer deltas.

        Raises:
            RuntimeError: If no observable reference price can be inferred.
        """

        if self._reference_price_usd is not None:
            return float(self._reference_price_usd)
        total_msrp = self._to_float(self._latest_raw_observation.get("total_msrp_delta_usd"))
        if total_msrp is None:
            raise RuntimeError(
                "TTA requires `raw_observation.total_msrp_delta_usd` to infer reference price."
            )
        self._reference_price_usd = float(total_msrp * self._initial_offer_markup)
        return float(self._reference_price_usd)

    def _price_to_delta_idx(self, price_usd: float) -> int:
        """Converts a concrete offer price to the nearest delta bin index."""

        reference = self._current_reference_price()
        delta_steps = int(round((float(price_usd) - reference) / self._price_step_usd))
        return int(np.clip(self._delta_center + delta_steps, 0, self._codec.delta_bin_count - 1))

    def _should_consider_walkaway(self) -> bool:
        """Returns whether walkaway should be included as a candidate."""

        round_idx = int(self._latest_raw_observation.get("round_idx", 1) or 1)
        return round_idx >= 4 or float(self._walkaway_risk_est) >= 0.85

    def _infer_reference_price(self, raw: Mapping[str, Any]) -> float | None:
        """Infers current reference price from observable negotiation state."""

        counter = self._to_float(raw.get("last_consumer_offer_usd"))
        if counter is not None:
            return counter
        last_offer = self._to_float(raw.get("last_agent_offer_usd"))
        if last_offer is not None:
            return last_offer
        return None

    @staticmethod
    def _extract_raw_observation(payload: Mapping[str, Any] | None) -> Mapping[str, Any]:
        """Extracts `raw_observation` mapping from step/reset payload."""

        if not isinstance(payload, Mapping):
            return {}
        raw = payload.get("raw_observation")
        if isinstance(raw, Mapping):
            return raw
        return {}

    @staticmethod
    def _to_float(value: Any) -> float | None:
        """Safely casts one numeric-like value to float."""

        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _required_float(raw: Mapping[str, Any], key: str) -> float:
        """Reads one required numeric value from `raw_observation`.

        Args:
            raw: Raw observation mapping.
            key: Required key name.

        Returns:
            Parsed float value.

        Raises:
            RuntimeError: If the key is missing or invalid.
        """

        value = raw.get(key)
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"TTA requires numeric `raw_observation.{key}`.") from exc

    @staticmethod
    def _round_progress(raw: Mapping[str, Any]) -> float:
        """Returns normalized round progress in `[0, 1]`."""

        round_idx = int(raw.get("round_idx", 1) or 1)
        remaining_rounds = int(raw.get("remaining_rounds", 0) or 0)
        max_rounds = max(1, round_idx + remaining_rounds)
        return float(np.clip((round_idx - 1) / max_rounds, 0.0, 1.0))

    @staticmethod
    def _ema(*, prev: float, new_value: float, alpha: float) -> float:
        """Computes one-step exponential moving average."""

        return float((1.0 - alpha) * float(prev) + alpha * float(new_value))

    @classmethod
    def _ema_optional(cls, *, prev: float | None, new_value: float, alpha: float) -> float:
        """Computes EMA for optional previous value."""

        if prev is None:
            return float(new_value)
        return cls._ema(prev=float(prev), new_value=float(new_value), alpha=float(alpha))
