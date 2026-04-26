"""Minimal negotiation environment using NegMAS as the only backend."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import random
from typing import Any, Dict, List, Optional, Tuple

from .catalog import Catalog, load_catalog
from .negmas_backend import NEGMA_AVAILABLE, NegMASRoundBackend, NegMASSession
from .persona import load_persona_sampler
from .types import CatalogOption, EnvAction, EnvObservation, EpisodeMetrics, ObservationDict, PersonaProfile
from .wtp import compute_seller_utility, compute_wtp_breakdown

class NegotiationEnv:
    """Runs episode-level pricing negotiations for customization options.

    This class runs the negotiation loop through NegMAS for every offer step.
    """

    def __init__(
        self,
        catalog_path: str | Path,
        persona_config_path: str | Path,
        persona_bank_path: str | Path | None = None,
        persona_bank_split: str = "train",
    ) -> None:
        """Initializes the environment.

        Args:
            catalog_path: YAML path for canonical customization catalog.
            persona_config_path: YAML path for persona sampling config.
            persona_bank_path: Optional JSONL persona-bank path.
            persona_bank_split: Split label used when persona-bank sampling is enabled.
        """

        if not NEGMA_AVAILABLE:
            raise RuntimeError(
                "NegMAS is required but unavailable in this environment. "
                "Install `negmas` before creating NegotiationEnv."
            )

        self.catalog: Catalog = load_catalog(catalog_path)
        self.seed, self.persona_config, self.persona_sampler = load_persona_sampler(
            persona_config_path,
            persona_bank_path=persona_bank_path,
            persona_bank_split=persona_bank_split,
        )
        self.rng = random.Random(self.seed)

        self.max_rounds = int(self.persona_config["sampling"]["max_rounds"])
        self.value_markup = float(self.persona_config["sampling"]["value_markup"])
        self.time_penalty = float(self.persona_config["sampling"]["time_penalty_usd_per_round"])
        self.noise_sigma = float(self.persona_config["sampling"]["noise_sigma_usd"])
        self._negmas_backend = NegMASRoundBackend(n_steps_per_episode=max(16, self.max_rounds * 4))

        self._selected_options: List[CatalogOption] = []
        self._persona: Optional[PersonaProfile] = None
        self._session: Optional[NegMASSession] = None
        self._round_idx = 0
        self._done = False
        self._last_offer: Optional[float] = None
        self._last_response = "none"
        self._last_counter_offer: Optional[float] = None
        self._last_consumer_offer: Optional[float] = None
        self._metrics: Optional[EpisodeMetrics] = None
        self._episode_trace: List[Dict[str, Any]] = []

    def reset(
        self,
        selected_option_keys: Optional[List[str]] = None,
        persona: Optional[PersonaProfile] = None,
    ) -> ObservationDict:
        """Resets the environment and starts a new episode.

        Args:
            selected_option_keys: Optional fixed configuration keys.
            persona: Optional fixed persona for deterministic experiments.

        Returns:
            Initial observation dictionary.
        """

        self._selected_options = self._resolve_configuration(selected_option_keys)
        self._persona = persona if persona is not None else self.persona_sampler.sample(self.rng)
        self._round_idx = 1
        self._done = False
        self._last_offer = None
        self._last_response = "none"
        self._last_counter_offer = None
        self._last_consumer_offer = None
        self._metrics = None
        self._episode_trace = []
        self._session = self._create_negmas_session()
        return self._obs_dict()

    def step(self, action: EnvAction) -> Tuple[ObservationDict, float, bool, Dict[str, Any]]:
        """Executes one negotiation step.

        Args:
            action: Environment action.

        Returns:
            Tuple of observation, reward, done flag, and extra info.
        """

        if self._done:
            raise RuntimeError("Episode already terminated. Call reset() before step().")
        if self._persona is None:
            raise RuntimeError("Environment is not initialized. Call reset() before step().")
        if self._session is None:
            raise RuntimeError("NegMAS session is not initialized. Call reset() before step().")
        if action.move not in {"offer", "accept", "walkaway"}:
            raise ValueError("Invalid move. Expected one of: offer, accept, walkaway.")

        info: Dict[str, Any] = {"backend": "negmas"}
        info["persona_id"] = self._persona.persona_id
        info["persona_source"] = self._persona.persona_source
        info["persona_split"] = self._persona.persona_split

        total_msrp_delta = self.catalog.total_msrp_delta(self._selected_options)
        total_impl_cost = self.catalog.total_implementation_cost(self._selected_options)
        info["total_msrp_delta_usd"] = float(total_msrp_delta)
        info["total_implementation_cost_usd"] = float(total_impl_cost)
        aesth = self.catalog.aesthetic_proxy(self._selected_options)
        feature_signals = self._feature_signals(self._selected_options)
        feature_match_score = sum(
            float(self._persona.feature_weight_vector.get(key, 0.0)) * float(feature_signals.get(key, 0.0))
            for key in ("safety", "comfort", "performance", "tech", "aesthetics")
        )
        tech_signal = float(feature_signals.get("tech", 0.0))
        wtp_snapshot = compute_wtp_breakdown(
            persona=self._persona,
            total_cost_usd=total_msrp_delta,
            aesthetic_proxy_score=aesth,
            feature_match_score=feature_match_score,
            tech_signal=tech_signal,
            value_markup=self.value_markup,
            time_penalty_usd_per_round=self.time_penalty,
            round_idx=self._round_idx,
            noise_sigma_usd=self.noise_sigma,
            rng=self.rng,
        )
        wtp = wtp_snapshot.wtp_usd
        reward = 0.0
        deal_reached = False
        walkaway = False
        deal_profit_usd: Optional[float] = None
        trace_event: Dict[str, Any] = {
            "round_idx": int(self._round_idx),
            "agent_move": str(action.move),
            "agent_offer_usd": None,
            "consumer_response": None,
            "consumer_counter_offer_usd": None,
            "seller_utility_usd": None,
            "termination_cause": None,
        }

        if action.move == "walkaway":
            self._last_response = "agent_walkaway"
            self._done = True
            walkaway = True
            reward = -50.0
            self._session.close("agent_walkaway")
            trace_event["consumer_response"] = "agent_walkaway"
            trace_event["termination_cause"] = "agent_walkaway"

        elif action.move == "accept":
            if self._last_counter_offer is None:
                self._last_response = "invalid_accept"
                reward = -20.0
                trace_event["consumer_response"] = "invalid_accept"
            else:
                final_price = self._last_counter_offer
                profit = final_price - total_impl_cost
                seller_utility = compute_seller_utility(
                    price_usd=final_price,
                    customization_cost_usd=total_impl_cost,
                    time_penalty_usd_per_round=self.time_penalty,
                    round_idx=self._round_idx,
                )
                info["seller_utility_usd"] = seller_utility
                self._last_response = "accept"
                self._done = True
                deal_reached = True
                reward = profit
                deal_profit_usd = float(profit)
                info["final_price_usd"] = final_price
                self._session.close("agent_accept_counter")
                trace_event["consumer_response"] = "accept"
                trace_event["consumer_counter_offer_usd"] = final_price
                trace_event["seller_utility_usd"] = seller_utility
                trace_event["termination_cause"] = "agent_accept_counter"

        else:  # offer
            self._last_offer = float(action.price_offer_usd)
            trace_event["agent_offer_usd"] = self._last_offer
            result = self._session.offer_round(
                agent_offer_usd=self._last_offer,
                wtp_usd=wtp,
                walkaway_threshold=self._persona.walkaway_threshold,
                counter_strength=self._persona.counter_strength,
                price_sensitivity=self._persona.price_sensitivity,
                customization_cost_usd=total_impl_cost,
                time_penalty_usd_per_round=self.time_penalty,
                round_idx=self._round_idx,
            )
            info["negmas_session_step"] = result.mechanism_step
            info["history_len"] = result.history_len
            if result.seller_utility_at_offer is not None:
                info["seller_utility_at_offer_usd"] = result.seller_utility_at_offer
                trace_event["seller_utility_usd"] = result.seller_utility_at_offer
            if result.termination_cause is not None:
                info["termination_cause"] = result.termination_cause
                trace_event["termination_cause"] = result.termination_cause
            trace_event["consumer_response"] = result.response
            trace_event["consumer_counter_offer_usd"] = result.counter_offer_usd
            if result.response == "accept" and result.deal_price_usd is not None:
                self._last_response = "accept"
                self._done = True
                deal_reached = True
                reward = result.deal_price_usd - total_impl_cost
                deal_profit_usd = float(reward)
                info["final_price_usd"] = result.deal_price_usd
                seller_utility = compute_seller_utility(
                    price_usd=result.deal_price_usd,
                    customization_cost_usd=total_impl_cost,
                    time_penalty_usd_per_round=self.time_penalty,
                    round_idx=self._round_idx,
                )
                info["seller_utility_usd"] = seller_utility
                trace_event["seller_utility_usd"] = seller_utility
            elif result.walked_away:
                self._last_response = "walkaway"
                self._done = True
                walkaway = True
                reward = -30.0
            elif result.response == "timeout":
                self._last_response = "timeout"
                self._done = True
                walkaway = True
                reward = -25.0
                self._last_counter_offer = None
                self._last_consumer_offer = None
            else:
                self._last_counter_offer = result.counter_offer_usd
                self._last_consumer_offer = result.counter_offer_usd
                self._last_response = "counter" if result.counter_offer_usd is not None else "reject"
                reward = -5.0 if result.counter_offer_usd is not None else -8.0
                if result.counter_offer_usd is not None:
                    info["counter_offer_usd"] = result.counter_offer_usd

        if not self._done:
            self._round_idx += 1
            if self._round_idx > min(self.max_rounds, self._persona.patience):
                self._done = True
                walkaway = True
                self._last_response = "timeout"
                reward += -25.0
                if self._session is not None:
                    self._session.close("env_round_timeout")
                if trace_event["consumer_response"] is None:
                    trace_event["consumer_response"] = "timeout"
                trace_event["termination_cause"] = "env_round_timeout"

        if trace_event["consumer_response"] is None:
            trace_event["consumer_response"] = self._last_response
        self._episode_trace.append(dict(trace_event))
        info["trace_event"] = dict(trace_event)

        if self._done:
            self._metrics = EpisodeMetrics(
                profit_usd=float(deal_profit_usd if deal_reached and deal_profit_usd is not None else 0.0),
                deal_reached=deal_reached,
                rounds_used=self._round_idx,
                walkaway=walkaway,
            )
            info["episode_metrics"] = asdict(self._metrics)
            info["episode_trace"] = [dict(event) for event in self._episode_trace]
            info["trace_len"] = len(self._episode_trace)

        return self._obs_dict(), float(reward), self._done, info

    def current_persona_metadata(self) -> Dict[str, str]:
        """Returns current episode persona metadata.

        Returns:
            Persona metadata dictionary. Empty if episode is not initialized.
        """

        if self._persona is None:
            return {}
        return {
            "persona_id": self._persona.persona_id,
            "persona_source": self._persona.persona_source,
            "persona_split": self._persona.persona_split,
        }

    def latest_metrics(self) -> Optional[EpisodeMetrics]:
        """Returns metrics for the latest finished episode.

        Returns:
            Episode metrics if the episode has finished, else None.
        """

        return self._metrics

    def _resolve_configuration(self, selected_option_keys: Optional[List[str]]) -> List[CatalogOption]:
        """Resolves selected options from keys or random sampling.

        Args:
            selected_option_keys: Optional list of fixed option keys.

        Returns:
            Selected option list.
        """

        if selected_option_keys is None:
            return self.catalog.sample_configuration(self.rng)

        by_key = {option.key: option for option in self.catalog.options}
        missing = [key for key in selected_option_keys if key not in by_key]
        if missing:
            raise ValueError(f"Unknown option keys: {missing}")
        return [by_key[key] for key in selected_option_keys]

    def _obs_dict(self) -> ObservationDict:
        """Builds current observation dictionary."""

        history_len = self._session.history_len if self._session is not None else 0
        observation = EnvObservation(
            round_idx=self._round_idx,
            remaining_rounds=max(0, self.max_rounds - self._round_idx + 1),
            last_agent_offer_usd=self._last_offer,
            last_consumer_response=self._last_response,
            last_consumer_offer_usd=self._last_consumer_offer,
            history_len=history_len,
            selected_option_keys=[option.key for option in self._selected_options],
            total_customization_cost_usd=self.catalog.total_cost(self._selected_options),
            total_msrp_delta_usd=self.catalog.total_msrp_delta(self._selected_options),
            aesthetic_proxy_score=self.catalog.aesthetic_proxy(self._selected_options),
            persona_age_band=self._persona.age_band if self._persona is not None else "unknown",
            persona_income_band=self._persona.income_band if self._persona is not None else "unknown",
            persona_household_stage=self._persona.household_stage if self._persona is not None else "unknown",
            persona_ownership_stage=self._persona.ownership_stage if self._persona is not None else "unknown",
            persona_primary_use_case=self._persona.primary_use_case if self._persona is not None else "unknown",
        )
        return asdict(observation)

    def _create_negmas_session(self) -> NegMASSession:
        """Creates one persistent NegMAS session for the current episode.

        Returns:
            Initialized NegMAS session.
        """

        if self._persona is None:
            raise RuntimeError("Persona must be initialized before creating a session.")
        total_msrp_delta = self.catalog.total_msrp_delta(self._selected_options)
        issue_min = max(100.0, total_msrp_delta * 0.4)
        issue_max = max(issue_min + 500.0, total_msrp_delta * 3.0, self._persona.reservation_price_base * 4.0)
        return self._negmas_backend.create_session(
            issue_min_price=issue_min,
            issue_max_price=issue_max,
            rng=self.rng,
        )

    def _feature_signals(self, selected_options: List[CatalogOption]) -> Dict[str, float]:
        """Builds normalized feature-channel signals from selected options.

        Args:
            selected_options: Selected customization options.

        Returns:
            Normalized feature-channel strengths in [0, 1].
        """

        category_by_dimension = {
            "paint_color": "aesthetics",
            "wheels": "aesthetics",
            "exterior_style": "aesthetics",
            "lighting": "aesthetics",
            "upholstery": "comfort",
            "trim": "comfort",
            "comfort": "comfort",
            "audio": "comfort",
            "technology": "tech",
            "safety": "safety",
            "performance": "performance",
        }
        signals = {"safety": 0.0, "comfort": 0.0, "performance": 0.0, "tech": 0.0, "aesthetics": 0.0}
        total_weight = 0.0
        for option in selected_options:
            category = category_by_dimension.get(option.dimension)
            if category is None:
                continue
            # Keep zero-cost options visible by assigning a minimum structural weight.
            weight = max(300.0, float(option.price_delta_usd))
            signals[category] += weight
            total_weight += weight

        if total_weight <= 1e-9:
            return {key: 0.0 for key in signals}
        return {key: value / total_weight for key, value in signals.items()}
