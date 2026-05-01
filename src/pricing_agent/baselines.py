"""Checkpoint-free heuristic policies for pricing negotiation."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any, Mapping

from pricing_env.types import EnvAction


@dataclass
class PolicyMetrics:
    """Aggregated metrics for one baseline policy run."""

    episodes: int
    deal_rate: float
    avg_profit_usd: float
    avg_rounds: float
    walkaway_rate: float
    avg_trace_len: float
    avg_env_reward: float


class RandomPolicy:
    """Random seller policy used as a sanity baseline."""

    def __init__(self, *, rng: random.Random, offer_min: float, offer_max: float) -> None:
        self._rng = rng
        self._offer_min = float(offer_min)
        self._offer_max = float(offer_max)

    def act(self, observation: Mapping[str, Any]) -> EnvAction:
        """Returns one random environment action."""

        has_counter = observation.get("last_consumer_offer_usd") is not None
        if has_counter and self._rng.random() < 0.12:
            return EnvAction(move="accept", price_offer_usd=0.0)
        if self._rng.random() < 0.08:
            return EnvAction(move="walkaway", price_offer_usd=0.0)
        offer = self._rng.uniform(self._offer_min, self._offer_max)
        return EnvAction(move="offer", price_offer_usd=float(offer))


class ConcessionPolicy:
    """Simple concession policy used as the main heuristic reference."""

    def __init__(self, *, rng: random.Random, offer_min: float, offer_max: float) -> None:
        self._rng = rng
        self._offer_min = float(offer_min)
        self._offer_max = float(offer_max)

    def act(self, observation: Mapping[str, Any]) -> EnvAction:
        """Returns one concession-based environment action."""

        round_idx = int(observation.get("round_idx", 1))
        remaining = int(observation.get("remaining_rounds", 1))
        max_rounds = max(1, round_idx + remaining - 1)
        total_msrp_delta = float(
            observation.get(
                "total_msrp_delta_usd",
                observation.get("total_customization_cost_usd", 0.0),
            )
        )
        last_counter = observation.get("last_consumer_offer_usd")
        counter_offer = float(last_counter) if last_counter is not None else None

        floor_price = max(self._offer_min, total_msrp_delta * 1.10)
        ceiling_price = min(self._offer_max, max(floor_price + 200.0, total_msrp_delta * 2.2))

        # Linear concession keeps the policy interpretable as a non-learning reference.
        progress = float(max(0, round_idx - 1)) / float(max(1, max_rounds - 1))
        target_offer = ceiling_price + (floor_price - ceiling_price) * progress
        target_offer += self._rng.gauss(0.0, 100.0)

        if counter_offer is not None:
            if round_idx >= max_rounds and counter_offer >= floor_price:
                return EnvAction(move="accept", price_offer_usd=0.0)
            target_offer = max(counter_offer + 120.0, 0.62 * target_offer + 0.38 * counter_offer)

        target_offer = float(min(self._offer_max, max(self._offer_min, target_offer)))
        return EnvAction(move="offer", price_offer_usd=target_offer)
