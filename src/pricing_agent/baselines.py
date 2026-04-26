"""Baseline policies and evaluation utilities for Gym wrapper experiments."""
# We utilize RecurrentPPO from sb3

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any, Dict, Mapping

from pricing_env.gym_wrapper import GYMNASIUM_AVAILABLE, PricingNegotiationGymWrapper


if GYMNASIUM_AVAILABLE:
    import numpy as np
else:  # pragma: no cover - optional dependency path
    np = None  # type: ignore[assignment]


@dataclass
class PolicyMetrics:
    """Aggregated metrics for one baseline policy run.

    Attributes:
        episodes: Number of evaluated episodes.
        deal_rate: Fraction of episodes that reached agreement.
        avg_profit_usd: Mean per-episode profit.
        avg_rounds: Mean used rounds.
        walkaway_rate: Fraction of no-deal walkaways.
        avg_trace_len: Mean number of logged trace events.
        terminated_rate: Fraction ended with `terminated=True`.
        truncated_rate: Fraction ended with `truncated=True`.
    """

    episodes: int
    deal_rate: float
    avg_profit_usd: float
    avg_rounds: float
    walkaway_rate: float
    avg_trace_len: float
    terminated_rate: float
    truncated_rate: float


class RandomPolicy:
    """Random negotiation policy for wrapper-level sanity checks."""

    def __init__(self, *, rng: random.Random, offer_min: float, offer_max: float):
        """Initializes random policy.

        Args:
            rng: Random generator.
            offer_min: Minimum offer value.
            offer_max: Maximum offer value.
        """

        self._rng = rng
        self._offer_min = float(offer_min)
        self._offer_max = float(offer_max)

    def act(self, info: Mapping[str, Any]) -> Dict[str, Any]:
        """Returns one random action.

        Args:
            info: Wrapper info containing `raw_observation`.

        Returns:
            Wrapper action dictionary.
        """

        raw_obs = info.get("raw_observation", {})
        has_counter = raw_obs.get("last_consumer_offer_usd") is not None
        if has_counter and self._rng.random() < 0.12:
            return {"move": 1, "price": [0.0]}
        if self._rng.random() < 0.08:
            return {"move": 2, "price": [0.0]}
        offer = self._rng.uniform(self._offer_min, self._offer_max)
        return {"move": 0, "price": [offer]}


class ConcessionPolicy:
    """Simple concession-based policy for non-learning baseline."""

    def __init__(self, *, rng: random.Random, offer_min: float, offer_max: float):
        """Initializes concession policy.

        Args:
            rng: Random generator.
            offer_min: Minimum offer value.
            offer_max: Maximum offer value.
        """

        self._rng = rng
        self._offer_min = float(offer_min)
        self._offer_max = float(offer_max)

    def act(self, info: Mapping[str, Any]) -> Dict[str, Any]:
        """Returns one concession action.

        Args:
            info: Wrapper info containing `raw_observation`.

        Returns:
            Wrapper action dictionary.
        """

        raw_obs = info.get("raw_observation", {})
        round_idx = int(raw_obs.get("round_idx", 1))
        remaining = int(raw_obs.get("remaining_rounds", 1))
        max_rounds = max(1, round_idx + remaining - 1)
        total_msrp_delta = float(
            raw_obs.get(
                "total_msrp_delta_usd",
                raw_obs.get("total_customization_cost_usd", 0.0),
            )
        )
        last_counter = raw_obs.get("last_consumer_offer_usd")
        counter_offer = float(last_counter) if last_counter is not None else None

        floor_price = max(self._offer_min, total_msrp_delta * 1.10)
        ceiling_price = min(self._offer_max, max(floor_price + 200.0, total_msrp_delta * 2.2))

        # Linear concession from high margin to lower margin over round progression.
        progress = float(max(0, round_idx - 1)) / float(max(1, max_rounds - 1))
        target_offer = ceiling_price + (floor_price - ceiling_price) * progress
        target_offer += self._rng.gauss(0.0, 100.0)

        if counter_offer is not None:
            if round_idx >= max_rounds and counter_offer >= floor_price:
                return {"move": 1, "price": [0.0]}
            target_offer = max(counter_offer + 120.0, 0.62 * target_offer + 0.38 * counter_offer)

        target_offer = float(min(self._offer_max, max(self._offer_min, target_offer)))
        return {"move": 0, "price": [target_offer]}


def evaluate_policy(
    *,
    env: PricingNegotiationGymWrapper,
    policy_name: str,
    episodes: int,
    seed: int,
) -> PolicyMetrics:
    """Evaluates one baseline policy on the wrapped environment.

    Args:
        env: Gym wrapper environment.
        policy_name: One of `random` or `concession`.
        episodes: Number of episodes.
        seed: Global random seed.

    Returns:
        Aggregated policy metrics.
    """

    if not GYMNASIUM_AVAILABLE:
        raise RuntimeError("Gymnasium is required to evaluate wrapper policies.")

    price_space = env.action_space["price"]
    offer_min = float(price_space.low[0])
    offer_max = float(price_space.high[0])
    policy_rng = random.Random(seed)
    if policy_name == "random":
        policy = RandomPolicy(rng=policy_rng, offer_min=offer_min, offer_max=offer_max)
    elif policy_name == "concession":
        policy = ConcessionPolicy(rng=policy_rng, offer_min=offer_min, offer_max=offer_max)
    else:
        raise ValueError(f"Unsupported policy name: {policy_name}")

    deals = 0
    total_profit = 0.0
    total_rounds = 0
    walkaways = 0
    total_trace_len = 0
    terminated_count = 0
    truncated_count = 0

    for episode_idx in range(episodes):
        _obs, info = env.reset(seed=seed + episode_idx)
        terminated = False
        truncated = False
        while not (terminated or truncated):
            action = policy.act(info)
            _obs, _reward, terminated, truncated, info = env.step(action)

        terminated_count += int(terminated)
        truncated_count += int(truncated)
        metrics = info.get("episode_metrics", {})
        deals += int(metrics.get("deal_reached", False))
        total_profit += float(metrics.get("profit_usd", 0.0))
        total_rounds += int(metrics.get("rounds_used", 0))
        walkaways += int(metrics.get("walkaway", False))
        total_trace_len += int(info.get("trace_len", 0))

    denom = float(max(episodes, 1))
    return PolicyMetrics(
        episodes=episodes,
        deal_rate=float(deals) / denom,
        avg_profit_usd=float(total_profit) / denom,
        avg_rounds=float(total_rounds) / denom,
        walkaway_rate=float(walkaways) / denom,
        avg_trace_len=float(total_trace_len) / denom,
        terminated_rate=float(terminated_count) / denom,
        truncated_rate=float(truncated_count) / denom,
    )
