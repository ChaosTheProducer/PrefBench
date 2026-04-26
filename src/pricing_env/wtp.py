"""WTP and consumer response utilities for phase-1 environment."""

from __future__ import annotations

from dataclasses import dataclass
import random

from .types import PersonaProfile


@dataclass(frozen=True)
class WTPBreakdown:
    """Represents WTP decomposition terms used by the current round.

    Attributes:
        r_base_usd: Baseline reservation value term.
        v_custom_usd: Functional customization value term.
        v_aesthetic_usd: Aesthetic premium term.
        v_brand_tech_usd: Brand/tech premium term (placeholder in v1 runtime).
        v_fatigue_usd: Round fatigue penalty term.
        epsilon_usd: Stochastic round noise term.
        wtp_usd: Final willingness to pay after all terms are combined.
    """

    r_base_usd: float
    v_custom_usd: float
    v_aesthetic_usd: float
    v_brand_tech_usd: float
    v_fatigue_usd: float
    epsilon_usd: float
    wtp_usd: float


def compute_wtp_breakdown(
    persona: PersonaProfile,
    total_cost_usd: float,
    aesthetic_proxy_score: float,
    feature_match_score: float,
    tech_signal: float,
    value_markup: float,
    time_penalty_usd_per_round: float,
    round_idx: int,
    noise_sigma_usd: float,
    rng: random.Random,
) -> WTPBreakdown:
    """Computes round-level WTP decomposition.

    This follows the v1 formula documented in `WTP_UTILITY_SPEC.md`:
    `WTP_t = R_base + V_custom + V_aesthetic + V_brand_tech - V_fatigue + epsilon_t`.

    Args:
        persona: Sampled consumer persona.
        total_cost_usd: Sum of selected option MSRP deltas (consumer-facing value anchor).
        aesthetic_proxy_score: Mean aesthetics signal from selected options.
        feature_match_score: Dot-product score between selected options and
            persona hidden feature weights.
        tech_signal: Technology-intensity proxy in [0, 1].
        value_markup: Markup factor on functional value.
        time_penalty_usd_per_round: Linear patience penalty base.
        round_idx: Current round index.
        noise_sigma_usd: Gaussian noise scale.
        rng: Random generator.

    Returns:
        One immutable round-level WTP decomposition.
    """

    sensitivity_scale = 1.0 / max(0.55, float(persona.price_sensitivity))
    r_base = float(persona.reservation_price_base) * sensitivity_scale
    feature_alignment = min(1.0, max(0.0, float(feature_match_score)))
    v_custom = float(total_cost_usd) * float(value_markup) * (0.55 + 0.9 * feature_alignment) * sensitivity_scale
    v_aesthetic = float(aesthetic_proxy_score) * 1400.0 * float(persona.aesthetic_sensitivity)
    tech_band_factor = {"low": 0.65, "medium": 1.0, "high": 1.35}.get(persona.tech_affinity_band, 1.0)
    v_brand_tech = float(persona.brand_loyalty) * 900.0 + min(1.0, max(0.0, float(tech_signal))) * 800.0 * tech_band_factor
    round_offset = max(round_idx - 1, 0)
    patience_scale = 5.0 / max(1.0, float(persona.patience))
    v_fatigue = float(time_penalty_usd_per_round) * float(round_offset) * patience_scale * (0.8 + 0.9 * float(persona.impulsivity))
    noise_sigma = float(noise_sigma_usd) * (0.7 + 0.6 * float(persona.belief_obscurity))
    epsilon = rng.gauss(0.0, noise_sigma)
    wtp = max(1000.0, r_base + v_custom + v_aesthetic + v_brand_tech - v_fatigue + epsilon)
    return WTPBreakdown(
        r_base_usd=r_base,
        v_custom_usd=v_custom,
        v_aesthetic_usd=v_aesthetic,
        v_brand_tech_usd=v_brand_tech,
        v_fatigue_usd=v_fatigue,
        epsilon_usd=epsilon,
        wtp_usd=wtp,
    )


def compute_wtp(
    persona: PersonaProfile,
    total_cost_usd: float,
    aesthetic_proxy_score: float,
    feature_match_score: float,
    tech_signal: float,
    value_markup: float,
    time_penalty_usd_per_round: float,
    round_idx: int,
    noise_sigma_usd: float,
    rng: random.Random,
) -> float:
    """Computes willingness-to-pay for current round.

    Args:
        persona: Sampled consumer persona.
        total_cost_usd: Sum of selected option MSRP deltas (consumer-facing value anchor).
        aesthetic_proxy_score: Mean aesthetics signal from selected options.
        feature_match_score: Dot-product score between selected options and
            persona hidden feature weights.
        tech_signal: Technology-intensity proxy in [0, 1].
        value_markup: Markup factor on functional value.
        time_penalty_usd_per_round: Linear patience penalty.
        round_idx: Current round index.
        noise_sigma_usd: Gaussian noise scale.
        rng: Random generator.

    Returns:
        Estimated willingness-to-pay in USD.
    """

    return compute_wtp_breakdown(
        persona=persona,
        total_cost_usd=total_cost_usd,
        aesthetic_proxy_score=aesthetic_proxy_score,
        feature_match_score=feature_match_score,
        tech_signal=tech_signal,
        value_markup=value_markup,
        time_penalty_usd_per_round=time_penalty_usd_per_round,
        round_idx=round_idx,
        noise_sigma_usd=noise_sigma_usd,
        rng=rng,
    ).wtp_usd


def compute_buyer_utility(wtp_usd: float, price_usd: float) -> float:
    """Computes buyer utility for one offered price.

    Args:
        wtp_usd: Current buyer willingness to pay.
        price_usd: Offered transaction price.

    Returns:
        Buyer utility value.
    """

    return float(wtp_usd) - float(price_usd)


def compute_seller_time_cost(
    *,
    time_penalty_usd_per_round: float,
    round_idx: int,
) -> float:
    """Computes round-dependent seller time cost.

    Args:
        time_penalty_usd_per_round: Per-round time cost.
        round_idx: Current round index starting from 1.

    Returns:
        Cumulative time cost applied at this round.
    """

    return float(time_penalty_usd_per_round) * float(max(round_idx - 1, 0))


def compute_seller_utility(
    *,
    price_usd: float,
    customization_cost_usd: float,
    time_penalty_usd_per_round: float,
    round_idx: int,
) -> float:
    """Computes seller utility for one offered price.

    Args:
        price_usd: Offered transaction price.
        customization_cost_usd: Total selected customization cost.
        time_penalty_usd_per_round: Per-round time cost.
        round_idx: Current round index starting from 1.

    Returns:
        Seller utility value.
    """

    time_cost = compute_seller_time_cost(
        time_penalty_usd_per_round=time_penalty_usd_per_round,
        round_idx=round_idx,
    )
    return float(price_usd) - float(customization_cost_usd) - time_cost


def estimate_counter_offer(
    wtp_usd: float,
    agent_offer_usd: float,
    counter_strength: float,
    rng: random.Random,
) -> float:
    """Estimates consumer counter-offer when rejecting an offer.

    Args:
        wtp_usd: Consumer willingness-to-pay.
        agent_offer_usd: Agent's proposed price.
        counter_strength: Higher means tougher bargaining.
        rng: Random generator.

    Returns:
        Counter-offer in USD.
    """

    midpoint = (wtp_usd + agent_offer_usd) / 2.0
    toughness_shift = (agent_offer_usd - wtp_usd) * (0.2 + 0.3 * counter_strength)
    jitter = rng.gauss(0.0, 80.0)
    return max(500.0, midpoint - toughness_shift + jitter)
