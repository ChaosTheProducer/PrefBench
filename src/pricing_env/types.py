"""Core data types for the pricing negotiation environment."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class CatalogOption:
    """Represents one canonical customization option.

    Attributes:
        key: Stable option identifier.
        dimension: Logical dimension group.
        price_delta_usd: MSRP delta relative to base vehicle.
        aesthetic_weight: Scalar aesthetics proxy used by the simulator.
    """

    key: str
    dimension: str
    price_delta_usd: float
    aesthetic_weight: float


@dataclass
class PersonaProfile:
    """Represents one sampled consumer persona for an episode.

    Attributes:
        persona_id: Stable persona identifier for experiment traceability.
        persona_source: Persona source tag (e.g., `distribution_hidden_mapping_v1`, `persona_bank`).
        persona_split: Dataset split tag (`train`, `val`, `test`).
        age_band: Coarse consumer age group visible to the agent.
        income_band: Coarse household income band visible to the agent.
        household_stage: Household stage visible to the agent.
        ownership_stage: Ownership stage visible to the agent.
        primary_use_case: Primary vehicle usage scenario visible to the agent.
        decision_style: Hidden decision style category.
        tech_affinity_band: Hidden tech-affinity category.
        stated_priority_top2: Hidden top-2 stated priorities.
        reservation_price_base: Hidden baseline reservation price.
        price_sensitivity: Hidden price-sensitivity coefficient.
        aesthetic_sensitivity: Hidden aesthetics preference coefficient.
        patience: Hidden maximum tolerated rounds.
        counter_strength: Hidden counter aggressiveness in [0, 1].
        walkaway_threshold: Hidden base walkaway probability in [0, 1].
        belief_obscurity: Hidden behavioral obscurity coefficient in [0, 1].
        brand_loyalty: Hidden brand-premium tendency in [0, 1].
        impulsivity: Hidden impulsivity coefficient in [0, 1].
        feature_weight_vector: Hidden normalized feature weights over
            `safety`, `comfort`, `performance`, `tech`, `aesthetics`.
    """

    persona_id: str = "generated"
    persona_source: str = "distribution_hidden_mapping_v1"
    persona_split: str = "train"
    age_band: str = "26-35"
    income_band: str = "60-100k"
    household_stage: str = "couple"
    ownership_stage: str = "replacement"
    primary_use_case: str = "mixed"
    decision_style: str = "balanced"
    tech_affinity_band: str = "medium"
    stated_priority_top2: List[str] = field(default_factory=lambda: ["comfort", "tech"])
    reservation_price_base: float = 9000.0
    price_sensitivity: float = 1.0
    aesthetic_sensitivity: float = 0.7
    patience: int = 5
    walkaway_threshold: float = 0.1
    counter_strength: float = 0.6
    belief_obscurity: float = 0.4
    brand_loyalty: float = 0.5
    impulsivity: float = 0.4
    feature_weight_vector: Dict[str, float] = field(
        default_factory=lambda: {
            "safety": 0.2,
            "comfort": 0.25,
            "performance": 0.15,
            "tech": 0.2,
            "aesthetics": 0.2,
        }
    )


@dataclass(frozen=True)
class EnvAction:
    """Represents one environment action.

    Attributes:
        move: One of `offer`, `accept`, `walkaway`.
        price_offer_usd: Agent proposed price in USD.
    """

    move: str
    price_offer_usd: float


@dataclass
class EnvObservation:
    """Represents the observation returned by the environment.

    Attributes:
        round_idx: Current round index starting from 1.
        remaining_rounds: Rounds left before forced termination.
        last_agent_offer_usd: Last offer price from the agent.
        last_consumer_response: Consumer response token.
        last_consumer_offer_usd: Last counter-offer from consumer if available.
        history_len: Number of negotiation events in current session.
        selected_option_keys: Selected canonical options for this episode.
        total_customization_cost_usd: Sum of selected option implementation-cost proxies.
        total_msrp_delta_usd: Sum of selected option MSRP deltas.
        aesthetic_proxy_score: Mean aesthetics score for selected options.
        persona_age_band: Observable persona age band.
        persona_income_band: Observable persona income band.
        persona_household_stage: Observable persona household stage.
        persona_ownership_stage: Observable persona ownership stage.
        persona_primary_use_case: Observable persona use-case tag.
    """

    round_idx: int
    remaining_rounds: int
    last_agent_offer_usd: Optional[float]
    last_consumer_response: str
    last_consumer_offer_usd: Optional[float]
    history_len: int
    selected_option_keys: List[str]
    total_customization_cost_usd: float
    total_msrp_delta_usd: float
    aesthetic_proxy_score: float
    persona_age_band: str
    persona_income_band: str
    persona_household_stage: str
    persona_ownership_stage: str
    persona_primary_use_case: str


@dataclass
class EpisodeMetrics:
    """Aggregated metrics for one completed episode.

    Attributes:
        profit_usd: Final profit for the episode.
        deal_reached: Whether a deal was reached.
        rounds_used: Number of rounds consumed.
        walkaway: Whether consumer exited without a deal.
    """

    profit_usd: float
    deal_reached: bool
    rounds_used: int
    walkaway: bool


ObservationDict = Dict[str, float | int | str | List[str] | List[float] | None]
