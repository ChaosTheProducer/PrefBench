"""NegMAS-backed persistent negotiation session backend."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any, Dict, List

try:
    from negmas import SAOMechanism, make_issue
    from negmas.preferences import MappingUtilityFunction
    from negmas.sao import ResponseType, SAONegotiator

    NEGMA_AVAILABLE = True
except Exception:  # pragma: no cover
    NEGMA_AVAILABLE = False
    SAOMechanism = None  # type: ignore[assignment]
    make_issue = None  # type: ignore[assignment]
    MappingUtilityFunction = None  # type: ignore[assignment]
    SAONegotiator = object  # type: ignore[assignment]
    ResponseType = None  # type: ignore[assignment]

from .wtp import compute_buyer_utility, compute_seller_utility


@dataclass(frozen=True)
class NegMASRoundResult:
    """Represents one NegMAS session-step output.

    Attributes:
        response: Negotiation response token.
        deal_price_usd: Deal price if accepted, otherwise None.
        counter_offer_usd: Counter-offer if generated, otherwise None.
        walked_away: Whether buyer ended negotiation.
        mechanism_step: Internal NegMAS step counter after this update.
        history_len: Session event count after this update.
        buyer_utility_at_offer: Buyer utility for the seller offer of this round.
        seller_utility_at_offer: Seller utility for the seller offer of this round.
        termination_cause: Human-readable cause for terminal states.
    """

    response: str
    deal_price_usd: float | None
    counter_offer_usd: float | None
    walked_away: bool
    mechanism_step: int
    history_len: int
    buyer_utility_at_offer: float | None = None
    seller_utility_at_offer: float | None = None
    termination_cause: str | None = None


class _SellerOfferNegotiator(SAONegotiator):
    """Seller negotiator that proposes the latest agent offer."""

    def __init__(
        self,
        offer_usd: float,
        customization_cost_usd: float,
        time_penalty_usd_per_round: float,
        round_idx: int,
        **kwargs,
    ):
        super().__init__(ufun=MappingUtilityFunction(self._utility_for_outcome), **kwargs)
        self._offer = (int(round(offer_usd)),)
        self._customization_cost_usd = float(customization_cost_usd)
        self._time_penalty_usd_per_round = float(time_penalty_usd_per_round)
        self._round_idx = int(round_idx)

    def set_offer(self, offer_usd: float) -> None:
        """Updates the seller offer used in the next mechanism step.

        Args:
            offer_usd: Seller offer in USD.
        """

        self._offer = (int(round(offer_usd)),)

    def set_context(
        self,
        *,
        customization_cost_usd: float,
        time_penalty_usd_per_round: float,
        round_idx: int,
    ) -> None:
        """Updates seller utility context.

        Args:
            customization_cost_usd: Cost of selected customization bundle.
            time_penalty_usd_per_round: Seller per-round time penalty.
            round_idx: Current round index.
        """

        self._customization_cost_usd = float(customization_cost_usd)
        self._time_penalty_usd_per_round = float(time_penalty_usd_per_round)
        self._round_idx = int(round_idx)

    def utility_of_price(self, price_usd: float) -> float:
        """Computes seller utility for one price."""

        return compute_seller_utility(
            price_usd=float(price_usd),
            customization_cost_usd=self._customization_cost_usd,
            time_penalty_usd_per_round=self._time_penalty_usd_per_round,
            round_idx=self._round_idx,
        )

    def _utility_for_outcome(self, outcome: Any) -> float:
        """Maps an outcome tuple to seller utility."""

        if outcome is None:
            return float("-inf")
        return self.utility_of_price(float(outcome[0]))

    def propose(self, state, dest=None):
        """Returns fixed seller offer."""

        return self._offer

    def respond(self, state, source=None):
        """Rejects buyer counters to preserve explicit `accept` action semantics."""

        return ResponseType.REJECT_OFFER


class _BuyerPersonaNegotiator(SAONegotiator):
    """Buyer negotiator driven by persona-like WTP and bargaining parameters."""

    def __init__(
        self,
        wtp_usd: float,
        walkaway_threshold: float,
        counter_strength: float,
        price_sensitivity: float,
        rng: random.Random,
        **kwargs,
    ):
        super().__init__(ufun=MappingUtilityFunction(self._utility_for_outcome), **kwargs)
        self.wtp_usd = float(wtp_usd)
        self.walkaway_threshold = float(walkaway_threshold)
        self.counter_strength = float(counter_strength)
        self.price_sensitivity = float(price_sensitivity)
        self.rng = rng
        self.walked_away = False

    def set_context(
        self,
        *,
        wtp_usd: float,
        walkaway_threshold: float,
        counter_strength: float,
        price_sensitivity: float,
    ) -> None:
        """Updates buyer context for the next offer evaluation.

        Args:
            wtp_usd: Updated willingness to pay.
            walkaway_threshold: Updated base walkaway tendency.
            counter_strength: Updated counter aggressiveness.
            price_sensitivity: Consumer sensitivity to offered prices.
        """

        self.wtp_usd = float(wtp_usd)
        self.walkaway_threshold = float(walkaway_threshold)
        self.counter_strength = float(counter_strength)
        self.price_sensitivity = float(price_sensitivity)
        self.walked_away = False

    def utility_of_price(self, price_usd: float) -> float:
        """Computes buyer utility for one price."""

        return compute_buyer_utility(self.wtp_usd, float(price_usd))

    def _utility_for_outcome(self, outcome: Any) -> float:
        """Maps an outcome tuple to buyer utility."""

        if outcome is None:
            return float("-inf")
        return self.utility_of_price(float(outcome[0]))

    def respond(self, state, source=None):
        """Responds to seller offer with accept/reject/end decisions."""

        if state.current_offer is None:
            return ResponseType.REJECT_OFFER

        seller_offer = float(state.current_offer[0])
        offer_utility = self.utility_of_price(seller_offer)
        if offer_utility >= 0.0:
            return ResponseType.ACCEPT_OFFER

        utility_gap_ratio = max(0.0, -offer_utility) / max(self.wtp_usd, 1.0)
        sensitivity_multiplier = min(1.6, max(0.7, self.price_sensitivity))
        p_walkaway = min(
            0.95,
            self.walkaway_threshold + utility_gap_ratio * 0.55 * sensitivity_multiplier,
        )
        if self.rng.random() < p_walkaway:
            self.walked_away = True
            return ResponseType.END_NEGOTIATION
        return ResponseType.REJECT_OFFER

    def propose(self, state, dest=None):
        """Returns a counter-offer if seller offer is above WTP."""

        if state.current_offer is None:
            anchor = self.wtp_usd - (160.0 + 380.0 * self.counter_strength)
            jitter = self.rng.gauss(0.0, 80.0)
            return (int(max(500.0, anchor + jitter)),)

        seller_offer = float(state.current_offer[0])
        concession = max(0.0, seller_offer - self.wtp_usd) * (0.12 + 0.28 * (1.0 - self.counter_strength))
        target = self.wtp_usd - (120.0 + 500.0 * self.counter_strength) - concession
        jitter = self.rng.gauss(0.0, 70.0)
        counter = max(500.0, min(seller_offer - 80.0, target + jitter))
        return (int(round(counter)),)


class NegMASRoundBackend:
    """Creates persistent NegMAS sessions for one environment episode."""

    def __init__(self, n_steps_per_episode: int = 64):
        """Initializes backend.

        Args:
            n_steps_per_episode: SAO mechanism step budget per episode.
        """

        if not NEGMA_AVAILABLE:
            raise RuntimeError("NegMAS is not available in this environment.")
        self.n_steps_per_episode = int(n_steps_per_episode)

    def create_session(
        self,
        *,
        issue_min_price: float,
        issue_max_price: float,
        rng: random.Random,
    ) -> "NegMASSession":
        """Creates one persistent NegMAS session.

        Args:
            issue_min_price: Lower bound of the negotiation price issue.
            issue_max_price: Upper bound of the negotiation price issue.
            rng: Random generator.

        Returns:
            A persistent NegMAS session object.
        """

        min_price = int(max(100.0, issue_min_price))
        max_price = int(max(min_price + 500, issue_max_price))
        return NegMASSession(
            min_price=min_price,
            max_price=max_price,
            n_steps=max(8, self.n_steps_per_episode),
            rng=rng,
        )


class NegMASSession:
    """Represents one persistent buyer-seller NegMAS negotiation session."""

    def __init__(self, *, min_price: int, max_price: int, n_steps: int, rng: random.Random) -> None:
        """Initializes a session.

        Args:
            min_price: Lower issue bound.
            max_price: Upper issue bound.
            n_steps: Mechanism step budget.
            rng: Random generator shared with environment.
        """

        if not NEGMA_AVAILABLE:
            raise RuntimeError("NegMAS is not available in this environment.")

        self._rng = rng
        self._history: List[Dict[str, Any]] = []
        self._closed = False
        self._closed_reason: str | None = None
        self._latest_counter_offer_usd: float | None = None

        self._mechanism = SAOMechanism(
            issues=[make_issue((min_price, max_price), name="price")],
            n_steps=int(n_steps),
            one_offer_per_step=True,
            offering_is_accepting=True,
        )
        self._seller = _SellerOfferNegotiator(
            offer_usd=min_price,
            customization_cost_usd=0.0,
            time_penalty_usd_per_round=0.0,
            round_idx=1,
            name="seller",
        )
        self._buyer = _BuyerPersonaNegotiator(
            wtp_usd=min_price,
            walkaway_threshold=0.1,
            counter_strength=0.5,
            price_sensitivity=1.0,
            rng=self._rng,
            name="buyer",
        )
        self._mechanism.add(self._seller)
        self._mechanism.add(self._buyer)

    @property
    def history_len(self) -> int:
        """Returns the number of recorded session events."""

        return len(self._history)

    @property
    def latest_counter_offer_usd(self) -> float | None:
        """Returns the latest buyer counter offer."""

        return self._latest_counter_offer_usd

    @property
    def mechanism_step(self) -> int:
        """Returns current NegMAS mechanism step."""

        return int(self._mechanism.state.step)

    def close(self, reason: str) -> None:
        """Closes the session explicitly.

        Args:
            reason: Reason token for closure.
        """

        self._closed = True
        self._closed_reason = reason

    def offer_round(
        self,
        *,
        agent_offer_usd: float,
        wtp_usd: float,
        walkaway_threshold: float,
        counter_strength: float,
        price_sensitivity: float,
        customization_cost_usd: float,
        time_penalty_usd_per_round: float,
        round_idx: int,
    ) -> NegMASRoundResult:
        """Runs one seller->buyer exchange on the persistent mechanism.

        Args:
            agent_offer_usd: Seller offer for this round.
            wtp_usd: Buyer willingness to pay for this round.
            walkaway_threshold: Buyer base walkaway tendency.
            counter_strength: Buyer counter aggressiveness.
            price_sensitivity: Buyer price sensitivity.
            customization_cost_usd: Seller customization cost for this episode.
            time_penalty_usd_per_round: Seller per-round time cost.
            round_idx: Current environment round index.

        Returns:
            Structured result containing accept/counter/reject/walkaway outcome.
        """

        if self._closed:
            return NegMASRoundResult(
                response="closed",
                deal_price_usd=None,
                counter_offer_usd=self._latest_counter_offer_usd,
                walked_away=False,
                mechanism_step=self.mechanism_step,
                history_len=self.history_len,
                buyer_utility_at_offer=None,
                seller_utility_at_offer=None,
                termination_cause=self._closed_reason or "closed",
            )

        self._seller.set_offer(agent_offer_usd)
        self._seller.set_context(
            customization_cost_usd=customization_cost_usd,
            time_penalty_usd_per_round=time_penalty_usd_per_round,
            round_idx=round_idx,
        )
        self._buyer.set_context(
            wtp_usd=wtp_usd,
            walkaway_threshold=walkaway_threshold,
            counter_strength=counter_strength,
            price_sensitivity=price_sensitivity,
        )
        seller_utility_at_offer = self._seller.utility_of_price(agent_offer_usd)
        buyer_utility_at_offer = self._buyer.utility_of_price(agent_offer_usd)

        # Step 1: seller proposes current offer in the ongoing session.
        state = self._mechanism.step()
        # Step 2: buyer responds (accept/reject/end) and may propose counter.
        if not state.broken and not state.timedout and state.agreement is None:
            state = self._mechanism.step()

        if state.agreement is not None:
            agreed = float(state.agreement[0])
            self._closed = True
            self._closed_reason = "agreement"
            self._latest_counter_offer_usd = None
            self._history.append(
                {
                    "seller_offer_usd": float(agent_offer_usd),
                    "buyer_offer_usd": None,
                    "response": "accept",
                    "mechanism_step": int(state.step),
                    "buyer_utility_at_offer": buyer_utility_at_offer,
                    "seller_utility_at_offer": seller_utility_at_offer,
                }
            )
            return NegMASRoundResult(
                response="accept",
                deal_price_usd=agreed,
                counter_offer_usd=None,
                walked_away=False,
                mechanism_step=int(state.step),
                history_len=self.history_len,
                buyer_utility_at_offer=buyer_utility_at_offer,
                seller_utility_at_offer=seller_utility_at_offer,
                termination_cause="agreement",
            )

        if self._buyer.walked_away:
            self._closed = True
            self._closed_reason = "buyer_walkaway"
            self._latest_counter_offer_usd = None
            self._history.append(
                {
                    "seller_offer_usd": float(agent_offer_usd),
                    "buyer_offer_usd": None,
                    "response": "walkaway",
                    "mechanism_step": int(state.step),
                    "buyer_utility_at_offer": buyer_utility_at_offer,
                    "seller_utility_at_offer": seller_utility_at_offer,
                }
            )
            return NegMASRoundResult(
                response="walkaway",
                deal_price_usd=None,
                counter_offer_usd=None,
                walked_away=True,
                mechanism_step=int(state.step),
                history_len=self.history_len,
                buyer_utility_at_offer=buyer_utility_at_offer,
                seller_utility_at_offer=seller_utility_at_offer,
                termination_cause="buyer_walkaway",
            )

        if state.timedout:
            self._closed = True
            self._closed_reason = "mechanism_timeout"
            self._latest_counter_offer_usd = None
            self._history.append(
                {
                    "seller_offer_usd": float(agent_offer_usd),
                    "buyer_offer_usd": None,
                    "response": "timeout",
                    "mechanism_step": int(state.step),
                    "buyer_utility_at_offer": buyer_utility_at_offer,
                    "seller_utility_at_offer": seller_utility_at_offer,
                }
            )
            return NegMASRoundResult(
                response="timeout",
                deal_price_usd=None,
                counter_offer_usd=None,
                walked_away=True,
                mechanism_step=int(state.step),
                history_len=self.history_len,
                buyer_utility_at_offer=buyer_utility_at_offer,
                seller_utility_at_offer=seller_utility_at_offer,
                termination_cause="mechanism_timeout",
            )

        counter = None
        if state.current_offer is not None and state.current_proposer == self._buyer.id:
            current = float(state.current_offer[0])
            if abs(current - float(agent_offer_usd)) > 1e-6:
                counter = current

        self._latest_counter_offer_usd = counter
        response = "counter" if counter is not None else "reject"
        self._history.append(
            {
                "seller_offer_usd": float(agent_offer_usd),
                "buyer_offer_usd": counter,
                "response": response,
                "mechanism_step": int(state.step),
                "buyer_utility_at_offer": buyer_utility_at_offer,
                "seller_utility_at_offer": seller_utility_at_offer,
            }
        )
        return NegMASRoundResult(
            response=response,
            deal_price_usd=None,
            counter_offer_usd=counter,
            walked_away=False,
            mechanism_step=int(state.step),
            history_len=self.history_len,
            buyer_utility_at_offer=buyer_utility_at_offer,
            seller_utility_at_offer=seller_utility_at_offer,
            termination_cause=None,
        )
