"""LLM-facing observation rendering and strict action parsing."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Dict, Mapping

from pricing_env.catalog import Catalog
from pricing_env.types import EnvAction, ObservationDict


ALLOWED_LLM_MOVES = ("offer", "accept", "walkaway")


@dataclass(frozen=True)
class ParsedLLMAction:
    """Parsed LLM action with explicit invalid-output metadata."""

    valid: bool
    action: EnvAction | None
    reason: str
    invalid_type: str | None = None
    error: str | None = None
    payload: Dict[str, Any] | None = None


def render_llm_observation(observation: ObservationDict, catalog: Catalog, prompt_version: str = "v1") -> str:
    """Renders the environment observation into the fixed LLM prompt."""

    selected_keys = [str(key) for key in observation.get("selected_option_keys", [])]
    option_by_key = {option.key: option for option in catalog.options}
    selected_options = [
        {
            "key": key,
            "dimension": option_by_key[key].dimension,
            "msrp_delta_usd": option_by_key[key].price_delta_usd,
        }
        for key in selected_keys
    ]
    if prompt_version not in {"v1", "v2"}:
        raise ValueError("prompt_version must be one of: v1, v2.")
    policy_contract = {
        "role": "You are the seller in a personalized pricing benchmark.",
        "objective": "Maximize expected seller profit from the customization bundle.",
        "profit_formula": "profit_usd = deal_price_usd - estimated_implementation_cost_usd",
        "information_limit": (
            "Use only the observable buyer profile, bundle information, and negotiation history below. "
            "Hidden willingness-to-pay, hidden preferences, patience, and walkaway tendency are not observable."
        ),
        "action_meanings": {
            "offer": "Propose a customization-bundle price in USD. This is not the full vehicle price.",
            "accept": "Accept the buyer's last counter-offer. Use only when last_consumer_offer_usd is not null.",
            "walkaway": "End the negotiation without a deal.",
        },
        "action_validity": "Choosing accept when last_consumer_offer_usd is null is invalid.",
        "decision_guidance": [
            "Prefer profitable deals over no deal.",
            "Do not optimize only for immediate acceptance.",
            "When several rounds remain, use the opportunity to make a profitable but ambitious opening offer.",
            "A first-round offer may be above the expected settlement price if it is still plausible for the bundle.",
            "Avoid offering below estimated_implementation_cost_usd unless strategically necessary.",
            "If the buyer made a counter-offer, compare it with implementation cost and remaining rounds.",
            "If remaining rounds are low, make a realistic final offer or accept a profitable counter.",
            "Treat the observable buyer profile as weak evidence only.",
        ],
    }
    state_description: Dict[str, Any] | None = None
    if prompt_version == "v2":
        policy_contract.update(
            {
                "scenario": (
                    "The product is a fixed vehicle customization bundle selected before the negotiation. "
                    "You price only the customization bundle, not the base vehicle."
                ),
                "evaluation_note": (
                    "High deal rate alone is not sufficient; low-price immediate acceptance can reduce profit."
                ),
                "interaction_dynamics": (
                    "After an offer, the buyer may accept, reject, make a counter-offer, or walk away. "
                    "A reject or counter-offer creates a new observation in the next round if the episode continues."
                ),
            }
        )
        state_description = {
            "round_idx": "Current seller decision turn, starting from 1.",
            "remaining_rounds": "Number of seller decision turns left after the current one.",
            "selected_options": "Customization options included in the fixed bundle being negotiated.",
            "total_msrp_delta_usd": "Reference retail price increase for the selected customization bundle.",
            "estimated_implementation_cost_usd": "Estimated seller-side cost to provide the selected customization bundle.",
            "aesthetic_proxy_score": (
                "Coarse visible distinctiveness/style proxy for the selected bundle, not a hidden preference."
            ),
            "buyer_observable_profile": (
                "Observable buyer demographic and usage-context signals. These are weak evidence, not hidden preferences."
            ),
            "last_agent_offer_usd": "Your previous seller offer, or null if no offer has been made.",
            "last_consumer_response": "Buyer response to the previous seller action.",
            "last_consumer_offer_usd": "Buyer counter-offer if one was made; otherwise null.",
            "history_len": "Number of completed negotiation steps so far.",
        }
    output_contract = {
        "instruction": "Return exactly one JSON object and nothing else.",
        "allowed_actions": list(ALLOWED_LLM_MOVES),
        "schema": {
            "move": "one of: offer, accept, walkaway",
            "price_offer_usd": "non-negative number required for offer; use 0 for accept or walkaway",
            "reason": "brief string for trace only",
        },
        "example": {
            "move": "offer",
            "price_offer_usd": 5200,
            "reason": "profitable offer adjusted for buyer profile and remaining rounds",
        },
    }
    state = {
        "round": {
            "round_idx": int(observation["round_idx"]),
            "remaining_rounds": int(observation["remaining_rounds"]),
        },
        "bundle": {
            "selected_options": selected_options,
            "selected_option_keys": selected_keys,
            "total_msrp_delta_usd": float(observation["total_msrp_delta_usd"]),
            "estimated_implementation_cost_usd": float(observation["total_customization_cost_usd"]),
            "aesthetic_proxy_score": float(observation["aesthetic_proxy_score"]),
        },
        "buyer_observable_profile": {
            "age_band": str(observation["persona_age_band"]),
            "income_band": str(observation["persona_income_band"]),
            "household_stage": str(observation["persona_household_stage"]),
            "ownership_stage": str(observation["persona_ownership_stage"]),
            "primary_use_case": str(observation["persona_primary_use_case"]),
        },
        "negotiation_state": {
            "last_agent_offer_usd": observation.get("last_agent_offer_usd"),
            "last_consumer_response": str(observation["last_consumer_response"]),
            "last_consumer_offer_usd": observation.get("last_consumer_offer_usd"),
            "history_len": int(observation["history_len"]),
        },
    }
    payload = {
        "prompt_version": prompt_version,
        "policy_contract": policy_contract,
        "output_contract": output_contract,
        "current_state": state,
    }
    if state_description is not None:
        payload["state_description"] = state_description
    return (
        "Return only valid JSON. Do not include Markdown, code fences, or extra text.\n\n"
        f"{json.dumps(payload, indent=2, sort_keys=True)}"
    )


def parse_llm_action(response_text: str) -> ParsedLLMAction:
    """Parses a strict JSON LLM response into an environment action."""

    try:
        payload = json.loads(response_text)
    except json.JSONDecodeError as exc:
        return ParsedLLMAction(
            valid=False,
            action=None,
            reason="",
            invalid_type="invalid_json",
            error=str(exc),
        )

    if not isinstance(payload, dict):
        return ParsedLLMAction(
            valid=False,
            action=None,
            reason="",
            invalid_type="invalid_action",
            error="LLM response must be a JSON object.",
        )

    move = str(payload.get("move", "")).strip().lower()
    reason = str(payload.get("reason", "")).strip()
    if move not in ALLOWED_LLM_MOVES:
        return ParsedLLMAction(
            valid=False,
            action=None,
            reason=reason,
            invalid_type="invalid_action",
            error=f"Unsupported move: {move}",
            payload=payload,
        )

    if move == "offer":
        if "price_offer_usd" not in payload:
            return ParsedLLMAction(
                valid=False,
                action=None,
                reason=reason,
                invalid_type="invalid_action",
                error="`price_offer_usd` is required for offer.",
                payload=payload,
            )
        price_raw = payload["price_offer_usd"]
        if not isinstance(price_raw, (int, float)) or isinstance(price_raw, bool):
            return ParsedLLMAction(
                valid=False,
                action=None,
                reason=reason,
                invalid_type="invalid_action",
                error="`price_offer_usd` must be numeric for offer.",
                payload=payload,
            )
        price = float(price_raw)
        if price < 0.0:
            return ParsedLLMAction(
                valid=False,
                action=None,
                reason=reason,
                invalid_type="invalid_action",
                error="`price_offer_usd` must be non-negative.",
                payload=payload,
            )
        action = EnvAction(move="offer", price_offer_usd=price)
        return ParsedLLMAction(valid=True, action=action, reason=reason, payload=payload)

    action = EnvAction(move=move, price_offer_usd=0.0)
    return ParsedLLMAction(valid=True, action=action, reason=reason, payload=payload)
