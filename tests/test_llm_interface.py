from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pricing_agent.llm_interface import parse_llm_action, render_llm_observation
from pricing_env.catalog import Catalog
from pricing_env.types import CatalogOption


def test_parse_valid_offer() -> None:
    parsed = parse_llm_action('{"move": "offer", "price_offer_usd": 18000, "reason": "margin"}')

    assert parsed.valid
    assert parsed.action is not None
    assert parsed.action.move == "offer"
    assert parsed.action.price_offer_usd == 18000.0
    assert parsed.reason == "margin"


def test_parse_valid_accept_zeroes_price() -> None:
    parsed = parse_llm_action('{"move": "accept", "price_offer_usd": 12345, "reason": "ok"}')

    assert parsed.valid
    assert parsed.action is not None
    assert parsed.action.move == "accept"
    assert parsed.action.price_offer_usd == 0.0


def test_parse_invalid_json() -> None:
    parsed = parse_llm_action("```json\n{\"move\":\"offer\"}\n```")

    assert not parsed.valid
    assert parsed.invalid_type == "invalid_json"


def test_parse_unsupported_move() -> None:
    parsed = parse_llm_action('{"move": "discount", "price_offer_usd": 10000}')

    assert not parsed.valid
    assert parsed.invalid_type == "invalid_action"


def test_parse_offer_requires_numeric_price() -> None:
    missing = parse_llm_action('{"move": "offer"}')
    text_price = parse_llm_action('{"move": "offer", "price_offer_usd": "18000"}')

    assert not missing.valid
    assert missing.invalid_type == "invalid_action"
    assert not text_price.valid
    assert text_price.invalid_type == "invalid_action"


def test_render_observation_excludes_hidden_fields() -> None:
    catalog = Catalog(
        options=[
            CatalogOption(
                key="paint_color.paint_metallic",
                dimension="paint_color",
                price_delta_usd=750.0,
                aesthetic_weight=0.45,
            )
        ]
    )
    observation = {
        "round_idx": 1,
        "remaining_rounds": 5,
        "last_agent_offer_usd": None,
        "last_consumer_response": "none",
        "last_consumer_offer_usd": None,
        "history_len": 0,
        "selected_option_keys": ["paint_color.paint_metallic"],
        "total_customization_cost_usd": 375.0,
        "total_msrp_delta_usd": 750.0,
        "aesthetic_proxy_score": 0.45,
        "persona_age_band": "36-50",
        "persona_income_band": "100-180k",
        "persona_household_stage": "family",
        "persona_ownership_stage": "replacement",
        "persona_primary_use_case": "family",
    }

    prompt = render_llm_observation(observation, catalog)

    assert "paint_color.paint_metallic" in prompt
    assert "reservation_price_base" not in prompt
    assert "price_sensitivity" not in prompt
    assert "counter_strength" not in prompt
    assert "walkaway_threshold" not in prompt
    assert "feature_weight_vector" not in prompt


def test_render_observation_supports_prompt_versions() -> None:
    catalog = Catalog(
        options=[
            CatalogOption(
                key="paint_color.paint_metallic",
                dimension="paint_color",
                price_delta_usd=750.0,
                aesthetic_weight=0.45,
            )
        ]
    )
    observation = {
        "round_idx": 1,
        "remaining_rounds": 5,
        "last_agent_offer_usd": None,
        "last_consumer_response": "none",
        "last_consumer_offer_usd": None,
        "history_len": 0,
        "selected_option_keys": ["paint_color.paint_metallic"],
        "total_customization_cost_usd": 375.0,
        "total_msrp_delta_usd": 750.0,
        "aesthetic_proxy_score": 0.45,
        "persona_age_band": "36-50",
        "persona_income_band": "100-180k",
        "persona_household_stage": "family",
        "persona_ownership_stage": "replacement",
        "persona_primary_use_case": "family",
    }

    prompt_v1 = render_llm_observation(observation, catalog, prompt_version="v1")
    prompt_v2 = render_llm_observation(observation, catalog, prompt_version="v2")

    assert '"prompt_version": "v1"' in prompt_v1
    assert '"prompt_version": "v2"' in prompt_v2
    assert "state_description" not in prompt_v1
    assert "state_description" in prompt_v2
