"""Persona sampling and persona-bank loading for runtime episodes."""

from __future__ import annotations

import json
from pathlib import Path
import random
from typing import Any, Dict, List, Optional, Tuple

import yaml

from .types import PersonaProfile


FEATURE_KEYS = ("safety", "comfort", "performance", "tech", "aesthetics")
OWNERSHIP_STAGES = ("first_time", "replacement", "additional")
DEFAULT_PERSONA_SPLIT = "train"


def _clamp01(value: float) -> float:
    """Clamps a numeric value into [0, 1].

    Args:
        value: Input numeric value.

    Returns:
        Clamped value.
    """

    return min(1.0, max(0.0, float(value)))


def _normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    """Normalizes feature weights to sum to one.

    Args:
        weights: Raw feature weights.

    Returns:
        Normalized feature weights.
    """

    total = sum(max(0.0, float(value)) for value in weights.values())
    if total <= 1e-9:
        uniform = 1.0 / float(len(FEATURE_KEYS))
        return {key: uniform for key in FEATURE_KEYS}
    return {key: max(0.0, float(weights.get(key, 0.0))) / total for key in FEATURE_KEYS}


class PersonaSampler:
    """Samples `Persona v2` profiles from explicit observable/hidden mappings."""

    def __init__(self, config: Dict[str, Any], *, config_path: Path | None = None):
        """Initializes a persona sampler.

        Args:
            config: Parsed persona YAML configuration.
            config_path: Optional source path used to resolve external YAML hooks.
        """

        self._config = config
        self._config_path = config_path
        self._schema_version = str(config.get("schema_version", "")).strip().lower()
        if self._schema_version not in {"persona_v2", "v2"}:
            raise ValueError("PersonaSampler requires `schema_version: persona_v2`.")

        self._sampling = config.get("sampling", {})
        self._observable_hook = config.get("observable_distribution", {})
        self._hidden_hook = config.get("hidden_mapping", {})

        self._observable_distribution = self._load_external_yaml_config(
            hook=self._observable_hook,
            expected_schema="us_buyer_distribution_v2",
        )
        self._hidden_mapping = self._load_external_yaml_config(
            hook=self._hidden_hook,
            expected_schema="persona_hidden_mapping_v1",
        )

    def sample(self, rng: random.Random) -> PersonaProfile:
        """Samples one persona profile.

        Args:
            rng: Random generator for reproducible sampling.

        Returns:
            A sampled persona profile.
        """

        age_band, income_band, household_stage, ownership_stage, primary_use_case = self._sample_observable_profile(
            rng=rng,
        )
        hidden = self._sample_hidden_profile(
            rng=rng,
            age_band=age_band,
            income_band=income_band,
            ownership_stage=ownership_stage,
            primary_use_case=primary_use_case,
        )

        return PersonaProfile(
            persona_id=f"cfg_v2_{rng.randrange(10**10):010d}",
            persona_source=str(hidden.get("persona_source", "distribution_hidden_mapping_v1")),
            persona_split=DEFAULT_PERSONA_SPLIT,
            age_band=age_band,
            income_band=income_band,
            household_stage=household_stage,
            ownership_stage=ownership_stage,
            primary_use_case=primary_use_case,
            decision_style=str(hidden["decision_style"]),
            tech_affinity_band=str(hidden["tech_affinity_band"]),
            stated_priority_top2=[str(hidden["stated_priority_top2"][0]), str(hidden["stated_priority_top2"][1])],
            reservation_price_base=float(hidden["reservation_price_base"]),
            price_sensitivity=float(hidden["price_sensitivity"]),
            aesthetic_sensitivity=float(hidden["aesthetic_sensitivity"]),
            patience=int(hidden["patience"]),
            counter_strength=_clamp01(float(hidden["counter_strength"])),
            walkaway_threshold=_clamp01(float(hidden["walkaway_threshold"])),
            belief_obscurity=_clamp01(float(hidden["belief_obscurity"])),
            brand_loyalty=_clamp01(float(hidden["brand_loyalty"])),
            impulsivity=_clamp01(float(hidden["impulsivity"])),
            feature_weight_vector=dict(hidden["feature_weight_vector"]),
        )

    def _sample_observable_profile(self, *, rng: random.Random) -> Tuple[str, str, str, str, str]:
        """Samples observable persona fields from the US buyer distribution config.

        Args:
            rng: Random generator.

        Returns:
            Observable tuple ordered as
            `(age_band, income_band, household_stage, ownership_stage, primary_use_case)`.
        """

        priors = self._observable_distribution["observable_priors"]
        use_conditionals = bool(self._observable_hook.get("use_conditionals", True))
        conditionals = self._observable_distribution.get("joint_approximation", {}).get("conditionals", {})

        age_band = str(self._sample_categorical(rng, priors["age_band"]))
        if use_conditionals:
            income_band = str(
                self._sample_from_conditional(
                    rng=rng,
                    conditional_table=conditionals.get("income_band_given_age_band", {}),
                    parent_value=age_band,
                    ordered_values=[str(v) for v in priors["income_band"]["values"]],
                )
            )
            household_stage = str(
                self._sample_from_conditional(
                    rng=rng,
                    conditional_table=conditionals.get("household_stage_given_age_band", {}),
                    parent_value=age_band,
                    ordered_values=[str(v) for v in priors["household_stage"]["values"]],
                )
            )
            ownership_stage = str(
                self._sample_from_conditional(
                    rng=rng,
                    conditional_table=conditionals.get("ownership_stage_given_age_band", {}),
                    parent_value=age_band,
                    ordered_values=[str(v) for v in priors["ownership_stage"]["values"]],
                )
            )
            primary_use_case = str(
                self._sample_from_conditional(
                    rng=rng,
                    conditional_table=conditionals.get("primary_use_case_given_household_stage", {}),
                    parent_value=household_stage,
                    ordered_values=[str(v) for v in priors["primary_use_case"]["values"]],
                )
            )
            return age_band, income_band, household_stage, ownership_stage, primary_use_case

        income_band = str(self._sample_categorical(rng, priors["income_band"]))
        household_stage = str(self._sample_categorical(rng, priors["household_stage"]))
        ownership_stage = str(self._sample_categorical(rng, priors["ownership_stage"]))
        primary_use_case = str(self._sample_categorical(rng, priors["primary_use_case"]))
        return age_band, income_band, household_stage, ownership_stage, primary_use_case

    def _sample_hidden_profile(
        self,
        *,
        rng: random.Random,
        age_band: str,
        income_band: str,
        ownership_stage: str,
        primary_use_case: str,
    ) -> Dict[str, Any]:
        """Samples hidden persona fields from explicit mapping config.

        Args:
            rng: Random generator.
            age_band: Sampled age band.
            income_band: Sampled income band.
            ownership_stage: Sampled ownership stage.
            primary_use_case: Sampled use-case.

        Returns:
            Hidden profile dictionary used to build `PersonaProfile`.
        """

        mapping = self._hidden_mapping
        hidden_conditionals = mapping["hidden_conditionals"]
        numeric_mixtures = mapping["numeric_mixtures"]
        use_conditionals = bool(self._hidden_hook.get("use_conditionals", True))

        if use_conditionals:
            decision_style = str(
                self._sample_categorical_from_map(
                    rng=rng,
                    conditional_map=hidden_conditionals["decision_style_given_primary_use_case"],
                    parent_key=primary_use_case,
                )
            )
            tech_affinity_band = str(
                self._sample_categorical_from_map(
                    rng=rng,
                    conditional_map=hidden_conditionals["tech_affinity_band_given_age_band"],
                    parent_key=age_band,
                )
            )
            stated_priority_top2 = self._sample_priority_pair_from_map(
                rng=rng,
                conditional_map=hidden_conditionals["stated_priority_top2_given_primary_use_case"],
                parent_key=primary_use_case,
            )
        else:
            decision_style = str(self._sample_categorical(rng, hidden_conditionals["decision_style_global"]))
            tech_affinity_band = str(self._sample_categorical(rng, hidden_conditionals["tech_affinity_band_global"]))
            stated_priority_top2 = self._sample_priority_pair(rng, hidden_conditionals["stated_priority_top2_global"])

        reservation_price_base = self._sample_reservation_price_base_from_table(
            rng=rng,
            income_band=income_band,
            table=mapping["reservation_price_base_by_income"],
        )
        price_sensitivity = float(self._sample_categorical(rng, numeric_mixtures["price_sensitivity"]))
        aesthetic_sensitivity = float(self._sample_categorical(rng, numeric_mixtures["aesthetic_sensitivity"]))
        patience = int(self._sample_categorical(rng, numeric_mixtures["patience"]))
        counter_strength = float(self._sample_categorical(rng, numeric_mixtures["counter_strength"]))
        walkaway_threshold = float(self._sample_categorical(rng, numeric_mixtures["walkaway_threshold"]))
        belief_obscurity = float(self._sample_categorical(rng, numeric_mixtures["belief_obscurity"]))
        brand_loyalty = float(self._sample_categorical(rng, numeric_mixtures["brand_loyalty"]))
        impulsivity = float(self._sample_categorical(rng, numeric_mixtures["impulsivity"]))

        feature_weight_vector = self._build_feature_weights(
            rng=rng,
            primary_use_case=primary_use_case,
            decision_style=decision_style,
            stated_priority_top2=stated_priority_top2,
            templates=mapping["feature_weight_templates"],
        )

        shifts = mapping["conditional_shifts"]
        primary_shift = shifts.get("by_primary_use_case", {}).get(primary_use_case, {})
        ownership_shift = shifts.get("by_ownership_stage", {}).get(ownership_stage, {})
        tech_shift = shifts.get("by_tech_affinity_band", {}).get(tech_affinity_band, {})
        price_priority_shift = shifts.get("by_priority_contains", {}).get("price", {})

        price_sensitivity += float(primary_shift.get("price_sensitivity", 0.0))
        brand_loyalty += float(primary_shift.get("brand_loyalty", 0.0))
        aesthetic_sensitivity += float(primary_shift.get("aesthetic_sensitivity", 0.0))

        price_sensitivity += float(ownership_shift.get("price_sensitivity", 0.0))
        brand_loyalty += float(ownership_shift.get("brand_loyalty", 0.0))
        walkaway_threshold += float(ownership_shift.get("walkaway_threshold", 0.0))
        patience += int(ownership_shift.get("patience", 0))
        aesthetic_sensitivity += float(ownership_shift.get("aesthetic_sensitivity", 0.0))

        brand_loyalty += float(tech_shift.get("brand_loyalty", 0.0))
        if "price" in stated_priority_top2:
            price_sensitivity += float(price_priority_shift.get("price_sensitivity", 0.0))

        coupling = mapping["hidden_coupling"]
        reservation_cfg = coupling["reservation_price_base_from_price_sensitivity"]
        reservation_multiplier = float(reservation_cfg["base"]) + float(reservation_cfg["slope"]) * (price_sensitivity - 1.0)
        reservation_multiplier = self._clip_to_bounds(
            reservation_multiplier,
            [float(reservation_cfg["min"]), float(reservation_cfg["max"])],
        )
        reservation_price_base *= reservation_multiplier

        walkaway_threshold += (
            float(coupling["walkaway_from_price_sensitivity"]["slope"])
            * (price_sensitivity - float(coupling["walkaway_from_price_sensitivity"]["center"]))
        )
        walkaway_threshold += (
            float(coupling["walkaway_from_patience"]["slope"])
            * (patience - float(coupling["walkaway_from_patience"]["center"]))
        )
        counter_strength += (
            float(coupling["counter_strength_from_belief_obscurity"]["slope"])
            * (belief_obscurity - float(coupling["counter_strength_from_belief_obscurity"]["center"]))
        )

        bounds = mapping["bounds"]
        price_sensitivity = self._clip_to_bounds(price_sensitivity, bounds["price_sensitivity"])
        aesthetic_sensitivity = self._clip_to_bounds(aesthetic_sensitivity, bounds["aesthetic_sensitivity"])
        patience = int(round(self._clip_to_bounds(float(patience), bounds["patience"])))
        counter_strength = self._clip_to_bounds(counter_strength, bounds["counter_strength"])
        walkaway_threshold = self._clip_to_bounds(walkaway_threshold, bounds["walkaway_threshold"])
        belief_obscurity = self._clip_to_bounds(belief_obscurity, bounds["belief_obscurity"])
        brand_loyalty = self._clip_to_bounds(brand_loyalty, bounds["brand_loyalty"])
        impulsivity = self._clip_to_bounds(impulsivity, bounds["impulsivity"])

        return {
            "decision_style": decision_style,
            "tech_affinity_band": tech_affinity_band,
            "stated_priority_top2": stated_priority_top2,
            "reservation_price_base": reservation_price_base,
            "price_sensitivity": price_sensitivity,
            "aesthetic_sensitivity": aesthetic_sensitivity,
            "patience": patience,
            "counter_strength": counter_strength,
            "walkaway_threshold": walkaway_threshold,
            "belief_obscurity": belief_obscurity,
            "brand_loyalty": brand_loyalty,
            "impulsivity": impulsivity,
            "feature_weight_vector": feature_weight_vector,
            "persona_source": "distribution_hidden_mapping_v1",
        }

    def _build_feature_weights(
        self,
        *,
        rng: random.Random,
        primary_use_case: str,
        decision_style: str,
        stated_priority_top2: List[str],
        templates: Dict[str, Any],
    ) -> Dict[str, float]:
        """Builds normalized feature weights for WTP construction.

        Args:
            rng: Random generator.
            primary_use_case: Sampled use-case.
            decision_style: Sampled hidden decision style.
            stated_priority_top2: Sampled hidden priorities.
            templates: Use-case weight templates.

        Returns:
            Normalized feature-weight dictionary.
        """

        base = dict(templates.get(primary_use_case, templates.get("mixed", {})))
        weights = {key: float(base.get(key, 0.2)) for key in FEATURE_KEYS}

        if decision_style == "analytic":
            weights["safety"] += 0.06
            weights["tech"] += 0.06
        elif decision_style == "expressive":
            weights["aesthetics"] += 0.08
            weights["performance"] += 0.04

        for label in stated_priority_top2:
            mapped = self._map_priority_to_feature(label)
            if mapped is not None:
                weights[mapped] += 0.08
            elif label == "price":
                weights["safety"] += 0.03
                weights["comfort"] += 0.03
                weights["performance"] = max(0.05, weights["performance"] - 0.03)
                weights["aesthetics"] = max(0.05, weights["aesthetics"] - 0.03)

        jitter_std = float(self._sampling.get("feature_jitter_std", 0.02))
        for key in FEATURE_KEYS:
            weights[key] = max(0.02, weights[key] + rng.gauss(0.0, jitter_std))

        return _normalize_weights(weights)

    def _load_external_yaml_config(
        self,
        *,
        hook: Any,
        expected_schema: str,
    ) -> Dict[str, Any]:
        """Loads external YAML config and validates schema.

        Args:
            hook: Hook block from persona config.
            expected_schema: Required schema version label.

        Returns:
            Parsed config dictionary.
        """

        if not isinstance(hook, dict) or not bool(hook.get("enabled", False)):
            raise ValueError(f"Hook for `{expected_schema}` must be enabled.")
        path_raw = hook.get("path")
        if not path_raw:
            raise ValueError(f"Hook for `{expected_schema}` requires `path`.")

        path = Path(str(path_raw))
        if not path.is_absolute():
            base = self._config_path.parent if self._config_path is not None else Path(".")
            path = base / path

        payload = yaml.safe_load(path.read_text())
        schema = str(payload.get("schema_version", "")).strip().lower()
        if schema != expected_schema:
            raise ValueError(
                f"External config `{path}` has schema `{schema}`, expected `{expected_schema}`."
            )
        return payload

    def _sample_reservation_price_base_from_table(
        self,
        *,
        rng: random.Random,
        income_band: str,
        table: Dict[str, Any],
    ) -> float:
        """Samples reservation base price from an explicit income table.

        Args:
            rng: Random generator.
            income_band: Income-band key.
            table: Mapping from income band to `{mean, std}`.

        Returns:
            Reservation base price in USD.
        """

        if income_band not in table:
            return float(max(1000.0, rng.gauss(10000.0, 1200.0)))
        mean = float(table[income_band]["mean"])
        std = float(table[income_band]["std"])
        return float(max(1000.0, rng.gauss(mean, std)))

    @staticmethod
    def _sample_from_conditional(
        *,
        rng: random.Random,
        conditional_table: Dict[str, Any],
        parent_value: str,
        ordered_values: List[str],
    ) -> Any:
        """Samples from a strict conditional probability row.

        Args:
            rng: Random generator.
            conditional_table: Parent-keyed table of probability vectors.
            parent_value: Parent sampled value.
            ordered_values: Ordered categorical value list.

        Returns:
            One sampled value.
        """

        row = conditional_table.get(parent_value)
        if not isinstance(row, list):
            raise ValueError(f"Missing conditional row for parent value `{parent_value}`.")
        if len(row) != len(ordered_values):
            raise ValueError(
                f"Conditional row length mismatch for parent value `{parent_value}`: "
                f"expected {len(ordered_values)}, got {len(row)}."
            )
        return rng.choices(ordered_values, weights=[float(p) for p in row], k=1)[0]

    @staticmethod
    def _sample_categorical_from_map(
        *,
        rng: random.Random,
        conditional_map: Dict[str, Any],
        parent_key: str,
    ) -> Any:
        """Samples a categorical value from a parent-keyed distribution map.

        Args:
            rng: Random generator.
            conditional_map: Parent-keyed distribution map.
            parent_key: Parent sampled token.

        Returns:
            One sampled value.
        """

        dist = conditional_map.get(parent_key)
        if not isinstance(dist, dict):
            raise ValueError(f"Missing conditional distribution for parent key `{parent_key}`.")
        return rng.choices(dist["values"], weights=dist["probs"], k=1)[0]

    @staticmethod
    def _sample_priority_pair_from_map(
        *,
        rng: random.Random,
        conditional_map: Dict[str, Any],
        parent_key: str,
    ) -> List[str]:
        """Samples one priority pair from a parent-keyed distribution map.

        Args:
            rng: Random generator.
            conditional_map: Parent-keyed distribution map.
            parent_key: Parent sampled token.

        Returns:
            Two-item priority list.
        """

        raw = PersonaSampler._sample_categorical_from_map(
            rng=rng,
            conditional_map=conditional_map,
            parent_key=parent_key,
        )
        if not isinstance(raw, list) or len(raw) != 2:
            raise ValueError("Priority pair distributions must emit two labels.")
        return [str(raw[0]), str(raw[1])]

    @staticmethod
    def _sample_priority_pair(rng: random.Random, dist: Dict[str, Any]) -> List[str]:
        """Samples one top-2 priority pair.

        Args:
            rng: Random generator.
            dist: Distribution dictionary with `values` and `probs`.

        Returns:
            Two priority labels.
        """

        raw = rng.choices(dist["values"], weights=dist["probs"], k=1)[0]
        if not isinstance(raw, list) or len(raw) != 2:
            raise ValueError("`stated_priority_top2` must contain pairs with length 2.")
        return [str(raw[0]), str(raw[1])]

    @staticmethod
    def _sample_categorical(rng: random.Random, dist: Dict[str, Any]) -> Any:
        """Samples a categorical value.

        Args:
            rng: Random generator.
            dist: Dict with `values` and `probs`.

        Returns:
            Sampled value.
        """

        return rng.choices(dist["values"], weights=dist["probs"], k=1)[0]

    @staticmethod
    def _map_priority_to_feature(priority_label: str) -> str | None:
        """Maps a priority token to runtime feature channel.

        Args:
            priority_label: Priority label from persona profile.

        Returns:
            Feature channel key if applicable, otherwise None.
        """

        mapping = {
            "safety": "safety",
            "comfort": "comfort",
            "performance": "performance",
            "tech": "tech",
            "aesthetics": "aesthetics",
        }
        return mapping.get(priority_label)

    @staticmethod
    def _clip_to_bounds(value: float, bounds: List[float]) -> float:
        """Clips a value to `[lower, upper]`.

        Args:
            value: Raw scalar value.
            bounds: Two-element bounds list.

        Returns:
            Clipped scalar value.
        """

        if not isinstance(bounds, list) or len(bounds) != 2:
            return float(value)
        lower = float(bounds[0])
        upper = float(bounds[1])
        return max(lower, min(upper, float(value)))


class PersonaBankSampler:
    """Samples personas from an offline persona-bank file."""

    def __init__(self, profiles_by_split: Dict[str, List[PersonaProfile]], default_split: str) -> None:
        """Initializes a persona-bank sampler.

        Args:
            profiles_by_split: Parsed persona profiles grouped by split.
            default_split: Preferred sampling split.
        """

        self._profiles_by_split = profiles_by_split
        self._default_split = default_split

    @classmethod
    def from_jsonl(cls, path: str | Path, *, split: str = DEFAULT_PERSONA_SPLIT) -> "PersonaBankSampler":
        """Loads a persona bank from JSONL.

        Args:
            path: Persona bank file path.
            split: Preferred split for runtime sampling.

        Returns:
            A configured persona-bank sampler.
        """

        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Persona bank file not found: {file_path}")

        profiles_by_split: Dict[str, List[PersonaProfile]] = {}
        for line_idx, line in enumerate(file_path.read_text().splitlines(), start=1):
            raw = line.strip()
            if not raw:
                continue
            record = json.loads(raw)
            profile_split, profile = cls._parse_record(record, line_idx=line_idx)
            profiles_by_split.setdefault(profile_split, []).append(profile)

        if not profiles_by_split:
            raise ValueError(f"Persona bank is empty: {file_path}")

        target_split = split.strip().lower()
        if target_split not in profiles_by_split:
            available = sorted(profiles_by_split.keys())
            raise ValueError(
                f"Requested persona split `{target_split}` is unavailable. "
                f"Available splits: {available}."
            )
        return cls(profiles_by_split=profiles_by_split, default_split=target_split)

    def sample(self, rng: random.Random) -> PersonaProfile:
        """Samples one persona profile from the configured split.

        Args:
            rng: Random generator for reproducible sampling.

        Returns:
            A sampled persona profile.
        """

        bucket = self._profiles_by_split[self._default_split]
        if not bucket:
            raise RuntimeError(f"Persona split `{self._default_split}` has no records.")
        return rng.choice(bucket)

    @staticmethod
    def _parse_record(record: Dict[str, Any], *, line_idx: int) -> Tuple[str, PersonaProfile]:
        """Parses and validates one persona-bank record.

        Args:
            record: Raw JSON dictionary.
            line_idx: Source line index in the JSONL file.

        Returns:
            Tuple of split label and validated persona profile.
        """

        schema_version = str(record.get("schema_version", "")).strip().lower()
        if schema_version not in {"persona_v2", "v2"}:
            raise ValueError(
                f"Persona bank line {line_idx}: unsupported schema_version `{schema_version}`."
            )

        persona_id = str(record.get("persona_id", "")).strip()
        if not persona_id:
            raise ValueError(f"Persona bank line {line_idx}: missing `persona_id`.")

        split = str(record.get("split", DEFAULT_PERSONA_SPLIT)).strip().lower()
        source = str(record.get("source", "persona_bank")).strip() or "persona_bank"

        observable = record.get("observable")
        hidden = record.get("hidden")
        if not isinstance(observable, dict) or not isinstance(hidden, dict):
            raise ValueError(
                f"Persona bank line {line_idx}: `observable` and `hidden` must be objects."
            )

        priorities = hidden.get("stated_priority_top2", [])
        if not isinstance(priorities, list) or len(priorities) != 2:
            raise ValueError(
                f"Persona bank line {line_idx}: `stated_priority_top2` must contain exactly two values."
            )
        stated_priority_top2 = [str(priorities[0]), str(priorities[1])]

        ownership_stage = str(observable.get("ownership_stage", "replacement")).strip()
        if ownership_stage not in OWNERSHIP_STAGES:
            ownership_stage = "replacement"

        raw_weights = hidden.get("feature_weight_vector", {})
        if not isinstance(raw_weights, dict):
            raise ValueError(f"Persona bank line {line_idx}: `feature_weight_vector` must be an object.")
        weights = {key: float(raw_weights.get(key, 0.0)) for key in FEATURE_KEYS}

        return split, PersonaProfile(
            persona_id=persona_id,
            persona_source=source,
            persona_split=split,
            age_band=str(observable["age_band"]),
            income_band=str(observable["income_band"]),
            household_stage=str(observable["household_stage"]),
            ownership_stage=ownership_stage,
            primary_use_case=str(observable["primary_use_case"]),
            decision_style=str(hidden["decision_style"]),
            tech_affinity_band=str(hidden["tech_affinity_band"]),
            stated_priority_top2=stated_priority_top2,
            reservation_price_base=float(hidden["reservation_price_base"]),
            price_sensitivity=float(hidden["price_sensitivity"]),
            aesthetic_sensitivity=float(hidden["aesthetic_sensitivity"]),
            patience=int(hidden["patience"]),
            walkaway_threshold=_clamp01(float(hidden["walkaway_threshold"])),
            counter_strength=_clamp01(float(hidden["counter_strength"])),
            belief_obscurity=_clamp01(float(hidden["belief_obscurity"])),
            brand_loyalty=_clamp01(float(hidden["brand_loyalty"])),
            impulsivity=_clamp01(float(hidden["impulsivity"])),
            feature_weight_vector=_normalize_weights(weights),
        )


def load_persona_sampler(
    path: str | Path,
    *,
    persona_bank_path: Optional[str | Path] = None,
    persona_bank_split: str = DEFAULT_PERSONA_SPLIT,
) -> Tuple[int, Dict[str, Any], PersonaSampler | PersonaBankSampler]:
    """Loads persona config and builds a sampler.

    Args:
        path: YAML file path.
        persona_bank_path: Optional JSONL persona-bank file path.
        persona_bank_split: Optional split name used with persona-bank sampling.

    Returns:
        Tuple of global seed, parsed config, and persona sampler.
    """

    config_path = Path(path)
    config = yaml.safe_load(config_path.read_text())
    if str(config.get("schema_version", "")).strip().lower() not in {"persona_v2", "v2"}:
        raise ValueError("Only `persona_v2` config is supported.")

    seed = int(config["seed"])
    bank_cfg = config.get("persona_bank", {})

    resolved_bank_path: Optional[Path] = None
    if persona_bank_path is not None:
        resolved_bank_path = Path(persona_bank_path)
    elif bool(bank_cfg.get("enabled", False)) and bank_cfg.get("path"):
        cfg_path = Path(str(bank_cfg["path"]))
        resolved_bank_path = cfg_path if cfg_path.is_absolute() else (config_path.parent / cfg_path)

    if resolved_bank_path is None:
        return seed, config, PersonaSampler(config, config_path=config_path)

    split = str(bank_cfg.get("split", persona_bank_split)) if persona_bank_path is None else persona_bank_split
    sampler = PersonaBankSampler.from_jsonl(path=resolved_bank_path, split=split)
    return seed, config, sampler
