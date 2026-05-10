# PrefBench Simulator Card

This document records the simulator assumptions for the current PrefBench
arXiv prototype. It is intended to make the benchmark auditable without
claiming that the simulator is a calibrated real-market model.

## Purpose

PrefBench simulates seller-side personalized pricing under hidden buyer
preferences. Each episode contains:

- one fixed customization bundle sampled from the catalog;
- one buyer persona sampled from a frozen persona bank;
- a short multi-round negotiation over the bundle price;
- hidden buyer variables that affect willingness to pay and bargaining behavior.

The simulator is designed for controlled agent evaluation, not for deployment
or real-market price prediction.

## Product Space

The product substrate is the fixed E350 customization catalog in
`catalog/e350_core_catalog.yaml`. Each option has:

- a stable option key;
- a customization dimension;
- a consumer-facing MSRP delta;
- an aesthetics proxy score.

Seller-side implementation cost is proxied as a fixed ratio of MSRP delta by
the catalog loader. The arXiv prototype keeps the bundle fixed within an
episode and sampled by the simulator. Agents do not choose the bundle.

## Buyer Population

The buyer population is semi-synthetic.

Observable fields are sampled from `configs/us_buyer_distribution_v2.yaml`.
Those distributions are anchored in public U.S. population and mobility
statistics plus task-specific new-vehicle-buyer adjustments recorded in the
config comments. Observable fields are:

- `age_band`
- `income_band`
- `household_stage`
- `ownership_stage`
- `primary_use_case`

Hidden fields are generated through structured conditional rules in
`configs/persona_hidden_mapping_v1.yaml`. These are benchmark-defined modeling
assumptions rather than directly observed real consumer records. Hidden fields
include:

- `decision_style`
- `tech_affinity_band`
- `stated_priority_top2`
- `reservation_price_base`
- `price_sensitivity`
- `aesthetic_sensitivity`
- `patience`
- `counter_strength`
- `walkaway_threshold`
- `belief_obscurity`
- `brand_loyalty`
- `impulsivity`
- `feature_weight_vector`

The released persona bank in `datasets/persona_bank/bank50k_s123/` freezes this
semi-synthetic population into 35k train, 7.5k validation, and 7.5k test
records. The train split is retained for future trainable methods. The current
zero-shot LLM evaluation path uses the fixed 500-record
`llm_test_500.jsonl` subset sampled from the test split to keep API evaluation
reproducible and cost-controlled.

## WTP And Response Model

The current willingness-to-pay formula is frozen for the arXiv prototype:

```text
WTP_t = R_base + V_custom + V_aesthetic + V_brand_tech - V_fatigue + epsilon_t
```

The terms are:

- `R_base`: income-linked reservation value adjusted by price sensitivity;
- `V_custom`: feature value based on bundle price and hidden feature alignment;
- `V_aesthetic`: aesthetics premium from option aesthetics and persona taste;
- `V_brand_tech`: brand and technology premium;
- `V_fatigue`: round-dependent patience and impulsivity penalty;
- `epsilon_t`: stochastic noise scaled by belief obscurity.

The buyer accepts offers at or below the current WTP, may counter-offer when
rejecting, and may walk away when the offered price is too far above WTP. The
negotiation backend uses NegMAS to execute each round while preserving the
PrefBench action contract.

## Metrics

The benchmark reports outcome metrics rather than claiming behavioral realism:

- `avg_profit_usd`
- `deal_rate`
- `walkaway_rate`
- `avg_rounds`
- `avg_trace_len`
- `avg_env_reward`

These metrics are useful for comparing agents under the same simulator, but
they should not be interpreted as real-world profit estimates.

## Known Limits

- The hidden preference model is hand-designed.
- The benchmark has no human reference study in the current release.
- The WTP formula is a controlled evaluation mechanism, not an econometric
  estimate.
- The current release has no OOD persona suite.
- Agents do not select product bundles in the current action space.

These limits are intentional for the prototype release. Future versions can add
human reference data, OOD splits, calibration studies, and richer action spaces.
