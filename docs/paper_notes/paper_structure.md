# Updated arXiv Paper Structure Notes

These notes define the agreed structure for the PrefBench arXiv paper after the
completion of the zero-shot LLM experiments. The target is a 10--11 page main
text, excluding references. The paper should not be a short workshop-style
writeup, but it also should not reproduce the thesis-level benchmark definition
in full.

## Core Positioning

The thesis-stage work already defines the PrefBench task, POMDP formulation,
consumer simulator, persona bank, negotiation environment, and benchmark
protocol. The arXiv paper should use that benchmark foundation compactly and
focus its new contribution on the LLM-facing protocol and completed zero-shot
LLM evaluation.

Recommended one-sentence thesis:

> PrefBench is a simulator-grounded benchmark prototype for hidden-preference
> personalized pricing, and its zero-shot LLM evaluation shows a gap between
> structured action compliance and strategic pricing competence.

The paper should be framed as an artifact-oriented prototype benchmark and
baseline study. It should not claim to be a fully human-validated pricing
benchmark or a complete model of real consumer behavior.

## Target Length

Main text target: 10--11 pages.

Approximate allocation:

| Section | Target pages | Role |
|---|---:|---|
| Abstract | 0.3--0.5 | Compact artifact and result summary |
| Introduction | 1.2 | Motivation, gap, contributions |
| Related Work | 1.0 | Pricing, negotiation agents, simulator benchmarks, LLM agent evaluation |
| Task Formulation | 1.0 | Episode, observability, actions, objective |
| Simulator and Benchmark Assets | 1.5 | Catalog, persona bank, hidden variables, WTP and response model |
| LLM Protocol and Baselines | 1.3 | Prompt rendering, JSON actions, policies, API protocol |
| Experiments | 2.0 | Main table, behavior distributions, prompt and reasoning ablations |
| Findings and Discussion | 1.0 | Interpretation of LLM behavior |
| Limitations and Future Work | 0.8 | Scope boundaries and extension path |
| Conclusion | 0.2--0.3 | Short closing statement |

## Proposed Main Structure

### 1. Abstract

The abstract should now report completed LLM experiments, not a planned LLM
path.

Include:

- PrefBench as a benchmark prototype for hidden-preference personalized pricing.
- Fixed customization bundle, observable buyer profile, hidden buyer variables,
  and multi-round negotiation.
- Structured LLM action protocol with strict JSON moves.
- Main result: zero-shot LLMs reliably follow the action protocol and achieve
  high deal rates, but tend to settle immediately and earn lower seller profit
  than a simple concession heuristic.
- Scope: simulator-based and semi-synthetic, not human-validated.

Avoid:

- "planned LLM evaluation"
- claims of real-world pricing validity
- broad claims that LLMs are generally poor at pricing

### 2. Introduction

The introduction should motivate the benchmark through the gap between LLM
agent compliance and profit-sensitive strategic decision making.

Suggested paragraph functions:

1. Personalized pricing and negotiation require decisions under hidden buyer
   preferences, not only demand prediction.
2. LLM agents are increasingly used in structured decision interfaces, but
   following an action format is different from bargaining strategically.
3. Existing pricing and negotiation evaluations do not directly isolate this
   hidden-preference seller-side pricing setting.
4. PrefBench provides a reproducible simulator-grounded task with fixed assets,
   hidden buyer variables, and shared metrics.
5. This paper extends the thesis-stage benchmark foundation with an LLM-facing
   protocol and reports zero-shot LLM baselines.

Contribution framing:

1. A hidden-preference personalized-pricing negotiation task for LLM agents.
2. A public-data-informed semi-synthetic persona bank and simulator over a
   fixed vehicle-customization product substrate.
3. A structured LLM interaction protocol with strict JSON actions and explicit
   hidden-information boundaries.
4. Baseline results showing that zero-shot LLMs comply with the protocol but
   under-exploit multi-round bargaining opportunities.
5. An open-source artifact and evaluation protocol that can be extended to
   broader pricing-agent benchmarks.

### 3. Related Work

Keep related work concise. The section should situate the task without becoming
a thesis literature review.

Suggested subsections or paragraphs:

- Dynamic and personalized pricing.
- Automated negotiation and bargaining environments.
- Simulation benchmarks and synthetic research environments.
- LLM agent and negotiation evaluation.

The related-work gap should be scoped:

> PrefBench differs by combining hidden buyer preferences, configurable product
> pricing, seller-side profit metrics, and an LLM-facing structured action
> protocol under one reproducible simulator.

### 4. Task Formulation

This section should compress the thesis POMDP formulation into a paper-sized
definition.

Include:

- One episode is one seller-buyer negotiation over one fixed customization
  bundle.
- The seller observes only public profile fields, bundle descriptors, and
  negotiation history.
- Hidden variables include reservation value, price sensitivity, patience,
  counter strength, walkaway threshold, and feature weights.
- Actions are `offer`, `accept`, and `walkaway`.
- The episode ends with deal, buyer walkaway, seller walkaway, invalid action,
  or horizon exhaustion.
- Primary objective:

```text
profit_usd = deal_price_usd - estimated_implementation_cost_usd
```

Use the POMDP language lightly. It is useful to mention the finite-horizon POMDP
view, but the arXiv paper should not reproduce the full thesis tuple in detail.

### 5. Simulator and Benchmark Assets

This section should establish credibility and transparency for the benchmark
without spending too much space.

Include:

- Vehicle customization catalog as the current product substrate.
- Bundle-level MSRP delta, estimated implementation cost, and aesthetic proxy.
- Public-data-informed observable buyer distribution.
- Semi-synthetic hidden variables generated from structured conditional rules.
- Frozen persona bank: 50k records, 35k train, 7.5k validation, 7.5k test.
- Main LLM and heuristic reports use the 7.5k test split.
- The optional `llm_test_500` subset should be described only as a
  cost-controlled subset for development or optional future use, not as the main
  reported evaluation.
- WTP and response model summarized in words, with the formula if space allows:

```text
WTP_t = R_base + V_custom + V_aesthetic + V_brand-tech - V_fatigue + epsilon_t
```

Important wording:

- "public-data-informed"
- "semi-synthetic"
- "controlled simulator"
- "benchmark-defined hidden variables"

Avoid:

- "realistic consumer behavior" without qualification
- "validated human preference model"
- "real-market profit estimate"

### 6. LLM Protocol and Baselines

This is one of the new central sections of the arXiv paper.

Include the environment-to-LLM flow:

1. `NegotiationEnv` returns a structured observation.
2. The observation is rendered into an LLM prompt.
3. The model is called through an OpenAI-compatible chat-completions API.
4. The model returns one JSON action.
5. The JSON is parsed into the shared environment action.
6. The environment executes the action and records the result.

Define the LLM-visible state:

- Current round and remaining seller turns.
- Selected customization options and dimensions.
- Bundle MSRP delta.
- Estimated implementation cost.
- Aesthetic proxy score.
- Observable buyer profile fields.
- Last seller offer.
- Last consumer response.
- Last consumer counter-offer if present.
- Negotiation history length.

State clearly that LLMs do not see:

- Reservation price base.
- Price sensitivity.
- Feature weight vector.
- Counter strength.
- Walkaway threshold.
- True willingness to pay.

Baselines:

- Random.
- Concession heuristic.
- DeepSeek V4 Flash.
- Kimi K2.6.
- Qwen3.6 Plus.

Supporting ablations:

- DeepSeek V4 Flash prompt v2.
- DeepSeek V4 Pro reasoning smoke100.

Do not put PPO or Dreamer in the main result table. They belong to the broader
thesis-stage benchmark foundation and can be mentioned as supported future or
separate trainable-agent paths.

### 7. Experiments

The experiment section should report the new completed results.

Main setup:

- Test split: `datasets/persona_bank/bank50k_s123/test.jsonl`.
- Episodes: 7,500.
- Seed: 123.
- Main LLM prompt: prompt v1.
- Main LLM mode: non-thinking / cost-effective provider configurations.
- Hidden buyer variables are not exposed to policies.

Main table columns:

- Method.
- Episodes.
- Deal rate.
- Avg profit USD.
- Avg rounds.
- Walkaway rate.
- Invalid rate.
- Tokens per episode for LLMs.

Main results:

| Method | Episodes | Deal Rate | Avg Profit USD | Avg Rounds | Walkaway Rate | Invalid Rate | Tokens / Ep |
|---|---:|---:|---:|---:|---:|---:|---:|
| Random | 7,500 | 0.5769 | 6,572.33 | 1.3541 | 0.4231 | n/a | n/a |
| Concession Heuristic | 7,500 | 0.7268 | 14,774.11 | 1.7123 | 0.2732 | n/a | n/a |
| DeepSeek V4 Flash | 7,500 | 0.9903 | 6,749.21 | 1.0313 | 0.0097 | 0.0000 | 1,399.22 |
| Kimi K2.6 | 7,500 | 1.0000 | 4,514.43 | 1.0000 | 0.0000 | 0.0000 | 1,297.97 |
| Qwen3.6 Plus | 7,500 | 0.9985 | 5,979.55 | 1.0036 | 0.0015 | 0.0000 | 1,473.91 |

Supporting table:

| Run | Episodes | Deal Rate | Avg Profit USD | Avg Rounds | Invalid Rate | Tokens / Ep | Purpose |
|---|---:|---:|---:|---:|---:|---:|---|
| DeepSeek V4 Flash prompt v2 | 7,500 | 0.9993 | 4,142.73 | 1.0008 | 0.0000 | 1,724.47 | Prompt-clarity ablation |
| DeepSeek V4 Pro Reasoning | 100 | 0.9900 | 5,231.50 | 1.0000 | 0.0100 | 2,036.81 | Small reasoning ablation |

The text should not frame the table as a model leaderboard. It should frame the
results as behavioral evidence about the benchmark.

### 8. Findings and Discussion

This section should carry the main interpretation.

Core findings:

1. LLMs follow the structured protocol reliably.
2. Zero-shot LLMs tend to settle immediately.
3. High deal rate does not imply high profit.
4. The concession heuristic remains strong because it encodes an explicit
   pricing prior.
5. More detailed prompt explanations did not improve strategy.
6. A small reasoning-enabled run did not automatically solve bargaining.

Main discussion claim:

> PrefBench exposes a gap between action-format compliance and strategic pricing
> competence.

Important interpretation:

- LLMs are not failing at JSON compliance.
- LLMs are also not failing to reach agreements.
- The interesting failure mode is that they reach agreements too softly and too
  quickly from the seller-profit perspective.
- The concession heuristic should not be described as a generally superior
  agent. It is a useful anchor because it encodes a hand-designed aggressive
  concession prior.

### 9. Limitations and Future Work

This section should stay honest and direct.

Limitations:

- No human validation.
- Semi-synthetic hidden behavior model.
- Single product domain.
- Prompt sensitivity.
- Hosted API instability.
- No fine-tuning or learned LLM policy evaluation.
- No full fairness, privacy, or legal analysis.
- Reasoning-model evaluation is only a small smoke run.

Future work:

- Human buyer or seller validation.
- Additional product and pricing domains.
- Agent-selected bundles rather than fixed sampled bundles.
- Subscription pricing, promotion design, auction-like selling, bundle pricing,
  and inventory-aware dynamic pricing.
- Richer metrics: consumer surplus, fairness, robustness, inventory effects,
  and long-term customer value.
- Larger reasoning-model evaluations.
- Fine-tuned, tool-augmented, or learned pricing agents.
- Versioned benchmark releases with frozen prompts, data, API settings, and
  schemas.

### 10. Conclusion

Keep the conclusion short.

Suggested function:

- Restate PrefBench as a concrete simulator-grounded benchmark prototype.
- Restate the main finding: LLMs comply with the structured action protocol but
  do not yet exploit the hidden-preference bargaining structure well.
- Close with the artifact value and extension path.

## Claims To Avoid

- Do not claim PrefBench is a complete standardized pricing benchmark.
- Do not claim the simulator fully models real consumer behavior.
- Do not claim the benchmark is human-validated.
- Do not claim LLMs are generally bad at pricing.
- Do not claim heuristic policies and LLM policies have identical interfaces.
- Do not claim reasoning models have been fully evaluated.
- Do not put PPO or Dreamer in the main LLM result story unless the paper is
  explicitly broadened beyond the zero-shot LLM artifact.

## Mapping From Current arXiv Draft

The current draft needs these main updates:

- `00_abstract.tex`: replace "planned zero-shot LLM evaluation path" with
  completed LLM results and main finding.
- `01_introduction.tex`: update contributions around LLM protocol and completed
  evaluation.
- `02_benchmark.tex`: split or expand into task formulation plus simulator and
  benchmark assets; align main reported evaluation with the 7,500-episode test
  split.
- `03_agents.tex`: replace "LLM-facing extension" with the actual LLM protocol
  and zero-shot baselines.
- `04_experiments.tex`: replace heuristic-only 5,000-episode table with the
  7,500-episode main table and supporting ablation table.
- `05_limitations.tex`: remove statements that LLM evaluation is future work;
  keep limitations around simulator scope, prompt sensitivity, API instability,
  and missing human validation.
- `06_conclusion.tex`: update to the completed LLM benchmark finding.

