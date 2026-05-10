# Targeted Related Work Search Notes

Date: 2026-05-07

Purpose: sharpen the Introduction and Related Work gap for PrefBench after the
zero-shot LLM experiments.

## Main Positioning Gap

Closest work falls into three groups:

1. LLM negotiation and bargaining benchmarks.
2. General LLM agent / tool-use benchmarks.
3. Retail, pricing, and simulator-based evaluation environments.

PrefBench should be positioned at the intersection but not as a replacement for
any one group. The key missing setting is:

> seller-side profit-sensitive pricing over a fixed configurable bundle, under
> hidden buyer variables, multi-round negotiation, and a structured LLM action
> protocol.

## LLM Negotiation and Bargaining Benchmarks

Already in bibliography:

- `xia2024MeasuringBargainingAbilities`
  - Evaluates LLM bargaining abilities as an asymmetric incomplete-information
    bargaining task.
  - Strongly relevant because it studies buyer/seller bargaining with profit
    metrics.
  - Difference: PrefBench focuses on personalized pricing over configurable
    product bundles with a simulator-defined hidden buyer persona and fixed
    seller-side pricing protocol.

- `chan2024NegotiationToMBenchmarkStresstesting`
  - Evaluates theory-of-mind ability around negotiation scenarios.
  - Relevant for hidden mental states and negotiation reasoning.
  - Difference: PrefBench evaluates action-level seller pricing outcomes rather
    than ToM question answering or mental-state inference.

Possible additional related work but not yet added:

- PACT / conversational bargaining benchmarks.
- AgreeMate / training LLMs to haggle.

These are useful if we later broaden the LLM negotiation paragraph, but the two
existing citations may be enough for the Introduction.

## General LLM Agent and Tool-Use Benchmarks

Added to bibliography:

- `liu2023AgentBenchEvaluatingLLMs`
  - General benchmark for evaluating LLMs as agents across interactive
    environments.
  - Useful for motivating LLM-as-agent evaluation.

- `yao2024TauBenchToolAgentUser`
  - Tool-agent-user interaction benchmark in realistic domains.
  - Useful for structured action, tool/API interaction, and reliability under
    domain rules.

- `qin2023ToolLLMFacilitatingLarge`
  - Tool-use benchmark and training framework with real-world APIs.
  - Useful for positioning JSON/tool/action compliance as a known evaluation
    axis.

Gap statement:

> These benchmarks evaluate multi-step interaction, tool use, and rule
> following, but they do not specifically test whether an LLM seller can convert
> hidden buyer-preference signals into profit-sensitive pricing decisions.

## Retail and Pricing Simulation

Already in bibliography:

- `xia2023RetailSynthSyntheticData`
  - Synthetic retail simulator for evaluating retail AI systems with
    heterogeneity and price sensitivity.
  - Useful for justifying simulator-based retail/pricing evaluation.
  - Difference: PrefBench is interactive negotiation over fixed customization
    bundles with hidden buyer state and LLM action protocol.

Existing dynamic/personalized pricing citations in the bibliography remain
useful:

- `choi2023SemiParametricContextualPricing`
- `liu2021DynamicPricingEcommerce`
- `pandey2020DeepReinforcementLearning`
- `biggs2021ModelDistillationRevenue`
- `dube2017PersonalizedPricingConsumer`
- `priester2020SpecialPriceJust`

Gap statement:

> Dynamic and personalized pricing work studies demand response, revenue
> optimization, and customer heterogeneity, but usually not a benchmark where
> an LLM seller acts through a constrained multi-round negotiation protocol
> while hidden buyer variables are retained by the evaluator.

## Model Citations

Added official documentation citations:

- `deepseek2026V4PreviewRelease`
- `deepseek2026ModelsPricing`
- `moonshot2026KimiModelList`
- `alibaba2026QwenModelStudioModels`

Use these in the experiments or LLM protocol section when listing tested model
IDs. Prefer official API/model documentation over news articles.

## Introduction Revision Implication

The Introduction should not say only that existing work "touches different
parts" of the problem. A sharper paragraph is:

> Pricing benchmarks and retail simulators study demand response and customer
> heterogeneity; negotiation benchmarks study bargaining and mental-state
> reasoning; and LLM-agent benchmarks study tool use, task completion, and rule
> following. None of these directly isolates the combination needed here:
> hidden buyer variables, seller-side profit, multi-round pricing actions, and
> structured LLM outputs under one fixed evaluation protocol.

