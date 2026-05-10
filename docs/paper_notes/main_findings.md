# Main Findings Notes

These notes collect the paper-facing findings supported by the completed
experiments. They are intentionally restrained and should not be overstated.

## Finding 1: LLMs Follow the Structured Protocol Reliably

Across the main full-test runs, non-thinking LLM agents produced valid JSON
actions with zero invalid episodes:

- DeepSeek V4 Flash: invalid rate `0.0000`
- Kimi K2.6: invalid rate `0.0000`
- Qwen3.6 Plus: invalid rate `0.0000`

This supports the claim that the benchmark protocol is usable with
OpenAI-compatible LLM APIs and that current LLMs can follow the structured
action interface.

Do not overclaim this as strategic competence. It only shows protocol
compliance.

## Finding 2: Zero-Shot LLMs Tend to Settle Immediately

The tested zero-shot LLMs strongly prefer immediate agreements:

- DeepSeek V4 Flash: `7408 / 7500` episodes ended in one round.
- Kimi K2.6: `7500 / 7500` episodes ended in one round.
- Qwen3.6 Plus: `7489 / 7500` episodes ended in one round.

The models receive remaining-round and negotiation-history information, and
prompt v1 explicitly says not to optimize only for immediate acceptance.
Despite this, they rarely exploit multi-round bargaining opportunities.

Suggested wording:

> The tested LLMs reliably participate in the structured protocol, but their
> zero-shot policies are strongly biased toward immediate settlement.

## Finding 3: High Deal Rate Does Not Imply High Profit

The LLMs achieve very high deal rates:

- DeepSeek V4 Flash: `0.9903`
- Kimi K2.6: `1.0000`
- Qwen3.6 Plus: `0.9985`

However, their average profits are much lower than the concession heuristic:

- Concession heuristic: `14,774.11`
- DeepSeek V4 Flash: `6,749.21`
- Qwen3.6 Plus: `5,979.55`
- Kimi K2.6: `4,514.43`

This is a core benchmark result: models can get deals, but often at low prices.

Suggested wording:

> The LLM baselines optimize for agreement-like behavior rather than aggressive
> seller profit, producing high deal rates but substantially lower profit than a
> simple hand-designed concession strategy.

## Finding 4: Hand-Designed Pricing Priors Remain Strong

The concession heuristic achieves the highest profit but accepts more walkaway
risk:

- Avg profit: `14,774.11`
- Deal rate: `0.7268`
- Walkaway rate: `0.2732`
- Avg rounds: `1.7123`

This does not mean the heuristic is a better general agent. It encodes an
explicit aggressive-concession pricing prior. The comparison is useful as a
reference policy showing that the environment rewards strategic price
anchoring and concession behavior.

Suggested wording:

> The concession heuristic demonstrates that the simulator can reward more
> aggressive bargaining strategies, even though the zero-shot LLMs rarely
> discover such behavior.

## Finding 5: More Detailed Prompt Explanations Did Not Improve Strategy

DeepSeek V4 Flash prompt v2 added more explicit scenario and observable-state
explanations. It did not improve performance:

- Prompt v1 avg profit: `6,749.21`
- Prompt v2 avg profit: `4,142.73`
- Prompt v1 counters: `235`
- Prompt v2 counters: `6`

Interpretation:

- Clearer field descriptions may improve interface clarity.
- They do not necessarily induce better strategic pricing.
- In this case, the model became more conservative and settled even faster.

This should be treated as a small prompt ablation, not as a general claim about
prompt engineering.

## Finding 6: Reasoning Does Not Automatically Solve the Task

DeepSeek V4 Pro reasoning was tested on a small 100-episode smoke run:

- Avg profit: `5,231.50`
- Avg rounds: `1.0000`
- Invalid rate: `0.0100`
- Tokens per episode: `2,036.81`

This small run did not show improved multi-round bargaining. It also used more
tokens and produced one empty-response invalid JSON episode.

Suggested wording:

> A small reasoning-enabled run did not automatically improve bargaining
> behavior, suggesting that hidden-preference pricing may require more than
> generic reasoning mode. Full-scale reasoning-model evaluation is left to
> future work due to cost.

## Overall Takeaway

The strongest paper-facing takeaway is:

> PrefBench exposes a gap between structured action compliance and strategic
> pricing competence. Current zero-shot LLM agents can follow the interaction
> protocol, but they under-exploit multi-round bargaining opportunities and
> produce lower seller profit than a simple heuristic with hand-designed pricing
> priors.

This supports the benchmark's value as a prototype evaluation environment. The
results should not be framed as a claim that LLMs are bad at pricing in
general; they show that this hidden-preference, multi-round pricing setting is
nontrivial for zero-shot LLM agents.
