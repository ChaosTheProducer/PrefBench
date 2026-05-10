# Limitations and Future Work Notes

These notes record the intended paper positioning for PrefBench as an
open-source prototype benchmark, not a fully validated human-preference
benchmark.

## Intended Positioning

PrefBench is a simulator-grounded prototype for evaluating zero-shot LLM agents
in personalized pricing negotiations. It provides a fixed persona bank, a
hidden-preference consumer simulator, a structured LLM interaction protocol,
and baseline evaluation scripts.

The paper should present PrefBench as an extensible research artifact. It
should not claim that the simulator fully captures real consumer behavior or
that the benchmark has been validated against human negotiation data.

## Current Scope

- The benchmark uses a single configurable-product setting: vehicle
  customization pricing.
- Consumer personas are public-data-informed but semi-synthetic.
- Hidden preference and behavior variables are partly designed rather than
  directly estimated from human negotiation traces.
- LLM agents receive only observable buyer profile fields, bundle information,
  and negotiation history.
- LLM agents act through a strict JSON action schema: `offer`, `accept`, or
  `walkaway`.
- The main experiments should focus on checkpoint-free heuristic baselines and
  zero-shot LLM baselines.
- RL baselines from the thesis-stage code should not be central to this
  artifact-oriented version.

## Limitations

- **No human validation:** The current benchmark does not include human buyer
  studies, human seller studies, or direct comparison against real negotiation
  transcripts.
- **Semi-synthetic behavior model:** The simulator is grounded in public
  observable distributions but still relies on designed hidden variables and
  hand-crafted behavior mappings.
- **Single product domain:** Current experiments use vehicle customization as
  the only product substrate, so conclusions may not transfer to other pricing
  contexts.
- **Prompt sensitivity:** LLM behavior depends on the fixed zero-shot prompt.
  The benchmark should report this prompt as part of the protocol rather than
  claiming prompt-invariant model ability.
- **API instability:** Hosted LLM outputs may change over time because provider
  models, serving stacks, and pricing can change.
- **No fine-tuning evaluation:** The current scope does not evaluate supervised
  fine-tuning, RLHF, preference optimization, or tool-augmented pricing agents.
- **Limited strategic validation:** The benchmark can produce multi-round
  negotiations, but it does not prove that the simulated bargaining dynamics
  match real strategic human behavior.
- **No fairness or policy audit:** Personalized pricing raises fairness and
  regulatory concerns, but the current artifact focuses on decision behavior
  and does not provide a full fairness, privacy, or legal compliance analysis.

## Future Work

- Position PrefBench as one task family within a broader pricing-agent
  benchmark suite.
- Validate simulator behavior against human buyer or seller responses.
- Collect small-scale human negotiation traces to calibrate acceptance,
  counter-offer, and walkaway behavior.
- Extend the benchmark to additional product or service domains.
- Add other pricing scenarios, such as subscription pricing, promotion design,
  auction-like selling, inventory-aware dynamic pricing, and bundle pricing.
- Allow agents to choose or design product bundles instead of negotiating over
  a fixed sampled bundle.
- Expand metrics beyond deal rate and seller profit to include consumer surplus,
  fairness, robustness, inventory effects, and long-term customer value.
- Add richer buyer observations, such as dialogue history, preference
  statements, or multimodal product context.
- Evaluate larger and more diverse LLM families under the same fixed prompt
  protocol.
- Study reasoning-enabled LLM variants as a controlled ablation, while keeping
  prompt and simulator settings fixed.
- Add learned agents or fine-tuned LLM policies once sufficient training data
  or simulated trajectories are available.
- Introduce fairness and robustness metrics for personalized pricing behavior.
- Provide versioned benchmark releases with frozen data, prompts, API settings,
  and result schemas.
- Separate public benchmark evaluation from private API credentials and
  provider-specific execution details.

## Suggested Paper Framing

Use restrained claims:

- "PrefBench is a prototype benchmark for studying zero-shot LLM decision
  behavior in hidden-preference personalized pricing."
- "The simulator is public-data-informed and semi-synthetic."
- "The benchmark provides a reproducible protocol and baseline scripts rather
  than a fully human-validated pricing environment."

Avoid overclaims:

- Do not claim that PrefBench measures real-world pricing performance.
- Do not claim that the hidden preference model is empirically complete.
- Do not claim that results generalize across all personalized pricing domains.
- Do not claim that LLM agents are aligned with human consumer preferences.

## Broader Benchmark Vision

PrefBench can be framed as one slice of a broader pricing-agent benchmark. A
complete benchmark suite could contain multiple pricing task families, including
personalized product customization, subscription pricing, promotion design,
bundle construction, auction-like selling, and inventory-aware dynamic pricing.

Under this broader view, the current vehicle-customization task focuses on
hidden-preference negotiation over a fixed sampled bundle. Future tasks could
allow agents to choose bundles, reason over inventory and implementation costs,
trade off seller profit with consumer-facing objectives, and evaluate robustness
across domains.

The paper can explicitly present the current artifact as a seed for this
broader direction rather than a finished benchmark suite. This makes the
contribution more honest: the project contributes a concrete task formulation,
simulator, persona bank, LLM interaction protocol, and baseline scripts, while
leaving larger-scale validation and benchmark expansion to future work.
