# V2 Workplan

This document records the currently agreed direction for turning the existing
benchmark-v1 codebase into a release-ready benchmark prototype, with a longer
benchmark agenda that can be continued later if more resources become
available.

It should be read together with:

- [Agents.md](../Agents.md)
- [V1_ROADMAP.md](./V1_ROADMAP.md)

## Core Positioning

The future paper should not be positioned as:

- a POMDP benchmark paper
- an RL benchmark paper
- a paper about making PPO or Dreamer stronger

The future paper should be positioned as:

- a benchmark for personalized sequential negotiation under hidden buyer
  preferences
- a diagnostic benchmark for decision-making under incomplete buyer
  information
- a unified benchmark that supports heuristic, RL, and LLM agents under a
  common contract

Operationally, `POMDP` remains a valid technical formalization, but it should
not be the main selling point in the title, abstract, or contributions.

## Strategic Decisions

The following decisions are now treated as settled:

- LLM agents must be introduced into the benchmark.
- The near-term LLM work should use one or two inexpensive API-based models
  under a fixed zero-shot or simple prompted protocol.
- Human studies and LLM fine-tuning are not required for the near-term release.
- POMDP should be kept as background formalization rather than headline
  framing.
- The benchmark should be written as a benchmark paper, not as an RL
  algorithm paper.
- The benchmark-v1 core should remain the stable anchor, while new benchmark-v2
  value should come from new evaluation layers rather than rewrites of the
  frozen v1 core.

## Main Paper Claim

The near-term paper should aim to support a claim close to the following:

> We present a working benchmark prototype for personalized pricing and
> negotiation under hidden buyer preferences, together with a reproducible
> evaluation framework, baseline agents, and a clear agenda for future
> diagnostic, LLM, and robustness extensions.

The paper should not claim to be a complete top-tier benchmark unless the
future agenda items are implemented and validated. For a workshop or arXiv
release, the correct framing is `benchmark prototype` or `toward benchmarking`
rather than `comprehensive benchmark`.

## Release Scope

These items define the current low-resource publication target. The goal is a
complete, runnable, and citeable benchmark artifact, not a finished full-scale
benchmark.

### A. Stable Benchmark Artifact

Required deliverables:

- runnable environment and benchmark scripts
- fixed catalog, persona bank, and split definitions
- documented observation, action, reward, and metric contracts
- structured result files suitable for tables and later comparison
- clear quickstart instructions and reproducibility notes

This is the core value of the release. The paper should make clear that the
artifact exists and can be used by others.

### B. Baseline Result Set

The release paper should report a compact baseline set rather than attempt a
wide model comparison.

Required baselines:

- heuristic baselines already supported by the repository
- existing RL baselines, such as PPO and Dreamer, if their results are
  reproducible enough to report

Strongly recommended:

- one or two inexpensive API-based LLM baselines evaluated zero-shot or with a
  simple fixed prompt
- cost and episode count for LLM runs

The goal of the LLM result is to demonstrate that the benchmark can admit LLM
agents under a common contract. It does not need to be a broad LLM study.

### C. Minimal LLM-Compatible Interface

The release should include a minimal LLM-facing interface if feasible.

Required properties:

- natural-language observation rendering
- structured JSON-style action schema
- deterministic parsing and execution rules
- compatibility with the existing environment and benchmark scripts

This should stay simple. The release does not require prompt tuning,
fine-tuning, tool-use agents, or multi-model leaderboard experiments.

### D. Honest Limitations And Future Agenda

The paper should explicitly state the current limits:

- semi-synthetic buyer population
- hand-designed latent preference model
- limited or absent human reference evaluation
- limited LLM evaluation
- no broad OOD suite unless implemented later
- no fine-tuned LLM agents

These limitations are acceptable for a prototype release if they are framed as
future work and not hidden.

## Future Benchmark Agenda

These are the additions required to turn the prototype into a stronger
benchmark line suitable for a top-tier benchmark paper.

### 1. Expanded LLM-Compatible Agent Interface

Extend the minimal release interface into a stronger benchmark interface that
allows more LLM agents to participate under the same underlying environment
contract.

Required properties:

- natural-language observation rendering
- structured action schema
- deterministic parsing and execution rules
- compatibility with the existing environment and benchmark scripts

The goal is not to turn the benchmark into a free-form dialogue task. The goal
is to let LLM agents enter the same benchmark protocol cleanly.

### 2. Belief / Hidden-State Probing

Add a diagnostic layer that measures whether an agent can infer latent buyer
properties from interaction history.

Initial target labels should include:

- `price_sensitivity`
- `patience`
- `counter_strength`
- `wtp_bucket`

Desired properties:

- per-round evaluation, not only end-of-episode evaluation
- partial-history probing to test whether belief improves with interaction
- probe-friendly episode export format with trace plus hidden labels

This is the most important addition for making the benchmark scientifically
strong.

### 3. Robustness / OOD Benchmark Suite

Add explicit generalization tests to show that agents are not simply exploiting
one frozen handcrafted distribution.

Initial OOD directions should include:

- observable-prior shift
- hidden-mapping perturbation
- rare or edge-persona splits
- segment-wise reporting by buyer group

This is required to answer the likely reviewer question:

> Is this benchmark only measuring overfitting to one simulator distribution?

## Future Support Work

These additions are not the first blocking items, but they are highly valuable
for a top-tier benchmark paper.

### 4. Trace-Based Diagnostics

Use episode traces to produce more interpretable benchmark outputs.

Recommended outputs:

- no-deal cause taxonomy
- offer / counter / accept / walkaway summaries
- failure-mode slices
- policy behavior clustering or family-level behavior comparison

This should be presented as diagnostic analysis rather than as the core
benchmark definition.

### 5. Benchmark Stability Analysis

Show that the benchmark is itself a stable measurement platform rather than a
one-off leaderboard snapshot.

Recommended analyses:

- repeated benchmark reruns
- variance across seeds
- variance across sampled persona subsets
- interpretation guidance on what score gaps are meaningful

### 6. Packaging and Release Protocol

Package the benchmark more explicitly as a reusable benchmark release.

Required deliverables:

- standardized result schema
- leaderboard-style report format
- evaluation protocol document
- release checklist
- clear split and version naming conventions

## Support Layer

These items are not the headline innovations, but they are important support
work for making the benchmark convincing as a stronger benchmark paper.

### 7. Reference Baselines

Benchmark results need interpretable anchors, not only model-vs-model ranking.

Recommended references:

- stronger heuristic reference baselines
- human-inspired baseline when feasible
- small human reference study only if external resources become available
- one limited-information oracle-style or segment-aware reference if practical

Without this layer, benchmark scores are harder to interpret.

### 8. Benchmark Validity Evidence

The benchmark should include evidence that it is measuring the intended
capabilities rather than accidental shortcuts.

Recommended evidence:

- relationships between belief-probing quality and negotiation outcomes
- evidence that harder splits are genuinely harder
- evidence that segment differences are meaningful rather than noise
- analysis that hidden-preference tasks cannot be solved well from trivial
  priors alone

### 9. Fairness Across Agent Families

Heuristic, RL, and LLM agents should be compared under aligned contracts as
much as possible.

Required checks:

- comparable observation access
- comparable action authority
- clearly documented budget constraints
- clearly documented parsing and execution rules for LLM agents

This is necessary to defend the benchmark against fairness objections.

### 10. Cost and Efficiency Reporting

Once LLM agents are introduced, raw task performance alone is not sufficient.

Recommended reporting:

- token cost or API cost when relevant
- runtime latency
- action budget usage
- cost-normalized or budget-aware performance views when useful

### 11. Failure Taxonomy

Beyond top-line metrics, the benchmark should expose recognizable failure
modes.

Recommended slices:

- too-high pricing and buyer walkaway
- too-early acceptance and low-profit deals
- weak adaptation to counter-offers
- poor long-horizon negotiation consistency
- segment-specific failure patterns

### 12. Simulator Assumption Audit

Because this is a simulator-based benchmark, the paper should explicitly state
what is data-anchored and what remains modeling assumption.

Recommended outputs:

- assumption inventory
- discussion of external validity limits
- discussion of what conclusions should and should not be drawn from benchmark
  results

### 13. Data Realism and Semi-Synthetic Validation

The benchmark should explicitly treat its data pipeline as semi-synthetic rather
than imply that all buyer behavior is directly observed from real-world records.

Current structure:

- observable profile distributions are anchored in public statistics
- hidden preference and bargaining variables are benchmark-defined latent
  variables generated through structured conditional rules

Benchmark-v2 should strengthen this layer through:

- clearer terminology for the observable-versus-latent split
- explicit validation of whether the generated population behaves plausibly
- sensitivity analysis over latent-generation rules
- challenge splits that test agents beyond one smooth in-distribution bank
- written discussion of external-validity limits

The goal is not to eliminate modeling assumptions. The goal is to make them
auditable, defensible, and experimentally stress-tested.

### 14. Latent-Model Calibration

The hidden layer should become better informed, but it should not depend on an
unrealistic requirement that every latent variable have a directly observed
public-data distribution.

Recommended calibration sources:

- vehicle-attribute preference and WTP studies
- conjoint or discrete-choice studies on vehicle features
- consumer bargaining and negotiated-price studies
- small-scale internal or future data-collection studies when feasible

Recommended outputs:

- bounded latent-variable ranges with justification
- conditional directions with literature support where possible
- explicit identification of variables that are only weakly anchored
- plan for future refinement if small real-data calibration becomes available

### 15. Dual-Track Evaluation Design

The benchmark should not assume that all agent families use the same training
assets in the same way.

Benchmark-v2 should separate at least two evaluation tracks:

- `trainable track` for RL or other methods that use benchmark training data
- `zero-shot track` for LLM-style agents evaluated without benchmark training

This implies corresponding data roles:

- train bank for trainable methods
- core evaluation bank for standard testing
- challenge / OOD banks for robustness testing across all agent families

This separation prevents the benchmark from over-centering on train-set scale
when some important agent families are inference-only.

## Baseline Policy Coverage

Future benchmark tables should not center only on PPO and Dreamer.

The intended baseline family is:

- heuristic baselines
- RL baselines
- LLM baselines

If possible, include one stronger reference-style baseline as well, such as a
segment-aware heuristic or a limited-information oracle-style reference, to
improve score interpretability.

The benchmark should also report which baseline belongs to which evaluation
track, especially when comparing trainable and zero-shot systems.

## What Is Not A Mainline Priority

The following should not be treated as main contributions for the near-term
release:

- further PPO tuning
- further Dreamer tuning
- trying to make RL beat the concession heuristic by tuning alone
- more reward-engineering variants
- making POMDP the paper identity
- training-curve presentation work as a main effort
- collecting a large human negotiation dataset
- running a full human-subject study
- fine-tuning LLMs
- building a broad commercial-LLM leaderboard

These may still happen in support of cleaner baseline runs, but they should not
drive project direction.

## Writing Guidance

The intended paper language should emphasize:

- hidden buyer preferences
- incomplete-information negotiation
- standardized evaluation
- verifiable evaluation
- diagnostic benchmark design
- robustness and generalization
- unified support for heuristic, RL, and LLM agents

The paper should avoid centering language such as:

- POMDP benchmark
- RL benchmark
- world-model benchmark
- reward-design benchmark

These can still appear in technical sections where appropriate, but not as the
headline identity.

## Recommended Order Of Execution

The current recommended order for a workshop or arXiv release is:

1. Finalize the release paper positioning around a working benchmark prototype.
2. Clean up documentation, quickstart, configs, and result schema.
3. Reproduce and report the compact baseline set.
4. Implement the minimal LLM-compatible interface.
5. Run one or two inexpensive API-based LLM baselines on a small fixed eval set.
6. Add cost and episode-count reporting for the LLM runs.
7. Write limitations and future agenda clearly.
8. Package the repository so the benchmark can be cited and reused.

After the prototype release, the recommended order for a stronger future
benchmark line is:

1. Implement belief-probing data export and evaluator.
2. Implement robustness / OOD suites.
3. Add reference baselines and fairness checks across agent families.
4. Add trace diagnostics and benchmark stability studies.
5. Add semi-synthetic validation, latent-model calibration, and dual-track
   evaluation definitions.
6. Add validity evidence, richer cost reporting, and failure taxonomy.

## Definition Of Progress

Near-term progress should be measured by whether the release can answer the
following questions convincingly:

- Does the benchmark run from a clean checkout?
- Are the task contract, assets, and splits documented?
- Are baseline results reproducible enough to report?
- Can at least one LLM-style agent enter the benchmark through a clean
  structured interface?
- Are limitations stated directly rather than hidden behind overclaims?
- Is there a credible future agenda that explains how the benchmark can grow?

Progress toward a stronger future benchmark paper should be measured by whether
we can answer the following questions convincingly:

- Why does this benchmark matter beyond one simulator?
- What exactly is being measured besides final profit?
- Can different agent families be compared fairly?
- Can hidden-state inference be measured directly?
- Can generalization beyond the frozen in-distribution setting be measured?
- Is the semi-synthetic data pipeline sufficiently anchored and validated?
- Are trainable and zero-shot agent families handled under a fair benchmark
  design?
- Are results verifiable and reproducible?

If the first set is weak, the project is not ready even as a prototype release.
If the second set is weak, the benchmark line is not ready for a strong
benchmark submission.
