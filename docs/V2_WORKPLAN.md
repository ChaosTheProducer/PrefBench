# Current Workplan

This document records the current release plan for PrefBench after narrowing
the repository scope. The previous thesis-stage RL stack is no longer retained
in this repository. The active goal is a clean arXiv artifact centered on a
personalized-pricing simulator, checkpoint-free heuristic baselines, and a
minimal zero-shot LLM agent interface.

It should be read together with:

- [Agents.md](../Agents.md)

## Core Positioning

The arXiv report should not be positioned as:

- a POMDP benchmark paper
- an RL benchmark paper
- a paper about improving one learning algorithm
- a complete benchmark suite

The arXiv report should be positioned as:

- a benchmark prototype for personalized sequential negotiation under hidden
  buyer preferences
- a simulator-based pricing-agent evaluation artifact
- a checkpoint-free baseline package for heuristics and zero-shot LLM agents

Operationally, partial observability remains part of the simulator contract,
but it should not be the headline contribution.

## Settled Decisions

- The repository keeps the simulator, catalog, persona bank, persona-generation
  scripts, and heuristic baselines.
- PPO, Dreamer, TTA, Gymnasium wrappers, and CLIP-semantics code are removed
  from this repository.
- Main reported results should use heuristic baselines and zero-shot LLM
  baselines.
- Human studies and LLM fine-tuning are not required for the near-term arXiv
  release.
- The next code layer should add an LLM-facing interface without silently
  changing the simulator contract.

## Main Paper Claim

The report should support a claim close to:

> We present a working benchmark prototype for personalized pricing and
> negotiation under hidden buyer preferences, together with a reproducible
> simulator, checkpoint-free heuristic baselines, a zero-shot LLM evaluation
> path, and a clear agenda for future diagnostic and robustness extensions.

The paper should not claim to be a complete benchmark. The correct framing is
`benchmark prototype` or `toward benchmarking`, with limitations stated
directly.

## Release Scope

Required deliverables:

- runnable simulator and heuristic benchmark script
- fixed catalog, persona bank, and split definitions
- documented observation, action, reward, and metric contracts
- structured JSON result files suitable for paper tables and future comparison
- minimal LLM-compatible observation/action interface
- zero-shot LLM benchmark runner
- invalid-action or invalid-JSON reporting for LLM runs
- cost/token reporting when API models are used

Out of scope for the near-term release:

- fine-tuning LLMs
- broad commercial-LLM leaderboard experiments
- large human-subject studies
- broad OOD benchmark suites
- new training-algorithm development

## Baseline Coverage

The intended main arXiv table is:

- Random
- Concession heuristic
- zero-shot LLM baselines under a fixed prompt and JSON action schema

If time allows, add one stronger reference heuristic, such as a segment-aware
policy or limited-information oracle-style reference. This is useful only if it
improves interpretation without expanding the project into algorithm work.

## LLM-Compatible Interface

The minimal LLM layer should provide:

- natural-language observation rendering from the existing environment state
- a fixed JSON action schema
- deterministic parsing and execution rules
- explicit invalid-output handling
- per-episode trace export
- compatibility with the same fixed persona/test split used by heuristics

The benchmark should not become a free-form dialogue task. The LLM should choose
pricing actions under the same simulator contract as the heuristic policies.

## Validity And Limitations

The paper should explicitly state:

- buyer population is semi-synthetic
- observable variables are configured from plausible population assumptions
- hidden preference and bargaining variables are benchmark-defined latent
  variables generated through structured conditional rules
- no human reference study is included in the near-term release
- benchmark scores should not be interpreted as deployable real-world pricing
  performance

The value of the artifact is controlled, reproducible evaluation under hidden
buyer preferences, not real-market calibration.

## Future Agenda

The most valuable future extensions are:

- belief / hidden-state probing for price sensitivity, patience, bargaining
  strength, and willingness-to-pay buckets
- robustness and OOD persona splits
- trace-based diagnostics and failure taxonomy
- benchmark stability analysis across seeds and persona subsets
- semi-synthetic validation and latent-model calibration
- optional human reference comparison if resources become available

## Recommended Order Of Execution

1. Keep the simulator and heuristic benchmark runnable after cleanup.
2. Update documentation and paper claims to match the narrowed repository.
3. Implement the minimal LLM-compatible interface.
4. Run heuristic baselines on the selected evaluation size.
5. Run several inexpensive or free zero-shot LLM baselines on the same fixed
   evaluation stream.
6. Report metrics, invalid-output rate, and cost/token metadata.
7. Write limitations and future agenda clearly.
8. Package the repository so the benchmark can be cited and reused.

## Definition Of Progress

Near-term progress should be measured by whether the release can answer:

- Does the benchmark run from a clean checkout?
- Are the task contract, assets, and splits documented?
- Are heuristic and LLM baseline results reproducible enough to report?
- Can at least one LLM-style agent enter through a clean structured interface?
- Are invalid LLM actions measured rather than hidden?
- Are limitations stated directly rather than hidden behind overclaims?

If these are weak, the project is not ready even as a prototype release.
