# Repository Agent Rules

This file defines the working contract for agents collaborating in this
repository. The repository includes both code work and research writing work.

## Project Identity

- Treat this repository as the benchmark-v1 baseline for customized
  car-feature pricing under partial observability.
- Preserve reproducibility and benchmark comparability by default.
- Do not treat the repository as an open-ended sandbox for arbitrary RL tuning.

## Frozen Vs. Extensible

The following should be treated as frozen unless there is a clear critical bug:

- `catalog/`
- `datasets/`
- benchmark-v1 environment contract
- retained thesis-stage benchmark configs used for reported results

Extensions are allowed when they are added as new layers without silently
changing the v1 baseline. Preferred extension directions include:

- LLM-facing observation/action wrappers
- belief and hidden-state probing
- robustness and OOD evaluation
- trace-based diagnostics
- benchmark packaging and release tooling
- paper-writing and documentation artifacts tied to the real implementation

If new benchmark assets or evaluation suites are added, put them in new
versioned locations instead of overwriting v1 assets.

## Code Rules

- Read the relevant code, config, and data contract before editing.
- Prefer minimal, sufficient changes over large refactors.
- Preserve existing interfaces unless the task explicitly requires interface
  changes.
- Reuse the existing pipeline shape when possible:
  `catalog -> persona/data -> env -> wrapper -> agent adapter -> eval/benchmark`.
- Keep experiment entry points config-driven and seed-controlled.
- Keep benchmark outputs structured and machine-readable, preferably JSON.
- Do not silently change reward semantics, observation meaning, split
  definitions, or evaluation contracts.

### Coding Style

- Prefer direct implementations over layered indirection.
- Do not add excessive defensive programming.
- Do not add unnecessary fallback paths.
- Avoid "just in case" abstractions, retries, compatibility shims, and
  branch-heavy handling unless there is a concrete need in this repository.
- Use validation at true boundaries, but do not turn normal code paths into
  overprotected frameworks.
- Fail clearly when assumptions are violated instead of masking issues with
  broad fallback behavior.

### Experiment Discipline

- New experiments should have explicit config files or clearly documented
  command-line settings.
- Expensive actions such as retraining, rebuilding large data artifacts, or
  rerunning major benchmark suites should be done intentionally, with clear
  purpose and expected outputs.
- New datasets, persona banks, probes, or OOD suites should have generation
  scripts and manifests when appropriate.

## Writing Rules

- Write only what is supported by the repository, its configs, and actual
  experiment outputs.
- Explicitly separate:
  - implemented behavior
  - validated results
  - planned extensions
- Do not claim features, diagnostics, or guarantees that are not present.
- Do not invent citations, results, or implementation details.
- Keep terminology stable across code and writing. Important terms include:
  - benchmark v1
  - benchmark v2
  - partial observability
  - belief probing
  - robustness / OOD
  - LLM-compatible interface
- Frame future papers as benchmark papers first, not as benchmark wrappers
  around one tuned agent.

## Collaboration Rules

- When analyzing or reviewing, focus first on bugs, regressions, missing
  evidence, and evaluation gaps.
- When uncertain, say what is directly supported by the repository and what is
  inference.
- If an uncertainty would change the research claim, experiment contract,
  dataset semantics, or user-facing scope, ask before deciding.
- Do not add unrequested features, files, abstractions, experiments, or writing
  sections.
- When modifying existing work, change only the part needed to satisfy the
  request.
- For nontrivial work, provide acceptance criteria in addition to the
  implementation plan.
- Keep code changes, experiment outputs, and writing claims aligned.
- If an interface or experimental contract changes, update the related config,
  script, and documentation together.
- Do not overwrite user work or unrelated local changes.

## Subagent Usage

Subagents should be used only when the task is naturally parallel or clearly
separable. They are useful for bounded research, codebase exploration, and
independent implementation slices, but they should not replace the main agent's
responsibility for final integration and consistency.

Good uses:

- one subagent surveys related benchmark papers while the main agent works on
  repository design;
- one subagent inspects a specific code path, such as persona generation or
  benchmark evaluation;
- one subagent drafts a bounded writing section while another task continues;
- separate workers implement disjoint modules, such as an LLM wrapper and a
  probe exporter, when their file ownership does not overlap.

Avoid using subagents for:

- vague brainstorming without a concrete output;
- editing the same files in parallel;
- decisions that change benchmark identity, data contracts, or paper claims
  without main-agent review;
- expensive experiments or data regeneration unless the scope and outputs are
  explicit.

Subagent outputs should be treated as inputs for review, not as automatically
accepted conclusions. The main agent should reconcile terminology, file
changes, and paper claims before considering the task complete.

## Preferred Direction

High-value work for this repository or its direct successor is:

- stronger benchmark science
- stronger diagnostics
- stronger generalization evaluation
- standardized interfaces for future LLM agents

Lower-value work by default is:

- endless PPO tuning
- endless Dreamer tuning
- reward-engineering churn without benchmark value
- cosmetic presentation changes without scientific gain

## Definition Of Done

A task is not done just because code compiles. It should also satisfy the
relevant subset of the following:

- implementation matches the intended benchmark contract
- outputs are reproducible or at least traceable
- configs and scripts are updated consistently
- documentation or writing reflects the real state of the code
- claims are proportional to available evidence
