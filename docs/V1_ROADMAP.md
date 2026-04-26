# V1 Roadmap

## What this repository is

This repository is the thesis-stage, benchmark-v1 implementation of a simulator-based benchmark for customized car-feature pricing under partial observability.

At its core, it provides:

- a fixed Mercedes-Benz E350 customization catalog;
- a frozen persona bank with observable and hidden buyer attributes;
- a multi-round negotiation environment with a Gymnasium-style interface;
- comparable heuristic, PPO, and Dreamer agent pipelines;
- optional CLIP semantics and Dreamer-side TTA extensions;
- reproducible YAML-driven experiment configurations.

This repository should now be treated as a stable baseline rather than an actively evolving research sandbox.

## What is already complete

### Fixed benchmark assets

The benchmark-v1 package already contains the main frozen assets needed to reproduce the retained experiments:

- `catalog/e350_core_catalog.yaml`
- `datasets/persona_bank/bank50k_s123/`
- `datasets/clip_semantics/e350_clip_text_v1.json`

These assets define the benchmark input space and should be preserved as the v1 reference setup.

### Core runtime implementation

The runtime environment and agent adapters are already in place:

- `src/pricing_env/`
- `src/pricing_agent/`

The environment side includes catalog loading, persona loading and hidden-trait logic, willingness-to-pay computation, the multi-round negotiation backend, and the Gymnasium wrapper. The agent side includes heuristic baselines, PPO support, Dreamer integration, and Dreamer-side TTA logic.

### Main experiment entry points

The main runnable scripts are already organized and usable:

- `scripts/agents/`
- `scripts/common/`
- `scripts/data/`

These are the primary retained entry points for training, benchmarking, evaluation, and data preparation.

### Baseline and ablation coverage

The repository already supports the benchmark-v1 baseline family:

- `random`
- `concession`
- `PPO`
- `Dreamer`
- `Dreamer + CLIP`
- `Dreamer + TTA`

It also contains the retained reward-ablation and TTA-ablation configurations that were used to support the thesis analysis.

### Release-style packaging

The repository has already been partially cleaned toward release form:

- `README.md`
- `requirements.txt`
- `LICENSE`
- `Lei_Yingjie/` submission package preparation
- code-listing alignment work

This means the current codebase is suitable to freeze as a benchmark-v1 package.

## What should be treated as frozen

The following should be treated as the stable benchmark-v1 reference and should not be modified further in this repository unless there is a critical bug:

- the fixed benchmark assets under `catalog/` and `datasets/`;
- the benchmark-v1 environment interface;
- the retained PPO and Dreamer benchmark configurations used in the thesis;
- the thesis-stage release packaging and submission materials.

The main reason is reproducibility: the current repository now functions as a thesis-backed baseline package, and changing it further would blur the distinction between the completed benchmark-v1 system and the next research phase.

## What should not be the focus anymore

The following directions are not the highest-value use of effort in this repository anymore:

- further PPO hyperparameter tuning;
- further Dreamer hyperparameter tuning;
- trying to force Dreamer to beat the concession heuristic;
- adding more reward-engineering variants;
- extending thesis-stage TTA sweeps;
- spending more time on training-curve presentation.

These may matter for algorithm papers, but they are no longer the most important route for making the benchmark publishable as a strong benchmark paper.

## Main limitations of benchmark v1

Although benchmark-v1 is already substantial, it still has limitations if the goal is a stronger benchmark publication at venues such as AAAI or NeurIPS:

1. It does not yet include a strong diagnostic suite for hidden-state inference.
2. It does not yet include a proper robustness / OOD benchmark layer.
3. It does not yet expose a standardized LLM-facing observation layer under the same benchmark contract.
4. It still leans more toward environment-and-baseline benchmarking than toward diagnostic benchmark science.

These are not thesis-blocking weaknesses, but they are the natural frontier for the next-stage benchmark paper.

## Recommended next-phase direction

The next repository should be treated as a benchmark-v2 line rather than as a continuation of thesis-stage cleanup.

The goal should be:

- keep the benchmark core stable;
- add diagnostic and generalization tooling;
- support future LLM agents without changing the benchmark contract;
- make the benchmark more clearly benchmark-paper-oriented rather than algorithm-paper-oriented.

## High-priority roadmap for the next repository

### 1. LLM-compatible benchmark interface

Add an LLM-facing observation wrapper while preserving the same underlying Gymnasium-style action contract.

Recommended direction:

- natural-language observation rendering;
- structured action output schema;
- no free-form dialogue runtime as the main benchmark interface.

This allows future LLM agents to enter the same benchmark protocol without turning the benchmark into a prompt-engineering task.

### 2. Belief / hidden-state probing

Add a diagnostic layer that evaluates whether an agent can infer latent buyer properties from interaction history.

Recommended first probe targets:

- price sensitivity;
- patience;
- counter strength;
- a discretized WTP-related bucket rather than the raw continuous hidden value.

This is likely the single most important extension for strengthening the benchmark as a genuine POMDP benchmark.

### 3. Robustness and OOD evaluation

Add explicit robustness suites to show that agents are not merely exploiting one frozen handcrafted distribution.

Recommended directions:

- prior perturbation tests;
- edge-persona test sets;
- adversarial or rare personas;
- segment-wise robustness reporting.

This is essential for answering the natural question of whether the benchmark is overly tied to one handcrafted conditional distribution.

### 4. Trace-based diagnostics

Use the existing trace logs to derive benchmark-level interpretability signals.

Recommended directions:

- no-deal cause analysis;
- policy behavior clustering;
- offer/accept/walkaway pattern summaries.

This should be treated as a diagnostic layer, not as the main benchmark contribution.

### 5. Stronger benchmark packaging and release protocol

In the next repository, package the benchmark more explicitly as a reusable benchmark release rather than as a thesis codebase.

Recommended deliverables:

- a clearer leaderboard-style output schema;
- a benchmark release checklist;
- standardized split definitions and evaluation rules;
- a documented baseline package for heuristics, PPO, Dreamer, and future LLM agents.

## Suggested priorities

### Must-have

- LLM-compatible observation wrapper with structured actions
- belief / hidden-state probing
- robustness / OOD benchmark suite

### Should-have

- trace-based diagnostics
- clearer leaderboard-style benchmark release conventions

### Defer

- more reward engineering
- more PPO/Dreamer tuning
- larger Dreamer model sweeps
- multi-objective reward variants
- broad horizon sweeps as a main benchmark claim

## Recommended handoff principle

Use this repository as the frozen benchmark-v1 anchor.

Use the next repository for:

- benchmark-v2 diagnostic extensions;
- LLM-agent support under the same interface;
- robustness and belief-evaluation tooling;
- publication-oriented benchmark packaging.

In short:

- this repository = stable benchmark-v1 baseline;
- next repository = benchmark-v2 research line.
