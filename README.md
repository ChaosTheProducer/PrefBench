# PrefBench

PrefBench is a simulator-based benchmark prototype for personalized pricing agents under hidden buyer preferences. The benchmark models a seller negotiating over a fixed configurable product bundle with heterogeneous consumer personas, then evaluates policies under shared seeds and shared episode contracts.

The current repository is scoped to a clean arXiv artifact centered on checkpoint-free heuristic baselines and a zero-shot LLM agent interface. The previous thesis-stage PPO, Dreamer, TTA, Gymnasium, and CLIP-semantics code has been removed from this repository to keep the release focused.

## Overview

PrefBench contains two main components:

- **Consumer simulator**: samples structured buyer personas with observable fields and hidden behavioral variables, then uses a utility-driven negotiation backend to respond to seller actions.
- **Pricing agents**: currently includes random and concession-style heuristic policies plus a zero-shot LLM policy wrapper with a natural-language observation and structured JSON action schema.

The current benchmark uses a Mercedes-Benz E350 customization catalog as a fixed product substrate. Each episode samples one customization bundle, then runs a bounded multi-round price negotiation. The seller sees only observable profile/context information and negotiation history; latent willingness-to-pay and hidden behavioral traits remain inside the simulator.

## Installation

Prerequisites:

- Linux
- Python 3.10
- Conda or Miniconda

Create and activate the environment:

```bash
conda create -n pricing-agent python=3.10 -y
conda activate pricing-agent
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Data

The benchmark expects fixed assets under `catalog/`, `configs/`, and `datasets/`:

- `catalog/e350_core_catalog.yaml`: fixed customization catalog.
- `configs/personas_v2.yaml`: persona sampling entry config.
- `configs/us_buyer_distribution_v2.yaml`: observable buyer-profile distribution.
- `configs/persona_hidden_mapping_v1.yaml`: hidden preference and behavior mapping.
- `datasets/persona_bank/bank50k_s123/`: generated persona bank with train/validation/test splits.
- `datasets/persona_bank/bank50k_s123/llm_test_500.jsonl`: fixed cost-controlled test subset for main zero-shot LLM evaluation.

Regenerate the persona bank if needed:

```bash
python scripts/data/build_persona_bank.py \
  --count 50000 \
  --seed 123 \
  --output-dir datasets/persona_bank/bank50k_s123 \
  --overwrite
```

Regenerate the fixed 500-episode LLM evaluation subset:

```bash
python scripts/data/build_eval_subset.py \
  --source-path datasets/persona_bank/bank50k_s123/test.jsonl \
  --output-path datasets/persona_bank/bank50k_s123/llm_test_500.jsonl \
  --metadata-path datasets/persona_bank/bank50k_s123/llm_test_500_metadata.json \
  --count 500 \
  --seed 20260501 \
  --subset-name llm_test_500 \
  --overwrite
```

Summarize the frozen persona bank for simulator sanity checks:

```bash
python scripts/data/summarize_simulator.py \
  --persona-bank-path datasets/persona_bank/bank50k_s123/persona_bank.jsonl \
  --output artifacts/simulator/sanity_summary.json
```

Simulator assumptions are documented in `docs/SIMULATOR_CARD.md`.

## Run Heuristic Benchmark

Run the checkpoint-free heuristic benchmark on the fixed test split:

```bash
python scripts/agents/run_benchmark.py \
  --episodes 100 \
  --seed 123 \
  --policies random,concession \
  --report-out artifacts/heuristic/benchmark_test.json
```

The report is written as JSON and includes profit, deal rate, walkaway rate, average rounds, average trace length, seed metadata, and shared-episode verification.

## Run LLM Benchmark

The LLM runner uses an OpenAI-compatible `/chat/completions` API. Copy the
example config, paste your provider settings into the local JSON file, then
select one named LLM run on the fixed 500-episode subset:

```bash
cp configs/llm_api.example.json configs/llm_api.local.json
# Edit configs/llm_api.local.json. Each entry under "runs" defines one LLM.
# For DeepSeek V4, keep response_format={"type": "json_object"} and
# thinking={"type": "disabled"} for strict action JSON.

python scripts/agents/run_llm_benchmark.py \
  --api-config configs/llm_api.local.json \
  --llm-run-name deepseek_v4_flash \
  --prompt-version v1 \
  --episodes 500 \
  --persona-bank-path datasets/persona_bank/bank50k_s123/llm_test_500.jsonl \
  --report-out artifacts/llm/deepseek_v4_flash_llm_test_500.json
```

`configs/llm_api.local.json` is gitignored so real API keys are not committed.
Only `configs/llm_api.example.json` is tracked as the shareable template.

The LLM must return a single JSON object with `move`, `price_offer_usd`, and
optional `reason`. Invalid JSON or invalid actions terminate that episode and
are counted in the report; the runner does not execute a fallback action.

The summary report stores aggregate metrics only. Per-episode records, per-step
traces, and prompts are written to sidecar JSONL files derived from
`--report-out`, for example `*_episodes.jsonl`, `*_trace.jsonl`, and
`*_prompts.jsonl`.

Long LLM runs flush sidecar JSONL files periodically. If a provider error stops
a run, rerun the same command with the same output paths; the runner resumes
from the completed episode rows already present in `*_episodes.jsonl`.

The main experiments use `--prompt-version v1`. `v2` adds more explicit
observable-state descriptions and can be used as a prompt-clarity ablation.

## Repository Structure

```text
PrefBench/
├── catalog/                      # Fixed product catalog
├── configs/                      # Persona and simulator configuration
├── datasets/                     # Persona bank and split files
├── scripts/
│   ├── agents/                   # Benchmark entry points
│   └── data/                     # Persona-bank generation
├── src/
│   ├── pricing_env/              # Simulator, WTP, persona, NegMAS backend
│   └── pricing_agent/            # Checkpoint-free heuristic policies
├── arxiv_paper/                  # arXiv report draft
├── docs/                         # Planning and roadmap notes
├── requirements.txt
└── Agents.md
```

Generated outputs such as reports and logs are written under `artifacts/`.

## Current Scope

Implemented:

- fixed customization catalog;
- semi-synthetic persona bank and split generation;
- multi-round negotiation simulator;
- random and concession heuristic baselines;
- structured JSON benchmark report.

Planned next:

- LLM-facing observation renderer;
- structured JSON action parser;
- zero-shot LLM benchmark runner;
- invalid-action and cost/token reporting for LLM runs.

## Acknowledgements

This codebase builds on or interfaces with:

- [NegMAS](https://github.com/yasserfarouk/negmas): negotiation mechanism support.
- [PyYAML](https://pyyaml.org/) and standard Python tooling for reproducible simulation and data generation.

The benchmark customization catalog is derived from publicly available Mercedes-Benz E350 configuration information. Mercedes-Benz and related model names are used only as an anchor for research benchmark construction and remain the property of their respective owners.

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.
