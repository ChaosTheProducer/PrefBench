# Lei Yingjie - Customized Car-Feature Pricing Benchmark

This repository contains the executable code for a simulator-based benchmark for **customized car-feature pricing under partial observability**. The system models a seller pricing agent negotiating with heterogeneous consumer personas over a fixed vehicle customization bundle, then evaluates different pricing policies under shared seeds and shared episode contracts.

The repository focuses on the runtime benchmark code, configuration files, and reproducible experiment scripts. Thesis drafts, review notes, and long-form research documents are not required to run the code release.

## Overview

The benchmark implements a controlled dynamic-pricing task with two interacting components:

- **Consumer simulator**: samples structured buyer personas with observable fields and hidden behavioural variables, then uses a utility-driven negotiation backend to respond to seller actions.
- **Pricing agents**: compare heuristic policies, PPO, and DreamerV3-style world-model agents under the same observation/action/reward/termination contract.

The current benchmark uses a Mercedes-Benz E350 customization catalog as the fixed product space. Each episode samples one fixed customization bundle, then runs a bounded multi-round price negotiation. The seller sees only observable profile/context information and negotiation history; latent willingness-to-pay and hidden behavioural traits remain inside the simulator.

Main supported features:

This repository is intended as the stable benchmark core for a future LLM-facing benchmark extension built on the same environment contract.

- fixed E350 customization catalog;
- offline persona-bank generation;
- Gymnasium-style negotiation environment wrapper;
- heuristic baselines, PPO baseline, and DreamerV3 integration;
- optional offline CLIP text-semantics features;
- YAML-driven experiment configuration and reproducible artifact output.

## Installation

### Prerequisites

- Linux
- NVIDIA GPU
- CUDA 12.x-compatible driver/runtime
- Conda or Miniconda
- Python 3.10

This repository is designed for a GPU-enabled setup. The full benchmark pipeline, including PPO, DreamerV3, CLIP semantics, and benchmark evaluation, assumes an NVIDIA GPU with CUDA 12.x.

If Conda is not installed, install Miniconda from:

```text
https://docs.conda.io/en/latest/miniconda.html
```

### Setup Environment

Create and activate the project environment:

```bash
conda create -n pricing-agent python=3.10 -y
conda activate pricing-agent
```

Install all required dependencies with a single command:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Verify GPU-backed PyTorch and JAX

Check that PyTorch can see the GPU:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
```

Check that Dreamer/JAX is installed and that JAX sees a CUDA device:

```bash
python -c "import dreamerv3, embodied, jax; print('dreamerv3 ok'); print(jax.__version__); print(jax.devices())"
```

Expected output should include a CUDA device, for example `CudaDevice(id=0)`. If either PyTorch or JAX only reports CPU execution, the repository is not installed in the intended configuration.

## Benchmark Data

The benchmark expects fixed data under `datasets/`:

- `datasets/persona_bank/bank50k_s123/`: fixed persona bank and train/validation/test splits.
- `datasets/clip_semantics/e350_clip_text_v1.json`: offline CLIP text-semantics features for catalog options.

A release package should include these files so users can run the benchmark directly. If the files are missing, or if you want to rebuild them from configuration, regenerate them with the commands below.

Regenerate the persona bank:

```bash
python scripts/data/build_persona_bank.py \
  --count 50000 \
  --seed 123 \
  --output-dir datasets/persona_bank/bank50k_s123 \
  --overwrite
```

Regenerate offline CLIP text semantics:

```bash
python scripts/data/build_clip_semantics.py \
  --catalog-path catalog/e350_core_catalog.yaml \
  --output-path datasets/clip_semantics/e350_clip_text_v1.json \
  --model-name ViT-B-32 \
  --pretrained openai \
  --seed 123 \
  --overwrite
```

## Expected Repository Structure

After setup, the runtime project structure should look like:

```text
car-customization-pricing-llm-benchmark/
├── catalog/                      # E350 customization catalog
│   └── e350_core_catalog.yaml
├── configs/                      # Persona, PPO, Dreamer, and reward configs
│   ├── dreamer/
│   ├── ppo/
│   ├── personas_v2.yaml
│   ├── persona_hidden_mapping_v1.yaml
│   └── us_buyer_distribution_v2.yaml
├── datasets/                     # Generated persona banks and CLIP semantics
│   ├── persona_bank/
│   └── clip_semantics/
├── scripts/                      # Executable entrypoints
│   ├── agents/                   # Train/eval/benchmark scripts
│   ├── common/                   # Shared config loaders
│   └── data/                     # Dataset and CLIP precompute scripts
├── src/                          # Main Python packages
│   ├── pricing_env/              # Environment, persona, WTP, NegMAS backend
│   └── pricing_agent/            # Baselines, PPO adapter, and Dreamer adapter
├── requirements.txt
├── LICENSE
├── README.md
├── docs/
│   ├── V1_ROADMAP.md
│   └── ...
```

Generated outputs such as checkpoints, reports, and logs are written under `artifacts/` at runtime.

## Running Experiments

Make sure the environment is active before running any script:

```bash
conda activate pricing-agent
```

### PPO Baseline

Train, evaluate, and benchmark the PPO agent with the following commands:

```bash
python scripts/agents/train_ppo.py \
  --config-path configs/ppo/ablations/bank50k/clip.yaml \
  --seed 123

python scripts/agents/eval_ppo.py \
  --config-path configs/ppo/ablations/bank50k/clip.yaml \
  --seed 123

python scripts/agents/run_benchmark.py \
  --config-path configs/ppo/ablations/bank50k/clip.yaml \
  --seed 123
```

PPO reports and checkpoints are written under the `output_root` configured in the YAML, typically inside `artifacts/ppo/...`.

### DreamerV3 Agent

Train, evaluate, and benchmark the DreamerV3 agent with the following commands:

```bash
python scripts/agents/train_dreamer.py \
  --config-path configs/dreamer/ablations/bank50k/clip.yaml \
  --seed 123

python scripts/agents/eval_dreamer.py \
  --config-path configs/dreamer/ablations/bank50k/clip.yaml \
  --seed 123

python scripts/agents/run_benchmark_dreamer.py \
  --config-path configs/dreamer/ablations/bank50k/clip.yaml \
  --seed 123
```

Dreamer reports, logs, and checkpoints are written under the configured `output_root`, typically inside `artifacts/dreamer/...`.

## Acknowledgements

This codebase builds on or interfaces with the following open-source projects and research tools:

- [NegMAS](https://github.com/yasserfarouk/negmas): negotiation mechanism support.
- [Gymnasium](https://github.com/Farama-Foundation/Gymnasium): environment interface conventions.
- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) and [SB3-Contrib](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib): PPO and recurrent PPO baselines.
- [DreamerV3](https://github.com/danijar/dreamerv3): world-model agent implementation.
- [JAX](https://github.com/google/jax): GPU-accelerated numerical backend for Dreamer experiments.
- [PyTorch](https://pytorch.org/) and [OpenCLIP](https://github.com/mlfoundations/open_clip): offline CLIP text-semantics precomputation.

The benchmark customization catalog is derived from publicly available Mercedes-Benz E350 configuration information. Mercedes-Benz and related model names are used only as an anchor for research benchmark construction and remain the property of their respective owners.

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.
