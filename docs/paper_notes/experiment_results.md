# Experiment Results Notes

These notes summarize the completed PrefBench experiments for paper writing.
They are based on the current artifact outputs under `artifacts/`.

## Evaluation Setup

- Test split: `datasets/persona_bank/bank50k_s123/test.jsonl`
- Test episodes: 7,500 for main full-test runs.
- Seed: 123.
- LLM prompt: `prompt_v1` for main LLM results.
- LLM mode: non-thinking / cost-effective configurations for the main table.
- LLM outputs: strict JSON action schema with `offer`, `accept`, and
  `walkaway`.
- Hidden variables are not exposed to policies.

## Main Results

| Method | Episodes | Deal Rate | Avg Profit USD | Avg Rounds | Walkaway Rate | Invalid Rate | Tokens / Ep |
|---|---:|---:|---:|---:|---:|---:|---:|
| Random | 7,500 | 0.5769 | 6,572.33 | 1.3541 | 0.4231 | n/a | n/a |
| Concession Heuristic | 7,500 | 0.7268 | 14,774.11 | 1.7123 | 0.2732 | n/a | n/a |
| DeepSeek V4 Flash | 7,500 | 0.9903 | 6,749.21 | 1.0313 | 0.0097 | 0.0000 | 1,399.22 |
| Kimi K2.6 | 7,500 | 1.0000 | 4,514.43 | 1.0000 | 0.0000 | 0.0000 | 1,297.97 |
| Qwen3.6 Plus | 7,500 | 0.9985 | 5,979.55 | 1.0036 | 0.0015 | 0.0000 | 1,473.91 |

## Supporting Runs

| Run | Episodes | Deal Rate | Avg Profit USD | Avg Rounds | Invalid Rate | Tokens / Ep | Purpose |
|---|---:|---:|---:|---:|---:|---:|---|
| DeepSeek V4 Flash prompt v2 | 7,500 | 0.9993 | 4,142.73 | 1.0008 | 0.0000 | 1,724.47 | Prompt-clarity ablation |
| DeepSeek V4 Pro Reasoning | 100 | 0.9900 | 5,231.50 | 1.0000 | 0.0100 | 2,036.81 | Small reasoning ablation |

## Behavior Distributions

DeepSeek V4 Flash, prompt v1:

- Consumer responses: `accept=7427`, `counter=235`, `walkaway=52`
- Rounds: `1=7408`, `2=32`, `3=15`, `4=18`, `5=16`, `6=11`

Kimi K2.6, prompt v1:

- Consumer responses: `accept=7500`
- Rounds: `1=7500`

Qwen3.6 Plus, prompt v1:

- Consumer responses: `accept=7489`, `counter=27`, `walkaway=8`
- Rounds: `1=7489`, `2=5`, `3=1`, `5=5`

DeepSeek V4 Flash, prompt v2:

- Consumer responses: `accept=7495`, `counter=6`, `walkaway=5`
- Rounds: `1=7496`, `2=2`, `3=2`

DeepSeek V4 Pro Reasoning, prompt v1, smoke100:

- Consumer responses: `accept=99`
- Rounds: `1=100`
- One invalid JSON episode caused by an empty raw response.

## Important Interpretation Notes

- Heuristic and LLM policies share the same environment, episode stream, action
  space, and hidden-information restriction.
- The comparison is not meant to equalize policy class or input modality:
  heuristics use structured observations with hand-designed pricing priors,
  while LLMs receive the observable state through a prompt and act zero-shot.
- The concession heuristic achieves much higher profit by encoding an explicit
  aggressive-concession pricing prior, at the cost of lower deal rate and
  higher walkaway rate.
- The LLM baselines achieve very high deal rates and near-zero invalid rates,
  but they tend to settle immediately and produce lower seller profit.
- `avg_env_reward` should not be used for resumed Qwen reporting. Use
  `avg_profit_usd` as the main seller objective metric.
