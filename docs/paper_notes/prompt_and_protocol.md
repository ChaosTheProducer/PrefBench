# Prompt and Protocol Notes

These notes describe the LLM-facing protocol used by PrefBench. Use this file
when writing the benchmark or experiment sections.

## Environment-to-LLM Flow

Each LLM agent interacts with the same `NegotiationEnv` used by heuristic
policies:

1. `NegotiationEnv.reset()` or `NegotiationEnv.step()` returns an observation
   dictionary.
2. `render_llm_observation(observation, env.catalog, prompt_version=...)`
   renders the observable state into a JSON-style prompt.
3. The OpenAI-compatible chat-completions API is called with one system message
   and one user prompt.
4. The model returns a JSON object.
5. `parse_llm_action()` converts the JSON into the shared `EnvAction`.
6. `env.step(action)` executes the action in the simulator.

## Observable State

The LLM prompt exposes only observable state:

- Current round index.
- Remaining seller decision turns.
- Selected customization option keys and dimensions.
- Bundle MSRP delta.
- Estimated seller implementation cost.
- Aesthetic proxy score.
- Observable buyer profile fields:
  age band, income band, household stage, ownership stage, and primary use case.
- Last seller offer.
- Last consumer response.
- Last consumer counter-offer if available.
- Negotiation history length.

The LLM prompt does not expose hidden variables:

- Reservation price base.
- Price sensitivity.
- Feature weight vector.
- Counter strength.
- Walkaway threshold.
- True willingness-to-pay.

## Action Schema

The LLM must return exactly one JSON object:

```json
{
  "move": "offer",
  "price_offer_usd": 5200,
  "reason": "profitable offer adjusted for buyer profile and remaining rounds"
}
```

Allowed moves:

- `offer`: propose a customization-bundle price in USD.
- `accept`: accept the buyer's last counter-offer. This is valid only when a
  buyer counter-offer exists.
- `walkaway`: end the negotiation without a deal.

Invalid JSON or invalid actions terminate only the current episode and are
counted in the report. The runner does not execute fallback actions.

## Prompt Versions

The runner supports `--prompt-version v1` and `--prompt-version v2`.

### Prompt v1

Prompt v1 is the main experiment protocol. It states:

- The model is the seller in a personalized pricing benchmark.
- The objective is to maximize seller profit from the customization bundle.
- Profit is `deal_price_usd - estimated_implementation_cost_usd`.
- Hidden buyer variables are not observable.
- The action meanings and JSON schema.
- The model should not optimize only for immediate acceptance.
- When several rounds remain, it may make an ambitious but plausible opening
  offer.
- If a buyer counter-offer exists, the model should compare it with cost and
  remaining rounds.

Prompt v1 is shorter than v2 and produced stronger DeepSeek Flash results.

### Prompt v2

Prompt v2 adds more explicit explanation of the scenario and each observable
state field. It clarifies that the product is a fixed vehicle customization
bundle and that the agent prices only the customization bundle, not the full
vehicle.

In experiments, v2 made DeepSeek V4 Flash more conservative:

- Avg profit dropped from 6,749.21 to 4,142.73.
- Avg rounds dropped from 1.0313 to 1.0008.
- Consumer counters dropped from 235 to 6.

For the main paper results, use prompt v1. Prompt v2 can be described as a
prompt-clarity ablation showing that more explicit field explanations did not
automatically improve strategic bargaining.

## API Configuration

The runner uses direct HTTP requests to OpenAI-compatible `/chat/completions`
endpoints. It does not import the OpenAI Python SDK.

Provider-specific request fields are configured in
`configs/llm_api.local.json`:

- DeepSeek non-thinking uses `response_format={"type": "json_object"}` and
  `thinking={"type": "disabled"}`.
- Kimi K2.6 uses `thinking={"type": "disabled"}` and its required
  non-thinking temperature.
- Qwen3.6 Plus uses `extra_body={"enable_thinking": false}`.
- DeepSeek V4 Pro reasoning uses
  `extra_body={"reasoning_effort": "high", "thinking": {"type": "enabled"}}`.

API keys are stored only in `configs/llm_api.local.json`, which is gitignored.
The repository tracks only `configs/llm_api.example.json`.
