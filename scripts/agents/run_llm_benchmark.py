"""Run an OpenAI-compatible LLM policy on PrefBench."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Mapping, Sequence
from urllib import error, request

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pricing_agent.llm_interface import ParsedLLMAction, parse_llm_action, render_llm_observation
from pricing_env.negotiation_env import NegotiationEnv


DEFAULT_CATALOG_PATH = ROOT / "catalog" / "e350_core_catalog.yaml"
DEFAULT_PERSONA_CONFIG_PATH = ROOT / "configs" / "personas_v2.yaml"
DEFAULT_PERSONA_BANK_PATH = ROOT / "datasets" / "persona_bank" / "bank50k_s123" / "llm_test_500.jsonl"
DEFAULT_API_CONFIG_PATH = ROOT / "configs" / "llm_api.local.json"
DEFAULT_REPORT_OUT = ROOT / "artifacts" / "llm" / "benchmark_llm.json"


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments."""

    parser = argparse.ArgumentParser(description="Benchmark one OpenAI-compatible LLM pricing policy.")
    parser.add_argument("--catalog-path", type=Path, default=DEFAULT_CATALOG_PATH)
    parser.add_argument("--persona-config-path", type=Path, default=DEFAULT_PERSONA_CONFIG_PATH)
    parser.add_argument("--persona-bank-path", type=Path, default=DEFAULT_PERSONA_BANK_PATH)
    parser.add_argument("--persona-bank-split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--run-name", type=str, default="llm_test_500")
    parser.add_argument("--report-out", type=Path, default=DEFAULT_REPORT_OUT)
    parser.add_argument("--episodes-out", type=Path, default=None)
    parser.add_argument("--trace-out", type=Path, default=None)
    parser.add_argument("--prompts-out", type=Path, default=None)
    parser.add_argument("--flush-every", type=int, default=25)
    parser.add_argument("--prompt-version", type=str, default="v1", choices=["v1", "v2"])
    parser.add_argument("--api-config", type=Path, default=DEFAULT_API_CONFIG_PATH)
    parser.add_argument("--llm-run-name", type=str, required=True)
    return parser.parse_args()


def _resolve_repo_path(path: Path) -> Path:
    """Resolves a path relative to the repository root."""

    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def _derive_output_path(report_out: Path, suffix: str) -> Path:
    """Derives sidecar output paths from the summary report path."""

    return report_out.with_name(f"{report_out.stem}_{suffix}.jsonl")


def _truncate_for_error(raw: str, limit: int = 800) -> str:
    """Returns a bounded provider response snippet for error messages."""

    return raw[:limit]


def _load_api_config(path: Path, llm_run_name: str) -> Dict[str, Any]:
    """Loads one named LLM API configuration from the local JSON file."""

    if not path.exists():
        raise FileNotFoundError(
            f"LLM API config not found: {path}. Copy `configs/llm_api.example.json` "
            "to `configs/llm_api.local.json` and fill in your API settings."
        )
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError("LLM API config must be a JSON object.")
    runs = payload.get("runs")
    if not isinstance(runs, list) or not runs:
        raise ValueError("LLM API config must contain a non-empty `runs` list.")
    matched = [run for run in runs if isinstance(run, dict) and run.get("run_name") == llm_run_name]
    if not matched:
        available = [run.get("run_name") for run in runs if isinstance(run, dict) and run.get("run_name")]
        raise ValueError(f"LLM run `{llm_run_name}` not found. Available runs: {available}")
    config = matched[0]
    required = ("run_name", "base_url", "api_key", "model", "temperature", "max_tokens", "timeout_sec")
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"LLM API config `{llm_run_name}` missing required keys: {missing}")
    for key in ("run_name", "base_url", "api_key", "model"):
        if not isinstance(config[key], str) or not config[key].strip():
            raise ValueError(f"LLM API config `{key}` must be a non-empty string.")
    if int(config["max_tokens"]) <= 0:
        raise ValueError("LLM API config `max_tokens` must be positive.")
    if float(config["timeout_sec"]) <= 0.0:
        raise ValueError("LLM API config `timeout_sec` must be positive.")
    for key in ("response_format", "thinking", "extra_body"):
        if key in config and not isinstance(config[key], dict):
            raise ValueError(f"LLM API config `{key}` must be a JSON object.")
    return config


class OpenAICompatibleClient:
    """Minimal OpenAI-compatible chat-completions client."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        temperature: float,
        max_tokens: int,
        timeout_sec: float,
        response_format: Mapping[str, Any] | None,
        thinking: Mapping[str, Any] | None,
        extra_body: Mapping[str, Any] | None,
    ) -> None:
        self._endpoint = f"{base_url.rstrip('/')}/chat/completions"
        self._api_key = api_key
        self._model = model
        self._temperature = float(temperature)
        self._max_tokens = int(max_tokens)
        self._timeout_sec = float(timeout_sec)
        self._response_format = dict(response_format) if response_format is not None else None
        self._thinking = dict(thinking) if thinking is not None else None
        self._extra_body = dict(extra_body) if extra_body is not None else None

    def complete(self, prompt: str) -> Dict[str, Any]:
        """Calls the chat-completions API and returns raw response metadata."""

        payload = {
            "model": self._model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a pricing agent. Return exactly one JSON object matching the requested schema. "
                        "Do not include Markdown or explanatory text outside JSON."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
        }
        if self._response_format is not None:
            payload["response_format"] = self._response_format
        if self._thinking is not None:
            payload["thinking"] = self._thinking
        if self._extra_body is not None:
            payload.update(self._extra_body)
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            self._endpoint,
            data=body,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self._timeout_sec) as response:
                raw = response.read().decode("utf-8")
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"LLM API HTTP error {exc.code}: {detail}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"LLM API request failed: {exc}") from exc

        data = json.loads(raw)
        if isinstance(data, dict) and "error" in data:
            raise RuntimeError(f"LLM API returned error payload: {_truncate_for_error(raw)}")
        choices = data.get("choices", [])
        if not isinstance(choices, list) or not choices:
            raise RuntimeError(f"LLM API response missing `choices`: {_truncate_for_error(raw)}")
        message = choices[0].get("message", {})
        content = message.get("content")
        if not isinstance(content, str):
            raise RuntimeError("LLM API response missing string `message.content`.")
        return {
            "content": content.strip(),
            "usage": data.get("usage", {}),
            "response_id": data.get("id"),
            "finish_reason": choices[0].get("finish_reason"),
        }


def _new_metric() -> Dict[str, float]:
    """Creates one metric accumulator."""

    return {
        "episodes": 0.0,
        "deal_count": 0.0,
        "total_profit_usd": 0.0,
        "total_rounds": 0.0,
        "walkaway_count": 0.0,
        "total_trace_len": 0.0,
        "total_env_reward": 0.0,
        "invalid_json_count": 0.0,
        "invalid_action_count": 0.0,
        "invalid_episode_count": 0.0,
        "api_request_count": 0.0,
        "prompt_tokens": 0.0,
        "completion_tokens": 0.0,
        "total_tokens": 0.0,
    }


def _finalize_metric(acc: Mapping[str, float]) -> Dict[str, float | int]:
    """Finalizes metric accumulators."""

    episodes = int(acc["episodes"])
    denom = float(max(1, episodes))
    return {
        "episodes": episodes,
        "deal_rate": float(acc["deal_count"]) / denom,
        "avg_profit_usd": float(acc["total_profit_usd"]) / denom,
        "avg_rounds": float(acc["total_rounds"]) / denom,
        "walkaway_rate": float(acc["walkaway_count"]) / denom,
        "avg_trace_len": float(acc["total_trace_len"]) / denom,
        "avg_env_reward": float(acc["total_env_reward"]) / denom,
        "invalid_json_count": int(acc["invalid_json_count"]),
        "invalid_action_count": int(acc["invalid_action_count"]),
        "invalid_episode_count": int(acc["invalid_episode_count"]),
        "invalid_episode_rate": float(acc["invalid_episode_count"]) / denom,
        "api_request_count": int(acc["api_request_count"]),
        "prompt_tokens": int(acc["prompt_tokens"]),
        "completion_tokens": int(acc["completion_tokens"]),
        "total_tokens": int(acc["total_tokens"]),
    }


def _record_usage(acc: Dict[str, float], usage: Mapping[str, Any]) -> None:
    """Adds API token usage if the provider reports it."""

    acc["prompt_tokens"] += float(usage.get("prompt_tokens", 0) or 0)
    acc["completion_tokens"] += float(usage.get("completion_tokens", 0) or 0)
    acc["total_tokens"] += float(usage.get("total_tokens", 0) or 0)


def _invalid_episode_metrics(observation: Mapping[str, Any]) -> Dict[str, Any]:
    """Builds metrics for an invalid-output terminated episode."""

    return {
        "episode_metrics": {
            "profit_usd": 0.0,
            "deal_reached": False,
            "rounds_used": int(observation["round_idx"]),
            "walkaway": False,
            "invalid_terminated": True,
        },
        "trace_len": int(observation["round_idx"]),
    }


def _run_llm_policy(
    *,
    client: OpenAICompatibleClient,
    episode_seeds: Sequence[int],
    args: argparse.Namespace,
    start_episode_idx: int,
) -> Dict[str, Any]:
    """Runs the LLM policy on the shared episode stream."""

    env = NegotiationEnv(
        catalog_path=args.catalog_path,
        persona_config_path=args.persona_config_path,
        persona_bank_path=args.persona_bank_path,
        persona_bank_split=args.persona_bank_split,
    )
    acc = _new_metric()
    episode_results: List[Dict[str, Any]] = []
    trace_rows: List[Dict[str, Any]] = []
    prompt_rows: List[Dict[str, Any]] = []
    trace_offset = _count_jsonl_rows(args.trace_out)
    prompt_offset = _count_jsonl_rows(args.prompts_out)

    for local_episode_idx, ep_seed in enumerate(episode_seeds):
        episode_idx = int(start_episode_idx) + int(local_episode_idx)
        env.rng.seed(int(ep_seed))
        observation = env.reset()
        persona_meta = env.current_persona_metadata()
        done = False
        total_reward = 0.0
        llm_trace: List[Dict[str, Any]] = []
        final_info: Dict[str, Any] = {}

        while not done:
            prompt = render_llm_observation(observation, env.catalog, prompt_version=args.prompt_version)
            prompt_idx = prompt_offset + len(prompt_rows)
            prompt_rows.append(
                {
                    "episode_idx": int(episode_idx),
                    "episode_seed": int(ep_seed),
                    "round_idx": int(observation["round_idx"]),
                    "prompt": prompt,
                }
            )
            api_response = client.complete(prompt)
            acc["api_request_count"] += 1.0
            _record_usage(acc, api_response.get("usage", {}))
            raw_response = str(api_response["content"])
            parsed = parse_llm_action(raw_response)
            if not parsed.valid:
                acc["invalid_episode_count"] += 1.0
                if parsed.invalid_type == "invalid_json":
                    acc["invalid_json_count"] += 1.0
                else:
                    acc["invalid_action_count"] += 1.0
                final_info = _invalid_episode_metrics(observation)
                invalid_trace = {
                    "valid": False,
                    "invalid_type": parsed.invalid_type,
                    "error": parsed.error,
                    "payload": parsed.payload,
                    "reason": parsed.reason,
                }
                trace_rows.append(
                    {
                        "episode_idx": int(episode_idx),
                        "episode_seed": int(ep_seed),
                        "round_idx": int(observation["round_idx"]),
                        "prompt_idx": int(prompt_idx),
                        "raw_response": raw_response,
                        "usage": api_response.get("usage", {}),
                        "response_id": api_response.get("response_id"),
                        "finish_reason": api_response.get("finish_reason"),
                        "parsed": invalid_trace,
                        "env_trace_event": None,
                        "termination_cause": "invalid_llm_output",
                    }
                )
                llm_trace.append(
                    {
                        "round_idx": int(observation["round_idx"]),
                        "prompt_idx": int(prompt_idx),
                        "trace_idx": trace_offset + len(trace_rows) - 1,
                    }
                )
                done = True
                break

            assert parsed.action is not None
            next_observation, reward, done, step_info = env.step(parsed.action)
            trace_event = {
                "episode_idx": int(episode_idx),
                "episode_seed": int(ep_seed),
                "round_idx": int(observation["round_idx"]),
                "prompt_idx": int(prompt_idx),
                "raw_response": raw_response,
                "usage": api_response.get("usage", {}),
                "response_id": api_response.get("response_id"),
                "finish_reason": api_response.get("finish_reason"),
                "parsed": {
                    "valid": True,
                    "move": parsed.action.move,
                    "price_offer_usd": parsed.action.price_offer_usd,
                    "reason": parsed.reason,
                    "payload": parsed.payload,
                },
                "env_trace_event": step_info.get("trace_event"),
            }
            trace_rows.append(trace_event)
            llm_trace.append(
                {
                    "round_idx": trace_event["round_idx"],
                    "prompt_idx": trace_event["prompt_idx"],
                    "trace_idx": trace_offset + len(trace_rows) - 1,
                }
            )
            total_reward += float(reward)
            observation = next_observation
            final_info = dict(step_info)

        metrics = final_info.get("episode_metrics", {})
        acc["episodes"] += 1.0
        acc["deal_count"] += float(bool(metrics.get("deal_reached", False)))
        acc["total_profit_usd"] += float(metrics.get("profit_usd", 0.0))
        acc["total_rounds"] += float(metrics.get("rounds_used", 0))
        acc["walkaway_count"] += float(bool(metrics.get("walkaway", False)))
        acc["total_trace_len"] += float(final_info.get("trace_len", len(llm_trace)))
        acc["total_env_reward"] += total_reward

        episode_results.append(
            {
                "episode_idx": episode_idx,
                "episode_seed": int(ep_seed),
                "persona": persona_meta,
                "selected_option_keys": list(observation.get("selected_option_keys", [])),
                "metrics": metrics,
                "total_env_reward": float(total_reward),
                "trace_indices": [int(event["trace_idx"]) for event in llm_trace if "trace_idx" in event],
            }
        )
        if int(args.flush_every) > 0 and len(episode_results) >= int(args.flush_every):
            _append_jsonl(args.episodes_out, episode_results)
            _append_jsonl(args.trace_out, trace_rows)
            _append_jsonl(args.prompts_out, prompt_rows)
            episode_results.clear()
            trace_rows.clear()
            prompt_rows.clear()
            trace_offset = _count_jsonl_rows(args.trace_out)
            prompt_offset = _count_jsonl_rows(args.prompts_out)

    _append_jsonl(args.episodes_out, episode_results)
    _append_jsonl(args.trace_out, trace_rows)
    _append_jsonl(args.prompts_out, prompt_rows)
    return {
        "metrics": _finalize_metric(acc),
    }


def _append_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Appends records as JSONL."""

    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _count_jsonl_rows(path: Path) -> int:
    """Counts existing JSONL rows for resume-aware trace indices."""

    if not path.exists():
        return 0
    with path.open() as handle:
        return sum(1 for _ in handle)


def _load_completed_episode_count(path: Path) -> int:
    """Returns completed episode rows already present in the sidecar."""

    return _count_jsonl_rows(path)


def _summarize_completed_sidecars(episodes_path: Path, trace_path: Path) -> Dict[str, float | int]:
    """Summarizes completed episode and trace sidecars."""

    acc = _new_metric()
    with episodes_path.open() as handle:
        for line in handle:
            row = json.loads(line)
            metrics = row.get("metrics", {})
            acc["episodes"] += 1.0
            acc["deal_count"] += float(bool(metrics.get("deal_reached", False)))
            acc["total_profit_usd"] += float(metrics.get("profit_usd", 0.0))
            acc["total_rounds"] += float(metrics.get("rounds_used", 0))
            acc["walkaway_count"] += float(bool(metrics.get("walkaway", False)))
            acc["total_trace_len"] += float(len(row.get("trace_indices", [])))
            acc["total_env_reward"] += float(row.get("total_env_reward", metrics.get("profit_usd", 0.0)))
            acc["invalid_episode_count"] += float(bool(metrics.get("invalid_terminated", False)))

    with trace_path.open() as handle:
        for line in handle:
            row = json.loads(line)
            acc["api_request_count"] += 1.0
            _record_usage(acc, row.get("usage", {}))
            parsed = row.get("parsed", {})
            if not parsed.get("valid", False):
                if parsed.get("invalid_type") == "invalid_json":
                    acc["invalid_json_count"] += 1.0
                else:
                    acc["invalid_action_count"] += 1.0

    return _finalize_metric(acc)


def main() -> None:
    """Runs the LLM benchmark and writes the report."""

    args = parse_args()
    args.catalog_path = _resolve_repo_path(args.catalog_path)
    args.persona_config_path = _resolve_repo_path(args.persona_config_path)
    args.persona_bank_path = _resolve_repo_path(args.persona_bank_path)
    args.report_out = _resolve_repo_path(args.report_out)
    args.episodes_out = _resolve_repo_path(args.episodes_out) if args.episodes_out is not None else _derive_output_path(args.report_out, "episodes")
    args.trace_out = _resolve_repo_path(args.trace_out) if args.trace_out is not None else _derive_output_path(args.report_out, "trace")
    args.prompts_out = _resolve_repo_path(args.prompts_out) if args.prompts_out is not None else _derive_output_path(args.report_out, "prompts")
    args.api_config = _resolve_repo_path(args.api_config)
    api_config = _load_api_config(args.api_config, args.llm_run_name)
    if int(args.episodes) <= 0:
        raise ValueError("`episodes` must be positive.")
    if int(args.flush_every) < 0:
        raise ValueError("`flush_every` must be non-negative.")

    client = OpenAICompatibleClient(
        base_url=str(api_config["base_url"]),
        api_key=str(api_config["api_key"]),
        model=str(api_config["model"]),
        temperature=float(api_config["temperature"]),
        max_tokens=int(api_config["max_tokens"]),
        timeout_sec=float(api_config["timeout_sec"]),
        response_format=api_config.get("response_format"),
        thinking=api_config.get("thinking"),
        extra_body=api_config.get("extra_body"),
    )
    completed_episodes = _load_completed_episode_count(args.episodes_out)
    if completed_episodes > int(args.episodes):
        raise ValueError(
            f"Existing episode sidecar has {completed_episodes} rows, "
            f"which exceeds requested episodes={args.episodes}."
        )
    episode_seeds = [int(args.seed) + idx for idx in range(int(args.episodes))]
    remaining_seeds = episode_seeds[completed_episodes:]
    result = _run_llm_policy(
        client=client,
        episode_seeds=remaining_seeds,
        args=args,
        start_episode_idx=completed_episodes,
    )
    total_metrics = result["metrics"]
    if completed_episodes > 0:
        total_metrics = _summarize_completed_sidecars(args.episodes_out, args.trace_out)
    report = {
        "schema_version": "llm_benchmark_v1",
        "run_name": str(args.run_name),
        "llm_run_name": str(api_config["run_name"]),
        "api_config_path": str(args.api_config),
        "model": str(api_config["model"]),
        "base_url": str(api_config["base_url"]),
        "temperature": float(api_config["temperature"]),
        "max_tokens": int(api_config["max_tokens"]),
        "timeout_sec": float(api_config["timeout_sec"]),
        "response_format": api_config.get("response_format"),
        "thinking": api_config.get("thinking"),
        "extra_body": api_config.get("extra_body"),
        "seed": int(args.seed),
        "episodes": int(args.episodes),
        "episode_seed_range": [episode_seeds[0], episode_seeds[-1]],
        "completed_episodes_before_run": int(completed_episodes),
        "catalog_path": str(args.catalog_path),
        "persona_config_path": str(args.persona_config_path),
        "persona_bank_path": str(args.persona_bank_path),
        "persona_bank_split": str(args.persona_bank_split),
        "episodes_out": str(args.episodes_out),
        "trace_out": str(args.trace_out),
        "prompts_out": str(args.prompts_out),
        "prompt_version": str(args.prompt_version),
        "metrics": total_metrics,
    }
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report["metrics"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
