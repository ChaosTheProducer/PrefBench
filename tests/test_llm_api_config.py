import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.agents.run_llm_benchmark import _load_api_config


def test_load_api_config_selects_named_run(tmp_path):
    config_path = tmp_path / "llm_api.local.json"
    config_path.write_text(
        json.dumps(
            {
                "runs": [
                    {
                        "run_name": "deepseek_v4_flash",
                        "base_url": "https://api.deepseek.com",
                        "api_key": "deepseek-key",
                        "model": "deepseek-v4-flash",
                        "temperature": 0.0,
                        "max_tokens": 256,
                        "timeout_sec": 60.0,
                        "response_format": {"type": "json_object"},
                        "thinking": {"type": "disabled"},
                        "extra_body": {"enable_thinking": False},
                        "extra_body": {"enable_thinking": False},
                    },
                    {
                        "run_name": "qwen_turbo",
                        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                        "api_key": "qwen-key",
                        "model": "qwen-turbo",
                        "temperature": 0.0,
                        "max_tokens": 256,
                        "timeout_sec": 60.0,
                    },
                ]
            }
        )
    )

    config = _load_api_config(config_path, "qwen_turbo")

    assert config["run_name"] == "qwen_turbo"
    assert config["model"] == "qwen-turbo"
    assert config["api_key"] == "qwen-key"


def test_load_api_config_preserves_provider_request_options(tmp_path):
    config_path = tmp_path / "llm_api.local.json"
    config_path.write_text(
        json.dumps(
            {
                "runs": [
                    {
                        "run_name": "deepseek_v4_flash",
                        "base_url": "https://api.deepseek.com",
                        "api_key": "deepseek-key",
                        "model": "deepseek-v4-flash",
                        "temperature": 0.0,
                        "max_tokens": 512,
                        "timeout_sec": 60.0,
                        "response_format": {"type": "json_object"},
                        "thinking": {"type": "disabled"},
                        "extra_body": {"enable_thinking": False},
                    }
                ]
            }
        )
    )

    config = _load_api_config(config_path, "deepseek_v4_flash")

    assert config["response_format"] == {"type": "json_object"}
    assert config["thinking"] == {"type": "disabled"}
    assert config["extra_body"] == {"enable_thinking": False}


def test_load_api_config_rejects_unknown_run(tmp_path):
    config_path = tmp_path / "llm_api.local.json"
    config_path.write_text(
        json.dumps(
            {
                "runs": [
                    {
                        "run_name": "deepseek_v4_flash",
                        "base_url": "https://api.deepseek.com",
                        "api_key": "deepseek-key",
                        "model": "deepseek-v4-flash",
                        "temperature": 0.0,
                        "max_tokens": 256,
                        "timeout_sec": 60.0,
                    }
                ]
            }
        )
    )

    with pytest.raises(ValueError, match="not found"):
        _load_api_config(config_path, "missing_model")
