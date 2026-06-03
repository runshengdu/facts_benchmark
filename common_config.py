from __future__ import annotations

import csv
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-not-found]


def ensure_parent_dir(path: str) -> None:
    Path(path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)


def load_yaml_config(path: str, model_name: str) -> dict[str, Any] | None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        models = data.get("models", []) if isinstance(data, dict) else []
        for model in models:
            if model.get("name") == model_name:
                config = model.copy()
                if "api_key" in config:
                    config["api_key"] = os.path.expandvars(str(config["api_key"]))
                    if (
                        isinstance(config["api_key"], str)
                        and config["api_key"].startswith("${")
                        and config["api_key"].endswith("}")
                    ):
                        var_name = config["api_key"][2:-1]
                        config["api_key"] = os.environ.get(var_name, "")
                return config
    except Exception as e:
        print(f"Error loading config from {path}: {e}")
    return None


CHAT_COMPLETION_TOKEN_LIMIT_KEYS = ("max_completion_tokens", "max_tokens")


def token_limit_kwargs_from_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return token limit kwargs using the key(s) declared in YAML (no renaming)."""
    return {
        key: config[key]
        for key in CHAT_COMPLETION_TOKEN_LIMIT_KEYS
        if key in config and config[key] is not None
    }


def get_client_params(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    params = {
        "api_key": config.get("api_key"),
        "base_url": config.get("base_url"),
    }
    exclude_keys = {"name", "api_key", "base_url", *CHAT_COMPLETION_TOKEN_LIMIT_KEYS}
    chat_kwargs = {k: v for k, v in config.items() if k not in exclude_keys}
    chat_kwargs.update(token_limit_kwargs_from_config(config))
    return params, chat_kwargs


def make_chat_completion_create_kwargs(
    model_cfg: dict[str, Any],
    messages: list[dict[str, Any]],
) -> dict[str, Any]:
    _, chat_kwargs = get_client_params(model_cfg)
    return {
        "model": model_cfg["name"],
        "messages": messages,
        **chat_kwargs,
    }


def sanitize_path_component(value: str) -> str:
    cleaned = re.sub(r"[<>:\"/\\|?*\x00-\x1f]", "_", str(value))
    cleaned = cleaned.strip().strip(".")
    return cleaned or "unknown"


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def has_non_empty_text(value: Any) -> bool:
    return normalize_text(value) != ""


def build_hash_key(query: Any, gold_answer: Any) -> str:
    material = f"{normalize_text(query)}\n{normalize_text(gold_answer)}"
    return hashlib.sha1(material.encode("utf-8")).hexdigest()


def build_hash_key_from_dataset_row(row: dict[str, Any]) -> str:
    return build_hash_key(row.get("query"), row.get("answer"))


def build_hash_key_from_result_row(row: dict[str, Any]) -> str | None:
    existing_hash_key = normalize_text(row.get("hash_key"))
    if existing_hash_key:
        return existing_hash_key
    query = row.get("query")
    gold_answer = row.get("gold_answer", row.get("answer"))
    if not has_non_empty_text(query) and not has_non_empty_text(gold_answer):
        return None
    return build_hash_key(query, gold_answer)


def order_result_payload(data: Any) -> Any:
    if not isinstance(data, dict) or "results" not in data:
        return data
    preferred_keys = ["model_config", "judge_config", "calculate_mean_score", "results"]
    ordered = {key: data[key] for key in preferred_keys if key in data}
    for key, value in data.items():
        if key not in ordered:
            ordered[key] = value
    return ordered


def load_simpleqa_dataset(path: str, num_tasks: int | None) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if num_tasks:
        rows = rows[:num_tasks]
    normalized: list[dict[str, Any]] = []
    for row in rows:
        query = str(row.get("query", ""))
        gold_answer = str(row.get("answer", ""))
        hash_key = build_hash_key_from_dataset_row({"query": query, "answer": gold_answer})
        normalized.append(
            {
                "hash_key": hash_key,
                "query": query,
                "gold_answer": gold_answer,
                "topic": row.get("topic"),
                "token": row.get("token"),
                "urls": row.get("urls"),
            }
        )
    return normalized


def load_simpleqa_output(path: str) -> dict[str, Any]:
    p = Path(path)
    if not p.is_file():
        return {"results": []}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {"results": []}
    if isinstance(data, dict) and isinstance(data.get("results"), list):
        return data
    return {"results": []}


def read_existing_generated_hash_keys_simpleqa(path: str) -> set[str]:
    payload = load_simpleqa_output(path)
    hash_keys: set[str] = set()
    for item in payload.get("results", []):
        if not (isinstance(item, dict) and has_non_empty_text(item.get("llm_answer"))):
            continue
        hash_key = build_hash_key_from_result_row(item)
        if hash_key:
            hash_keys.add(hash_key)
    return hash_keys


def upsert_simpleqa_results(existing: list[dict[str, Any]], updates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    filtered_existing: list[dict[str, Any]] = []
    for item in existing:
        if not isinstance(item, dict) or has_non_empty_text(item.get("llm_answer")):
            filtered_existing.append(item)
            continue
        hash_key = item.get("hash_key") or build_hash_key_from_result_row(item)
        print(f"skip existing empty llm_answer before writing JSON: hash_key={hash_key}")
    existing = filtered_existing

    index_by_hash: dict[str, int] = {}
    for idx, item in enumerate(existing):
        if not isinstance(item, dict):
            continue
        hash_key = build_hash_key_from_result_row(item)
        if hash_key:
            index_by_hash[hash_key] = idx

    for item in updates:
        if not has_non_empty_text(item.get("llm_answer")):
            hash_key = item.get("hash_key") or build_hash_key_from_result_row(item)
            print(f"skip empty llm_answer before writing JSON: hash_key={hash_key}")
            continue
        hash_key = str(item.get("hash_key", "")).strip()
        if hash_key in index_by_hash:
            existing[index_by_hash[hash_key]] = item
        else:
            if hash_key:
                index_by_hash[hash_key] = len(existing)
            existing.append(item)
    return existing
