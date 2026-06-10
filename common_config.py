from __future__ import annotations

import csv
import hashlib
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-not-found]

DEFAULT_DATASET_PATH = "dataset/FACTS-Parametric-public.csv"


def ensure_parent_dir(path: str) -> None:
    Path(path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)


def load_json(path: Path | str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path: Path | str, data: Any, *, indent: int = 2) -> None:
    p = Path(path)
    ensure_parent_dir(str(p))
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def save_text(path: Path | str, text: str) -> None:
    p = Path(path)
    ensure_parent_dir(str(p))
    p.write_text(text, encoding="utf-8")


def write_json_output(output_path: Path | str, payload: dict[str, Any]) -> None:
    save_json(output_path, order_result_payload(payload))


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


def strip_sensitive_config(config: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in config.items() if key != "api_key"}


def sanitize_path_component(value: str) -> str:
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", str(value))
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
    return build_hash_key(row.get("query"), row.get("answer", row.get("gold_answer")))


def build_hash_key_from_result_row(row: dict[str, Any]) -> str | None:
    existing_hash_key = normalize_text(row.get("hash_key"))
    if existing_hash_key:
        return existing_hash_key
    query = row.get("query")
    gold_answer = row.get("gold_answer", row.get("answer"))
    if not has_non_empty_text(query) and not has_non_empty_text(gold_answer):
        return None
    return build_hash_key(query, gold_answer)


def has_required_generation_fields(result: dict[str, Any]) -> bool:
    required_fields = ["query", "llm_answer", "gold_answer"]
    for key in required_fields:
        if key not in result or result[key] is None or str(result[key]).strip() == "":
            return False
    return True


def filter_results_with_non_empty_answers(
    results: list[Any],
    context: str,
) -> list[Any]:
    filtered: list[Any] = []
    for idx, result in enumerate(results):
        if not isinstance(result, dict):
            filtered.append(result)
            continue
        if has_non_empty_text(result.get("llm_answer")):
            filtered.append(result)
            continue
        hash_key = result.get("hash_key") or build_hash_key_from_result_row(result)
        print(f"[{context}] Skip empty llm_answer at index {idx}, hash_key={hash_key}")
    return filtered


def order_result_payload(data: Any) -> Any:
    if not isinstance(data, dict) or "results" not in data:
        return data
    preferred_keys = ["model_config", "judge_config", "calculate_mean_score", "results"]
    ordered = {key: data[key] for key in preferred_keys if key in data}
    for key, value in data.items():
        if key not in ordered:
            ordered[key] = value
    return ordered


def build_default_result_path(
    model_id: str,
    dataset_path: str,
    save_to: str | None = None,
) -> Path:
    if save_to:
        return Path(save_to)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_slug = model_id.replace("/", "_")
    test_set_slug = Path(dataset_path).stem
    return Path(f"result/{model_slug}/{test_set_slug}/{timestamp}.json")


def build_generation_output_payload(
    model_config: dict[str, Any],
    results: list[dict[str, Any]],
    base_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = dict(base_payload) if base_payload else {}
    if "calculate_mean_score" not in payload:
        payload["calculate_mean_score"] = None
    payload["model_config"] = strip_sensitive_config(model_config)
    payload["results"] = results
    return order_result_payload(payload)


def load_facts_dataset(path: str, num_tasks: int | None) -> list[dict[str, Any]]:
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
            }
        )
    return normalized


def load_generation_output(path: str) -> dict[str, Any]:
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


def read_existing_generated_hash_keys(path: str) -> set[str]:
    payload = load_generation_output(path)
    hash_keys: set[str] = set()
    for item in payload.get("results", []):
        if not (isinstance(item, dict) and has_non_empty_text(item.get("llm_answer"))):
            continue
        hash_key = build_hash_key_from_result_row(item)
        if hash_key:
            hash_keys.add(hash_key)
    return hash_keys


def upsert_generation_results(
    existing: list[dict[str, Any]],
    updates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
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


def custom_id_for_key(key: int) -> str:
    return str(int(key))


def extract_text_from_file_content(content: Any) -> str:
    if hasattr(content, "text"):
        return str(content.text)
    if hasattr(content, "read"):
        raw = content.read()
        if isinstance(raw, bytes):
            return raw.decode("utf-8")
        return str(raw)
    return str(content)


def _extract_message_content_from_body(body: Any) -> str:
    if not isinstance(body, dict):
        return ""
    choices = body.get("choices")
    if isinstance(choices, list) and choices:
        choice = choices[0]
        if isinstance(choice, dict):
            message = choice.get("message") or {}
            if isinstance(message, dict):
                content = message.get("content")
                if content is not None:
                    return str(content)
            text = choice.get("text")
            if text is not None:
                return str(text)
    for field in ("output", "content"):
        value = body.get(field)
        if value is not None:
            return str(value)
    return ""


def _extract_response_text_from_batch_object(obj: dict[str, Any]) -> str:
    if obj.get("error"):
        return ""

    response = obj.get("response")
    if isinstance(response, dict):
        if response.get("error"):
            return ""
        status = response.get("status_code")
        if status is not None and int(status) >= 400:
            return ""
        body = response.get("body")
        if body is not None:
            if isinstance(body, str):
                try:
                    body = json.loads(body)
                except json.JSONDecodeError:
                    return body
            text = _extract_message_content_from_body(body)
            if text:
                return text
        text = _extract_message_content_from_body(response)
        if text:
            return text

    body = obj.get("body")
    if isinstance(body, dict):
        text = _extract_message_content_from_body(body)
        if text:
            return text
    return ""


def parse_batch_output(
    output_text: str,
    key_payloads: dict[str, dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    key_results: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    custom_id_to_sk: dict[str, str] = {}
    for sk, payload in key_payloads.items():
        key = payload.get("key")
        if key is not None:
            custom_id_to_sk[custom_id_for_key(int(key))] = sk
        custom_id_to_sk[str(sk)] = sk

    for line in output_text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(str(exc))
            continue
        if not isinstance(obj, dict):
            continue

        custom_id = str(obj.get("custom_id", ""))
        sk = custom_id_to_sk.get(custom_id)
        if sk is None and custom_id in key_payloads:
            sk = custom_id
        if sk is None:
            errors.append(f"unknown custom_id: {custom_id!r}")
            continue

        key_results[sk] = {"response": _extract_response_text_from_batch_object(obj)}

    return key_results, errors
