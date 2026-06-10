import argparse
import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI
from tqdm.asyncio import tqdm

from common_config import (
    DEFAULT_DATASET_PATH,
    build_default_result_path,
    build_hash_key_from_dataset_row,
    build_hash_key_from_result_row,
    filter_results_with_non_empty_answers,
    get_client_params,
    has_non_empty_text,
    has_required_generation_fields,
    load_facts_dataset,
    load_yaml_config,
    strip_sensitive_config,
    write_json_output,
)
from prompts import GRADER_TEMPLATE, QUERY_TEMPLATE

EVALUATORS_CONFIG_PATH = "evaluators.yaml"
MODELS_CONFIG_PATH = "models.yaml"


def extract_classification(judgment: str) -> str:
    """Extract the classification from the judgment text."""
    judgment = judgment.strip()

    if "MISTAKE" in judgment:
        return "MISTAKE"
    if "CORRECT" in judgment:
        return "CORRECT"
    if "NOT_ATTEMPTED" in judgment:
        return "NOT_ATTEMPTED"
    return "UNKNOWN"


def calculate_score(judgments: list) -> float:
    """Calculate a final score based on judgments.

    All judgments must be 'CORRECT' to get a score of 1.0, otherwise 0.0.
    """
    if all(j == "CORRECT" for j in judgments):
        return 1.0
    return 0.0


def calculate_mean_score(run_results):
    """Calculate the mean score from run results."""
    scores = []
    for result in run_results:
        if "final_score" in result and result["final_score"] is not None:
            scores.append(result["final_score"])
        elif "dictResult" in result:
            score = result["dictResult"].get("score", 0.0)
            if score is not None:
                scores.append(score)

    if len(scores) > 0:
        return sum(scores) / len(scores)
    return 0.0


async def call_api_with_retry(
    client: AsyncOpenAI,
    messages: List[Dict],
    model: str,
    request_label: str = "",
    **kwargs
) -> str:
    """Calls OpenAI API with retries using streaming."""
    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            kwargs["stream"] = True
            stream = await client.chat.completions.create(
                model=model,
                messages=messages,
                **kwargs
            )

            collected_content = []
            async for chunk in stream:
                content = chunk.choices[0].delta.content
                if content:
                    collected_content.append(content)

            return "".join(collected_content)

        except Exception as e:
            if attempt == max_attempts - 1:
                label = f"{request_label} " if request_label else ""
                raise RuntimeError(
                    f"{label}API call failed after {max_attempts} attempts: {e}"
                ) from e
            await asyncio.sleep(1 * (attempt + 1))
    return ""


async def generate_task(
    evaluator_sem: asyncio.Semaphore,
    item: Dict[str, Any],
    evaluator_client: AsyncOpenAI,
    evaluator_model: str,
    evaluator_kwargs: Dict
) -> Optional[Dict[str, Any]]:
    """Generates model output for a single task."""
    try:
        query = item["query"]
        item_hash_key = item.get("hash_key") or build_hash_key_from_dataset_row(item)
        gold_answer = item["gold_answer"]

        formatted_prompt = QUERY_TEMPLATE.format(question=query)
        try:
            async with evaluator_sem:
                prediction = await call_api_with_retry(
                    evaluator_client,
                    [{"role": "user", "content": formatted_prompt}],
                    model=evaluator_model,
                    request_label=f"[generation] hash_key={item_hash_key}",
                    **evaluator_kwargs
                )
        except Exception as e:
            print(f"[generation] Skip hash_key={item_hash_key}, query={query!r}, error={e}")
            return None

        if not has_non_empty_text(prediction):
            print(f"[generation] Skip hash_key={item_hash_key}, query={query!r}, error=empty API response")
            return None

        return {
            "hash_key": item_hash_key,
            "query": query,
            "llm_answer": prediction,
            "gold_answer": gold_answer,
            "final_score": None
        }

    except Exception as e:
        print(f"Task {item.get('hash_key')} failed: {e}")
        return None


async def evaluate_existing_result(
    judge_sem: asyncio.Semaphore,
    result_idx: int,
    item: Dict[str, Any],
    judge_client: AsyncOpenAI,
    judge_model: str,
    judge_kwargs: Dict
) -> Optional[tuple]:
    """Evaluates an already generated result entry and returns (index, final_score)."""
    try:
        if item.get("final_score") is not None:
            return None

        query = item["query"]
        gold_answer = item["gold_answer"]
        prediction = item["llm_answer"]
        item_hash_key = item.get("hash_key") or build_hash_key_from_result_row(item)

        async def run_judge(seed: int) -> str:
            grader_prompt = GRADER_TEMPLATE.format(
                question=query,
                gold_answer=gold_answer,
                prediction=prediction
            )
            async with judge_sem:
                return await call_api_with_retry(
                    judge_client,
                    [{"role": "user", "content": grader_prompt}],
                    model=judge_model,
                    request_label=f"[evaluation] hash_key={item_hash_key} seed={seed}",
                    seed=seed,
                    **judge_kwargs
                )

        raw_judgments = await asyncio.gather(*[run_judge(i) for i in range(3)])
        if not all(has_non_empty_text(j_text) for j_text in raw_judgments):
            print(f"[evaluation] Skip hash_key={item_hash_key}, error=empty judge API response")
            return None
        judgments = [extract_classification(j_text) for j_text in raw_judgments]
        final_score = calculate_score(judgments)
        return result_idx, final_score
    except Exception as e:
        print(f"[evaluation] Skip hash_key={item.get('hash_key')}, error={e}")
        return None


async def run_generation_mode(args) -> None:
    model_to_evaluate = args.model_id
    max_concurrent_evaluator_tasks = args.gen_workers

    if not model_to_evaluate:
        print("Generation mode requires --model-id.")
        return

    output_path = build_default_result_path(
        model_id=model_to_evaluate,
        dataset_path=DEFAULT_DATASET_PATH,
        save_to=args.save_to,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model_config = load_yaml_config(MODELS_CONFIG_PATH, model_to_evaluate)
    if not model_config:
        print(f"Could not load config for model: {model_to_evaluate}")
        return
    model_config_safe = strip_sensitive_config(model_config)

    eval_params, eval_kwargs = get_client_params(model_config)
    evaluator_client = AsyncOpenAI(**eval_params)

    try:
        dataset = load_facts_dataset(DEFAULT_DATASET_PATH, args.num_tasks)
    except FileNotFoundError:
        print(f"Dataset not found at {DEFAULT_DATASET_PATH}")
        return

    existing_results = []
    completed_hash_keys = set()
    if output_path.exists():
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                content = json.load(f)
            if isinstance(content, dict) and "results" in content:
                existing_results = content["results"]
            elif isinstance(content, list):
                existing_results = content

            existing_results = filter_results_with_non_empty_answers(
                existing_results,
                "resume generation",
            )
            for res in existing_results:
                if not isinstance(res, dict) or not has_non_empty_text(res.get("llm_answer")):
                    continue
                hash_key = build_hash_key_from_result_row(res)
                if hash_key:
                    completed_hash_keys.add(hash_key)
            print(f"Resuming generation... {len(completed_hash_keys)} tasks already completed.")
        except json.JSONDecodeError:
            print("Output file exists but is not valid JSON. Starting fresh.")

    tasks_to_run = [item for item in dataset if item["hash_key"] not in completed_hash_keys]
    print(f"Total tasks: {len(dataset)}")
    print(f"Already completed (Skipped): {len(completed_hash_keys)}")
    print(f"Tasks to run: {len(tasks_to_run)}")

    evaluator_sem = asyncio.Semaphore(max_concurrent_evaluator_tasks)
    tasks = [
        generate_task(
            evaluator_sem=evaluator_sem,
            item=item,
            evaluator_client=evaluator_client,
            evaluator_model=model_to_evaluate,
            evaluator_kwargs=eval_kwargs
        )
        for item in tasks_to_run
    ]

    all_results = list(existing_results)
    if tasks:
        for future in tqdm.as_completed(tasks, total=len(tasks), desc="Generating"):
            result = await future
            if not result:
                continue
            all_results.append(result)
            final_output = {
                "calculate_mean_score": None,
                "model_config": model_config_safe,
                "results": all_results
            }
            try:
                write_json_output(output_path, final_output)
            except Exception as e:
                print(f"Error saving progress: {e}")

    print(f"Generation complete. Saved to {output_path}")


async def run_evaluation_mode(args) -> None:
    output_path = Path(args.evaluate_file)
    if not output_path.exists():
        print(f"Evaluate file not found: {output_path}")
        return

    judge_model = args.judge_model
    max_concurrent_judge_tasks = args.eval_workers
    judge_config = load_yaml_config(EVALUATORS_CONFIG_PATH, judge_model)
    if not judge_config:
        print(f"Could not load config for judge model: {judge_model}")
        return
    judge_config_safe = strip_sensitive_config(judge_config)

    judge_params, judge_kwargs = get_client_params(judge_config)
    judge_client = AsyncOpenAI(**judge_params)

    try:
        with open(output_path, "r", encoding="utf-8") as f:
            content = json.load(f)
    except json.JSONDecodeError:
        print(f"Evaluate file is not valid JSON: {output_path}")
        return

    if isinstance(content, dict):
        if "results" not in content or not isinstance(content["results"], list):
            print("Evaluate file must contain a 'results' list.")
            return
        output_content = content
        all_results = output_content["results"]
    elif isinstance(content, list):
        all_results = content
        output_content = {
            "calculate_mean_score": 0.0,
            "results": all_results
        }
    else:
        print("Evaluate file JSON root must be an object or a list.")
        return

    output_content["judge_config"] = judge_config_safe

    to_evaluate = []
    skipped_completed = 0
    skipped_invalid = 0
    for idx, result in enumerate(all_results):
        if result.get("final_score") is not None:
            skipped_completed += 1
            continue
        if not has_required_generation_fields(result):
            skipped_invalid += 1
            print(f"Skipping invalid result at index {idx}: missing or empty query/llm_answer/gold_answer")
            continue
        to_evaluate.append((idx, result))

    print(f"Total results: {len(all_results)}")
    print(f"Already scored (Skipped): {skipped_completed}")
    print(f"Invalid rows (Skipped): {skipped_invalid}")
    print(f"Rows to evaluate: {len(to_evaluate)}")

    judge_sem = asyncio.Semaphore(max_concurrent_judge_tasks)
    tasks = [
        evaluate_existing_result(
            judge_sem=judge_sem,
            result_idx=result_idx,
            item=result_item,
            judge_client=judge_client,
            judge_model=judge_model,
            judge_kwargs=judge_kwargs
        )
        for result_idx, result_item in to_evaluate
    ]

    if tasks:
        for future in tqdm.as_completed(tasks, total=len(tasks), desc="Evaluating"):
            evaluated = await future
            if not evaluated:
                continue
            result_idx, final_score = evaluated
            all_results[result_idx]["final_score"] = final_score

            output_content["calculate_mean_score"] = calculate_mean_score(all_results)
            try:
                write_json_output(output_path, output_content)
            except Exception as e:
                print(f"Error saving progress: {e}")

    output_content["calculate_mean_score"] = calculate_mean_score(all_results)
    write_json_output(output_path, output_content)
    print(f"Evaluation complete. Saved to {output_path}")
    print(f"Mean Score: {output_content['calculate_mean_score']}")


async def main_async():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-to", help="Path to save the results")
    parser.add_argument("--evaluate-file", help="Path to generated JSON file for in-place scoring")
    parser.add_argument("--num-tasks", type=int, help="Number of tasks to run from the start of the dataset")
    parser.add_argument("--model-id", help="Model ID to evaluate")
    parser.add_argument("--judge-model", default="deepseek-v4-pro", help="Judge Model ID")
    parser.add_argument("--gen-workers", type=int, default=50, help="Max concurrent generation tasks")
    parser.add_argument("--eval-workers", type=int, default=50, help="Max concurrent evaluation tasks")
    args = parser.parse_args()
    if args.evaluate_file:
        if args.save_to:
            print("--save-to is ignored in evaluation mode; results are written in-place.")
        await run_evaluation_mode(args)
    else:
        await run_generation_mode(args)


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
