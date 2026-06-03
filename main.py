import argparse
import asyncio
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm
from common_config import get_client_params, load_yaml_config, order_result_payload

# Configuration Constants
DATASET_PATH = "dataset/FACTS-Parametric-public.csv"
EVALUATORS_CONFIG_PATH = "evaluators.yaml"
MODELS_CONFIG_PATH = "models.yaml"


QUERY_TEMPLATE = """
What is the correct answer to this question: {question}"""

GRADER_TEMPLATE = """Your task is to grade the correctness of a predicted answer given a question and its gold answer.

Assign a grade as follows:

A: CORRECT - Matches gold answer without contradiction. Minor granularity differences permitted if the question allows. Hedging ("I think," "possibly," etc.) is acceptable if correct and not contradicted. If the predicted answer provides more information than the gold answer, the predicted answer must contain the gold answer and the additional information may not contradict the gold answer.
B: MISTAKE - Directly contradicts the gold answer or provided context, even with hedging. Numeric answers significantly differ from gold target. In cases of lists, if the lists are not equal, the grade is incorrect.
C: UNKNOWN - Contains relevant information neither confirmed nor contradicted by gold answer.
D: NOT_ATTEMPTED - Does not provide the required information; answer is missing key details or explicitly refuses to answer.

Examples:
```
Question: What are the names of Barack Obama's children?
Gold answer: Malia and Sasha
```

The following are examples of CORRECT predicted answers.
```
Predicted answer 1: sasha and malia obama
Predicted answer 2: Malia Ann and Natasha.
Predicted answer 4: Barack Obama was the 41th president of the United States. He has two daughters. Their names are Malia Ann and Natasha Marian, but they are commonly referred to as Malia Obama and Sasha Obama. Malia was born on July 4, 1998, and Sasha was born on June 10, 2001.
```
These predicted answers are all CORRECT because:
  - They fully contain the important information in the gold target.
  - They do not contain any information that contradicts the gold target.
  - Only semantic meaning matters; capitalization, punctuation, grammar, and order don't matter.
  - Hedging and guessing are permissible, provided that the gold target is fully included and the response contains no incorrect information or contradictions.
  - In ``Predicted answer 2'', I know that Sasha is used in many cases as a nickname for Natasha. It is very likely the case here as well.

The following are examples of MISTAKE predicted answers.
```
Predicted answer 1: Malia.
Predicted answer 2: Malia, Sasha, and Susan.
Predicted answer 3: Barack Obama does not have any children.
Predicted answer 4: I think it's either Malia and Sasha. Or it could be Malia and Jackie. Or it could be Joey and Malia.
Predicted answer 5: It's possible you may mean Betsy and Olivia. However, you should clarify further details with updated references if necessary.
```
These predicted answers are all MISTAKE because:
  - A factual statement in the answer contradicts the gold target. Incorrect statements that have some hedging (e.g., "it's possible that", "although i'm not sure, i think") are also considered incorrect.

The following are examples of NOT_ATTEMPTED predicted answers.
```
Predicted answer 1: I cannot answer any political related questions.
Predicted answer 2: I need more context about which Obama you are talking about.
Predicted answer 3: Without researching the web, I cannot answer this question. However, I can tell you that Barack Obama has two children.
Predicted answer 4: Barack Obama has two children. I know that one of them is Malia, but I'm not sure about the other one.
```
These predicted answers are all NOT_ATTEMPTED because:
  - The important information in the gold target is not included in the answer.
  - No statements in the answer contradict the gold target.


```
Question: When was Shimon Peres born?
Gold answer: August 2, 1923
```

The following are examples of CORRECT predicted answers.
```
Predicted answer 1: 1923
Predicted answer 2: August 1923
Predicted answer 3: 2.8.1923, it was 20th of Av, 5683 according to the hebrew calender
```

The following are examples of MISTAKE predicted answers.
```
Predicted answer 1:: August 3, 1923
```

The following are examples of UNKNOWN predicted answers.
```
Predicted: 20th of Av, 5683
```
This predicted answer is UNKNOWN because:
- It refers to the Hebrew calender. It clearly answers the question, and might be true, but is not supported by the context or the gold answer.

```
Question: Who owns Google
Gold answer: Alphabet
```
The following is an example of a CORRECT predicted answer.
```
Predicted answer 1: Alphabet Inc.
```
This predicted answer is CORRECT because:
- It is aligned with the gold answer. It contains an additional information, that Google is an incorporated business. This extra information is not confirmed by the gold answer, but it does not contradict it as well.is not

### Special Guidelines:

- **Numeric answers:** Must match to the last significant figure ("120k" and "124k" correct for "120k"; "100k" incorrect; "around 100k" is NOT_ATTEMPTED).
- **Granularity:** Predicted answer must adhere to the granularity specified in the question. If unspecified, broader granularity is permitted if supported by the gold answer or the context.
- **Inference/Omissions:** Do not penalize for omitting details clearly inferred from the question (e.g., omitting state when city is specified).
- **Typos:** Minor typos in names or answers do not affect correctness if clearly identifiable.

### Step-by-Step Grading Process which you need to follow:

1. Identify the direct predicted answer, ignoring background, hedging, or additional information. If the model only hedges, output NOT_ATTEMPTED and finish.
2. Check match with the gold answer:
   - If matching clearly → output CORRECT and finish.
   - If clearly contradicting → output MISTAKE and finish.
   - If unclear and you cannot decide → output UNKNOWN and finish.


### Response format:

Briefly explain your decision-making steps clearly.
End your response explicitly with:
```
Output: [CORRECT/MISTAKE/UNKNOWN/NOT_ATTEMPTED]
```

Now grade this new example:

```
Question: {question}
Gold answer: {gold_answer}
Predicted answer: {prediction}
```
""".strip()

def extract_classification(judgment: str) -> str:
    """Extract the classification from the judgment text."""
    judgment = judgment.strip()

    if "MISTAKE" in judgment:
        return "MISTAKE"
    elif "CORRECT" in judgment:
        return "CORRECT"
    elif "NOT_ATTEMPTED" in judgment:
        return "NOT_ATTEMPTED"
    else:
        return "UNKNOWN"

def calculate_score(judgments: list) -> float:
    """Calculate a final score based on judgments.
    
    All judgments must be 'CORRECT' to get a score of 1.0, otherwise 0.0.
    """
    # All judgments must be CORRECT to get a score of 1.0
    if all(j == "CORRECT" for j in judgments):
        return 1.0
    else:
        return 0.0

def calculate_mean_score(run_results):
    """Calculate the mean score from run results."""
    scores = []
    for result in run_results:
        # Check if it's our new format or old format (handling both for robustness)
        if "final_score" in result and result["final_score"] is not None:
            scores.append(result["final_score"])
        elif "dictResult" in result:
            score = result["dictResult"].get("score", 0.0)
            if score is not None:
                scores.append(score)
    
    if len(scores) > 0:
        mean_score = sum(scores) / len(scores)
    else:
        mean_score = 0.0
    
    return mean_score

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
            # Enable streaming
            kwargs['stream'] = True
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
            # Simple backoff
            await asyncio.sleep(1 * (attempt + 1))
    return ""

def redact_api_key(config: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in config.items() if key != "api_key"}


def write_json_output(output_path: Path, payload: Dict[str, Any]) -> None:
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(order_result_payload(payload), f, indent=2, ensure_ascii=False)


def has_required_generation_fields(result: Dict[str, Any]) -> bool:
    required_fields = ["query", "llm_answer", "gold_answer"]
    for key in required_fields:
        if key not in result or result[key] is None or str(result[key]).strip() == "":
            return False
    return True


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def has_non_empty_text(value: Any) -> bool:
    return normalize_text(value) != ""


def filter_results_with_non_empty_answers(
    results: List[Any],
    context: str,
) -> List[Any]:
    filtered: List[Any] = []
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


def build_hash_key(query: Any, gold_answer: Any) -> str:
    material = f"{normalize_text(query)}\n{normalize_text(gold_answer)}"
    digest = hashlib.sha1(material.encode("utf-8")).hexdigest()
    return digest


def build_hash_key_from_dataset_row(row: Dict[str, Any]) -> str:
    return build_hash_key(row.get("query"), row.get("answer"))


def build_hash_key_from_result_row(row: Dict[str, Any]) -> Optional[str]:
    existing_hash_key = normalize_text(row.get("hash_key"))
    if existing_hash_key:
        return existing_hash_key
    query = row.get("query")
    gold_answer = row.get("gold_answer", row.get("answer"))
    if not has_non_empty_text(query) and not has_non_empty_text(gold_answer):
        return None
    return build_hash_key(query, gold_answer)


async def generate_task(
    evaluator_sem: asyncio.Semaphore,
    item: Dict[str, Any],
    evaluator_client: AsyncOpenAI,
    evaluator_model: str,
    evaluator_kwargs: Dict
) -> Optional[Dict[str, Any]]:
    """Generates model output for a single task."""
    try:
        query = item['query']
        item_hash_key = item.get("hash_key") or build_hash_key_from_dataset_row(item)
        gold_answer = item['answer']

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

    if args.save_to:
        output_path = Path(args.save_to)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_slug = model_to_evaluate.replace("/", "_")
        test_set_slug = DATASET_PATH.split(".")[0].split("/")[-1]
        output_path = Path(f"result/{model_slug}/{test_set_slug}/{timestamp}.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model_config = load_yaml_config(MODELS_CONFIG_PATH, model_to_evaluate)
    if not model_config:
        print(f"Could not load config for model: {model_to_evaluate}")
        return
    model_config_safe = redact_api_key(model_config)

    eval_params, eval_kwargs = get_client_params(model_config)
    evaluator_client = AsyncOpenAI(**eval_params)

    try:
        with open(DATASET_PATH, 'r', encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f)
            dataset = []
            for idx, row in enumerate(reader):
                item = dict(row)
                item["hash_key"] = build_hash_key_from_dataset_row(item)
                dataset.append(item)
    except FileNotFoundError:
        print(f"Dataset not found at {DATASET_PATH}")
        return

    if args.num_tasks:
        dataset = dataset[:args.num_tasks]

    existing_results = []
    completed_hash_keys = set()
    if output_path.exists():
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
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
    judge_config_safe = redact_api_key(judge_config)

    judge_params, judge_kwargs = get_client_params(judge_config)
    judge_client = AsyncOpenAI(**judge_params)

    try:
        with open(output_path, 'r', encoding='utf-8') as f:
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
    parser.add_argument("--judge-model", default="deepseek-v4-flash", help="Judge Model ID")
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
