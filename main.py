import argparse
import asyncio
import csv
import json
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import yaml
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm

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
        if "final_score" in result:
             scores.append(result["final_score"])
        elif "dictResult" in result:
            score = result["dictResult"].get("score", 0.0)
            scores.append(score)
    
    if len(scores) > 0:
        mean_score = sum(scores) / len(scores)
    else:
        mean_score = 0.0
    
    return mean_score

def load_yaml_config(path: str, model_name: str) -> Optional[Dict[str, Any]]:
    """Loads configuration for a specific model from a YAML file."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
            models = data.get('models', [])
            for model in models:
                if model.get('name') == model_name:
                    # Substitute environment variables
                    config = model.copy()
                    if 'api_key' in config:
                        config['api_key'] = os.path.expandvars(config['api_key'])
                        # Also handle ${VAR} syntax manually if expandvars doesn't catch it all
                        if config['api_key'].startswith('${') and config['api_key'].endswith('}'):
                            var_name = config['api_key'][2:-1]
                            config['api_key'] = os.environ.get(var_name, '')
                    return config
    except Exception as e:
        print(f"Error loading config from {path}: {e}")
    return None

async def call_api_with_retry(client: AsyncOpenAI, messages: List[Dict], model: str, **kwargs) -> str:
    """Calls OpenAI API with retries using streaming."""
    retries = 3
    for attempt in range(retries):
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
            if attempt == retries - 1:
                print(f"API call failed after {retries} attempts: {e}")
                raise e
            # Simple backoff
            await asyncio.sleep(1 * (attempt + 1))
    return ""

async def evaluate_task(
    evaluator_sem: asyncio.Semaphore,
    judge_sem: asyncio.Semaphore,
    item: Dict[str, Any],
    evaluator_client: AsyncOpenAI,
    judge_client: AsyncOpenAI,
    evaluator_model: str,
    judge_model: str,
    evaluator_kwargs: Dict,
    judge_kwargs: Dict
) -> Optional[Dict[str, Any]]:
    """Evaluates a single task."""
    try:
        query = item['query']
        gold_answer = item['answer']
        item_id = item['id']
    

        # 1. Get Prediction
        formatted_prompt = QUERY_TEMPLATE.format(question=query)
        try:
            async with evaluator_sem:
                prediction = await call_api_with_retry(
                    evaluator_client,
                    [{"role": "user", "content": formatted_prompt}],
                    model=evaluator_model, # Usually ignored by openrouter if base_url is specific, but good practice
                    **evaluator_kwargs
                )
        except Exception:
            return None # Fail silently/skip as requested

        # 2. Get Judgments (3 times)
        judgments = []
        
        # We can run judgments in parallel too if desired, but let's keep it simple or sequential per task
        # User said "api uses concurrent calling", which implies the tasks are concurrent.
        # Inside a task, we can also be concurrent for the 3 judgments.
        
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
                    seed=seed, # OpenAI supports seed
                    **judge_kwargs
                )

        judge_tasks = [run_judge(i) for i in range(3)]
        
        try:
            raw_judgments = await asyncio.gather(*judge_tasks)
        except Exception:
             return None

        for j_text in raw_judgments:
            judgments.append(extract_classification(j_text))

        final_score = calculate_score(judgments)

        return {
            "id": item_id,
            "query": query,
            "llm_answer": prediction,
            "gold_answer": gold_answer,
            "final_score": final_score
        }

    except Exception as e:
        print(f"Task {item.get('id')} failed: {e}")
        return None

async def main_async():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-to", help="Path to save the results")
    parser.add_argument("--num-tasks", type=int, help="Number of tasks to run from the start of the dataset")
    parser.add_argument("--model-id", help="Model ID to evaluate")
    parser.add_argument("--judge-model", default="deepseek-reasoner", help="Judge Model ID")
    parser.add_argument("--concurrency-eval", type=int, default=50, help="Max concurrent evaluator tasks")
    parser.add_argument("--concurrency-judge", type=int, default=50, help="Max concurrent judge tasks")
    args = parser.parse_args()

    model_to_evaluate = args.model_id 
    judge_model = args.judge_model 
    max_concurrent_evaluator_tasks = args.concurrency_eval 
    max_concurrent_judge_tasks = args.concurrency_judge 

    # 1. Determine Output Path
    if args.save_to:
        output_path = Path(args.save_to)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_slug = model_to_evaluate.replace("/", "_")
        test_set_slug = DATASET_PATH.split(".")[0].split("/")[-1]
        output_path = Path(f"result/{model_slug}/{test_set_slug}/{timestamp}.json")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 2. Load Configs
    model_config = load_yaml_config(MODELS_CONFIG_PATH, model_to_evaluate)
    judge_config = load_yaml_config(EVALUATORS_CONFIG_PATH, judge_model)

    if not model_config:
        print(f"Could not load config for model: {model_to_evaluate}")
        return
    if not judge_config:
        print(f"Could not load config for judge model: {judge_model}")
        return

    def redact_api_key(config: Dict[str, Any]) -> Dict[str, Any]:
        return {key: value for key, value in config.items() if key != "api_key"}

    model_config_safe = redact_api_key(model_config)
    judge_config_safe = redact_api_key(judge_config)

    # 3. Setup Clients
    # Extract params for OpenAI client
    def get_client_params(config):
        params = {
            "api_key": config.get("api_key"),
            "base_url": config.get("base_url"),
        }
        
        # Keys consumed by client init or internal logic, not to be passed to chat.completions.create
        exclude_keys = {"name", "api_key", "base_url"}
        
        # Filter kwargs for chat completion
        chat_kwargs = {}
        for key, value in config.items():
            if key not in exclude_keys:
                chat_kwargs[key] = value
                
        return params, chat_kwargs

    eval_params, eval_kwargs = get_client_params(model_config)
    judge_params, judge_kwargs = get_client_params(judge_config)

    evaluator_client = AsyncOpenAI(**eval_params)
    judge_client = AsyncOpenAI(**judge_params)

    # 4. Load Dataset
    try:
        with open(DATASET_PATH, 'r', encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f)
            dataset = []
            for idx, row in enumerate(reader):
                item = dict(row)
                item_id = item.get("id")
                if item_id is None or str(item_id).strip() == "":
                    item["id"] = idx
                dataset.append(item)
    except FileNotFoundError:
        print(f"Dataset not found at {DATASET_PATH}")
        return

    if args.num_tasks:
        dataset = dataset[:args.num_tasks]

    # 5. Resume Logic
    completed_ids = set()
    existing_results = []
    
    if output_path.exists():
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                content = json.load(f)
                # Check format. If it has 'results' list, use that.
                if isinstance(content, dict) and "results" in content:
                    existing_results = content["results"]
                elif isinstance(content, list):
                    # Legacy or simple list
                    existing_results = content
                
                for res in existing_results:
                    if "id" in res and res.get("final_score") is not None:
                        completed_ids.add(res["id"])
            print(f"Resuming... {len(completed_ids)} tasks already completed.")
        except json.JSONDecodeError:
            print("Output file exists but is not valid JSON. Starting fresh.")

    # 6. Run Tasks
    evaluator_sem = asyncio.Semaphore(max_concurrent_evaluator_tasks) # Limit evaluator concurrency to avoid hitting rate limits too hard
    judge_sem = asyncio.Semaphore(max_concurrent_judge_tasks) # Limit judge concurrency to avoid hitting rate limits too hard
    tasks = []
    
    # Identify tasks to run
    tasks_to_run = []
    for item in dataset:
        if item['id'] in completed_ids:
            continue
        tasks_to_run.append(item)
    
    print(f"Total tasks: {len(dataset)}")
    print(f"Already completed (Skipped): {len(completed_ids)}")
    print(f"Tasks to run: {len(tasks_to_run)}")

    # Create coroutines for tasks
    for item in tasks_to_run:
        tasks.append(
            evaluate_task(
                evaluator_sem,
                judge_sem,
                item, 
                evaluator_client, 
                judge_client, 
                model_to_evaluate,
                judge_model,
                eval_kwargs,
                judge_kwargs
            )
        )

    # Use as_completed to process results as they finish
    new_results = []
    all_results = list(existing_results) # Start with existing results
    
    if tasks:
        for future in tqdm.as_completed(tasks, total=len(tasks), desc="Evaluating"):
            result = await future
            if result:
                new_results.append(result)
                all_results.append(result)
                
                # Real-time saving
                mean_score = calculate_mean_score(all_results)
                final_output = {
                    "calculate_mean_score": mean_score,
                    "model_config": model_config_safe,
                    "judge_config": judge_config_safe,
                    "results": all_results
                }
                
                # Write to file (rewrite the whole file for safety and simplicity with JSON structure)
                # For very large datasets, append mode to a JSONL file is better, but user requested specific JSON structure.
                try:
                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(final_output, f, indent=2, ensure_ascii=False)
                except Exception as e:
                    print(f"Error saving progress: {e}")

    # Final calculation (already done in loop, but good to ensure consistency)
    mean_score = calculate_mean_score(all_results)
    
    print(f"Evaluation complete. Saved to {output_path}")
    print(f"Mean Score: {mean_score}")

def main():
    asyncio.run(main_async())

if __name__ == "__main__":
    main()
