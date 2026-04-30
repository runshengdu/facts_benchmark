# LLM Evaluation Framework

This project provides an automated framework for evaluating Large Language Models (LLMs) on factual questions using a "Model-as-a-Judge" approach. It compares model predictions against gold-standard answers and uses a judge model to grade the correctness.

## Features

- **Automated Evaluation**: Automatically queries models and grades their answers.
- **Async & Concurrent**: Uses `asyncio` for high-throughput concurrent API calls.
- **Streaming Support**: Handles streaming API responses for better compatibility and reliability.
- **Resume Capability**: Automatically detects existing results and resumes evaluation where it left off.
- **Flexible Configuration**: Define models and evaluators easily in YAML files.
- **Standardized Output**: Produces detailed JSON reports with queries, predictions, gold answers, and scores.

## Prerequisites

- Python 3.8+
- Required Python packages (install via pip):
  ```bash
  pip install openai pyyaml tqdm
  ```

## Configuration

The framework uses two main YAML configuration files:

### 1. `models.yaml`
Defines the models to be evaluated.
```yaml
models:
  - name: model-name-slug
    temperature: 1.0
    base_url: https://api.provider.com/v1
    api_key: "${ENV_VAR_NAME}"
    # Additional parameters can be added
```

### 2. `evaluators.yaml`
Defines the judge models used for grading.
```yaml
models:
  - name: judge-model-name
    temperature: 0.0
    base_url: https://api.provider.com/v1
    api_key: "${ENV_VAR_NAME}"
```

**Note**: Ensure required environment variables (e.g., `OPENROUTER_API_KEY`, `DEEPSEEK_API_KEY`) are set in your environment before running the script.

## Usage

Run the main script to start the evaluation:

```bash
python main.py [options]
```

### Arguments

- `--save-to`: (Optional) Custom path to save the results JSON file.
- `--evaluate-file`: (Optional) Enter evaluation mode and score an existing generated JSON file in-place.
- `--num-tasks`: (Optional) Number of tasks to run from the start of the dataset. Useful for testing.
- `--model-id`: Model ID used in generation mode.
- `--judge-model`: Judge model ID used in evaluation mode (default: `deepseek-chat`).

### Examples

Run generation on all tasks:
```bash
python main.py --model-id doubao-seed-2-0-pro-260215
```

Run only the first 10 tasks:
```bash
python main.py --model-id doubao-seed-2-0-pro-260215 --num-tasks 10
```

Save results to a specific file:
```bash
python main.py --model-id doubao-seed-2-0-pro-260215 --save-to results/my_generated.json
```

Evaluate an existing generated file in-place:
```bash
python main.py --evaluate-file result/doubao-seed-2-0-pro-260215/FACTS-Parametric-public/20260224_161817.json --judge-model deepseek-chat
```

## How It Works

1. **Generation Mode**:
   - Reads questions and gold answers from `dataset/FACTS-Parametric-public.csv`.
   - Queries the target model for `llm_answer`.
   - Writes each row with: `id`, `query`, `llm_answer`, `gold_answer`, `final_score` (`null` before scoring).
2. **Evaluation Mode**:
   - Reads an existing generated JSON via `--evaluate-file`.
   - Skips rows where `final_score` is already non-null.
   - Uses the judge model to score remaining rows and writes back to the same file.
3. **Grading**:
   - The judge model (defined as `JUDGE_MODEL`) compares the predicted answer with the gold answer.
   - Grading is performed 3 times for robustness.
   - Grades: `CORRECT`, `MISTAKE`, `UNKNOWN`, `NOT_ATTEMPTED`.
4. **Scoring**:
   - A score of `1.0` is assigned if **all 3** judgments are `CORRECT`.
   - Otherwise, the score is `0.0`.
5. **Output**:
   - Generation mode: `calculate_mean_score` is `null`.
   - Evaluation mode: `calculate_mean_score` is updated from non-null `final_score` rows only.

## Output Format

The output JSON file contains the mean score and a list of detailed results for each task:

```json
{
  "calculate_mean_score": 0.5,
  "results": [
    {
      "id": "task_id",
      "query": "Question text...",
      "llm_answer": "Model's predicted answer...",
      "gold_answer": "Correct answer...",
      "final_score": 1.0
    },
    ...
  ]
}
```

## Customization

To change the model being evaluated or the judge model, modify the constants in `main.py`:

```python
MODEL_TO_EVALUATE = "minimax-m2.1"
JUDGE_MODEL = "deepseek-chat"
```

Ensure these names correspond to entries in `models.yaml` and `evaluators.yaml`.
