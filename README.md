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
- `--num-tasks`: (Optional) Number of tasks to run from the start of the dataset. Useful for testing.

### Examples

Run evaluation on all tasks:
```bash
python main.py
```

Run only the first 10 tasks:
```bash
python main.py --num-tasks 10
```

Save results to a specific file:
```bash
python main.py --save-to results/my_eval.json
```

## How It Works

1. **Dataset Loading**: Reads questions and gold answers from `dataset/FACTS-Parametric-public.csv`.
2. **Prediction**: Queries the target model (defined in `main.py` as `MODEL_TO_EVALUATE`) for an answer.
3. **Grading**:
   - The judge model (defined as `JUDGE_MODEL`) compares the predicted answer with the gold answer.
   - Grading is performed 3 times for robustness.
   - Grades: `CORRECT`, `MISTAKE`, `UNKNOWN`, `NOT_ATTEMPTED`.
4. **Scoring**:
   - A score of `1.0` is assigned if **all 3** judgments are `CORRECT`.
   - Otherwise, the score is `0.0`.
5. **Output**: Results are saved incrementally to a JSON file.

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
