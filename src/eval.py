import os
import json
from tqdm import tqdm
import evaluate
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from utils.logger import logger


def compute_token_f1(pred_tokens, ref_tokens):
    """
    Compute token-based F1 score.

    Args:
        pred_tokens (list): Tokens from the model's generated response.
        ref_tokens (list): Tokens from the ground truth reference.

    Returns:
        float: F1 score.
    """
    pred_set = set(pred_tokens)
    ref_set = set(ref_tokens)

    tp = len(pred_set & ref_set)  # True positives
    precision = tp / len(pred_set) if pred_set else 0
    recall = tp / len(ref_set) if ref_set else 0
    f1 = (2 * precision * recall) / (precision + recall) if precision + recall > 0 else 0
    return f1


def visualize_results(evaluation_results, model_name, assets_dir):
    """
    Generate and save visualizations of evaluation metrics.

    Args:
        evaluation_results (pd.DataFrame): DataFrame containing detailed evaluation results.
        model_name (str): Name of the model being evaluated.
        assets_dir (str): Directory to save the visualization assets.
    """
    logger.info(f"Generating visualizations for {model_name}.")
    # Ensure assets directory exists
    os.makedirs(assets_dir, exist_ok=True)

    # Create bar chart of average metrics
    plt.figure(figsize=(10, 6))
    evaluation_results[["exact_match", "contains_match", "f1_score", "bert_score"]].mean().plot.bar()
    plt.title(f"Evaluation Metrics for {model_name}")
    plt.ylabel("Score")
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save the bar chart
    bar_chart_path = os.path.join(assets_dir, f"{model_name}_metrics_bar_chart.png")
    plt.savefig(bar_chart_path)
    plt.close()
    logger.info(f"Saved visualization to {bar_chart_path}.")


def evaluate_results(results_file, model_name="unknown", evals_dir="~/projects/SLLIM/data/evals"):
    """
    Evaluate the performance of a model based on its generated responses.

    Args:
        results_file (str): Path to the JSON file containing model responses.
        model_name (str): Name of the model being evaluated.
        evals_dir (str): Directory to save evaluation results.

    Returns:
        pd.DataFrame: DataFrame containing detailed evaluation results.
    """
    evals_dir = os.path.abspath(os.path.expanduser(evals_dir))
    os.makedirs(evals_dir, exist_ok=True)

    logger.info(f"Evaluating results for {model_name} from file: {results_file}.")

    # Load the results from the JSON file
    with open(results_file, "r") as infile:
        data = json.load(infile)

    responses = data.get("responses", [])

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # Initialize lists to store evaluation metrics
    exact_matches = []
    contains_matches = []
    f1_scores = []
    bert_scores = []

    # Load BERTScore metric
    bertscore_metric = evaluate.load("bertscore")

    # Prepare lists for BERTScore computation
    references = []
    candidates = []

    # Process each response
    for response in tqdm(responses, desc="Evaluating Responses"):
        answer = response.get("answer", "").strip()
        model_generated_answer = response.get("model_generated_answer", "").strip()

        # Skip invalid or empty responses
        if not answer or not model_generated_answer:
            logger.warning(f"Skipping invalid entry: {response}")
            exact_matches.append(0)
            contains_matches.append(0)
            f1_scores.append(0)
            references.append("")
            candidates.append("")
            continue

        # Tokenize predictions and references
        answer_tokens = tokenizer.tokenize(answer)
        model_generated_tokens = tokenizer.tokenize(model_generated_answer)

        # 1. Exact Match
        exact_matches.append(int(answer == model_generated_answer))

        # 2. Contains Match
        contains_matches.append(int(answer in model_generated_answer))

        # 3. Token-based F1 Score
        f1 = compute_token_f1(model_generated_tokens, answer_tokens)
        f1_scores.append(f1)

        # Append to lists for BERTScore
        references.append(answer)
        candidates.append(model_generated_answer)

    # 4. BERTScore
    bertscore_results = bertscore_metric.compute(predictions=candidates, references=references, lang="en")
    bert_scores = bertscore_results['f1']

    # Combine results into a DataFrame
    evaluation_results = pd.DataFrame({
        "question": [response.get("question", "") for response in responses],
        "answer": [response.get("answer", "") for response in responses],
        "model_generated_answer": [response.get("model_generated_answer", "") for response in responses],
        "exact_match": exact_matches,
        "contains_match": contains_matches,
        "f1_score": f1_scores,
        "bert_score": bert_scores,
    })

    # Calculate aggregated metrics
    aggregated_metrics = {
        "exact_match": evaluation_results["exact_match"].mean(),
        "contains_match": evaluation_results["contains_match"].mean(),
        "f1_score": evaluation_results["f1_score"].mean(),
        "bert_score": evaluation_results["bert_score"].mean(),
    }

    # Save aggregated metrics to a JSON file
    aggregated_metrics_file = os.path.join(evals_dir, f"{model_name}_aggregated_metrics.json")
    with open(aggregated_metrics_file, "w") as outfile:
        json.dump(aggregated_metrics, outfile, indent=4)
    logger.info(f"Aggregated metrics saved to {aggregated_metrics_file}.")

    # Save detailed results to a CSV file
    detailed_results_file = os.path.join(evals_dir, f"{model_name}_eval.csv")
    evaluation_results.to_csv(detailed_results_file, index=False)
    logger.info(f"Detailed eval saved to {detailed_results_file}.")

    return evaluation_results


if __name__ == "__main__":
    results_dir = "~/projects/SLLIM/data/results"
    results_dir = os.path.abspath(os.path.expanduser(results_dir))

    # Directories for evaluation results and visualizations
    evals_save_dir = "~/projects/SLLIM/data/evals"
    assets_save_dir = "~/projects/SLLIM/assets"
    evals_save_dir = os.path.abspath(os.path.expanduser(evals_save_dir))
    assets_save_dir = os.path.abspath(os.path.expanduser(assets_save_dir))

    # List results files
    result_files = [os.path.join(results_dir, f) for f in os.listdir(results_dir) if f.endswith(".json")]

    # Evaluate each file and generate visualizations
    for results_file in result_files:
        model_name = os.path.splitext(os.path.basename(results_file))[0]
        evaluation_results = evaluate_results(results_file, model_name=model_name, evals_dir=evals_save_dir)
        visualize_results(evaluation_results, model_name, assets_save_dir)
        logger.info(f"Completed evaluation for {model_name}.")
