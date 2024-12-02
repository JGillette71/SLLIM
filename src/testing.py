import os
import json
import torch
from tqdm import tqdm
from datetime import datetime
from transformers import pipeline, BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer
from utils.logger import logger
import random
from datasets import Dataset


def load_few_shot_examples(train_file, num_examples=2):
    """
    Load a few-shot example set from the training data file.

    Args:
        train_file (str): Path to the JSON training file.
        num_examples (int): Number of examples to load.

    Returns:
        list: List of formatted example strings.
    """
    train_file = os.path.abspath(os.path.expanduser(train_file))
    try:
        with open(train_file, "r") as infile:
            train_data = json.load(infile)

        # Randomly sample examples
        sampled_data = random.sample(train_data, k=min(num_examples, len(train_data)))
        logger.info(f"Loaded {len(sampled_data)} few-shot examples from {train_file}")

        return [
            {
                "context": ex["context"],
                "question": ex["question"],
                "answer": ex["answer"]
            }
            for ex in sampled_data
        ]
    except Exception as e:
        logger.error(f"Failed to load few-shot examples from {train_file}: {e}", exc_info=True)
        return []


def run_inference(model_id, input_path, output_path, train_file=None, few_shot=False):
    """
    Runs inference using the specified model on the given input data.

    Args:
        model_id (str): Hugging Face model identifier.
        input_path (str): Path to the JSON input file.
        output_path (str): Path to save the model responses.
        train_file (str): Path to the JSON training file for few-shot examples.
        few_shot (bool): Whether to use few-shot prompting.
    """
    # Ensure paths are properly expanded and absolute
    input_path = os.path.abspath(os.path.expanduser(input_path))
    output_path = os.path.abspath(os.path.expanduser(output_path))

    # Load few-shot examples if applicable
    few_shot_examples = []
    if few_shot and train_file:
        few_shot_examples = load_few_shot_examples(train_file)

    logger.info(f"Loading model: {model_id}")

    # Load tokenizer to explicitly set pad_token_id
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        pad_token_id = tokenizer.eos_token_id
    except Exception as e:
        logger.error(f"Failed to load tokenizer for {model_id}: {e}", exc_info=True)
        return

    # Configure quantization for 4-bit precision
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    try:
        # Load the model with quantization, ensuring full GPU utilization
        max_memory = {0: "15GB", "cpu": "1GB"}  # Use most of the GPU memory
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quant_config,
            device_map="auto",
            max_memory=max_memory,
            torch_dtype=torch.bfloat16,
        )
        model_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device_map="auto",
            pad_token_id=pad_token_id,
        )
    except Exception as e:
        logger.error(f"Failed to load model {model_id}: {e}", exc_info=True)
        return

    logger.info(f"Loading input data from: {input_path}")
    try:
        with open(input_path, "r") as infile:
            input_data = json.load(infile)
    except Exception as e:
        logger.error(f"Failed to load input file {input_path}: {e}", exc_info=True)
        return

    # Prepare results dictionary with metadata
    results = {
        "metadata": {
            "model_id": model_id,
            "test_type": "fewshot" if few_shot else "zeroshot",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
        "responses": []
    }

    logger.info(f"Processing {len(input_data)} entries...")

    # Run inference with batch size
    batch_size = 2
    few_shot_prompt = "\n\n".join(
        [f"Context: {ex['context']}\nQuestion: {ex['question']}\nAnswer: {ex['answer']}" for ex in few_shot_examples]
    ) + "\n\n" if few_shot_examples else ""

    def format_prompt(entry):
        return f"{few_shot_prompt}Context: {entry['context']}\nQuestion: {entry['question']}\nAnswer:"

    with tqdm(total=len(input_data), desc="Processing Batches") as pbar:
        for start_idx in range(0, len(input_data), batch_size):
            batch = input_data[start_idx:start_idx + batch_size]
            prompts = [format_prompt(entry) for entry in batch]

            try:
                batch_responses = model_pipeline(
                    prompts,
                    max_new_tokens=50,
                    truncation=True,
                    num_return_sequences=1,
                    temperature=0.5,
                    top_k=50,
                )
                for entry, response_list in zip(batch, batch_responses):
                    # Ensure response_list is processed correctly
                    if isinstance(response_list, list):
                        # Extract generated text from the first response
                        response_text = response_list[0].get("generated_text", "Error in generation")
                    else:
                        response_text = response_list.get("generated_text", "Error in generation")
                    
                    # Extract the last answer from the generated text
                    generated_answer = response_text.split("\nAnswer:")[-1].strip()
                    
                    # Append results in the updated structure
                    results["responses"].append({
                        "question": entry["question"],
                        "answer": entry["answer"],
                        "context": entry["context"],
                        "few_shot_examples": few_shot_examples if few_shot else [],
                        "model_generated_answer": generated_answer,
                    })
            except Exception as e:
                logger.error(f"Failed to generate responses for batch starting at index {start_idx}: {e}", exc_info=True)
                for entry in batch:
                    results["responses"].append({
                        "question": entry["question"],
                        "answer": entry["answer"],
                        "context": entry["context"],
                        "few_shot_examples": few_shot_examples if few_shot else [],
                        "model_generated_answer": "Error in generation",
                    })
            pbar.update(batch_size)

    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Saving responses to: {output_path}")
    try:
        with open(output_path, "w") as outfile:
            json.dump(results, outfile, indent=4)
    except Exception as e:
        logger.error(f"Failed to save responses: {e}", exc_info=True)


def main():
    # Define data directories
    raw_data_dir = "~/projects/SLLIM/data/processed"
    results_dir = "~/projects/SLLIM/data/results"
    raw_data_dir = os.path.abspath(os.path.expanduser(raw_data_dir))
    results_dir = os.path.abspath(os.path.expanduser(results_dir))

    # Ensure the results directory exists
    if not os.path.exists(results_dir):
        os.makedirs(results_dir, exist_ok=True)

    input_file = os.path.join(raw_data_dir, "test.json")
    train_file = os.path.join(raw_data_dir, "train.json")

    models = {
        "1": "meta-llama/Llama-3.1-8B",
        "2": "meta-llama/Llama-3.2-1B",
    }

    logger.info("Select a model to run inference:")
    for key, model_name in models.items():
        logger.info(f"{key}: {model_name}")

    model_choice = input("Enter the model number (1 or 2): ").strip()
    model_id = models.get(model_choice)

    if not model_id:
        logger.error("Invalid model choice. Exiting.")
        return

    # Ask if the user wants to use few-shot prompting
    use_few_shot = input("Use few-shot prompting? (yes/no): ").strip().lower() == "yes"

    # Determine test type for file naming
    test_type = "fewshot" if use_few_shot else "zeroshot"

    # Generate output file name with timestamp and test type
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = model_id.split("/")[-1]  # Extract the last part of the model ID
    output_file = os.path.join(results_dir, f"{model_name}_responses_{test_type}_{timestamp}.json")

    run_inference(model_id, input_file, output_file, train_file, few_shot=use_few_shot)

if __name__ == "__main__":
    main()
