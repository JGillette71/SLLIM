import os
import json
from typing import Dict, List
from logger import logger  # Import centralized logger

def consolidate_raw_data(raw_data_dir: str = "~/projects/SLLIM/data/raw", 
                         processed_data_dir: str = "~/projects/SLLIM/data/processed") -> None:
    """
    Transform raw JSON files into consolidated train, test, and dev JSON objects
    with a structure conducive to fine-tuning and evaluation, and save them to
    the processed directory.

    Args:
        raw_data_dir (str): Directory containing raw JSON files.
        processed_data_dir (str): Directory to save the processed JSON files.
    """
    logger.info("Starting data consolidation...")
    raw_data_dir = os.path.expanduser(raw_data_dir)
    processed_data_dir = os.path.expanduser(processed_data_dir)

    os.makedirs(processed_data_dir, exist_ok=True)
    consolidated_data = {"train": [], "dev": [], "test": []}

    for dataset_name in os.listdir(raw_data_dir):
        dataset_path = os.path.join(raw_data_dir, dataset_name)
        if not os.path.isdir(dataset_path):
            logger.warning("Skipping non-directory item: %s", dataset_path)
            continue

        logger.info("Processing dataset: %s", dataset_name)

        for role, aliases in {"train": ["train"], "dev": ["dev", "val"], "test": ["test"]}.items():
            for alias in aliases:
                raw_file_path = os.path.join(dataset_path, f"qa.json.{alias}")
                if os.path.exists(raw_file_path):
                    logger.info("Processing file: %s", raw_file_path)
                    with open(raw_file_path, "r") as f:
                        raw_data = [json.loads(line) for line in f.readlines()]

                    for item in raw_data:
                        transformed_item = {
                            "question": item["Question"],
                            "answer": item["Answer"],
                            "context": item["RawLog"],
                        }
                        consolidated_data[role].append(transformed_item)
                else:
                    logger.debug("File not found for alias %s: %s", alias, raw_file_path)

    for role, data in consolidated_data.items():
        output_file_path = os.path.join(processed_data_dir, f"{role}.json")
        with open(output_file_path, "w") as f:
            json.dump(data, f, indent=4)
        logger.info("Saved %s data to %s", role, output_file_path)

    logger.info("Data consolidation complete. Processed data saved to: %s", processed_data_dir)

def load_processed_data(processed_data_dir: str = "~/projects/SLLIM/data/processed") -> Dict[str, List[Dict]]:
    logger.info("Loading processed data...")
    processed_data_dir = os.path.expanduser(processed_data_dir)
    processed_data = {}

    for role in ["train", "dev", "test"]:
        file_path = os.path.join(processed_data_dir, f"{role}.json")
        if os.path.exists(file_path):
            logger.info("Loading file: %s", file_path)
            with open(file_path, "r") as f:
                processed_data[role] = json.load(f)
        else:
            logger.warning("File not found: %s. Initializing as empty.", file_path)
            processed_data[role] = []

    logger.info("Processed data successfully loaded.")
    return processed_data

if __name__ == "__main__":
    try:
        logger.info("Data Loader started.")
        
        # Transform and save raw data to processed format
        consolidate_raw_data()

        # Load processed data for use in fine-tuning or evaluation
        processed_data = load_processed_data()
        logger.info("Process data loader completed")
        print("Processed data: %s", json.dumps(processed_data, indent=4))
        
    except Exception as e:
        logger.error("An error occurred: %s", str(e), exc_info=True)
    finally:
        logger.info("Data Loader execution completed.")
