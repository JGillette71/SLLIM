import os
import json
from typing import Dict, List

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
    # Resolve paths in args
    raw_data_dir = os.path.expanduser(raw_data_dir)
    processed_data_dir = os.path.expanduser(processed_data_dir)

    # Ensure the processed data directory exists
    os.makedirs(processed_data_dir, exist_ok=True)

    consolidated_data = {"train": [], "dev": [], "test": []}

    # Iterate over raw dataset folders
    for dataset_name in os.listdir(raw_data_dir):
        dataset_path = os.path.join(raw_data_dir, dataset_name)
        if not os.path.isdir(dataset_path):
            continue

        # Consolidate data for each role (train, dev, test)
        for role in ["train", "dev", "test"]:
            raw_file_path = os.path.join(dataset_path, f"qa.json.{role}")
            if os.path.exists(raw_file_path):
                with open(raw_file_path, "r") as f:
                    raw_data = [json.loads(line) for line in f.readlines()]

                # Transform raw data into a consistent format
                for item in raw_data:
                    transformed_item = {
                        "question": item["Question"],
                        "answer": item["Answer"],
                        "context": item["RawLog"],
                    }
                    consolidated_data[role].append(transformed_item)

    # Save the consolidated data
    for role, data in consolidated_data.items():
        output_file_path = os.path.join(processed_data_dir, f"{role}.json")
        with open(output_file_path, "w") as f:
            json.dump(data, f, indent=4)

    print(f"Data transformation complete. Processed data saved to: {processed_data_dir}")

def load_processed_data(processed_data_dir: str = "~/projects/SLLIM/data/processed") -> Dict[str, List[Dict]]:
    """
    Load processed train, test, and dev JSON files and return them as dictionaries.

    Args:
        processed_data_dir (str): Directory containing processed JSON files.

    Returns:
        Dict[str, List[Dict]]: Dictionary containing train, dev, and test data.
    """
    # Resolve path
    processed_data_dir = os.path.expanduser(processed_data_dir)

    processed_data = {}
    for role in ["train", "dev", "test"]:
        file_path = os.path.join(processed_data_dir, f"{role}.json")
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                processed_data[role] = json.load(f)
        else:
            processed_data[role] = []

    return processed_data

# Example usage
if __name__ == "__main__":
    
    # Transform and save raw data to processed format
    consolidate_raw_data()

    # Load processed data for use in fine-tuning or evaluation
    processed_data = load_processed_data()
    print(json.dumps(processed_data, indent=4))
