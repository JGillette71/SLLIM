from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from logger import logger 


def download_model(model_name, local_dir):
    """
    Download a Hugging Face model and tokenizer to a local directory.

    Args:
        model_name (str): Hugging Face model name.
        local_dir (str): Path to save the downloaded model and tokenizer.
    """
    logger.info(f"Starting {model_name} download...")
    local_dir = os.path.expanduser(local_dir)
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=local_dir)
        model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=local_dir)
        logger.info(f"Model and tokenizer saved to {local_dir}")
    except Exception as e:
        logger.error("An error occurred during model download: %s", str(e), exc_info=True)
        raise

if __name__ == "__main__":
    model_name = "meta-llama/Llama-3.1-8b"
    local_dir = "~/projects/SLLIM/models/Llama-3.1-8b"
    download_model(model_name, local_dir)
