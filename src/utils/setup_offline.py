import os
import torch
from logger import logger 
import transformers

def download_model(model_name, local_dir):
    """
    Download and save a Hugging Face model and tokenizer for offline use.

    Args:
        model_name (str): Hugging Face model name.
        local_dir (str): Directory to save the downloaded tokenizer and model shards.
    """
    local_dir = os.path.expanduser(local_dir)
    os.makedirs(local_dir, exist_ok=True)  # Ensure the target directory exists

    try:
        logger.info(f"Starting download of {model_name}...")

        # Check if GPU is available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")

        # Download and save the tokenizer
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_name,
            token=os.getenv("HF_AUTH_TOKEN"),
            cache_dir=local_dir
        )
        tokenizer.save_pretrained(local_dir)
        logger.info(f"Tokenizer saved to {local_dir}")

        # Download the model and move it to GPU for efficient memory usage
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_name,
            token=os.getenv("HF_AUTH_TOKEN"),
            cache_dir=local_dir,
            device_map="auto",  # maps model shards to GPU if available
            quantization_config=transformers.BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type='nf4'
            )
        )
        logger.info(f"Model shards downloaded to {local_dir}")

    except Exception as e:
        logger.error("An error occurred while downloading the model: %s", str(e), exc_info=True)
        raise
    
if __name__ == "__main__":
    model_name = "meta-llama/Llama-3.1-8b"
    local_dir = "~/projects/SLLIM/models/Llama-3.1-8b"
    download_model(model_name, local_dir)

