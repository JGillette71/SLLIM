import logging
from pathlib import Path

# Set up log directory and file
LOG_DIR = Path("~/projects/SLLIM/data/logs").expanduser()
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "project.log"

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set log level
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),  # Log to file
        logging.StreamHandler(),       # Log to console
    ],
)

# Create a logger instance
logger = logging.getLogger("SLLIM")
logger.info("Logger initialized. Logs will be written to %s", LOG_FILE)
