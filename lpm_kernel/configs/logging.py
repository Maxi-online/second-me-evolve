import os
import sys
import logging
import logging.config
import logging.handlers
import datetime
import shutil

# Get project root directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

# Define log directories
LOG_BASE_DIR = os.path.join(PROJECT_ROOT, "logs")
TRAIN_LOG_DIR = os.path.join(LOG_BASE_DIR, "train")

# Define log file paths
APP_LOG_FILE = os.path.join(LOG_BASE_DIR, "app.log")
TRAIN_LOG_FILE = os.path.join(TRAIN_LOG_DIR, "train.log")

# Function to rename log file if it exists
def rename_existing_log_file():
    if os.path.exists(TRAIN_LOG_FILE):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_filename = f"train_{timestamp}.log"
        backup_path = os.path.join(TRAIN_LOG_DIR, backup_filename)
        
        shutil.move(TRAIN_LOG_FILE, backup_path)
        print(f"Existing train.log renamed to {backup_filename}")
        return True
    return False

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "standard",
            "stream": sys.stdout,
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "INFO",
            "formatter": "standard",
            "filename": APP_LOG_FILE,
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
            "encoding": "utf-8",
        },
        "train_process_file": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "INFO",
            "formatter": "standard",
            "filename": TRAIN_LOG_FILE,
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
            "encoding": "utf-8",
        },
    },
    "loggers": {
        "train_process": {
            "level": "INFO",
            "handlers": ["train_process_file", "console"],
            "propagate": False,
        },
    },
    "root": {  # root logger configuration
        "level": "INFO",
        "handlers": ["console", "file"],
    },
}

# Initialize logging configuration
def setup_logging():
    logging.config.dictConfig(LOGGING_CONFIG)

# Get train process logger
def get_train_process_logger():
    return logging.getLogger("train_process")
