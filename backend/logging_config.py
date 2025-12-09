# backend/logging_config.py
import logging
import sys
import os
from logging.handlers import RotatingFileHandler

def setup_logging():
    """Configures a rotating file logger and a stream handler."""
    # Define the format
    log_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )

    # Construct an absolute path for the log file inside the 'backend' directory
    log_file_dir = os.path.dirname(os.path.abspath(__file__))
    log_file_path = os.path.join(log_file_dir, "backend.log")

    # Create a rotating file handler
    file_handler = RotatingFileHandler(
        log_file_path, maxBytes=5*1024*1024, backupCount=5
    )
    file_handler.setFormatter(log_formatter)
    file_handler.setLevel(logging.INFO)

    # Create a console handler
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(log_formatter)
    stream_handler.setLevel(logging.INFO)

    # Get the root logger and add the handlers
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Clear existing handlers to avoid duplication
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
        
    root_logger.addHandler(file_handler)
    root_logger.addHandler(stream_handler)
    
    logging.info("Logging configured with file and stream handlers.")