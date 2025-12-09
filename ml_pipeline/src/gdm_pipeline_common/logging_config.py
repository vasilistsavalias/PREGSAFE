import sys
from loguru import logger
from pathlib import Path

def setup_logging(log_file_path: Path):
    """
    Configures the Loguru logger to output to both console and a file.
    """
    log_file_path.parent.mkdir(parents=True, exist_ok=True)

    logger.remove()  # Remove default handler

    # Console logger
    logger.add(
        sys.stderr,
        level="DEBUG",
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        )
    )

    # File logger
    logger.add(
        log_file_path,
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
        rotation="10 MB",
        retention="7 days",
        enqueue=True,
        backtrace=True,
        diagnose=True
    )
    logger.info("Logger has been configured.")
    return logger
