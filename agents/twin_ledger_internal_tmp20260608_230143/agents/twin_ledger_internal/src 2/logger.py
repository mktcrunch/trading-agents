"""
Unified logging configuration for all agents
"""
import logging
import logging.handlers
import os
from pathlib import Path
from src import config


def _cloud_run_mode() -> bool:
    return bool(os.getenv("K_SERVICE")) or os.getenv("LOG_TO_FILE", "true").lower() == "false"


def setup_logger(name: str, log_file: Path = None) -> logging.Logger:
    """
    Setup a logger with both file and console handlers

    Args:
        name: Logger name (typically __name__)
        log_file: Optional custom log file path

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, config.LOG_LEVEL))

    # Remove existing handlers to avoid duplicates
    logger.handlers = []

    formatter = logging.Formatter(config.LOG_FORMAT)
    level = getattr(logging, config.LOG_LEVEL)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if not _cloud_run_mode():
        log_path = log_file or config.LOG_FILE
        file_handler = logging.handlers.RotatingFileHandler(
            log_path,
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


if not _cloud_run_mode():
    logger = setup_logger("mktcrunch_agents")
