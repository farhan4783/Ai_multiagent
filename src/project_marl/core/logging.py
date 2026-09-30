"""
Standard structured logging configuration for the project_marl system.
"""

import logging
import sys


def setup_logger(name: str = "project_marl", level: int = logging.INFO) -> logging.Logger:
    """Configures and returns a consistent console logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-7s | [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
    return logger
