"""
Centralized logging configuration for the DBA agent system.

Call setup_logging() ONCE, as early as possible in your entrypoint
(e.g. top of Data/Seed_Data.py, or wherever your main graph is invoked).
Do NOT call logging.basicConfig() anywhere else in the project — the
first call to basicConfig() in a process wins, and any later calls
(in other modules) are silently ignored. This centralizes it so that
doesn't happen by accident.
"""

import logging
import os
from logging.handlers import RotatingFileHandler

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
LOG_FILE = os.path.join(LOG_DIR, "dba_agent.log")


def setup_logging(level: int = logging.INFO) -> None:
    os.makedirs(LOG_DIR, exist_ok=True)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    )

    # Console output
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # File output — rotates at 5MB, keeps 3 backups, so it doesn't grow forever
    file_handler = RotatingFileHandler(
        LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)

    # Avoid duplicate handlers if setup_logging() is accidentally called twice
    root_logger.handlers.clear()
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    logging.getLogger(__name__).info(f"Logging initialized. Writing to {LOG_FILE}")