"""
tools/logger.py
Централизованная настройка логирования.
"""

import logging
import os
from datetime import datetime

LOGS_DIR = "logs"
os.makedirs(LOGS_DIR, exist_ok=True)

LOG_FILE = os.path.join(LOGS_DIR, "agent.log")

_configured = False


def get_logger(name: str) -> logging.Logger:
    global _configured
    if not _configured:
        fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        datefmt = "%Y-%m-%d %H:%M:%S"

        logging.basicConfig(
            level=logging.INFO,
            format=fmt,
            datefmt=datefmt,
            handlers=[
                logging.FileHandler(LOG_FILE, encoding="utf-8"),
                logging.StreamHandler(),
            ],
        )
        _configured = True

    return logging.getLogger(name)
