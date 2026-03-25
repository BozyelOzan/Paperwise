"""
Structured JSON logger with rotating file handlers.
Outputs INFO+ to stdout, DEBUG+ to app.log, ERROR+ to error.log.
"""

import logging
import os
from logging.handlers import RotatingFileHandler

from pythonjsonlogger import jsonlogger

_LOG_DIR = "logs"
_MAX_BYTES = 5 * 1024 * 1024  # 5 MB
_BACKUP_COUNT = 3
_FMT = "%(asctime)s %(levelname)s %(name)s %(funcName)s %(message)s"
_DATE_FMT = "%Y-%m-%dT%H:%M:%S"


def setup_logger(name: str) -> logging.Logger:
    """Return a named logger. Safe to call multiple times — handlers are not duplicated."""
    os.makedirs(_LOG_DIR, exist_ok=True)

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)
    formatter = jsonlogger.JsonFormatter(fmt=_FMT, datefmt=_DATE_FMT)

    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    stream.setLevel(logging.INFO)

    app_handler = RotatingFileHandler(
        f"{_LOG_DIR}/app.log",
        maxBytes=_MAX_BYTES,
        backupCount=_BACKUP_COUNT,
        encoding="utf-8",
    )
    app_handler.setFormatter(formatter)
    app_handler.setLevel(logging.DEBUG)

    error_handler = RotatingFileHandler(
        f"{_LOG_DIR}/error.log",
        maxBytes=_MAX_BYTES,
        backupCount=_BACKUP_COUNT,
        encoding="utf-8",
    )
    error_handler.setFormatter(formatter)
    error_handler.setLevel(logging.ERROR)

    logger.addHandler(stream)
    logger.addHandler(app_handler)
    logger.addHandler(error_handler)
    return logger
