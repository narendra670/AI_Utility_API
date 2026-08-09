import logging
import sys

from config import get_settings

settings = get_settings()


def get_logger(name: str = "ai-utility-api") -> logging.Logger:
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(settings.log_format))
        logger.addHandler(handler)
        logger.setLevel(settings.log_level.upper())
        logger.propagate = False

    return logger


logger = get_logger()
