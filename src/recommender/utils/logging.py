import logging

from ..config import LOG_FORMAT


def get_logger(name: str) -> logging.Logger:
    """
    Create and return a logger with standardized formatting.

    Parameters:
        name: name of the logger (e.g. __name__)
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(LOG_FORMAT)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
