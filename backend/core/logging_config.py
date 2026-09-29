"""
Application-wide logging configuration.

Called once, at startup, from backend/main.py.
"""
import logging
import sys


def configure_logging(debug: bool = False) -> None:
    level = logging.DEBUG if debug else logging.INFO

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Avoid duplicate handlers if configure_logging() is called more than once
    # (e.g. under uvicorn --reload).
    root_logger.handlers.clear()
    root_logger.addHandler(handler)

    # Quiet down noisy third-party loggers a little.
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(
        logging.INFO if debug else logging.WARNING
    )
