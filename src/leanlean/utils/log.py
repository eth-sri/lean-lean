import logging
from pathlib import Path

from rich.logging import RichHandler


def _setup_root_logger(log_level = logging.INFO) -> None:
    logger = logging.getLogger("leanlean")
    logger.setLevel(log_level)
    logger.propagate = False
    _handler = RichHandler(
        show_path=False,
        show_time=False,
        show_level=False,
        markup=True,
    )
    _formatter = logging.Formatter("%(name)s: %(levelname)s: %(message)s")
    _handler.setFormatter(_formatter)
    logger.addHandler(_handler)
    return logger


def add_file_handler(path: Path | str, level: int = logging.DEBUG, *, print_path: bool = True) -> None:

    handler = logging.FileHandler(path)
    handler.setLevel(level)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    if print_path:
        print(f"Logging to '{path}'")


logger = _setup_root_logger() 

__all__ = ["logger"]
