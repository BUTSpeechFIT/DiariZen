# Licensed under the MIT license.
# Copy from https://github.com/haoxiangsnr/spiking-fullsubnet/blob/main/audiozen/logger.py
# Copyright 2024 Hong Kong Polytechnic University (author: Xiang Hao, haoxiangsnr@gmail.com)

import logging
import os
import time
from pathlib import Path

import toml
from torch.utils.tensorboard import SummaryWriter  # type: ignore


class TensorboardLogger(SummaryWriter):
    def __init__(self, log_dir: str = "") -> None:
        super().__init__(log_dir=log_dir, max_queue=5, flush_secs=30)

    def log_config(self, config: dict) -> None:
        self.add_text(
            tag="Configuration",
            text_string=f"<pre>  \n{toml.dumps(config)}  \n</pre>",
            global_step=1,
        )


def init_logging_logger(config):
    """Initialize logging logger with handlers.

    Args:
        log_fpath: Path to save log file.

    Examples:
        >>> # Call this function at the beginning of main file.
        >>> init_logger(log_fpath="log_path")
        >>> # Use this logger in other modules.
        >>> import logging
        >>> logger = logging.getLogger(__name__)
        >>> logger.info("info message")
    """
    # Parse log_fpath
    log_dir: Path = Path(config["meta"]["save_dir"]).expanduser().absolute() / config["meta"]["exp_id"]
    log_dir.mkdir(parents=True, exist_ok=True)

    # disable logging for libraries that use standard logging module
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("numba").setLevel(logging.ERROR)

    # Create logger
    logger = logging.getLogger()

    # Set the lowest level of root logger and controls logging via handlers' level
    logger.setLevel(logging.DEBUG)

    # Get log level from environment variable
    log_level = os.environ.get("LOG_LEVEL", "INFO").upper()

    # Create a console handler and set level to info
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level=log_level)

    # Create a file handler and set the logger level to debug
    time_now = time.strftime("%Y_%m_%d--%H_%M_%S")
    file_handler = logging.FileHandler(str(log_dir / f"{config['meta']['exp_id']}_{time_now}.log"))
    file_handler.setLevel(level=log_level)

    # Create formatters (file logger have more info)
    console_formatter = logging.Formatter(
        "%(asctime)s: %(message)s",
        datefmt="%m-%d %H:%M:%S",
    )
    file_formatter = logging.Formatter(
        "%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d]: %(message)s",
        datefmt="%m-%d %H:%M:%S",
    )

    # Add formatter to ch
    console_handler.setFormatter(console_formatter)
    file_handler.setFormatter(file_formatter)

    # Add ch to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    logger.info(f"Initialized logger with log file in {log_dir.as_posix()}.")
