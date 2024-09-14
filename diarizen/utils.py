# Licensed under the MIT license.
# Copy from https://github.com/haoxiangsnr/spiking-fullsubnet/blob/main/audiozen/utils.py
# Copyright 2024 Hong Kong Polytechnic University (author: Xiang Hao, haoxiangsnr@gmail.com)

import importlib
import os
import platform
import random
import time
from datetime import timedelta
from typing import Optional

import numpy as np
import psutil
import torch
import torch.distributed as dist
from accelerate import __version__ as version
from torch import Tensor


def print_env():
    pt_version = torch.__version__
    pt_cuda_available = torch.cuda.is_available()

    info = {
        "`Accelerate` version": version,
        "Platform": platform.platform(),
        "Python version": platform.python_version(),
        "Numpy version": np.__version__,
        "PyTorch version (GPU?)": f"{pt_version} ({pt_cuda_available})",
        "System RAM": f"{psutil.virtual_memory().total / 1024 ** 3:.2f} GB",
        "GPU Available": f"{pt_cuda_available}",
        "GPU IDs": f"{torch.cuda.device_count()}",
    }

    if pt_cuda_available:
        info["GPU type"] = torch.cuda.get_device_name()

    return "\n".join([f"- {prop}: {val}" for prop, val in info.items()])


def check_same_shape(est: Tensor, ref: Tensor) -> None:
    if est.shape != ref.shape:
        raise RuntimeError(f"Dimension mismatch: {est.shape=} vs {ref.shape=}.")


class Timer:
    """Count execution time.

    Examples:
        >>> timer = ExecutionTime()  # Start timer
        >>> print(f"Finished in {timer.duration()} seconds.")
    """

    def __init__(self):
        self.start_time = time.perf_counter()

    def duration(self, ndigits=3):
        """Get duration of execution.

        Args:
            ndigits: number of digits to round. Default: 3.
        """
        duration = round(time.perf_counter() - self.start_time, ndigits)
        return duration


def initialize_ddp(rank: int):
    """Initialize the process group"""
    torch.cuda.set_device(rank)

    # torchrun and multi-process distributed (single-node or multi-node) GPU training currently only achieves the best performance using the NCCL distributed backend.
    # The environment variables necessary to initialize a Torch process group are provided to you by this module, and no need for you to pass ``RANK`` manually.
    dist.init_process_group(backend="nccl", timeout=timedelta(seconds=3600))

    print(f"Initialized DistributedDataParallel process group on GPU {rank}.")


def instantiate(path: str, args: Optional[dict] = None, initialize: bool = True):
    """Load module or callable (like function) dynamically based on config string.

    Assume that the config items are as follows:

        [model]
            path = "model.FullSubNetModel"
            [model.args]
            n_frames = 32
            ...

    This function will:
        1. Load the "model" module from python search path.
        2. Load "model.FullSubNetModel" class or callable in the "model" module.
        3. If the "initialize" is set to True, instantiate (or call) class (or callable) with args (in "[model.args]").

    Args:
        path: Target class or callable path.
        args: Named arguments passed to class or callable.
        initialize: whether to initialize with args.

    Returns:
        If initialize is True, return the instantiated class or the return of the call.
        Otherwise, return the found class or callable

    Examples:
        >>> # Use official loss function
        >>> instantiate("torch.nn.CrossEntropyLoss", args={"label_smoothing": 0.2}, initialize=True)
        >>> # Use official optimizer
        >>> instantiate("torch.optim.Adam", args={"lr": 1e-3}, initialize=True)
        >>> # Use custom model in a recipe
        >>> instantiate("fsb.model.FullSubNetModel", args={"n_frames": 32}, initialize=True)
        >>> # Use custom loss function in audiozen
        >>> instantiate("audiozen.loss.CRMLoss", initialize=False)
    """
    # e.g., path = "fsb.model.FullSubNetModel"
    # module_path = "fsb.model"
    # class_or_function_name = "FullSubNetModel"
    splitted_path = path.split(".")

    if len(splitted_path) < 2:
        raise ValueError(f"Invalid path: {path}.")

    module_path = ".".join(splitted_path[:-1])
    class_or_function_name = splitted_path[-1]

    module = importlib.import_module(module_path)
    class_or_function = getattr(module, class_or_function_name)

    if initialize:
        if args:
            return class_or_function(**args)
        else:
            return class_or_function()
    else:
        return class_or_function


def set_random_seed(seed=3407):
    """Set random seed for reproducibility.

    Note:
        This function is used to control the reproducibility of the training process. Why the default value is 3407? It's just a joke.
        See https://arxiv.org/pdf/2109.08203.pdf.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def prepare_empty_dir(dirs, resume=False):
    """Prepare empty dirs.

    If resume a experiment, this function only assert that dirs should be exist.
    If does not resume a experiment, this function will set up new dirs.

    Args:
        dirs: a list of Path objects.
        resume: whether to resume a experiment. Default: False.
    """
    for dir_path in dirs:
        if resume:
            assert dir_path.exists(), "In resume mode, you must have an old experiment dir."
        else:
            dir_path.mkdir(parents=True, exist_ok=True)


def expand_path(path):
    return os.path.abspath(os.path.expanduser(path))


def clamp_inf_value(tensor):
    """Clamp inf value to a large number."""
    max_dtype = torch.finfo(tensor.dtype).max
    clamp_value = torch.where(torch.isinf(tensor).any(), max_dtype - 1000, tensor)
    tensor = torch.clamp(tensor, min=-clamp_value, max=clamp_value)
    return tensor
