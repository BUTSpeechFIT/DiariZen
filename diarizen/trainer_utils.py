# Licensed under the MIT license.
# Copy from https://github.com/haoxiangsnr/spiking-fullsubnet/blob/main/audiozen/trainer_utils.py
# Copyright 2024 Hong Kong Polytechnic University (author: Xiang Hao, haoxiangsnr@gmail.com)

import numpy as np
import torch
from accelerate.utils import set_seed


def seed_worker(_):
    """Helper function to set worker seed during Dataloader initialization.

    In recent check-ins, we may have no longer needed this function because PyTorch has already set the worker seed
    for numpy and random. But there is no adverse effect to keeping this function, since the initial_seed is
    inner_seed + worker_ids.
    """
    worker_seed = torch.initial_seed() % 2**32
    set_seed(worker_seed)


def has_length(dataset):
    """
    Checks if the dataset implements __len__() and it doesn't raise an error
    """
    try:
        return len(dataset) is not None
    except TypeError:
        # TypeError: len() of unsized object
        return False


class TrainerState:
    def __init__(self, save_max_score) -> None:
        self.epochs_trained = 0
        self.steps_trained = 0

        self.patience = 0

        self.best_score = -np.inf if save_max_score else np.inf
        self.best_score_epoch = 0

    def load_state_dict(self, state_dict: dict) -> None:
        self.epochs_trained = state_dict["epochs_trained"]
        self.steps_trained = state_dict["steps_trained"]

        self.best_score = state_dict["best_score"]
        self.best_score_epoch = state_dict["best_score_epoch"]

        self.patience = state_dict["patience"]

    def state_dict(self) -> dict:
        return {
            "epochs_trained": self.epochs_trained,
            "steps_trained": self.steps_trained,
            "patience": self.patience,
            "best_score": self.best_score,
            "best_score_epoch": self.best_score_epoch,
        }
