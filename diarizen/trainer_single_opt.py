# Licensed under the MIT license.
# Copyright 2024 Hong Kong Polytechnic University (author: Xiang Hao, haoxiangsnr@gmail.com)
# Copyright 2024 Brno University of Technology (author: Jiangyu Han, ihan@fit.vut.cz)

import math
import shutil
import sys
import time
from functools import partial
from pathlib import Path

import json
from os.path import join

import librosa
import pandas as pd
import toml
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from torch.utils.data import DataLoader
from torchinfo import summary
from tqdm.auto import tqdm

from diarizen.logger import TensorboardLogger
from diarizen.optimization import get_constant_schedule_with_warmup, get_linear_schedule_with_warmup
from diarizen.trainer_utils import TrainerState
from diarizen.utils import prepare_empty_dir, print_env

from diarizen.noam_updater import get_rate

from torch.optim.lr_scheduler import ReduceLROnPlateau, OneCycleLR

logger = get_logger(__name__)


class Trainer:
    def __init__(
        self,
        accelerator: Accelerator,
        config,
        resume,
        model,
        optimizer,
    ):
        """Create an instance of BaseTrainer for training, validation, and fine-tuning."""
        self.config = config
        self.resume = resume

        # Setup directories
        self._initialize_exp_dirs_and_paths(config)

        # GPU
        self.accelerator = accelerator
        self.rank = accelerator.device
        self.device = accelerator.device  # alias of rank

        # Model
        self.model = model
        self.optimizer = optimizer

        # Unwrap_Model
        self.unwrap_model = self.accelerator.unwrap_model(self.model)

        # Trainer.train args
        self.trainer_config = config["trainer"]["args"]
        self.debug = self.trainer_config.get("debug", False)
        self.max_steps = self.trainer_config.get("max_steps", 0)
        self.max_epochs = self.trainer_config.get("max_epochs", sys.maxsize)
        self.max_grad_norm = self.trainer_config.get("max_grad_norm", 0)
        self.save_max_score = self.trainer_config.get("save_max_score", True)
        self.save_ckpt_interval = self.trainer_config.get("save_ckpt_interval", 1)
        self.max_patience = self.trainer_config.get("max_patience", 10)
        self.plot_norm = self.trainer_config.get("plot_norm", True)
        self.plot_lr = self.trainer_config.get("plot_lr", False)
        self.validation_interval = self.trainer_config.get("validation_interval", 1)
        self.max_num_checkpoints = self.trainer_config.get("max_num_checkpoints", 10)
        self.scheduler_name = self.trainer_config.get("scheduler_name", "constant_schedule_with_warmup")
        self.warmup_steps = self.trainer_config.get("warmup_steps", 0)
        self.warmup_ratio = self.trainer_config.get("warmup_ratio", 0.0)
        self.gradient_accumulation_steps = self.trainer_config.get("gradient_accumulation_steps", 1)

        self.validation_before_training = self.trainer_config.get("validation_before_training", False)

        self.lr_decay = self.trainer_config.get("lr_decay", False)
        self.lr_decay_patience = self.trainer_config.get("lr_decay_patience", 2)

        self.use_one_cycle_lr = self.trainer_config.get("use_one_cycle_lr", False)

        self.gradient_percentile = self.trainer_config.get("gradient_percentile", 10)
        self.gradient_history_size = self.trainer_config.get("gradient_history_size", 1000)

        # wavlm
        self.freeze_wavlm = self.trainer_config.get("freeze_wavlm", False)
        # wavlm 
        if self.freeze_wavlm:
            logger.info("Freeze WavLM...")
            self.unwrap_model.freeze_by_name('wavlm_model')

        # Dataset
        self.dataset_config = config["train_dataset"]["args"]
        self.chunk_size = self.dataset_config.get("chunk_size", 500)

        # Finetune
        self.finetune_config = config["finetune"]
        self.finetune = self.finetune_config["finetune"]
        self.init_epochs = self.finetune_config.get("init_epochs", " ")
        self.ckpt_path = self.finetune_config.get("ckpt_path", " ")

        if self.max_steps > 0:
            logger.info(f"`max_steps` is set to {self.max_steps}. Ignoring `max_epochs`.")

        if self.validation_interval < 1:
            logger.info(f"`validation_interval` is set to {self.validation_interval}. It must be >= 1.")

        # Trainer states
        self.state = TrainerState(save_max_score=self.save_max_score)
        self.accelerator.register_for_checkpointing(self.state)  # Register accelerate objects

        # Others
        pd.set_option("display.float_format", lambda x: "%.3f" % x)

        if self.accelerator.is_local_main_process:
            prepare_empty_dir(
                [
                    self.save_dir,
                    self.exp_dir,
                    self.checkpoints_dir,
                    self.tb_log_dir,
                ],
                resume=resume,
            )

        self.writer = TensorboardLogger(self.tb_log_dir.as_posix())
        self.writer.log_config(config)

        with open(self.config_path.as_posix(), "w") as handle:
            toml.dump(config, handle)

        logger.info(f"Configuration file is saved to {self.config_path.as_posix()}.")

        logger.info(f"Environment information:\n{print_env()}")

        # Model summary
        logger.info(f"\n {summary(self.model, verbose=0)}")

    def _run_early_stop_check(self, score: float):
        should_stop = False

        if self._check_improvement(score, save_max_score=self.save_max_score):
            self.state.best_score = score
            self.state.best_score_epoch = self.state.epochs_trained
            self._save_checkpoint(self.state.epochs_trained, is_best_epoch=True)
            self.state.patience = 0
            logger.info(f"Found new best score: {score:.4f}, saving checkpoint...")
        else:
            logger.info(
                f"Score did not improve from {self.state.best_score:.4f} at epoch {self.state.best_score_epoch}."
            )
            self.state.patience += 1
            logger.info(f"Early stopping counter: {self.state.patience} out of {self.max_patience}")

            if self.state.patience >= self.max_patience:
                logger.info("Early stopping triggered, stopping training...")
                should_stop = True

        return should_stop

    @staticmethod
    def _get_time_now():
        return time.strftime("%Y_%m_%d--%H_%M_%S")

    def _initialize_exp_dirs_and_paths(self, config):
        """Initialize directories.

        Args:
            save_dir: the root directory to save all experiments.
            exp_id: the experiment id.

        Notes:
            - save_dir: /home/xhao/exp
            - exp_dir: /home/xhao/exp/fullsubnet_lr_0.1
            - checkpoints_dir: /home/xhao/exp/fullsubnet_lr_0.1/checkpoints
            - tb_log_dir: /home/xhao/exp/fullsubnet_lr_0.1/tb_log
            - src_source_code_dir: /home/xhao/diarizen
            - source_code_backup_dir: /home/xhao/exp/fullsubnet_lr_0.1/source_code__2023_01_07__17_19_57
            - config_path: /home/xhao/exp/fullsubnet_lr_0.1/config__2023_01_07__17_19_57.toml
        """
        self.save_dir = Path(config["meta"]["save_dir"]).expanduser().absolute()
        self.exp_dir = self.save_dir / config["meta"]["exp_id"]

        self.checkpoints_dir = self.exp_dir / "checkpoints"
        self.tb_log_dir = self.exp_dir / "tb_log"

        # Each run will have a unique source code, config, and log file.
        time_now = self._get_time_now()
        self.source_code_dir = Path(__file__).expanduser().absolute().parent.parent.parent
        self.source_code_backup_dir = self.exp_dir / f"source_code__{time_now}"
        self.config_path = self.exp_dir / f"config__{time_now}.toml"

    def _find_latest_ckpt_path(self):
        """Find the latest checkpoint path."""
        # Pick up all checkpoints with the format `epoch_*`
        checkpoints = sorted(self.checkpoints_dir.glob("epoch_" + ("[0-9]" * 4)))

        # Remove files that is not a checkpoint
        checkpoints = [ckpt for ckpt in checkpoints if ckpt.is_dir()]

        if len(checkpoints) == 0:
            raise FileNotFoundError(f"No checkpoints found in {self.checkpoints_dir.as_posix()}.")

        # Pick up the latest checkpoint
        ckpt_path = checkpoints[-1]

        return ckpt_path

    def _load_checkpoint(self, ckpt_path):
        """load a checkpoint from the checkpints directory.

        Args:
            ckpt_path: "best", "latest", or a path to a checkpoint file
        """
        if ckpt_path == "best":
            ckpt_path = self.checkpoints_dir / "best"
        elif ckpt_path == "latest":
            ckpt_path = self._find_latest_ckpt_path()
        else:
            ckpt_path = Path(ckpt_path).expanduser().absolute()

        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint {ckpt_path.as_posix()} not found.")

        self.accelerator.load_state(ckpt_path, map_location="cpu")

        logger.info(f"Checkpoint on epoch {self.state.epochs_trained} is loaded.")

    def _save_checkpoint(self, epoch, is_best_epoch):
        """Save checkpoint.

        Args:
            epoch: the current epoch.
            is_best_epoch: whether the current epoch is the best epoch.
        """
        # Save checkpoint
        if is_best_epoch:
            self.accelerator.save_state(self.checkpoints_dir / "best", safe_serialization=False)
        else:
            # Regular checkpoint
            ckpt_path = self.checkpoints_dir / f"epoch_{str(epoch).zfill(4)}"
            self.accelerator.save_state(ckpt_path.as_posix(), safe_serialization=False)

        # Find all regular checkpoints and only keep the latest `max_num_checkpoints` regular checkpoints
        checkpoints = sorted(self.checkpoints_dir.glob("epoch_*"))

        if epoch <= len(checkpoints):
            logger.warning(
                f"Current epoch is {epoch}, but found {len(checkpoints)} checkpoints. "
                f"This may be caused by you running the same experiment multiple times. "
                f"Recommend to run the experiment with a different `exp_id`."
            )

        if len(checkpoints) > self.max_num_checkpoints:
            logger.info(
                f"Found {len(checkpoints)} checkpoints, only keeping the latest {self.max_num_checkpoints} checkpoints."
            )
            for checkpoint_dir in checkpoints[: -self.max_num_checkpoints]:
                shutil.rmtree(checkpoint_dir.as_posix())
                logger.info(f"Checkpoint {checkpoint_dir.as_posix()} is removed.")


    @staticmethod
    def get_warmup_steps(warmup_steps, max_steps, warmup_ratio):
        if warmup_steps > 0:
            logger.info(f"warmup_steps={warmup_steps}. warmup_ratio will be ignored.")
            return warmup_steps
        else:
            return math.ceil(max_steps * warmup_ratio)

    def create_warmup_scheduler(self, optimizer, scheduler_name, max_steps: int):
        num_warmup_steps = self.get_warmup_steps(self.warmup_steps, max_steps, self.warmup_ratio)
        if scheduler_name == "constant_schedule_with_warmup":
            return get_constant_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=num_warmup_steps)
        elif scheduler_name == "linear_schedule_with_warmup":
            return get_linear_schedule_with_warmup(
                optimizer=optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=max_steps
            )

    def create_schedulers(self, max_steps: int):
        """Create schedulers.

        You can override this method to create your own schedulers. For example, in GAN training, you may want to
        create two schedulers for the generator and the discriminator.

        Args:
            max_steps: the maximum number of steps to train.
        """
        self.lr_scheduler = self.create_warmup_scheduler(
            optimizer=self.optimizer, scheduler_name=self.scheduler_name, max_steps=max_steps
        )
        self.lr_schedule = self.accelerator.prepare(self.lr_scheduler)

    def set_models_to_train_mode(self):
        """Set models to train mode.

        You can override this method to set your own models to train mode. For example, in GAN training, you may want to
        set the generator and the discriminator to train mode.
        """
        self.model.train()

    def set_models_to_eval_mode(self):
        self.model.eval()

    def get_optimizer_lr(self, optimizer):
        return optimizer.state_dict()['param_groups'][0]['lr']

    def lr_scheduler_step(self):
        """Step the lr scheduler.

        You can override this method to step your own lr scheduler. For example, in GAN training, you may want to
        step the lr scheduler of the generator and the discriminator.
        """
        self.lr_scheduler.step(self.state.steps_trained)

    def create_lr_decay_scheduler(self):
        self.lr_decay_scheduler = ReduceLROnPlateau(
            optimizer=self.optimizer,
            mode='min',
            factor=0.95,
            patience=self.lr_decay_patience,
            min_lr=1e-8
        )
        self.lr_decay_scheduler = self.accelerator.prepare(self.lr_decay_scheduler)

    def create_lr_one_cycle_scheduler(self, max_steps):
        self.lr_one_cycle_scheduler = OneCycleLR(
            optimizer=self.optimizer,
            max_lr=self.get_optimizer_lr(self.optimizer),
            total_steps=max_steps
        )
        self.lr_one_cycle_scheduler = self.accelerator.prepare(self.lr_one_cycle_scheduler)

    def create_bar_desc(self, loss_dict, norm):
        bar_desc = ""
        for k, v in loss_dict.items():
            bar_desc += f"{k}: {(v):.4f}, "
        bar_desc += f"norm: {norm:.4f}, " f"lr: {self.lr_scheduler.get_last_lr()[-1]:.10f}"

        # plot norm
        if self.plot_norm:
            self.writer.add_scalar("Train_Step/norm", norm, self.state.steps_trained)

        if self.plot_lr:
            self.writer.add_scalar("Train_Step/lr", self.lr_scheduler.get_last_lr()[-1], self.state.steps_trained)

        return bar_desc

    def train(self, train_dataloader: DataLoader, validation_dataloader):
        """Train loop entry point.

        Args:
            train_dataloader: the dataloader to train.
            validation_dataloades: the dataloader(s) to validate.

        Notes:
            You are responsible for calling ``.backward()``, ``.step()``, and ``.zero_grad()`` in your implementation
            of `training_step()`. Accelerate will automatically handle the gradient accumulation for you.
            It means that in gradient accumulation, the step() of optimizer and scheduler is called only when gradient_accumulation_steps is reached.

            The training step is implemented as follows:

            .. code-block:: python

                    self.optimizer.zero_grad()
                    loss = training_step(batch, batch_idx)
                    self.accelerator.backward(loss)
                    self.optimizer.step()

                    return {
                        "loss": loss,
                    }
        """
        early_stop_mark = torch.zeros(1, device=self.device)

        # Setting up training control variables
        steps_per_epoch = len(train_dataloader)
        update_steps_per_epoch = steps_per_epoch // self.gradient_accumulation_steps
        update_steps_per_epoch = max(update_steps_per_epoch, 1)

        if self.max_steps > 0:
            max_steps = self.max_steps
            max_epochs = self.max_steps // update_steps_per_epoch + int(self.max_steps % update_steps_per_epoch > 0)
        else:
            max_steps = self.max_epochs * update_steps_per_epoch
            max_epochs = self.max_epochs

        logger.info("Training control variables:")
        logger.info(f"`steps_per_epoch`: {steps_per_epoch}")
        logger.info(f"Gradient accumulation steps: {self.gradient_accumulation_steps}")
        logger.info(f"`update_steps_per_epoch`: {update_steps_per_epoch}")
        logger.info(f"`max_steps`: {max_steps}")
        logger.info(f"`max_epochs`: {max_epochs}")

        # Generator learning rate scheduler
        if self.warmup_steps > 0:
            self.create_schedulers(max_steps=max_steps)
        if self.use_one_cycle_lr:
            self.create_lr_one_cycle_scheduler(max_steps=max_steps * self.accelerator.num_processes)

        # Resume
        if self.resume:
            self._load_checkpoint(ckpt_path="latest")

        # validation 0 epoch performance
        if self.validation_before_training:
            with torch.no_grad():
                logger.info("Validation on ZERO epoch...")
                score = self.validate(validation_dataloader)
            if self.accelerator.is_local_main_process:
                self._save_checkpoint(epoch=0, is_best_epoch=False)

        for epoch in range(self.state.epochs_trained + 1, max_epochs + 1):
            logger.info(f"{'=' * 9} Epoch {epoch} out of {max_epochs} {'=' * 9}")
            logger.info("Begin training...")

            self.set_models_to_train_mode()
            if self.freeze_wavlm:
                self.unwrap_model.wavlm_model.eval()

            training_epoch_output = []

            # the iter number of progress bar increments by 1 by default whether gradient accumulation is used or not.
            # but we update the description of the progress bar only when the gradients are synchronized across all processes.
            dataloader_bar = tqdm(
                train_dataloader,
                desc="",
                dynamic_ncols=True,
                bar_format="{l_bar}{r_bar}",
                colour="green",
                disable=not self.accelerator.is_local_main_process,
                position=0,
                leave=True,
            )

            for batch_idx, batch in enumerate(dataloader_bar):
                # accumulate() will automatically skip synchronization if applicable loss is linearly scaled with the optimizer.grad
                # accumulate() will automatically divide the loss in backward by the number of gradient accumulation steps
                # However, it won't return this loss, so we need to manually divide the loss by the number of gradient accumulation steps.
                with self.accelerator.accumulate(self.model):
                    # You are responsible for calling `.backward()`, `.step()`, and `.zero_grad()` in your implementation
                    loss_dict = self.training_step(batch, batch_idx)
                    training_epoch_output.append(loss_dict)

                    if not self.accelerator.optimizer_step_was_skipped:
                        if self.warmup_steps > 0:
                            self.lr_scheduler_step()

                        if self.use_one_cycle_lr:
                            self.lr_one_cycle_scheduler.step() 

                self.state.steps_trained += 1
            self.state.epochs_trained += 1
            self.training_epoch_end(training_epoch_output)

            # Should save, evaluate, and early stop?
            if self.accelerator.is_local_main_process and epoch % self.save_ckpt_interval == 0:
                self._save_checkpoint(epoch, is_best_epoch=False)
    
            if epoch % self.validation_interval == 0:
                with torch.no_grad():
                    logger.info("Training finished, begin validation...")
                    score = self.validate(validation_dataloader)

                    if self.accelerator.is_local_main_process:

                        if self.lr_decay:
                            self.lr_decay_scheduler.step(score)

                        should_stop = self._run_early_stop_check(score)
                        if should_stop:
                            early_stop_mark += 1

                    logger.info("Validation finished.")

            self.accelerator.wait_for_everyone()

            # Reduces the `early_stop_mark` data across all processes
            # If `early_stop_mark` is 1 in any process, then `reduce_early_stop_mark` will be 1 in all processes.
            reduced_early_stop_mark = self.accelerator.reduce(early_stop_mark, reduction="sum")

            # If any process triggers early stopping, stop training
            if reduced_early_stop_mark != 0:
                break
        

    @torch.no_grad()
    def validate(self, dataloader):
        """Validate the model.

        Args:
            dataloaders: the dataloader(s) to validate.

        Returns:
            score: the metric score of the validation epoch.
        """
        logger.info("Begin validation...")

        self.set_models_to_eval_mode()

        validation_output = []
        for batch_idx, batch in enumerate(
            tqdm(
                dataloader,
                desc="",
                bar_format="{l_bar}{r_bar}",
                dynamic_ncols=True,
                disable=not self.accelerator.is_local_main_process,
            )
        ):
            # We recommend you directly calculate the metric score in the validation_step function and return the
            # metric score in the validation_step function, and then calculate the mean of the metric score
            # in the validation_epoch_end function.
            step_output = self.validation_step(batch, batch_idx)

            """
            {
                "metric_1": metric_1_score,
                "metric_2": metric_1_score,
                ...
            }
            """
            gathered_step_output = self.accelerator.gather_for_metrics(step_output)
            validation_output.append(gathered_step_output)

        logger.info("Validation inference finished, begin validation epoch end...")

        if self.accelerator.is_local_main_process:
            # only the main process will run validation_epoch_end
            score = self.validation_epoch_end(validation_output)
            return score
        else:
            return None

    def _check_improvement(self, score, save_max_score=True):
        """Check if the current model got the best metric score"""
        if save_max_score:
            if score > self.state.best_score:
                return True
            else:
                return False
        else:
            if score < self.state.best_score:
                return True
            else:
                return False

    def training_step(self, batch, batch_idx):
        """Implement a training step.

        Implement your own training step here.
        The input batch is from a training dataloader and the output of this function should be a loss tensor.
        Here is the persuade code for training a model:

        .. code-block:: python
            :emphasize-lines: 7

            for epoch in range(start_epoch, end_epoch):
                self.model.train()

                training_epoch_output = []
                for batch, batch_index in dataloader:
                    zero_grad()
                    loss = training_step(batch, batch_idx)
                    loss.backward()
                    optimizer.step()

                training_epoch_output.append(loss)
                training_epoch_end(training_epoch_output)

                save_checkpoint()

                if some_condition:
                    score = validate()
                    if score > best_score:
                        save_checkpoint(best=True)


        Args:
            batch: a batch of data, which passed from a custom training dataloader.
            batch_idx: the index of the current batch.

        Returns:
            loss: the loss of the batch.
        """
        raise NotImplementedError

    def training_epoch_end(self, training_epoch_output):
        """Implement the logic of the end of a training epoch. Please override this function if you want to do something.

        When the training epoch ends, this function will be called. The input is a list of the loss dict of each step
        in a training epoch. You may want to log the epoch-level training loss here.

        .. code-block:: python
            for epoch in range(start_epoch, end_epoch):
                self.model.train()

                training_epoch_output = []
                for batch, batch_index in dataloader:
                    loss = training_step(batch, batch_idx)
                    training_epoch_output.append(loss)

                training_epoch_end(training_epoch_output)

                save_checkpoint()

                if some_condition:
                    score = validate()
                    if score > best_score:
                        save_checkpoint(best=True)

        Args:
            training_epoch_output: the output of the training epoch. It may a list of the output of each batch.
        """
        loss_keys = training_epoch_output[0].keys()

        # Compute mean loss on all loss items on a epoch
        for key in loss_keys:
            loss_items = [step_out[key] for step_out in training_epoch_output]
            loss_mean = torch.mean(torch.tensor(loss_items))

            if self.accelerator.is_local_main_process:
                logger.info(f"Training Loss '{key}' on epoch {self.state.epochs_trained}: {loss_mean}")
                self.writer.add_scalar(f"Train_Epoch/{key}", loss_mean, self.state.epochs_trained)
                self.writer.add_scalar(f"Train_Epoch/lr", get_rate(self.optimizer), self.state.epochs_trained)

    def validation_step(self, batch, batch_idx):
        """Implement a validation step for validating a model on all processes.

        This function defines the validation step. The input batch is from a validation dataloader.
        Here is the persuade code for validating a model:

        .. code-block:: python
            :emphasize-lines: 4

            validation_output = []
            for dataloader_idx, dataloader in dataloaders:
                for batch_index, batch in dataloader:
                    loss_or_data = validation_step(batch, batch_idx)
                    validation_epoch_output.append(loss_or_data)

            score = validation_epoch_end(validation_epoch_output)
            return score

        Notes:
            **The validation step will be run on all processes.**

            About batch size:
            If your validation data have the same length, you may use a batch size larger than 1 to speed up the validation.
            For example, if you have 1000 samples in the validation set, and you have a batch size of 100, then you will
            have 10 batches in the validation set. However, if your data in the validation set has a different length, please
            use a batch size of 1. It still works for distributed validation. Otherwise, you will get an error.

            About distributed validation:
            The output of this function will be gathered across all processes. For example, if you have 4 processes, and
            you have a batch size of 1, then you will have 4 outputs from this function. The output of this function will
            be gathered across all processes. The first dimension of the result is num_processes multiplied by the first
            dimension of the input tensors. **Please make sure the first dimension of the input tensors is the batch size.**
            **The last dimension of the output will be padded to the length of the longest sample in the validation set.**
            It means that the output will be a tensor with the shape of [num_processes * batch_size, max_length]. If you
            calculate the metric score on the output, you should do a truncation to remove the padding. Otherwise, if you
            are using a metric that sensitive to the padding, you will get a wrong metric score. It is not easy to
            implement this truncation in the ``validation_epoch_end`` function. We recommend you directly calculate the metric
            score in the validation_step function. I guess the Accelerate team will implement a automatic truncation in the
            future. https://github.com/huggingface/accelerate/issues/226

        Args:
            batch: a batch of data.
            batch_idx: the index of the batch.
            dataloader_idx: the index of the dataloader.

        Returns:
            output: the output of the batch. It may enhanced audio signals.
        """
        raise NotImplementedError

    def validation_epoch_end(self, validation_epoch_output):
        """Validation epoch end.

        The input `validation_epoch_output` will be a list of list. For example, if you have two dataloaders, the `validation_epoch_output` will be:

        .. code-block:: python

            validation_epoch_output = [
                [dataloader_1_batch_1_output, dataloader_1_batch_2_output, ...],
                [dataloader_2_batch_1_output, dataloader_2_batch_2_output, ...],
                ...,
            ]


        The output of this function should be a metric score, which will be used to determine whether the current model is the best model.

        .. code-block:: python
            :emphasize-lines: 7

            validation_output = []
            for dataloader_idx, dataloader in dataloaders:
                for batch_index, batch in dataloader:
                    loss_or_data = validation_step(batch, batch_idx)
                    validation_epoch_output.append(loss_or_data)

            score = validation_epoch_end(validation_epoch_output)
            return score

        Args:
            validation_epoch_output: the output of the validation epoch. It is a list of list.

        Returns:
            score: the metric score of the validation epoch.
        """
        raise NotImplementedError
