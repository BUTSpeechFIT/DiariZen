# Licensed under the MIT license.
# Copyright 2024 Brno University of Technology (author: Jiangyu Han, ihan@fit.vut.cz)

import torch
import numpy as np

from accelerate.logging import get_logger

from pyannote.audio.utils.permutation import permutate
from pyannote.audio.utils.loss import nll_loss

from diarizen.trainer_single_opt import Trainer as BaseTrainer

logger = get_logger(__name__)


class Trainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.accelerator.print(self.model)

        # auto GN
        self.grad_history = []

    def compute_grad_norm(self, model):
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        return total_norm 

    def auto_clip_grad_norm_(self, model):
        grad_norm = self.compute_grad_norm(model)
        self.grad_history.append(grad_norm)
        if len(self.grad_history) > self.gradient_history_size:
            self.grad_history.pop(0)
        clip_value = np.percentile(self.grad_history, self.gradient_percentile)
        self.accelerator.clip_grad_norm_(model.parameters(), clip_value)  

    def training_step(self, batch, batch_idx):
        self.optimizer.zero_grad()

        xs, target = batch['xs'], batch['ts'] 
        y_pred = self.model(xs)
        # powerset
        multilabel = self.unwrap_model.powerset.to_multilabel(y_pred)
        permutated_target, _ = permutate(multilabel, target)
        permutated_target_powerset = self.unwrap_model.powerset.to_powerset(
            permutated_target.float()
        )
       
        loss = nll_loss(
            y_pred,
            torch.argmax(permutated_target_powerset, dim=-1)
        )
        
        # skip batch if something went wrong for some reason
        if torch.isnan(loss):
            return None
        
        self.accelerator.backward(loss)

        if self.accelerator.sync_gradients:
            # The gradients are added across all processes in this cumulative gradient accumulation step.
            self.auto_clip_grad_norm_(self.model)
                               
        self.optimizer.step()
        
        return {"Loss": loss}

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        xs, target = batch['xs'], batch['ts'] 
        sil_all_target = torch.zeros_like(target)

        y_pred = self.model(xs)
        # powerset
        multilabel = self.unwrap_model.powerset.to_multilabel(y_pred)
        permutated_target, _ = permutate(multilabel, target)
        permutated_target_powerset = self.unwrap_model.powerset.to_powerset(
            permutated_target.float()
        )

        loss = nll_loss(y_pred,
            torch.argmax(permutated_target_powerset, dim=-1)
        )
        val_metrics = self.unwrap_model.validation_metric(
            torch.transpose(multilabel, 1, 2),
            torch.transpose(target, 1, 2),
        )

        if not torch.equal(target, sil_all_target):
            val_DER = val_metrics['DiarizationErrorRate']
            val_FA = val_metrics['DiarizationErrorRate/FalseAlarm']
            val_Miss = val_metrics['DiarizationErrorRate/Miss']
            val_Confusion = val_metrics['DiarizationErrorRate/Confusion']
        else:
            # self.accelerator.print('Silent all the time. Ignore the metrics...')
            val_DER = torch.zeros_like(val_metrics['DiarizationErrorRate'])
            val_FA = torch.zeros_like(val_metrics['DiarizationErrorRate/FalseAlarm'])
            val_Miss = torch.zeros_like(val_metrics['DiarizationErrorRate/Miss'])
            val_Confusion = torch.zeros_like(val_metrics['DiarizationErrorRate/Confusion'])

        return {"Loss": loss, "DER": val_DER, "FA": val_FA, "Miss": val_Miss, "Confusion": val_Confusion}

    def validation_epoch_end(self, validation_epoch_output):
        metric_keys = validation_epoch_output[0].keys()
        # Compute mean loss on all loss items on a epoch
        for key in metric_keys:
            metric_items = [torch.mean(step_out[key]) for step_out in validation_epoch_output]
            metric_mean = torch.mean(torch.tensor(metric_items))
            if key == "Loss":
                Loss_val = metric_mean
            if key == "DER":
                DER_val = metric_mean
            self.writer.add_scalar(f"Validation_Epoch/{key}", metric_mean, self.state.epochs_trained)
        logger.info(f"Validation Loss/DER on epoch {self.state.epochs_trained}: {round(Loss_val.item(), 3)} / {round(DER_val.item(), 3)}")
        # metric reset
        self.unwrap_model.validation_metric.reset()
        return Loss_val