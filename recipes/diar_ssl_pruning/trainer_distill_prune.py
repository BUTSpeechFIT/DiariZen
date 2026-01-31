# Licensed under the MIT license.
# Copyright 2025 Brno University of Technology (author: Jiangyu Han, ihan@fit.vut.cz)

import torch
import numpy as np

from accelerate.logging import get_logger
from diarizen.trainer_distill_prune import Trainer as BaseTrainer

logger = get_logger(__name__)

class Trainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.accelerator.print(self.model)

        # auto GN
        self.grad_history = []

        self.lambda1, self.lambda2 = self.get_lambda()
        self.original_num_params = sum(
            p.numel() for p in self.unwrap_model.teacher_model.parameters()
        )
        logger.info(f'self.target_sparsity: {self.target_sparsity}')

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
    
    def get_target_sparsity(self):
        real_warmup_steps = self.state.steps_trained - self.pre_train_steps
        if real_warmup_steps >= self.sparsity_warmup_updates:
            return self.target_sparsity
        return self.target_sparsity * (real_warmup_steps / self.sparsity_warmup_updates)

    def get_lambda(self):
        for param_group in self.optimizer.param_groups:
            if param_group.get('name') == 'lambda':
                return param_group['params']
        return None, None

    def training_step(self, batch, batch_idx):
        self.optimizer.zero_grad()

        xs = batch['xs'][:, 0, :]     # B,C,T --> B,T; SDM
        student_hiddens, teacher_hiddens = self.model(xs)

        # distill loss
        loss_distill, (loss_mse, loss_l1, loss_cos) = self.distill_loss(student_hiddens, teacher_hiddens)

        # prune loss
        if self.use_reg and self.state.epochs_trained >= self.pre_train_epochs:
            cur_target_sparsity = self.get_target_sparsity()
            cur_expected_sparsity = 1. - self.unwrap_model.student_model.get_num_params() / self.original_num_params
            loss_reg = self.lambda1 * (cur_expected_sparsity - cur_target_sparsity) \
                + self.lambda2 * (cur_expected_sparsity - cur_target_sparsity)**2
        else:
            cur_target_sparsity = torch.Tensor([0.0])
            cur_expected_sparsity = torch.Tensor([0.0])            
            loss_reg = torch.Tensor([0.0]).to(loss_distill.device)

        loss = loss_distill + loss_reg

        # skip batch if something went wrong for some reason
        if torch.isnan(loss):
            return None
        
        self.accelerator.backward(loss)

        if self.accelerator.sync_gradients:
            # The gradients are added across all processes in this cumulative gradient accumulation step.
            self.auto_clip_grad_norm_(self.model)
                               
        self.optimizer.step()

        if not self.further_distill and batch_idx == self.update_steps_per_epoch - 1:
            self.target_sparsity_cur_epoch = round(cur_target_sparsity, 3)
            self.expected_sparsity_cur_epoch = round(cur_expected_sparsity.item(), 3)
        
        return {
            "Loss": loss.detach().float(), 
            "Loss_distill": loss_distill.detach().float(), 
            "Loss_l1": loss_l1.detach().float(), 
            "Loss_cos": loss_cos.detach().float(),
            "Loss_reg": loss_reg.detach().float(),
            "sparsity_expected": cur_expected_sparsity,
            "sparsity_target": cur_target_sparsity,
            "lambda1": self.lambda1,
            "lambda2": self.lambda2
        }

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        xs = batch['xs'][:, 0, :]     # B,C,T --> B,T; SDM
        student_hiddens, teacher_hiddens = self.model(xs)

        # distill loss
        loss_distill, (loss_mse, loss_l1, loss_cos) = self.distill_loss(student_hiddens, teacher_hiddens)
    
        # prune loss
        if self.use_reg and self.state.epochs_trained >= self.pre_train_epochs:
            cur_target_sparsity = self.get_target_sparsity()     
            cur_expected_sparsity = 1. - self.unwrap_model.student_model.get_num_params() / self.original_num_params
            loss_reg = self.lambda1 * (cur_expected_sparsity - cur_target_sparsity) \
                + self.lambda2 * (cur_expected_sparsity - cur_target_sparsity)**2
        else:          
            loss_reg = torch.Tensor([0.0]).to(loss_distill.device)
        
        loss = loss_distill + loss_reg
        
        return {
            "Loss": loss.detach().float(),
            "Loss_distill": loss_distill.detach().float(),
            "Loss_l1": loss_l1.detach().float(),
            "Loss_cos": loss_cos.detach().float(),
            "Loss_reg": loss_reg.detach().float(),
        }

    def validation_epoch_end(self, validation_epoch_output):
        metric_keys = validation_epoch_output[0].keys()
        # Compute mean loss on all loss items on a epoch
        for key in metric_keys:
            metric_items = [step_out[key] for step_out in validation_epoch_output]  # float
            metric_mean = sum(metric_items) / len(metric_items)
            if key == "Loss":
                Loss_val = metric_mean
            if key == "Loss_distill":
                Loss_distill_val = metric_mean
            if key == "Loss_reg":
                Loss_reg_val = metric_mean
            self.writer.add_scalar(f"Validation_Epoch/{key}", metric_mean, self.state.epochs_trained)
        logger.info(f"Validation Loss | Loss_distill | Loss_reg on epoch {self.state.epochs_trained}: {round(Loss_val, 3)} | {round(Loss_distill_val, 3)} | {round(Loss_reg_val, 3)}")
        return Loss_distill_val
