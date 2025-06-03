# Licensed under the MIT license.
# Copyright 2025 Brno University of Technology (author: Jiangyu Han, ihan@fit.vut.cz)
# This work is inspired by: https://github.com/pyf98/DPHuBERT

import os
import argparse

import torch
import torch.nn as nn

from transformers import WavLMModel

from diarizen.models.module.wav2vec2.model import wav2vec2_model
from diarizen.models.module.wav2vec2.utils.import_huggingface_wavlm import import_huggingface_model


class DistillLoss(nn.Module):
    def __init__(self, l2_weight, l1_weight, cos_weight, cos_type):
        super().__init__()
        self.l2_weight = l2_weight
        self.l1_weight = l1_weight
        self.cos_weight = cos_weight
        self.cos_type = cos_type
        assert cos_type in ["raw", "log_sig"], cos_type

        if l2_weight != 0:
            self.mse_loss = nn.MSELoss()
        if l1_weight != 0:
            self.l1_loss = nn.L1Loss()
        if cos_weight != 0:
            self.cos_sim = nn.CosineSimilarity(dim=-1)
    
    def __repr__(self) -> str:
        return "{}(l2={}, l1={}, {}_cos={})".format(
            self.__class__.__name__,
            self.l2_weight,
            self.l1_weight,
            self.cos_type,
            self.cos_weight,
        )

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        """
        Args:
            input: (batch, layer, time, feature)
            target: same shape as input
        """
        loss_mse = 0
        loss_l1 = 0
        loss_cos = 0
        if self.l2_weight != 0:
            loss_mse = self.mse_loss(input, target)
        if self.l1_weight != 0:
            loss_l1 = self.l1_loss(input, target)
        if self.cos_weight != 0:    # maximize cosine similarity
            if self.cos_type == "raw":
                loss_cos = -self.cos_sim(input, target).mean()
            elif self.cos_type == "log_sig":
                loss_cos = -self.cos_sim(input, target).sigmoid().log().mean()
            else:
                raise ValueError

        loss = self.l2_weight * loss_mse + self.l1_weight * loss_l1 + self.cos_weight * loss_cos

        return loss, (loss_mse, loss_l1, loss_cos)

def configure_optimizers(
    model: torch.nn.Module,
    distill_lr: float = 2e-4,
    reg_lr: float = 2e-2,
):
    main_params = [p for n, p in model.student_model.named_parameters() if "log_alpha" not in n]
    pgs = [
        {
            'params': main_params,
            'lr': distill_lr,
            'weight_decay': 0.0,
            'name': 'main_params',
        },
    ]
    if reg_lr is not None:
        lambda1 = nn.Parameter(torch.tensor(0.0))
        lambda2 = nn.Parameter(torch.tensor(0.0))
        pgs.extend(
            [
                {
                    'params': [p for n, p in model.student_model.named_parameters() if "log_alpha" in n],
                    'lr': reg_lr,
                    'weight_decay': 0.0,
                    'name': 'log_alpha',
                },
                {
                    'params': [lambda1, lambda2],
                    'lr': -reg_lr,
                    'weight_decay': 0.0,
                    'name': 'lambda',
                },
            ]
        )
    optimizer = torch.optim.AdamW(pgs)
    return optimizer

def convert_wavlm(hf_dir: str, output_dir: str):
    assert ('base' in hf_dir or 'large' in hf_dir), "hf_dir must contain 'base' or 'large'"
    model_name=os.path.basename(os.path.normpath(hf_dir))       # e.g. wavlm-base, wavlm-base-plus, wavlm-large
    out_name = f"{model_name}-converted.bin"
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, out_name)

    print(f"Loading WavLM model from: {hf_dir}")
    original_model = WavLMModel.from_pretrained(hf_dir)
    
    print("Converting model...")
    converted_model, config = import_huggingface_model(original_model)
    converted_model.eval()

    aux_config = {
        "aux_num_out": None,
        "extractor_prune_conv_channels": False,
        "encoder_prune_attention_heads": False,
        "encoder_prune_attention_layer": False,
        "encoder_prune_feed_forward_intermediate": False,
        "encoder_prune_feed_forward_layer": False,
    }
    config.update(aux_config)

    print(f"Saving converted model to: {output_path}")
    torch.save({
        "state_dict": converted_model.state_dict(),
        "config": config,
    }, output_path)

    print("Verifying saved checkpoint...")
    checkpoint = torch.load(output_path, map_location="cpu")
    model = wav2vec2_model(**checkpoint["config"])
    result = model.load_state_dict(checkpoint["state_dict"], strict=False)
    print("Checkpoint loaded with result:", result)