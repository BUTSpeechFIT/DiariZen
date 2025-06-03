# Licensed under the MIT license.
# Copyright 2025 Brno University of Technology (author: Jiangyu Han, ihan@fit.vut.cz)
# This work is inspired by: https://github.com/pyf98/DPHuBERT

import torch
import torch.nn as nn

from diarizen.models.module.wav2vec2.model import wav2vec2_model 

class Model(nn.Module):
    def __init__(
        self,
        teacher_ckpt: str,
        student_ckpt: str,
        pruning_units: str = "conv,head,interm",
        distill_layers: str = "0,4,8,12",
    ):
        super().__init__()

        self.distill_layers = [int(l) for l in distill_layers.split(",")]
        
        self.teacher_model = self.build_teacher(teacher_ckpt)
        self.student_model, self.student_config = self.build_student(student_ckpt, pruning_units)
        
    def build_teacher(self, teacher_ckpt):
        teacher_ckpt = torch.load(teacher_ckpt, map_location="cpu")
        teacher_model = wav2vec2_model(**teacher_ckpt["config"])
        teacher_result = teacher_model.load_state_dict(teacher_ckpt["state_dict"], strict=False)
        print(f"Load pretrained ckpt to teacher: missing {teacher_result.missing_keys}, unexpected {teacher_result.unexpected_keys}") 

        # freeze teacher 
        for p in teacher_model.parameters():
            p.requires_grad = False
        print("Freeze parameters of the teacher model by setting requires_grad=False")
        teacher_model.eval()
        return teacher_model

    def build_student(self, student_ckpt, pruning_units):
        student_ckpt = torch.load(student_ckpt, map_location="cpu")
        pruning_units = pruning_units.split(",")
        print(f"Pruning units: {pruning_units}")
        student_config = student_ckpt['config']
        student_config.update(
            dict(
                extractor_prune_conv_channels = "conv" in pruning_units,
                encoder_prune_attention_heads = "head" in pruning_units,
                encoder_prune_attention_layer = "attlayer" in pruning_units,
                encoder_prune_feed_forward_intermediate = "interm" in pruning_units,
                encoder_prune_feed_forward_layer = "ffnlayer" in pruning_units,
            )
        )
        student_model = wav2vec2_model(**student_config)
        student_result = student_model.load_state_dict(student_ckpt["state_dict"], strict=False)
        print(f"Load pretrained ckpt to student: missing {student_result.missing_keys}, unexpected {student_result.unexpected_keys}")    
        return student_model, student_config
        
    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        self.teacher_model.eval()
        with torch.no_grad():
            teacher_hiddens, _ = self.teacher_model.extract_features(waveforms)
            teacher_hiddens = torch.stack(
                [teacher_hiddens[idx] for idx in self.distill_layers], dim=1
            )  # (batch, layer, time, feature)
        
        student_hiddens, _ = self.student_model.extract_features(waveforms)
        student_hiddens = torch.stack(
            [student_hiddens[idx] for idx in self.distill_layers], dim=1
        )  # (batch, layer, time, feature)
        
        return student_hiddens, teacher_hiddens  