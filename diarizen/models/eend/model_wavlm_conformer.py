#!/usr/bin/env python3

# Licensed under the MIT license.
# Copyright 2020 CNRS (author: Herve Bredin, herve.bredin@irit.fr)
# Copyright 2024 Brno University of Technology (author: Jiangyu Han, ihan@fit.vut.cz)

import torch
import torch.nn as nn

from functools import lru_cache

from pyannote.audio.core.model import Model as BaseModel

from diarizen.models.module.conformer import ConformerEncoder
from diarizen.models.module.wavlm.WavLM import WavLM, WavLMConfig
from diarizen.models.module.wavlm.config import Config_WavLM_Base 


class Model(BaseModel):
    def __init__(
        self,
        wavlm_dir: str = None,
        wavlm_layer_num: int = 13,
        wavlm_feat_dim: int = 768,
        attention_in: int = 256,
        ffn_hidden: int = 1024,
        num_head: int = 4,
        num_layer: int = 4,
        kernel_size: int = 31,
        dropout: float = 0.1,
        use_posi: bool = False,
        output_activate_function: str = False,
        max_speakers_per_chunk: int = 4,
        chunk_size: int = 5,
        num_channels: int = 8,
        selected_channel: int = 0
    ):
        super().__init__(
            num_channels=num_channels,
            duration=chunk_size,
            max_speakers_per_chunk=max_speakers_per_chunk
        )
        
        self.chunk_size = chunk_size
        self.selected_channel = selected_channel

        # wavlm 
        self.wavlm_dir = wavlm_dir
        self.wavlm_model, self.wavlm_cfg = self.load_wavlm(wavlm_dir)
        self.weight_sum = nn.Linear(wavlm_layer_num, 1, bias=False)

        self.proj = nn.Linear(wavlm_feat_dim, attention_in)
        self.lnorm = nn.LayerNorm(attention_in)

        self.conformer = ConformerEncoder(
            attention_in=attention_in,
            ffn_hidden=ffn_hidden,
            num_head=num_head,
            num_layer=num_layer,
            kernel_size=kernel_size,
            dropout=dropout,
            use_posi=use_posi,
            output_activate_function=output_activate_function
        )

        self.classifier = nn.Linear(attention_in, self.dimension)
        self.activation = self.default_activation()

    def non_wavlm_parameters(self):
        return [
            *self.weight_sum.parameters(),
            *self.proj.parameters(),
            *self.lnorm.parameters(),
            *self.conformer.parameters(),
            *self.classifier.parameters(),
        ]

    @property
    def dimension(self) -> int:
        """Dimension of output"""
        if isinstance(self.specifications, tuple):
            raise ValueError("PyanNet does not support multi-tasking.")

        if self.specifications.powerset:
            return self.specifications.num_powerset_classes
        else:
            return len(self.specifications.classes)

    @lru_cache
    def num_frames(self, num_samples: int) -> int:
        """Compute number of output frames for a given number of input samples

        Parameters
        ----------
        num_samples : int
            Number of input samples

        Returns
        -------
        num_frames : int
            Number of output frames
        """

        return self.wavlm_model.num_frames(num_samples)

    def receptive_field_size(self, num_frames: int = 1) -> int:
        """Compute size of receptive field

        Parameters
        ----------
        num_frames : int, optional
            Number of frames in the output signal

        Returns
        -------
        receptive_field_size : int
            Receptive field size.
        """
        return self.wavlm_model.receptive_field_size(num_frames=num_frames)

    def receptive_field_center(self, frame: int = 0) -> int:
        """Compute center of receptive field

        Parameters
        ----------
        frame : int, optional
            Frame index

        Returns
        -------
        receptive_field_center : int
            Index of receptive field center.
        """

        return self.wavlm_model.receptive_field_center(frame=frame)
    
    @property
    def get_rf_info(self, sample_rate=16000):     
        """Return receptive field info to dataset
        """

        receptive_field_size = self.receptive_field_size(num_frames=1)
        receptive_field_step = (
            self.receptive_field_size(num_frames=2) - receptive_field_size
        )
        num_frames = self.num_frames(self.chunk_size * sample_rate)
        duration = receptive_field_size / sample_rate
        step=receptive_field_step / sample_rate
        return num_frames, duration, step

    def load_wavlm(self, wavlm_dir):
        if wavlm_dir is not None:
            checkpoint = torch.load(wavlm_dir)
            cfg = WavLMConfig(checkpoint['cfg'])
            model = WavLM(cfg)
            model.load_state_dict(checkpoint['model'])
        else:   # use the default config
            cfg = WavLMConfig(Config_WavLM_Base)
            model = WavLM(cfg)
        return model, cfg
    
    def wav2wavlm(self, in_wav, model, cfg):
        """
        transform wav to wavlm features
        """
        if cfg.normalize:
            in_wav = torch.nn.functional.layer_norm(in_wav, in_wav.shape)
        rep, layer_results = model.extract_features(in_wav, output_layer=model.cfg.encoder_layers, ret_layer_results=True)[0]
        layer_reps = [x.transpose(0, 1) for x, _ in layer_results]
        return torch.stack(layer_reps, dim=-1)
    
    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        """Pass forward

        Parameters
        ----------
        waveforms : (batch, channel, sample)

        Returns
        -------
        scores : (batch, frame, classes)
        """
        assert waveforms.dim() == 3
        waveforms = waveforms[:, self.selected_channel, :]

        wavlm_feat = self.wav2wavlm(waveforms, self.wavlm_model, self.wavlm_cfg)
        wavlm_feat = self.weight_sum(wavlm_feat)
        wavlm_feat = torch.squeeze(wavlm_feat, -1)

        outputs = self.proj(wavlm_feat)
        outputs = self.lnorm(outputs)
        
        outputs = self.conformer(outputs)

        outputs = self.classifier(outputs)
        outputs = self.activation(outputs)

        return outputs