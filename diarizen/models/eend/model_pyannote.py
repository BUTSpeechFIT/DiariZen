#!/usr/bin/env python3

# Licensed under the MIT license.
# Copyright 2020 CNRS (author: Herve Bredin, herve.bredin@irit.fr)
# Copyright 2024 Brno University of Technology (author: Jiangyu Han, ihan@fit.vut.cz)

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from functools import lru_cache

from pyannote.audio.core.model import Model as BaseModel
from pyannote.audio.models.blocks.sincnet import SincNet
from pyannote.audio.utils.receptive_field import (
    multi_conv_num_frames, 
    multi_conv_receptive_field_size, 
    multi_conv_receptive_field_center
)


class Model(BaseModel):
    def __init__(
        self,
        max_speakers_per_chunk: int = 4,
        chunk_size: int = 8,
        num_channels: int = 8,
        selected_channel: int = 0,
    ):
        super().__init__(
            num_channels=num_channels,
            duration=chunk_size,
            max_speakers_per_chunk=max_speakers_per_chunk
        )
        self.chunk_size = chunk_size
        self.selected_channel = selected_channel
        
        self.sincnet = SincNet(stride=10)
        
        self.lstm = nn.LSTM(
            input_size=60, 
            hidden_size=128,
            num_layers=4,
            batch_first=True,
            dropout=0.5,
            bidirectional=True
        )
        
        self.linear = nn.ModuleList()
        self.linear.append(nn.Linear(256, 128))
        self.linear.append(nn.Linear(128, 128))

        self.classifier = nn.Linear(128, self.dimension)
        self.activation = self.default_activation()

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

        kernel_size = [251, 3, 5, 3, 5, 3]
        stride = [10, 3, 1, 3, 1, 3]
        padding = [0, 0, 0, 0, 0, 0]
        dilation = [1, 1, 1, 1, 1, 1]

        return multi_conv_num_frames(
            num_samples,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )

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
        kernel_size = [251, 3, 5, 3, 5, 3]
        stride = [10, 3, 1, 3, 1, 3]
        dilation = [1, 1, 1, 1, 1, 1]

        return multi_conv_receptive_field_size(
            num_frames,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
        )

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

        kernel_size = [251, 3, 5, 3, 5, 3]
        stride = [10, 3, 1, 3, 1, 3]
        padding = [0, 0, 0, 0, 0, 0]
        dilation = [1, 1, 1, 1, 1, 1]

        return multi_conv_receptive_field_center(
            frame,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
    
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
        waveforms = torch.unsqueeze(waveforms, 1)
            
        outputs = self.sincnet(waveforms)
        outputs, _ = self.lstm(rearrange(outputs, "batch feature frame -> batch frame feature"))
        for linear in self.linear:
            outputs = F.leaky_relu(linear(outputs))
        return self.activation(self.classifier(outputs))