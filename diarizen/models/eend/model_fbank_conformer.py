#!/usr/bin/env python3

# Licensed under the MIT license.
# Copyright 2020 CNRS (author: Herve Bredin, herve.bredin@irit.fr)
# Copyright 2024 Brno University of Technology (author: Jiangyu Han, ihan@fit.vut.cz)

import torch
import torch.nn as nn

from functools import lru_cache

from pyannote.audio.core.model import Model as BaseModel

from diarizen.models.module.conformer import ConformerEncoder
from diarizen.models.module.speechbrain_feats import Fbank


class Model(BaseModel):
    def __init__(
        self,
        n_fft: int = 400,
        n_mels: int = 80,
        win_length: int = 25, # ms
        hop_length: int = 10, # ms
        sample_rate: int = 16000,
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
        selected_channel: int = 0,
    ):
        super().__init__(
            num_channels=num_channels,
            duration=chunk_size,
            max_speakers_per_chunk=max_speakers_per_chunk
        )
        
        self.chunk_size = chunk_size
        self.selected_channel = selected_channel

        self.n_fft = n_fft
        self.win_length_samples = win_length * sample_rate // 1000
        self.hop_length_samples = hop_length * sample_rate // 1000

        self.make_feats = Fbank(
            n_fft=n_fft,
            n_mels=n_mels,
            win_length=win_length,
            hop_length=hop_length
        )

        self.proj = nn.Linear(n_mels, attention_in)
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
    def num_frames(self, num_samples: int, center: bool = True) -> int:
        """Compute number of output frames for a given number of input samples

        Parameters
        ----------
        num_samples : int
            Number of input samples

        Returns
        -------
        num_frames : int
            Number of output frames

        Source
        ------
        https://pytorch.org/docs/stable/generated/torch.stft.html#torch.stft

        """

        if center:
            return 1 + num_samples // self.hop_length_samples
        else:
            return 1 + (num_samples - self.n_fft) // self.hop_length_samples

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

        return self.n_fft + (num_frames - 1) * self.hop_length_samples

    def receptive_field_center(self, frame: int = 0, center: bool = True) -> int:
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

        if center:
            return frame * self.hop_length_samples
        else:
            return frame * self.hop_length_samples + self.n_fft // 2
    
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

        wav_feat = self.make_feats(waveforms)

        outputs = self.proj(wav_feat)
        outputs = self.lnorm(outputs)
        
        outputs = self.conformer(outputs)

        outputs = self.classifier(outputs)
        outputs = self.activation(outputs)

        return outputs