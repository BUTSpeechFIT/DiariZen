# MIT License
#
# Copyright (c) 2020 CNRS
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from functools import lru_cache
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from pyannote.core.utils.generators import pairwise

from pyannote.audio.core.model import Model
from pyannote.audio.core.task import Task
from pyannote.audio.utils.params import merge_dict


from .modules import TransformerBlock

from .wavlm.WavLM import WavLM, WavLMConfig

MAX_CHANNEL = 7
WAVLM_PATH='/mnt/matylda3/ihan/project/DiariZen_dev/pyannote-audio/pyannote/audio/models/segmentation/wavlm/WavLM-Base+.pt'


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        in_size: int = 192,
        attention_in : int = 96,
        ffn_hidden: int = 192,
        num_head: int = 2,
        num_layer: int = 2,
        dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(in_size, attention_in)
        self.transformer_layer = nn.ModuleList([
            TransformerBlock(
                in_size=attention_in,
                ffn_hidden=ffn_hidden,
                num_head=num_head,
                dropout=dropout
            ) for _ in range(num_layer)
        ])
        self.lnorm = nn.LayerNorm(attention_in)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
        """
        x = self.linear(x)
        for layer in self.transformer_layer:
            x = layer(x)
        return self.lnorm(x)


class Model_wavlm(Model):
    def __init__(
        self,
        wavlm_dir: str = WAVLM_PATH,
        wavlm_layer_num: int = 13,
        wavlm_feat_dim: int = 768,
        attention_in: int = 256,
        ffn_hidden: int = 1024,
        num_head: int = 4,
        num_layer: int = 4,
        dropout: float = 0.1,
        sample_rate: int = 16000,
        num_channels: int = 1,
        task: Optional[Task] = None,
    ):
        super().__init__(sample_rate=sample_rate, num_channels=num_channels, task=task)
        self.save_hyperparameters()
        
        # wavlm 
        self.wavlm_dir = wavlm_dir
        self.wavlm_model, self.wavlm_cfg = self.load_wavlm(wavlm_dir)
        self.weight_sum = nn.Linear(wavlm_layer_num, 1, bias=False)

        # transformer
        self.transformer = TransformerEncoder(
            in_size=wavlm_feat_dim,
            attention_in=attention_in,
            ffn_hidden=ffn_hidden,
            num_head=num_head,
            num_layer=num_layer,
            dropout=dropout
        )     
        # self.classifier = nn.Linear(attention_in, 3)
        # self.activation = nn.Sigmoid()

    @property
    def dimension(self) -> int:
        """Dimension of output"""
        if isinstance(self.specifications, tuple):
            raise ValueError("PyanNet does not support multi-tasking.")

        if self.specifications.powerset:
            return self.specifications.num_powerset_classes
        else:
            return len(self.specifications.classes)

    def build(self):
        # if self.hparams.linear["num_layers"] > 0:
        #     in_features = self.hparams.linear["hidden_size"]
        # else:
        #     in_features = self.hparams.lstm["hidden_size"] * (
        #         2 if self.hparams.lstm["bidirectional"] else 1
        #     )

        self.classifier = nn.Linear(256, self.dimension)
        self.activation = self.default_activation()

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

    def load_wavlm(self, wavlm_dir):
        checkpoint = torch.load(wavlm_dir)
        cfg = WavLMConfig(checkpoint['cfg'])
        model = WavLM(cfg)
        model.load_state_dict(checkpoint['model'])
        model.eval()
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
        waveforms = torch.squeeze(waveforms, 1)
        wavlm_feat = self.wav2wavlm(waveforms, self.wavlm_model, self.wavlm_cfg)
        wavlm_feat = self.weight_sum(wavlm_feat.detach().to(waveforms.device))
        wavlm_feat = torch.squeeze(wavlm_feat, -1)

        outputs = self.transformer(wavlm_feat) 
        outputs = self.classifier(outputs)
        outputs = self.activation(outputs)

        return outputs

if __name__ == '__main__':
    nnet = Model_wavlm()
    x = torch.randn(2, 80000)
    y = nnet(x)