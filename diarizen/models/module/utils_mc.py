#!/usr/bin/env python3

# Licensed under the MIT license.
# Copyright 2025 Brno University of Technology (author: Jiangyu Han, ihan@fit.vut.cz)


import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class CrossChannelAttention(nn.Module):
    def __init__(
        self,
        n_units: int,
        h_units: int,
        h: int = 4,
        dropout: float = 0.1,
        init_mult: float = 1e-2,
    ) -> None:
        super().__init__()
        self.linearQ = nn.Linear(n_units, h_units)
        self.linearK = nn.Linear(n_units, h_units)
        self.linearV = nn.Linear(n_units, h_units)
        self.linearO = nn.Linear(h_units, n_units)

        self.ln_norm = nn.LayerNorm(n_units) # bias initialized to zeros
        self.ln_norm.weight.data *= init_mult
        
        self.d_k = h_units // h
        self.h = h
        self.dropout = nn.Dropout(p=dropout)
        self.att = None  # attention for plot

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, D)
        batch_size, channel, time, _ = x.shape 
        res = x
        x = torch.einsum('bctd->btcd', x)
        q = self.linearQ(x).reshape(-1, channel, self.h, self.d_k)
        k = self.linearK(x).reshape(-1, channel, self.h, self.d_k)
        v = self.linearV(x).reshape(-1, channel, self.h, self.d_k)

        q = q.transpose(1, 2)   # (..., head, channel, d_k)
        k = k.transpose(1, 2)   # (..., head, channel, d_k)
        v = v.transpose(1, 2)   # (..., head, channel, d_k)
        att_score = torch.matmul(q, k.transpose(-2, -1))
        
        scores = att_score / np.sqrt(self.d_k)
            
        # scores: (B, h, c, c)
        self.att = F.softmax(scores, dim=3)
        p_att = self.dropout(self.att)
        x = torch.matmul(p_att, v)      # (..., head, channel, d_k)
        x = x.permute(0, 2, 1, 3).reshape(batch_size, time, channel, self.h * self.d_k)
        x = torch.einsum('btcd->bctd', x)
        x = self.linearO(x)
        
        # fusion with input
        x = self.ln_norm(x) + res
        return x
    
class TACFusion(nn.Module):     
    """
    TAC-style fusion module.

    Reference:
        1. https://www.isca-archive.org/interspeech_2024/mosner24_interspeech.pdf
        2. https://github.com/BUTSpeechFIT/Wespeaker_MC_SSL/blob/main/wespeaker/models/fusion_modules.py
    """
    def __init__(
        self, 
        input_dim=768, 
        hidden_dim=1024, 
        activation=nn.PReLU, 
        norm_type=nn.LayerNorm, 
        gammma_init_mult=1e-2
    ):
        super().__init__()
        self.input_tf = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), activation()
        )
        self.avg_tf = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), activation()
        )
        self.concat_tf = nn.Sequential(
            nn.Linear(2 * hidden_dim, input_dim), activation()
        )
        self.norm = norm_type(input_dim) # bias initialized to zeros
        self.norm.weight.data *= gammma_init_mult
     
    def forward(self, x):
        """
        Args:
            x: (:class:`torch.Tensor`): Input multi-channel features.
                Shape: :(batch, mic_channels, frames, features).

        Returns:
            output (:class:`torch.Tensor`): features for each mic_channel after the TAC inter-channel processing.
                Shape: :(batch, mic_channels, frames, features).
        """
        assert x.dim() == 4     # x: (B, C, T, D)
        
        # First operation: transform the input for each frame and independently on each mic channel.
        output = self.input_tf(x) # (B, ch, frames, hidden_dim)

        # Mean pooling across channels
        mics_mean = output.mean(1) # (B, frames, hidden_dim)
        mics_mean = self.avg_tf(mics_mean) # (B, frames, hidden_dim)
        mics_mean = torch.unsqueeze(mics_mean, 1).expand_as(output)    # (B, ch, frames, hidden_dim)
        
        # concatnation
        output = torch.cat([output, mics_mean], -1)
        output = self.concat_tf(output)
        output = self.norm(output) # (B, ch, frames, input_dim)
        
        # residual connection
        output += x
        return output

if __name__ == '__main__':
    nnet = CrossChannelAttention(768, 128, 4, 0.1)
    x = torch.randn(12, 8, 400, 768)
    y = nnet(x)
    print(f'y: {y.shape}')