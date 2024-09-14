# Licensed under the MIT license.
# Copyright 2022 Brno University of Technology (author: Federico Landini, landini@fit.vut.cz)
# Copyright 2024 Brno University of Technology (author: Jiangyu Han, ihan@fit.vut.cz)

import os
from pathlib import Path

import torch
import torch.nn as nn

import copy

from typing import List, Dict


def average_checkpoints(
    model: nn.Module,
    checkpoint_list: str,
) -> nn.Module:
    states_dict_list = []
    for ckpt_data in checkpoint_list:
        ckpt_path = ckpt_data['bin_path']
        copy_model = copy.deepcopy(model)
        checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
        copy_model.load_state_dict(checkpoint)
        states_dict_list.append(copy_model.state_dict())
    avg_state_dict = average_states(states_dict_list, torch.device('cpu'))
    avg_model = copy.deepcopy(model)
    avg_model.load_state_dict(avg_state_dict)
    return avg_model

def average_states(
    states_list: List[Dict[str, torch.Tensor]],
    device: torch.device,
) -> List[Dict[str, torch.Tensor]]:
    qty = len(states_list)
    avg_state = states_list[0]
    for i in range(1, qty):
        for key in avg_state:
            avg_state[key] += states_list[i][key].to(device)
    for key in avg_state:
        avg_state[key] = avg_state[key] / qty
    return avg_state

def load_metric_summary(metric_file, ckpt_path):
    with open(metric_file, "r") as f:
        lines = f.readlines()
    out_lst = []
    for line in lines:
        assert "Validation Loss/DER" in line
        epoch = line.split()[4].split(':')[0]
        Loss, DER = line.split()[-3], line.split()[-1]
        bin_path = f"epoch_{str(epoch).zfill(4)}/pytorch_model.bin"
        out_lst.append({
            'epoch': int(epoch),
            'bin_path': ckpt_path / bin_path,
            'Loss': float(Loss),
            'DER': float(DER)
        })
    return out_lst

def average_ckpt(ckpt_dir, model, val_metric='Loss', avg_ckpt_num=5, val_mode="prev"):
    if 'checkpoints/epoch_' in ckpt_dir:
        print(f"No model averaging | Fine-tune model from: {ckpt_dir.split('/')[-1]}")
        ckpt_loaded = torch.load(os.path.join(ckpt_dir, 'pytorch_model.bin'), map_location=torch.device('cpu'))
        model.load_state_dict(ckpt_loaded)
        return model

    assert val_metric == "Loss" and val_mode == "prev"
    print(f'averaging previous {avg_ckpt_num} checkpoints to the converged moment...')

    ckpt_dir = Path(ckpt_dir).expanduser().absolute()
    ckpt_path = ckpt_dir / 'checkpoints'
    val_metric_path = ckpt_dir / 'val_metric_summary.lst'

    val_metric_lst = load_metric_summary(val_metric_path, ckpt_path)
    val_metric_lst_sorted = sorted(val_metric_lst, key=lambda i: i[val_metric])
    best_val_metric_idx = val_metric_lst.index(val_metric_lst_sorted[0])
    val_metric_lst_out = val_metric_lst[
            best_val_metric_idx - avg_ckpt_num + 1 :
            best_val_metric_idx + 1
    ]
    
    return average_checkpoints(model, val_metric_lst_out)
