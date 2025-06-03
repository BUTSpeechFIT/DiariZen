# Licensed under the MIT license.
# Copyright 2025 Brno University of Technology (author: Jiangyu Han, ihan@fit.vut.cz)

import os
import json
import toml
import argparse
from typing import List, Dict
from pathlib import Path

import torch

from diarizen.utils import instantiate
from diarizen.ckpt_utils import average_checkpoints
from diarizen.models.module.wav2vec2.model import wav2vec2_model


def load_metric_summary(metric_file, ckpt_path):
    with open(metric_file, "r") as f:
        lines = f.readlines()
    out_lst = []
    for line in lines:
        assert "Validation Loss | Loss_distill" in line
        epoch = line.split()[8].split(':')[0]
        Loss = line.split()[-5]
        bin_path = f"epoch_{str(epoch).zfill(4)}/pytorch_model.bin"
        out_lst.append({
            'epoch': int(epoch),
            'bin_path': ckpt_path / bin_path,
            'Loss': float(Loss),
        })
    return out_lst

def get_checkpoints(
    val_metric_summary: str,
    ckpt_path: str,
    val_metric: str = "Loss",
    avg_ckpt_num: int = 5,
) -> List[Dict[str, str]]:
    """ Select top-N checkpoints (with lowest validation loss) after the loss peak"""
    val_metric_lst = load_metric_summary(val_metric_summary, ckpt_path)

    # Find the peak loss as pruning will make loss increase firstly
    peak_entry = max(val_metric_lst, key=lambda x: x[val_metric])
    peak_index = val_metric_lst.index(peak_entry)

    # Now the sparsity won't increase. Training is stable
    post_peak_checkpoints = val_metric_lst[peak_index:]
    sorted_checkpoints = sorted(post_peak_checkpoints, key=lambda x: x[val_metric])
    return sorted_checkpoints[:avg_ckpt_num]

def run(args):
    config_path = Path(args.configuration).expanduser().absolute()
    config = toml.load(config_path.as_posix())
    config_name = str(config_path).split('/')[-2]

    ckpt_path = config_path.parent / 'checkpoints'
    checkpoints = get_checkpoints(
        val_metric_summary=args.val_metric_summary,
        ckpt_path=ckpt_path,
        avg_ckpt_num=args.avg_ckpt_num
    )

    print(f'checkpoints: {checkpoints}')

    model = instantiate(config["model"]["path"], args=config["model"]["args"])
    model = average_checkpoints(model, checkpoints)

    student_model = model.student_model

    if args.mode == "extract":
        student_num_params = sum(
            p.numel() for p in student_model.parameters()
        ) / 1e6
        print(f'student_num_params: {student_num_params} M')
        student_config = model.student_config.copy()
        print('saving student model...')
        student_out_path = os.path.join(args.out_dir, 'pytorch_model.bin')
        torch.save(
            {
                "state_dict": student_model.state_dict(),
                "config": student_config,
            },
            student_out_path
        )
        return student_model

    original_num_params = sum(
        p.numel() for p in model.teacher_model.parameters()
    ) / 1e6
    original_num_macs = model.teacher_model.get_num_macs() / 1e9
    print(f'original_num_params: {original_num_params} M')
    print(f'original_num_macs: {original_num_macs} G')

    conv_config, use_attention, use_feed_forward, num_heads, remaining_heads, ff_interm_features = student_model.prune()

    pruned_config = model.student_config.copy()
    if len(num_heads) == 0:     # specified for WavLM
        assert len(remaining_heads) > 0
        pruned_config.update({"encoder_remaining_heads": remaining_heads})
    else:
        pruned_config.update({"encoder_num_heads": num_heads})
    pruned_config.update(
        {
            "extractor_conv_layer_config": conv_config,
            "encoder_use_attention": use_attention,
            "encoder_use_feed_forward": use_feed_forward,
            "encoder_ff_interm_features": ff_interm_features,
            "extractor_prune_conv_channels": False,
            "encoder_prune_attention_heads": False,
            "encoder_prune_attention_layer": False,
            "encoder_prune_feed_forward_intermediate": False,
            "encoder_prune_feed_forward_layer": False,
            "use_layerwise_prune": False
        }
    )

    print('saving pruned model...')
    pruned_out_path = os.path.join(args.out_dir, 'pytorch_model.bin')
    torch.save(
        {
            "state_dict": student_model.state_dict(),
            "config": pruned_config,
        },
        pruned_out_path
    )

    pruned_config['wavlm_original_params'] = f'{original_num_params} M'
    pruned_config['wavlm_original_macs'] = f'{original_num_macs} G'

    return pruned_config

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply structured pruning to the pre-trained model")
    parser.add_argument(
        "-C",
        "--configuration",
        required=True,
        type=str,
        help="Configuration (*.toml).",
    )
    parser.add_argument(
        "-o",
        "--out_dir",
        type=str,
        default="./pruned_output",
        help="Path to output folder.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="prune",
        choices=["prune", "extract"],
        help="Pruning or extract student model",
    )
    parser.add_argument(
        "--avg_ckpt_num",
        type=int,
        default=5,
        help="The number of chckpoints of model averaging",
    )
    parser.add_argument(
        "--val_metric_summary",
        type=str,
        default="",
        required=False,
        help="Path to validation metric summary log file",
    )

    args = parser.parse_args()
    print(args)

    os.makedirs(args.out_dir, exist_ok=True)
    if args.mode == "extract":
        student_model = run(args)
        print(f"Successfully saved the student model weights and config to: {args.out_dir}")
    else:
        pruned_config = run(args)
        print(f"Successfully saved pruned model weights and config to: {args.out_dir}")

        # verify the saved ckpt
        pruned_model_path = os.path.join(args.out_dir, 'pytorch_model.bin')
        ckpt = torch.load(pruned_model_path, map_location="cpu")
        model = wav2vec2_model(**ckpt['config'])
        print(model.load_state_dict(ckpt['state_dict'], strict=False))

        # update current params and macs
        cur_num_params = sum(
            p.numel() for p in model.parameters()
        ) / 1e6
        cur_num_macs = model.get_num_macs() / 1e9
        print(f'current_num_params: {cur_num_params} M')
        print(f'current_num_macs: {cur_num_macs} G')

        pruned_config['wavlm_current_params'] = f'{cur_num_params} M'
        pruned_config['wavlm_current_macs'] = f'{cur_num_macs} G'

        json_out_path = os.path.join(args.out_dir, 'pruned_config.json')
        with open(json_out_path, 'w') as file:
            json.dump(pruned_config, file, indent=4)