import toml 
import argparse

import torch

import os.path
from pathlib import Path

from typing import *

from diarizen.utils import instantiate
from diarizen.ckpt_utils import load_metric_summary, average_checkpoints
from diarizen.models.module.wav2vec2.model import wav2vec2_model as wavlm_model


def get_checkpoints(
    val_metric_summary: str,
    ckpt_path: str,
    val_metric: str = "Loss",
    val_mode: str = "prev",     # "best", "prev"
    avg_ckpt_num: int = 5,
) -> List[Dict[str, str]]:
    assert val_mode in ["prev", "best"]
    assert val_metric in ["Loss", "DER"]
    val_metric_lst = load_metric_summary(val_metric_summary, ckpt_path)
    val_metric_lst_sorted = sorted(val_metric_lst, key=lambda i: i[val_metric])
    best_val_metric_idx = val_metric_lst.index(val_metric_lst_sorted[0])
    if val_mode == "prev":
        print(f'averaging previous {avg_ckpt_num} checkpoints to the converged moment...')
        segmentation = val_metric_lst[
            best_val_metric_idx - avg_ckpt_num + 1 :
            best_val_metric_idx + 1
        ]
    else:
        print(f'averaging the best {avg_ckpt_num} checkpoints...')
        segmentation = val_metric_lst_sorted[:avg_ckpt_num]

    assert len(segmentation) == avg_ckpt_num
    return segmentation


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract WavLM from DiariZen experimental directory"
    )
    parser.add_argument(
        "-C", "--configuration",
        required=True, type=str,
        help="Configuration (*.toml).",
    )
    parser.add_argument(
        "--val_metric_summary",
        type=str, required=True,
        help="Path to the validation metric summary file."
    )
    parser.add_argument(
        "--out_dir",
        type=str, required=True,
        help="Path to output folder.",  
    )
    parser.add_argument(
        "--out_affix",
        type=str, default=None, 
        help="Output filename prefix (affix)."
    )
    parser.add_argument(
        "--avg_ckpt_num",
        type=int, default=5, 
        help="Number of checkpoints to average."
    )
    parser.add_argument(
        "--val_metric",
        type=str, default="Loss", choices=["Loss", "DER"], 
        help="Validation metric to select checkpoints."
    )
    parser.add_argument(
        "--val_mode",
        type=str, default="best", choices=["best", "prev"], 
        help="Checkpoint selection mode."
    )


    args = parser.parse_args()
    print(args)

    config_path = Path(args.configuration).expanduser().absolute()
    config = toml.load(config_path.as_posix())

    ckpt_path = config_path.parent / 'checkpoints'
    checkpoints = get_checkpoints(
        val_metric_summary=args.val_metric_summary,
        ckpt_path=ckpt_path,
        val_metric=args.val_metric,
        val_mode=args.val_mode,    
        avg_ckpt_num=args.avg_ckpt_num,
    )
    print(f'checkpoints: {checkpoints}')

    # load finetuned wavlm
    model = instantiate(config["model"]["path"], args=config["model"]["args"])
    model = average_checkpoints(model, checkpoints)
    wavlm_finetuned = model.wavlm_model

    # load original wavlm
    wavlm_src = config["model"]["args"]["wavlm_src"]
    ckpt = torch.load(wavlm_src, map_location='cpu')
    conf_origin = ckpt["config"]
    wavlm_origin = wavlm_model(**conf_origin)

    # load finetuned params
    wavlm_origin.load_state_dict(wavlm_finetuned.state_dict(), strict=True)

    # params and macs
    num_params = sum(
        p.numel() for p in wavlm_finetuned.parameters()
    ) / 1e6
    num_macs = wavlm_finetuned.get_num_macs() / 1e9
    print(f'num_params: {num_params}M | num_macs: {num_macs}G')

    # save finetuned params
    if args.out_affix is not None:
        out_name = f'{args.out_affix}_{args.val_metric}_{args.val_mode}_{args.avg_ckpt_num}.pt'
    else:
        out_name = f'{args.val_metric}_{args.val_mode}_{args.avg_ckpt_num}.pt'

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, out_name)
    print(f'saving {out_path}...')
    torch.save(
        {
            "state_dict": wavlm_finetuned.state_dict(),
            "config": conf_origin
        },
        out_path
    )

    

