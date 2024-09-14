# Licensed under the MIT license.
# Copyright 2024 Hong Kong Polytechnic University (author: Xiang Hao, haoxiangsnr@gmail.com)
# Copyright 2024 Brno University of Technology (author: Jiangyu Han, ihan@fit.vut.cz)

import argparse
from pathlib import Path

import toml
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed
from torch.utils.data import DataLoader

from diarizen.logger import init_logging_logger
from diarizen.utils import instantiate
from diarizen.ckpt_utils import average_ckpt

from dataset import _collate_fn

def run(config, resume):
    init_logging_logger(config)

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=config["trainer"]["args"]["gradient_accumulation_steps"],
        kwargs_handlers=[ddp_kwargs],
    )

    set_seed(config["meta"]["seed"], device_specific=True)

    model = instantiate(config["model"]["path"], args=config["model"]["args"])
    model_num_frames, model_rf_duration, model_rf_step = model.get_rf_info

    if config["finetune"]["finetune"]:
        accelerator.print('fine-tuning...')
        model = average_ckpt(config["finetune"]["ckpt_dir"], model, avg_ckpt_num=config["finetune"]["avg_ckpt_num"])

    optimizer = instantiate(
        config["optimizer"]["path"],
        args={"params": model.parameters()}
        | config["optimizer"]["args"]
        | {"lr": config["optimizer"]["args"]["lr"]},
    )

    (model, optimizer) = accelerator.prepare(model, optimizer)

    # pass model receptive field info to dataset
    train_dataset_config = config["train_dataset"]["args"]
    train_dataset_config["model_num_frames"] = model_num_frames
    train_dataset_config["model_rf_duration"] = model_rf_duration
    train_dataset_config["model_rf_step"] = model_rf_step

    validate_dataset_config = config["validate_dataset"]["args"]
    validate_dataset_config["model_num_frames"] = model_num_frames
    validate_dataset_config["model_rf_duration"] = model_rf_duration
    validate_dataset_config["model_rf_step"] = model_rf_step

    if "train" in args.mode:
        train_dataset = instantiate(config["train_dataset"]["path"], args=train_dataset_config)
        train_dataloader = DataLoader(
            dataset=train_dataset, collate_fn=_collate_fn, shuffle=True, **config["train_dataset"]["dataloader"]
        )
        train_dataloader = accelerator.prepare(train_dataloader)

    if "train" in args.mode or "validate" in args.mode:
        validate_dataset = instantiate(config["validate_dataset"]["path"], args=validate_dataset_config)
        validate_dataloader = DataLoader(
            dataset=validate_dataset, collate_fn=_collate_fn, shuffle=False, **config["validate_dataset"]["dataloader"]
        )
        validate_dataloader = accelerator.prepare(validate_dataloader)
 
    trainer = instantiate(config["trainer"]["path"], initialize=False)(
        accelerator=accelerator,
        config=config,
        resume=resume,
        model=model,
        optimizer=optimizer
    )

    for flag in args.mode:
        if flag == "train":
            trainer.train(train_dataloader, validate_dataloader)
        elif flag == "validate":
            trainer.validate(validate_dataloader)
        else:
            raise ValueError(f"Unknown mode: {flag}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audio-ZEN based EEND framework")
    parser.add_argument(
        "-C",
        "--configuration",
        required=True,
        type=str,
        help="Configuration (*.toml).",
    )
    parser.add_argument(
        "-M",
        "--mode",
        nargs="+",
        type=str,
        default=["train"],
        choices=["train", "validate"],
        help="Mode of the experiment.",
    )
    parser.add_argument(
        "-R",
        "--resume",
        action="store_true",
        help="Resume the experiment from latest checkpoint.",
    )
    parser.add_argument(
        "-FT",
        "--finetune",
        action="store_true",
        help="Label of fine-tuning.",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        help="Checkpoint path for fine-tuning.",
    )

    args = parser.parse_args()

    config_path = Path(args.configuration).expanduser().absolute()
    config = toml.load(config_path.as_posix())

    config["meta"]["exp_id"] = config_path.stem
    config["meta"]["config_path"] = config_path.as_posix()

    run(config, args.resume)
