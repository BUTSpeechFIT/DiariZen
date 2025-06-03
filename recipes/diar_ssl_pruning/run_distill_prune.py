# Licensed under the MIT license.
# Copyright 2024 Hong Kong Polytechnic University (author: Xiang Hao, haoxiangsnr@gmail.com)
# Copyright 2025 Brno University of Technology (author: Jiangyu Han, ihan@fit.vut.cz)

import argparse
from pathlib import Path

import toml
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed
from torch.utils.data import DataLoader

from diarizen.logger import init_logging_logger
from diarizen.utils import instantiate
from diarizen.ckpt_utils import average_ckpt
from diarizen.models.pruning.utils import configure_optimizers

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

    if config["finetune"]["finetune"]:
        accelerator.print('fine-tuning...')
        model = average_ckpt(config["finetune"]["ckpt_dir"], model, avg_ckpt_num=config["finetune"]["avg_ckpt_num"])

    optimizer = configure_optimizers(
        model, 
        distill_lr=config["optimizer"]["args"]["distill_lr"],
        reg_lr=config["optimizer"]["args"]["reg_lr"],
    )

    distill_loss = instantiate(
        config["distill_loss"]["path"],
        args=config["distill_loss"]["args"],
    )

    (model, optimizer) = accelerator.prepare(model, optimizer)

    train_dataset_config = config["train_dataset"]["args"]
    validate_dataset_config = config["validate_dataset"]["args"]

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
        optimizer=optimizer,
        distill_loss=distill_loss,
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
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batchsize for each GPU.",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=20,
        help="Max epochs for further distillation.",
    )
    parser.add_argument(
        "--pruned_ckpt_path",
        type=str,
        default=None,
        help="Checkpoint path for the pruned model (pytorch_model.bin).",
    )
    args = parser.parse_args()

    config_path = Path(args.configuration).expanduser().absolute()
    config = toml.load(config_path.as_posix())

    config["meta"]["exp_id"] = config_path.stem
    config["meta"]["config_path"] = config_path.as_posix()
    
    # update model
    if args.pruned_ckpt_path is not None:
        # update model
        print(f'further distill from the pruned model...')
        model_config = config["model"]["args"]
        model_config["student_ckpt"] = args.pruned_ckpt_path
        model_config["pruning_units"] = " "
        
        # update trainer
        trainer_config = config["trainer"]["args"]
        trainer_config["use_reg"] = False
        trainer_config["further_distill"] = True 
        trainer_config["max_epochs"] = args.max_epochs 

        # update batch_size 
        data_train_config = config["train_dataset"]["dataloader"]  
        data_train_config["batch_size"] = args.batch_size

        del trainer_config["target_sparsity"]
        del trainer_config["pre_train_epochs"]
        del trainer_config["sparsity_warmup_epochs"]

    run(config, args.resume)
