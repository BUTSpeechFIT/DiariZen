"""Convert Hugging Face's WavLM to our format."""

import os
import argparse
import torch

from transformers import WavLMModel

from diarizen.models.module.wav2vec2.model import wav2vec2_model
from diarizen.models.module.wav2vec2.utils.import_huggingface_wavlm import import_huggingface_model


def convert_wavlm(hf_dir: str, output_dir: str):
    assert 'base' or 'large' in hf_dir
    out_name = 'wavlm-large-converted.bin' if 'large' in hf_dir else 'wavlm-base-converted.bin'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, out_name)

    print(f"Loading WavLM model from: {hf_dir}")
    original_model = WavLMModel.from_pretrained(hf_dir)
    
    print("Converting model...")
    converted_model, config = import_huggingface_model(original_model)
    converted_model.eval()

    aux_config = {
        "aux_num_out": None,
        "extractor_prune_conv_channels": False,
        "encoder_prune_attention_heads": False,
        "encoder_prune_attention_layer": False,
        "encoder_prune_feed_forward_intermediate": False,
        "encoder_prune_feed_forward_layer": False,
    }
    config.update(aux_config)

    print(f"Saving converted model to: {output_path}")
    torch.save({
        "state_dict": converted_model.state_dict(),
        "config": config,
    }, output_path)

    print("Verifying saved checkpoint...")
    checkpoint = torch.load(output_path, map_location="cpu")
    model = wav2vec2_model(**checkpoint["config"])
    result = model.load_state_dict(checkpoint["state_dict"], strict=False)
    print("Checkpoint loaded with result:", result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Hugging Face WavLM to custom format.")
    parser.add_argument("hf_dir", type=str, help="Path to Hugging Face WavLM directory.")
    parser.add_argument("out_dir", type=str, help="Path to output directory for converted model.")

    args = parser.parse_args()

    convert_wavlm(args.hf_dir, args.out_dir)