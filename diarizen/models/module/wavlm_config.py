# Licensed under the MIT license.
# Copyright 2025 Brno University of Technology (author: Jiangyu Han, ihan@fit.vut.cz)

"""
wavlm_config.py

Predefined WavLM model configurations for speaker diarization tasks.

Includes:
- wavlm_base
- wavlm_large
- wavlm_base_s80_md
- wavlm_large_s80_md

Usage:
    from diarizen.models.module.wavlm_config import get_config
    cfg = get_config("wavlm_base")
"""


def get_config(name: str) -> dict:
    """Retrieve a predefined WavLM configuration by name."""
    name = name.lower()
    configs = {
        "wavlm_base": WAVLM_BASE,
        "wavlm_large": WAVLM_LARGE,
        "wavlm_base_s80_md": WAVLM_BASE_S80_MD,     # multi-domain
        "wavlm_large_s80_md": WAVLM_LARGE_S80_MD,   # multi-domain
    }
    if name not in configs:
        raise ValueError(
            f"Unknown config name '{name}'. Available options: {', '.join(configs)}."
        )
    return configs[name]


# -------- Config Definitions --------
WAVLM_BASE = {
    "extractor_mode": "group_norm",
    "extractor_conv_layer_config": [
        (512, 10, 5),
        (512, 3, 2),
        (512, 3, 2),
        (512, 3, 2),
        (512, 3, 2),
        (512, 2, 2),
        (512, 2, 2),
    ],
    "extractor_conv_bias": False,
    "encoder_embed_dim": 768,
    "encoder_projection_dropout": 0.1,
    "encoder_pos_conv_kernel": 128,
    "encoder_pos_conv_groups": 16,
    "encoder_num_layers": 12,
    "encoder_use_attention": [True] * 12,
    "encoder_use_feed_forward": [True] * 12,
    "encoder_total_num_heads": [12] * 12,
    "encoder_remaining_heads": [[i for i in range(12)] for _ in range(12)],
    "encoder_num_buckets": 320,
    "encoder_max_distance": 800,
    "encoder_attention_dropout": 0.1,
    "encoder_ff_interm_features": [3072] * 12,
    "encoder_ff_interm_dropout": 0.0,
    "encoder_dropout": 0.1,
    "encoder_layer_norm_first": False,
    "encoder_layer_drop": 0.05,
    "aux_num_out": None,
    "normalize_waveform": False,
    "extractor_prune_conv_channels": False,
    "encoder_prune_attention_heads": False,
    "encoder_prune_attention_layer": False,
    "encoder_prune_feed_forward_intermediate": False,
    "encoder_prune_feed_forward_layer": False,
}

WAVLM_LARGE = {
    "extractor_mode": "layer_norm",
    "extractor_conv_layer_config": [
        (512, 10, 5),
        (512, 3, 2),
        (512, 3, 2),
        (512, 3, 2),
        (512, 3, 2),
        (512, 2, 2),
        (512, 2, 2),
    ],
    "extractor_conv_bias": False,
    "encoder_embed_dim": 1024,
    "encoder_projection_dropout": 0.1,
    "encoder_pos_conv_kernel": 128,
    "encoder_pos_conv_groups": 16,
    "encoder_num_layers": 24,
    "encoder_use_attention": [True] * 24,
    "encoder_use_feed_forward": [True] * 24,
    "encoder_total_num_heads": [16] * 24,
    "encoder_remaining_heads": [[i for i in range(16)] for _ in range(24)],
    "encoder_num_buckets": 320,
    "encoder_max_distance": 800,
    "encoder_attention_dropout": 0.1,
    "encoder_ff_interm_features": [4096] * 24,
    "encoder_ff_interm_dropout": 0.0,
    "encoder_dropout": 0.1,
    "encoder_layer_norm_first": True,
    "encoder_layer_drop": 0.1,
    "normalize_waveform": True,
    "aux_num_out": None,
    "extractor_prune_conv_channels": False,
    "encoder_prune_attention_heads": False,
    "encoder_prune_attention_layer": False,
    "encoder_prune_feed_forward_intermediate": False,
    "encoder_prune_feed_forward_layer": False,
}

WAVLM_BASE_S80_MD = {
    "extractor_mode": "group_norm",
    "extractor_conv_layer_config": [
        (90, 10, 5),
        (161, 3, 2),
        (173, 3, 2),
        (181, 3, 2),
        (351, 3, 2),
        (155, 2, 2),
        (137, 2, 2),
    ],
    "extractor_conv_bias": False,
    "encoder_embed_dim": 768,
    "encoder_projection_dropout": 0.1,
    "encoder_pos_conv_kernel": 128,
    "encoder_pos_conv_groups": 16,
    "encoder_num_layers": 12,
    "encoder_use_attention": [
        True, True, True, True, True, True, True, True, False, False, True, True
    ],
    "encoder_use_feed_forward": [True] * 12,
    "encoder_total_num_heads": [12] * 12,
    "encoder_remaining_heads": [
        [1, 6],
        [5, 7, 8],
        [0, 3, 9],
        [0, 1, 4, 8, 11],
        [6, 8],
        [0],
        [7, 8, 10, 11],
        [0, 1, 4, 8],
        [],
        [],
        [4, 7],
        [5],
    ],
    "encoder_num_buckets": 320,
    "encoder_max_distance": 800,
    "encoder_attention_dropout": 0.1,
    "encoder_ff_interm_features": [
        666, 660, 649, 1080, 237, 299, 437, 573, 53, 80, 211, 334
    ],
    "encoder_ff_interm_dropout": 0.0,
    "encoder_dropout": 0.1,
    "encoder_layer_norm_first": False,
    "encoder_layer_drop": 0.05,
    "aux_num_out": None,
    "normalize_waveform": False,
    "extractor_prune_conv_channels": False,
    "encoder_prune_attention_heads": False,
    "encoder_prune_attention_layer": False,
    "encoder_prune_feed_forward_intermediate": False,
    "encoder_prune_feed_forward_layer": False,
    "use_layerwise_prune": False,
}

WAVLM_LARGE_S80_MD = {
    "extractor_mode": "layer_norm",
    "extractor_conv_layer_config": [
        (512, 10, 5),
        (153, 3, 2),
        (224, 3, 2),
        (255, 3, 2),
        (302, 3, 2),
        (368, 2, 2),
        (211, 2, 2),
    ],
    "extractor_conv_bias": False,
    "encoder_embed_dim": 1024,
    "encoder_projection_dropout": 0.1,
    "encoder_pos_conv_kernel": 128,
    "encoder_pos_conv_groups": 16,
    "encoder_num_layers": 24,
    "encoder_use_attention": [
        True, True, True, True, True, True, True, True, True, False, True, True,
        False, True, True, True, False, False, True, True, True, True, True, True
    ],
    "encoder_use_feed_forward": [True] * 24,
    "encoder_total_num_heads": [16] * 24,
    "encoder_remaining_heads": [
        [1, 2, 4, 5, 6],
        [9, 10, 14],
        [0, 1, 2, 4, 5, 7],
        [1, 4, 7, 12, 13, 14],
        [0, 2, 3, 4, 13],
        [1, 7, 13, 14, 15],
        [11, 13, 15],
        [2, 3, 4, 8, 15],
        [2, 5, 6, 15],
        [],
        [0, 1],
        [1, 3, 5, 12],
        [],
        [4, 7, 11],
        [6, 9],
        [11],
        [],
        [],
        [14],
        [5, 15],
        [0, 2, 8, 11, 13, 15],
        [0, 1, 3, 4, 5, 6, 7, 10, 13],
        [0, 1, 3, 6, 7, 9, 10, 11, 12, 14],
        [1, 2, 3, 4, 7, 13, 14, 15],
    ],
    "encoder_num_buckets": 320,
    "encoder_max_distance": 800,
    "encoder_attention_dropout": 0.1,
    "encoder_ff_interm_features": [
        1092, 925, 759, 646, 745, 615, 684, 958, 286, 294,
        406, 377, 463, 542, 298, 236, 96, 104, 134, 211,
        473, 1011, 1770, 1316
    ],
    "encoder_ff_interm_dropout": 0.0,
    "encoder_dropout": 0.1,
    "encoder_layer_norm_first": True,
    "encoder_layer_drop": 0.1,
    "normalize_waveform": True,
    "aux_num_out": None,
    "extractor_prune_conv_channels": False,
    "encoder_prune_attention_heads": False,
    "encoder_prune_attention_layer": False,
    "encoder_prune_feed_forward_intermediate": False,
    "encoder_prune_feed_forward_layer": False,
    "use_layerwise_prune": False,
}
