# Licensed under the MIT license.
# Copyright 2020 CNRS (author: Herve Bredin, herve.bredin@irit.fr)
# Copyright 2025 Brno University of Technology (author: Jiangyu Han, ihan@fit.vut.cz)

import argparse
import os
from pathlib import Path
from typing import Optional, Dict, Any

import toml
import numpy as np
import torch
import torchaudio
from huggingface_hub import snapshot_download, hf_hub_download
from pyannote.audio.pipelines import SpeakerDiarization as SpeakerDiarizationPipeline
from pyannote.audio.utils.powerset import Powerset
from pyannote.database.protocol.protocol import ProtocolFile
from pyannote.audio.core.inference import Inference

from diarizen.pipelines.utils import (
    scp2path,
    reduce_powerset_to_top_speakers,
)


class DiariZenPipelineSaveReduced(SpeakerDiarizationPipeline):
    def __init__(
        self,
        diarizen_hub,
        embedding_model,
        config_parse: Optional[Dict[str, Any]] = None,
        out_dir: Optional[str] = None,
        num_speakers_to_keep: int = 2,
    ):
        config_path = Path(diarizen_hub / "config.toml")
        config = toml.load(config_path.as_posix())

        if config_parse is not None:
            print('Overriding with parsed config.')
            config["inference"]["args"] = config_parse["inference"]["args"]
            config["clustering"]["args"] = config_parse["clustering"]["args"]

        inference_config = config["inference"]["args"]
        clustering_config = config["clustering"]["args"]

        print(f'Loaded configuration: {config}')

        super().__init__(
            config=config,
            seg_duration=inference_config["seg_duration"],
            segmentation=str(Path(diarizen_hub / "pytorch_model.bin")),
            segmentation_step=inference_config["segmentation_step"],
            embedding=embedding_model,
            embedding_exclude_overlap=True,
            clustering=clustering_config["method"],
            embedding_batch_size=inference_config["batch_size"],
            segmentation_batch_size=inference_config["batch_size"],
            device=torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        )

        self.num_speakers_to_keep = num_speakers_to_keep

        if clustering_config["method"] == "AgglomerativeClustering":
            self.PIPELINE_PARAMS = {
                "clustering": {
                    "method": "centroid",
                    "min_cluster_size": clustering_config["min_cluster_size"],
                    "threshold": clustering_config["ahc_threshold"],
                }
            }
        elif clustering_config["method"] == "VBxClustering":
            self.PIPELINE_PARAMS = {
                "clustering": {
                    "ahc_criterion": clustering_config["ahc_criterion"],
                    "ahc_threshold": clustering_config["ahc_threshold"],
                    "Fa": clustering_config["Fa"],
                    "Fb": clustering_config["Fb"],
                }
            }
            self.clustering.plda_dir = str(Path(diarizen_hub / "plda"))
            self.clustering.lda_dim = clustering_config["lda_dim"]
            self.clustering.maxIters = clustering_config["max_iters"]
        else:
            raise ValueError(f"Unsupported clustering method: {clustering_config['method']}")

        self.instantiate(self.PIPELINE_PARAMS)

        if out_dir is not None:
            os.makedirs(out_dir, exist_ok=True)
        self.out_dir = out_dir

        assert self._segmentation.model.specifications.powerset is True

        # Create a separate inference object that returns powerset logits (no conversion)
        self._segmentation_powerset = Inference(
            model=self._segmentation.model,
            duration=self._segmentation.duration,
            step=self._segmentation.step,
            skip_aggregation=self._segmentation.skip_aggregation,
            skip_conversion=True,  # Return powerset logits directly
            batch_size=self._segmentation.batch_size,
            device=self._segmentation.device,
        )

        # Get the powerset conversion for later use
        specs = self._segmentation.model.specifications
        self.powerset = Powerset(len(specs.classes), specs.powerset_max_classes)
        self.original_mapping = self.powerset.mapping.cpu().numpy()

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str,
        cache_dir: str = None,
        out_dir: str = None,
        num_speakers_to_keep: int = 2,
    ) -> "DiariZenPipelineSaveReduced":
        diarizen_hub = snapshot_download(
            repo_id=repo_id,
            cache_dir=cache_dir,
            local_files_only=cache_dir is not None
        )

        embedding_model = hf_hub_download(
            repo_id="pyannote/wespeaker-voxceleb-resnet34-LM",
            filename="pytorch_model.bin",
            cache_dir=cache_dir,
            local_files_only=cache_dir is not None
        )

        return cls(
            diarizen_hub=Path(diarizen_hub).expanduser().absolute(),
            embedding_model=embedding_model,
            out_dir=out_dir,
            num_speakers_to_keep=num_speakers_to_keep,
        )

    def __call__(self, in_wav, sess_name=None):
        assert isinstance(in_wav, (str, ProtocolFile)), "input must be either a str or a ProtocolFile"
        in_wav = in_wav if not isinstance(in_wav, ProtocolFile) else in_wav['audio']

        print('Extracting segmentations.')
        waveform, sample_rate = torchaudio.load(in_wav)
        waveform = torch.unsqueeze(waveform[0], 0)  # force to use the SDM data

        # Get powerset logits directly (skip_conversion=True)
        segmentations_powerset = self._segmentation_powerset({"waveform": waveform, "sample_rate": sample_rate})

        # Flatten to (num_frames, num_powerset_classes)
        original_shape = segmentations_powerset.data.shape  # (num_chunks, num_frames, num_powerset_classes)
        logits_powerset = segmentations_powerset.data.reshape(-1, original_shape[-1])

        # Reduce to top-K speakers
        reduced_probs, top_speakers, reduced_mapping = reduce_powerset_to_top_speakers(
            logits_powerset,
            self.original_mapping,
            num_speakers_to_keep=self.num_speakers_to_keep,
            max_set_size=2,
        )

        print(f"Top {self.num_speakers_to_keep} speakers: {top_speakers}")
        print(f"Reduced powerset shape: {reduced_probs.shape} (was {logits_powerset.shape})")

        # Save reduced probabilities
        in_wavi = in_wav.split('/')[-1]
        out_npy = in_wavi.replace(".wav", ".npy")
        out_path = os.path.join(self.out_dir, out_npy)
        np.save(out_path, reduced_probs)

        # # Also save metadata about which speakers were kept
        # out_meta = in_wavi.replace(".wav", "_meta.npz")
        # out_meta_path = os.path.join(self.out_dir, out_meta)
        # np.savez(out_meta_path, top_speakers=top_speakers, reduced_mapping=reduced_mapping)

        # print(f"Saved: {out_path}")
        return reduced_probs, top_speakers


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        "Save reduced powerset probabilities (top-K speakers only)",
        add_help=True,
        usage="%(prog)s [options]",
    )

    # Required paths
    parser.add_argument(
        "--in_wav_scp",
        type=str,
        required=True,
        help="Path to wav.scp."
    )
    parser.add_argument(
        "--diarizen_hub",
        type=str,
        required=True,
        help="Path to DiariZen model hub directory."
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        required=True,
        help="Path to pretrained embedding model."
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Path to output directory for .npy files."
    )
    parser.add_argument(
        "--num_speakers_to_keep",
        type=int,
        default=2,
        help="Number of top speakers to keep (default: 2)."
    )

    # Inference parameters
    parser.add_argument(
        "--seg_duration",
        type=int,
        default=16,
        help="Segment duration in seconds.",
    )
    parser.add_argument(
        "--segmentation_step",
        type=float,
        default=0.1,
        help="Shifting ratio during segmentation",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Input batch size for inference.",
    )

    # Clustering parameters (needed for parent class init)
    parser.add_argument(
        "--clustering_method",
        type=str,
        default="AgglomerativeClustering",
        choices=["VBxClustering", "AgglomerativeClustering"],
        help="Clustering method to use.",
    )
    parser.add_argument(
        "--ahc_threshold",
        type=float,
        default=0.6,
        help="AHC threshold.",
    )
    parser.add_argument(
        "--min_cluster_size",
        type=int,
        default=13,
        help="Minimum cluster size (for AHC).",
    )

    args = parser.parse_args()
    print(args)

    inference_config = {
        "seg_duration": args.seg_duration,
        "segmentation_step": args.segmentation_step,
        "batch_size": args.batch_size,
        "apply_median_filtering": True
    }

    clustering_config = {
        "method": args.clustering_method,
        "min_speakers": 1,
        "max_speakers": 20,
        "ahc_threshold": args.ahc_threshold,
        "min_cluster_size": args.min_cluster_size,
    }

    config_parse = {
        "inference": {"args": inference_config},
        "clustering": {"args": clustering_config}
    }

    pipeline = DiariZenPipelineSaveReduced(
        diarizen_hub=Path(args.diarizen_hub),
        embedding_model=args.embedding_model,
        config_parse=config_parse,
        out_dir=args.out_dir,
        num_speakers_to_keep=args.num_speakers_to_keep,
    )

    audio_files = scp2path(args.in_wav_scp)
    for audio_file in audio_files:
        sess_name = Path(audio_file).stem.split('.')[0]
        print(f'Processing: {sess_name}')
        pipeline(audio_file, sess_name=sess_name)
