# Licensed under the MIT license.
# Copyright 2020 CNRS (author: Herve Bredin, herve.bredin@irit.fr)
# Copyright 2025 Brno University of Technology (author: Jiangyu Han, ihan@fit.vut.cz)

import argparse
import os
from pathlib import Path
from typing import Optional, Dict, Any
import pdb
import toml
import numpy as np
import torch
import torchaudio
import joblib
from scipy.ndimage import median_filter
from huggingface_hub import snapshot_download, hf_hub_download
from pyannote.audio.pipelines import SpeakerDiarization as SpeakerDiarizationPipeline
from pyannote.audio.utils.signal import Binarize
from pyannote.audio.utils.powerset import Powerset
from pyannote.database.protocol.protocol import ProtocolFile
from pyannote.audio.core.inference import Inference

from diarizen.pipelines.utils import scp2path

from cal_fusdiar.diarization.processing import (
    multilabel_probs_to_powerset_probs,
    powerset_probs_to_multilabel_probs
)


class DiariZenPipeline(SpeakerDiarizationPipeline):
    def __init__(
        self, 
        diarizen_hub,
        embedding_model,
        config_parse: Optional[Dict[str, Any]] = None,
        rttm_out_dir: Optional[str] = None,
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

        self.apply_median_filtering = inference_config["apply_median_filtering"]
        self.min_speakers = clustering_config["min_speakers"]
        self.max_speakers = clustering_config["max_speakers"]

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
        self.calibrator = joblib.load('/mnt/sd5/users/IA/calib/calibrator_displace_4spk.joblib')

        if rttm_out_dir is not None:
            os.makedirs(rttm_out_dir, exist_ok=True)
        self.rttm_out_dir = rttm_out_dir

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

    @classmethod
    def from_pretrained(
        cls, 
        repo_id: str, 
        cache_dir: str = None,
        rttm_out_dir: str = None,
    ) -> "DiariZenPipeline":
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
            rttm_out_dir=rttm_out_dir
        )

    def __call__(self, in_wav, sess_name=None):
        assert isinstance(in_wav, (str, ProtocolFile)), "input must be either a str or a ProtocolFile"
        in_wav = in_wav if not isinstance(in_wav, ProtocolFile) else in_wav['audio']

        print('Extracting segmentations.')
        waveform, sample_rate = torchaudio.load(in_wav)
        waveform = torch.unsqueeze(waveform[0], 0)      # force to use the SDM data

        # Get powerset logits directly (skip_conversion=True)
        segmentations_powerset = self._segmentation_powerset({"waveform": waveform, "sample_rate": sample_rate})
        # Apply calibration to powerset logits
        original_shape = segmentations_powerset.data.shape  # (num_chunks, num_frames, num_powerset_classes)
        logits_powerset = segmentations_powerset.data.reshape(-1, original_shape[-1])
        in_wavi = in_wav.split('/')[-1]
        out_npi = in_wavi.replace(".wav", ".npy")
        out_path = "/mnt/sd5/users/IA/calib/git/calibrated-fusion-diarization/data/displace2026/dev_train/sys1/"
        np.save(out_path + out_npi, logits_powerset)


    # def __call__(self, in_wav, sess_name=None):
    #     assert isinstance(in_wav, (str, ProtocolFile)), "input must be either a str or a ProtocolFile"
    #     in_wav = in_wav if not isinstance(in_wav, ProtocolFile) else in_wav['audio']
    #     print('Extracting segmentations.')
    #     waveform, sample_rate = torchaudio.load(in_wav) 
    #     waveform = torch.unsqueeze(waveform[0], 0)      # force to use the SDM data
    #     segmentations = self.get_segmentations({"waveform": waveform, "sample_rate": sample_rate}, soft=True)

    #     segmentations_flat = segmentations.data.reshape(-1, segmentations.data.shape[-1])
    #     in_wavi = in_wav.split('/')[-1]
    #     out_npi = in_wavi.replace(".wav", ".npy")
    #     out_path = "/mnt/sd5/users/IA/DISPLACE2026/DISPLACE-2026-Baselines/Track1_SD/out/out_powerset_probs/"
    #     np.save(out_path + out_npi, segmentations_flat)

        