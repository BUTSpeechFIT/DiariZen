# Licensed under the MIT license.
# Adopted from https://github.com/espnet/espnet/blob/master/egs2/chime8_task1/diar_asr1/local/pyannote_diarize.py
# Copyright 2025 Brno University of Technology (author: Jiangyu Han, ihan@fit.vut.cz)

import toml 
import numpy as np

import argparse
import os.path
from pathlib import Path

import torch
import torchaudio

from huggingface_hub import hf_hub_download

from pyannote.metrics.segmentation import Annotation, Segment
from pyannote.audio.pipelines import SpeakerDiarization as SpeakerDiarizationPipeline
from pyannote.audio.utils.signal import Binarize


def scp2path(scp_file):
    """ return path list """
    lines = [line.strip().split()[1] for line in open(scp_file)]
    return lines

def split_maxlen(utt_group, min_len=10):
    # merge if
    out = []
    stack = []
    for utt in utt_group:
        if not stack or (utt.end - stack[0].start) < min_len:
            stack.append(utt)
            continue

        out.append(Segment(stack[0].start, stack[-1].end))
        stack = [utt]

    if len(stack):
        out.append(Segment(stack[0].start, stack[-1].end))

    return out

def merge_closer(annotation, delta=1.0, max_len=60, min_len=10):
    name = annotation.uri
    speakers = annotation.labels()
    new_annotation = Annotation(uri=name)
    for spk in speakers:
        c_segments = sorted(annotation.label_timeline(spk), key=lambda x: x.start)
        stack = []
        for seg in c_segments:
            if not stack or abs(stack[-1].end - seg.start) < delta:
                stack.append(seg)
                continue  # continue

            # more than delta, save the current max seg
            if (stack[-1].end - stack[0].start) > max_len:
                # break into parts of 10 seconds at least
                for sub_seg in split_maxlen(stack, min_len):
                    new_annotation[sub_seg] = spk
                stack = [seg]
            else:
                new_annotation[Segment(stack[0].start, stack[-1].end)] = spk
                stack = [seg]

        if len(stack):
            new_annotation[Segment(stack[0].start, stack[-1].end)] = spk

    return new_annotation


class DiariZenPipeline:
    def __init__(
        self, 
        cfg_path, 
        pretrained_model,
        embedding_model,
        rttm_out_dir: str = None,
        segmentation_step: float = 0.1,
        batch_size: int = 32,
        min_cluster_size: int = 30,
        cluster_threshold: float = 0.7,
        min_n_speakers: int = 2,
        max_n_speakers: int = 8,
        merge_delta: float = 0.5,
        merge_max_length: int = 50
    ):
        config_path = Path(cfg_path).expanduser().absolute()
        config = toml.load(config_path.as_posix())
        
        if rttm_out_dir is not None:
            os.makedirs(rttm_out_dir, exist_ok=True)
        self.rttm_out_dir = rttm_out_dir

        self.min_n_speakers = min_n_speakers
        self.max_n_speakers = max_n_speakers
        
        self.merge_delta = merge_delta
        self.merge_max_length = merge_max_length

        self.PIPELINE_PARAMS = {
            "clustering": {
                "method": "centroid",
                "min_cluster_size": min_cluster_size,
                "threshold": cluster_threshold   
            },
            "segmentation": {
                "min_duration_off": 0.0    
            },
        }
        # create, instantiate and apply the pipeline
        self.pipeline = SpeakerDiarizationPipeline(
            config=config,      # model configurations 
            segmentation=pretrained_model,
            segmentation_step=segmentation_step,
            embedding=embedding_model,
            embedding_exclude_overlap=True,
            clustering="AgglomerativeClustering",
            embedding_batch_size=batch_size,
            segmentation_batch_size=batch_size,
            device=torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.pipeline.instantiate(self.PIPELINE_PARAMS)
        assert self.pipeline._segmentation.model.specifications.powerset is True

    @classmethod
    def from_pretrained(
        cls, 
        repo_id: str, 
        cache_dir: str = None,
        rttm_out_dir: str = None
    ) -> "DiariZenPipeline":
        config = hf_hub_download(
            repo_id=repo_id,
            filename="config.toml",
            cache_dir=cache_dir,
            local_files_only=cache_dir is not None
        )
        pretrained_model = hf_hub_download(
            repo_id=repo_id,
            filename="pytorch_model.bin",
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
            cfg_path=config,
            pretrained_model=pretrained_model,
            embedding_model=embedding_model,
            rttm_out_dir=rttm_out_dir
        )

    def __call__(self, in_wav, sess_name=None):
        print('Extracting segmentations.')
        waveform, sample_rate = torchaudio.load(in_wav)
        waveform = torch.unsqueeze(waveform[0], 0)      # force to use the SDM data
        segmentations = self.pipeline.get_segmentations({"waveform": waveform, "sample_rate": sample_rate}, soft=False)

        # binarize segmentation
        binarized_segmentations = segmentations     # powerset

        # estimate frame-level number of instantaneous speakers
        count = self.pipeline.speaker_count(
            binarized_segmentations,
            self.pipeline._segmentation.model._receptive_field,
            warm_up=(0.0, 0.0),
        )

        print("Extracting Embeddings.")
        embeddings = self.pipeline.get_embeddings(
            {"waveform": waveform, "sample_rate": sample_rate},
            binarized_segmentations,
            exclude_overlap=self.pipeline.embedding_exclude_overlap,
        )

        # shape: (num_chunks, local_num_speakers, dimension)
        print("Clustering.")
        hard_clusters, _, _ = self.pipeline.clustering(
            embeddings=embeddings,
            segmentations=binarized_segmentations,
            num_clusters=None,
            min_clusters=self.min_n_speakers,  # 4 for NSF
            max_clusters=self.max_n_speakers,  # max-speakers are ok
            file={
                "waveform": waveform, 
                "sample_rate": sample_rate
            },  # <== for oracle clustering
            frames=self.pipeline._segmentation.model._receptive_field,  # <== for oracle clustering
        )

        # during counting, we could possibly overcount the number of instantaneous
        # speakers due to segmentation errors, so we cap the maximum instantaneous number
        # of speakers by the `max_speakers` value
        count.data = np.minimum(count.data, self.max_n_speakers).astype(np.int8)

        # keep track of inactive speakers
        inactive_speakers = np.sum(binarized_segmentations.data, axis=1) == 0
        #   shape: (num_chunks, num_speakers)

        # reconstruct discrete diarization from raw hard clusters
        hard_clusters[inactive_speakers] = -2
        discrete_diarization, _ = self.pipeline.reconstruct(
            segmentations,
            hard_clusters,
            count,
        )

        # convert to annotation
        to_annotation = Binarize(
            onset=0.5,
            offset=0.5,
            min_duration_on=0.0,
            min_duration_off=self.pipeline.segmentation.min_duration_off
        )
        result = to_annotation(discrete_diarization)
        result.uri = sess_name
        result = merge_closer(
            result, delta=self.merge_delta, max_len=self.merge_max_length, min_len=10
        )
        
        if self.rttm_out_dir is not None:
            assert sess_name is not None
            rttm_out = os.path.join(self.rttm_out_dir, sess_name + ".rttm")
            with open(rttm_out, "w") as f:
                f.write(result.to_rttm())
        return result
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        "This script performs diarization using DiariZen pipeline ",
        add_help=True,
        usage="%(prog)s [options]",
    )
    parser.add_argument(
        "--cfg_path",
        required=True,
        type=str,
        help="Configuration (*.toml).",
    )
    parser.add_argument(
        "--in_wav_scp",
        type=str,
        default="",
        help="test wav.scp.",
        metavar="STR",
        dest="in_wav_scp",
    )
    parser.add_argument(
        "--pretrained_model",
        type=str,
        help="pretrained_model.",
        dest="pretrained_model",
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        help="embedding_model.",
        dest="embedding_model",
    )
    parser.add_argument(
        "--rttm_out_dir",
        type=str,
        default=None,
        required=False,
        help="Path to output folder.",
        metavar="STR",
        dest="rttm_out_dir",
    )

    args = parser.parse_args()
    print(args)
    # diarizen_pipeline = DiariZenPipeline(
    #     cfg_path=args.cfg_path,
    #     pretrained_model=args.pretrained_model,
    #     embedding_model=args.embedding_model,
    #     rttm_out_dir=args.rttm_out_dir
    # )

    # from Huggingface
    MODEL_NAME = "BUT-FIT/diarizen-meeting-base"
    cache_dir = "/PATH/hugging-face/hub"
    diarizen_pipeline = DiariZenPipeline.from_pretrained(
        MODEL_NAME, cache_dir=cache_dir, rttm_out_dir=args.rttm_out_dir)

    audio_f = scp2path(args.in_wav_scp)
    for audio_file in audio_f:
        sess_name = Path(audio_file).stem.split('.')[0]
        print(f'Prosessing: {sess_name}')
        diarizen_pipeline(audio_file, sess_name=sess_name)