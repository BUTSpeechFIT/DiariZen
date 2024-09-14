# Licensed under the MIT license.
# Copyright 2020 CNRS (author: Herve Bredin, herve.bredin@irit.fr)
# Copyright 2024 Brno University of Technology (author: Jiangyu Han, ihan@fit.vut.cz)

import os
 
import torch
import numpy as np

import soundfile as sf
from typing import Dict

from torch.utils.data import Dataset

def get_dtype(value: int) -> str:
    """Return the most suitable type for storing the
    value passed in parameter in memory.

    Parameters
    ----------
    value: int
        value whose type is best suited to storage in memory

    Returns
    -------
    str:
        numpy formatted type
        (see https://numpy.org/doc/stable/reference/arrays.dtypes.html)
    """
    # signe byte (8 bits), signed short (16 bits), signed int (32 bits):
    types_list = [(127, "b"), (32_768, "i2"), (2_147_483_648, "i")]
    filtered_list = [
        (max_val, type) for max_val, type in types_list if max_val > abs(value)
    ]
    if not filtered_list:
        return "i8"  # signed long (64 bits)
    return filtered_list[0][1]

def load_scp(scp_file: str) -> Dict[str, str]:
    """ return dictionary { rec: wav_rxfilename } """
    lines = [line.strip().split(None, 1) for line in open(scp_file)]
    return {x[0]: x[1] for x in lines}

def load_uem(uem_file: str) -> Dict[str, float]:
    """ returns dictionary { recid: duration }  """
    if not os.path.exists(uem_file):
        return None
    lines = [line.strip().split() for line in open(uem_file)]
    return {x[0]: [float(x[-2]), float(x[-1])] for x in lines}
    
def _gen_chunk_indices(
    init_posi: int,
    data_len: int,
    size: int,
    step: int,
) -> None:
    init_posi = int(init_posi + 1)
    data_len = int(data_len - 1)
    cur_len = data_len - init_posi
    assert cur_len > size
    num_chunks = int((cur_len - size + step) / step)
    
    for i in range(num_chunks):
        yield init_posi + (i * step), init_posi + (i * step) + size

def _collate_fn(batch, max_speakers_per_chunk=4) -> torch.Tensor:
    collated_x = []
    collated_y = []
    collated_names = []
    
    for x, y, name in batch:
        num_speakers = y.shape[-1]
        if num_speakers > max_speakers_per_chunk:
            # sort speakers in descending talkativeness order
            indices = np.argsort(-np.sum(y, axis=0), axis=0)
            # keep only the most talkative speakers
            y = y[:, indices[: max_speakers_per_chunk]]

        elif num_speakers < max_speakers_per_chunk:
            # create inactive speakers by zero padding
            y = np.pad(
                y,
                ((0, 0), (0, max_speakers_per_chunk - num_speakers)),
                mode="constant",
            )

        else:
            # we have exactly the right number of speakers
            pass
        collated_x.append(x)
        collated_y.append(y)
        collated_names.append(name)

    return {
        'xs': torch.from_numpy(np.stack(collated_x)).float(), 
        'ts': torch.from_numpy(np.stack(collated_y)), 
        'names': collated_names
    }        
        
        
class DiarizationDataset(Dataset):
    def __init__(
        self, 
        scp_file: str, 
        rttm_file: str,
        uem_file: str,
        model_num_frames: int,    # default: wavlm_base
        model_rf_duration: float,  # model.receptive_field.duration, seconds
        model_rf_step: float,  # model.receptive_field.step, seconds
        chunk_size: int = 5,  # seconds
        chunk_shift: int = 5, # seconds
        sample_rate: int = 16000
    ): 
        self.chunk_indices = []
        
        self.sample_rate = sample_rate
        
        self.model_rf_step = model_rf_step
        self.model_rf_duration = model_rf_duration
        self.model_num_frames = model_num_frames
        
        self.rec_scp = load_scp(scp_file)
        self.reco2dur = load_uem(uem_file)
        
        for rec, dur_info in self.reco2dur.items():
            start_sec, end_sec = dur_info   
            try:
                if chunk_size > 0:
                    for st, ed in _gen_chunk_indices(
                            start_sec,
                            end_sec,
                            chunk_size,
                            chunk_shift
                    ):
                        self.chunk_indices.append((rec, self.rec_scp[rec], st, ed))      # seconds
                else:
                    self.chunk_indices.append((rec, self.rec_scp[rec], start_sec, end_sec))
            except:
                print(f'Un-matched recording: {rec}')
                
        self.annotations = self.rttm2label(rttm_file)

    def get_session_idx(self, session):
        """
        convert session to session idex
        """
        session_keys = list(self.rec_scp.keys())
        return session_keys.index(session)
            
    def rttm2label(self, rttm_file):
        '''
        SPEAKER train100_306 1 15.71 1.76 <NA> <NA> 5456 <NA> <NA>
        '''
        annotations = []
        session_lst = []
        with open(rttm_file, 'r') as file:
            for seg_idx, line in enumerate(file):   
                line = line.split()
                session, start, dur = line[1], line[3], line[4]

                start = float(start)
                end = start + float(dur)
                spk = line[-2] if line[-2] != "<NA>" else line[-3]
                
                # new nession
                if session not in session_lst:      
                    unique_label_lst = []
                    session_lst.append(session)
                    
                if spk not in unique_label_lst:
                    unique_label_lst.append(spk)
                    
                label_idx = unique_label_lst.index(spk)
                
                annotations.append(
                    (
                        self.get_session_idx(session),
                        start,
                        end,
                        label_idx
                    )
                )
                
        segment_dtype = [
            (
                "session_idx",
                get_dtype(max(a[0] for a in annotations)),
            ),
            ("start", "f"),
            ("end", "f"),
            ("label_idx", get_dtype(max(a[3] for a in annotations))),
        ]
        
        return np.array(annotations, dtype=segment_dtype)

    def extract_wavforms(self, path, start, end, num_channels=8):
        start = int(start * self.sample_rate)
        end = int(end * self.sample_rate)
        data, sample_rate = sf.read(path, start=start, stop=end)
        assert sample_rate == self.sample_rate
        if data.ndim == 1:
            data = data.reshape(1, -1)
        else:
            data = np.einsum('tc->ct', data) 
        return data[:num_channels, :]

    def __len__(self):
        return len(self.chunk_indices)
    
    def __getitem__(self, idx):
        session, path, chunk_start, chunk_end = self.chunk_indices[idx]
        data = self.extract_wavforms(path, chunk_start, chunk_end)          # [start, end)
        
        # chunked annotations
        session_idx = self.get_session_idx(session)
        annotations_session = self.annotations[self.annotations['session_idx'] == session_idx]
        chunked_annotations = annotations_session[
            (annotations_session["start"] < chunk_end) & (annotations_session["end"] > chunk_start)
        ]
        
        # discretize chunk annotations at model output resolution
        step = self.model_rf_step
        half = 0.5 * self.model_rf_duration

        start = np.maximum(chunked_annotations["start"], chunk_start) - chunk_start - half
        start_idx = np.maximum(0, np.round(start / step)).astype(int)

        end = np.minimum(chunked_annotations["end"], chunk_end) - chunk_start - half
        end_idx = np.round(end / step).astype(int)
        
        # get list and number of labels for current scope
        labels = list(np.unique(chunked_annotations['label_idx']))
        num_labels = len(labels)

        mask_label = np.zeros((self.model_num_frames, num_labels), dtype=np.uint8)

        # map labels to indices
        mapping = {label: idx for idx, label in enumerate(labels)}
        for start, end, label in zip(
            start_idx, end_idx, chunked_annotations['label_idx']
        ):
            mapped_label = mapping[label]
            mask_label[start : end + 1, mapped_label] = 1
        
        return data, mask_label, session