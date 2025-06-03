# DiariZen EEND Module
This directory contains scripts for DiariZen EEND module training and global inference for speaker diarization. 


## Results (collar=0s)
| System     | Features       | AMI  | AISHELL-4 | AliMeeting |
|:------------|:----------------:|:------:|:------------:|:------------:|
| [Pyannote v3.1](https://github.com/pyannote/pyannote-audio)  | SincNet        | 22.4 | 12.2       | 24.4       |
| DiariZen   | Fbank          | 19.7 | 12.5       | 21.0       |
|            | WavLM-frozen   | 17.0 | 11.7       | 19.9       |
|            | WavLM-updated  | **15.4** | **11.7**       | **17.6**       |


## Citation
If you found this work helpful, please consider citing:
J. Han, F. Landini, J. Rohdin, A. Silnova, M. Diez, and L. Burget, [Leveraging Self-Supervised Learning for Speaker Diarization](https://arxiv.org/pdf/2409.09408), in Proc. ICASSP, 2025.
```
@inproceedings{han2025leveraging,
  title={Leveraging self-supervised learning for speaker diarization},
  author={Han, Jiangyu and Landini, Federico and Rohdin, Johan and Silnova, Anna and Diez, Mireia and Burget, Luk{\'a}{\v{s}}},
  booktitle={Proc. ICASSP},
  year={2025}
}

```
