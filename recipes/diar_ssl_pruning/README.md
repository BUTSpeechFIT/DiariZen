# Structured Pruning of WavLM 
This directory contains scripts for structured pruning of [WavLM](https://arxiv.org/pdf/2110.13900) applied to speaker diarization.

## How to run 
- Convert WavLM format from HuggingFace to our custom format: See `convert_wavlm_from_hf.py`.
- Fine-tune WavLM for diarization: See `../diar_ssl/run_stage.sh`.
- Start pruning training:
   `bash -i run_stage.sh`.
   

## Results (collar=0s)
| System         | Sparsity | Params | MACs | Speedup | AMI  | AISHELL-4 | AliMeeting | Macro |
|:----------------|:----------:|:----------:|:---------:|:---------:|:------:|:------------:|:-------------:|:--------:|
| Fbank          | -        | -        | -    | -   | 19.7 | 12.5       | 21.0        | 17.7   |
|  WavLM Base+  | 0%       | 94.4M   | 6.9G | -       | 15.6 | 11.8       | 17.7        | 15.0   |
|    | 80%      | 18.8M   | 1.1G | 4.0×    | 15.7 | 12.1       | 17.9        | 15.2   |
|        | 90%      | 9.4M    | 0.6G | 5.7×    | 17.2 | 12.1       | 19.2        | 16.1   |
| WavLM Large | 0%       | 316.6M  | 17.8G | -       | 14.8 | 11.3       | 16.3        | 14.1   |
|   | 80%      | 63.3M   | 3.8G | 2.6×    | 15.1 | 11.3       | 15.8        | 14.1   |
|                | 90%      | 30.6M   | 1.8G | 3.5×    | 15.7 | 11.2       | 17.6        | 14.8   |

## Citation
If you found this work helpful, please consider citing:
J. Han, F. Landini, J. Rohdin, A. Silnova, M. Diez, J. Cernocky and L. Burget, [Fine-tune Before Structured Pruning: Towards Compact and Accurate Self-Supervised Models for Speaker Diarization](https://arxiv.org/pdf/2505.24111), in Proc. INTERSPEECH, 2025.
```
@article{han2025fine,
  title={Fine-tune Before Structured Pruning: Towards Compact and Accurate Self-Supervised Models for Speaker Diarization},
  author={Han, Jiangyu and Landini, Federico and Rohdin, Johan and Silnova, Anna and Diez, Mireia and Cernocky, Jan and Burget, Lukas},
  journal={arXiv preprint arXiv:2505.24111},
  year={2025}
}

```

## Acknowledgments
We thank the authors of [DPHuBERT](https://github.com/pyf98/DPHuBERT) for open-sourcing their code.
