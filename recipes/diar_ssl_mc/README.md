# Spatially Aware WavLM 
This directory contains scripts for spatially aware [WavLM](https://arxiv.org/pdf/2110.13900) for multi-channel speaker diarization.
   

## Results (all channels, VBx clustering, collar=0s)
| System | AMI | AIS-4 | Ali | NSF-1 | CH-6 | Macro |
|:-------|:---:|:-----:|:---:|:-----:|:----:|:-----:|
| **Baselines** | | | | | | |
| Single-channel | 15.3 | 11.3 | 15.0 | 17.7 | 33.8 | 18.6 |
| DOVER-Lap | 14.7 | 10.9 | 13.5 | 17.1 | 30.9 | 17.4 |
| Average probs & embs | 14.9 | 11.0 | 14.0 | 17.5 | 28.8 | 17.2 |
| **Proposed methods** | | | | | | |
| ChAtt, DOVER-Lap | 14.8 | 11.0 | 12.8 | 17.4 | 31.3 | 17.5 |
| ChAtt, average embed. | 14.9 | 11.1 | 12.9 | 17.6 | 28.5 | 17.0 |
| ChAtt, att. argmax | 14.9 | 11.0 | 12.8 | 17.5 | 29.5 | 17.2 |
| ChAtt, att. weighted fusion | 14.8 | 11.2 | 12.8 | 17.4 | 27.5 | 16.7 |



## Citation
If you found this work helpful, please consider citing:
J. Han, R. Wang, Y. Masuyama, M. Delcroix, J. Rohdin, J. Du, L. Burget, [Spatially Aware Self-Supervised Models for Multi-Channel Neural Speaker Diarization](https://arxiv.org/pdf/2510.14551), in Proc. ICASSP, 2026.
```
@article{han2025spatially,
  title={Spatially Aware Self-Supervised Models for Multi-Channel Neural Speaker Diarization},
  author={Han, Jiangyu and Wang, Ruoyu and Masuyama, Yoshiki and Delcroix, Marc and Rohdin, Johan and Du, Jun and Burget, Lukas},
  journal={arXiv preprint arXiv:2510.14551},
  year={2025}
}
```
