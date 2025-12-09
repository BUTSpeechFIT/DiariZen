# DiariZen
DiariZen is a speaker diarization toolkit driven by [AudioZen](https://github.com/haoxiangsnr/spiking-fullsubnet) and [Pyannote 3.1](https://huggingface.co/pyannote/speaker-diarization-3.1). 


## Installation
```
# create virtual python environment
conda create --name diarizen python=3.10
conda activate diarizen

# install diarizen 
conda install pytorch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt && pip install -e .

# install pyannote-audio
cd pyannote-audio && pip install -e .[dev,testing]

# install dscore
git submodule init
git submodule update
```

## Usage
- For model training, see `recipes/diar_ssl/run_stage.sh`. 
- For model pruning, see `recipes/diar_ssl_pruning/run_stage.sh`. 
- For inference, our model supports for [Hugging Face](https://huggingface.co/BUT-FIT/diarizen-wavlm-large-s80-md) 🤗. See below: 
```python
from diarizen.pipelines.inference import DiariZenPipeline

# load pre-trained model
diar_pipeline = DiariZenPipeline.from_pretrained("BUT-FIT/diarizen-wavlm-large-s80-md")

# apply diarization pipeline
diar_results = diar_pipeline('./example/EN2002a_30s.wav')

# print results
for turn, _, speaker in diar_results.itertracks(yield_label=True):
    print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")
# start=0.0s stop=2.7s speaker_0
# start=0.8s stop=13.6s speaker_3
# start=5.8s stop=6.4s speaker_0
# ...

# load pre-trained model and save RTTM result
diar_pipeline = DiariZenPipeline.from_pretrained(
        "BUT-FIT/diarizen-wavlm-large-s80-md",
        rttm_out_dir='.'
)
# apply diarization pipeline
diar_results = diar_pipeline('./example/EN2002a_30s.wav', sess_name='EN2002a')
```


## Benchmark 
We train DiariZen models on a compound dataset composed of the datasets listed in the table below, followed by structured pruning to remove redundant parameters. For the results below: 
- AISHELL-4 was converted to mono using `sox in.wav -c 1 out.wav`.
- [NOTSOFAR-1](https://www.chimechallenge.org/challenges/chime8/task2/data) contains **only single-channel** recordings, e.g. `sc_plaza_0`, `sc_rockfall_0`.
- Diarization Error Rate (DER) is evaluated **without** applying a collar.
- **No domain adaptation** is applied to any individual dataset.
- All experiments use the **same clustering hyperparameters** across datasets.

| Dataset       | [Pyannote v3.1](https://github.com/pyannote/pyannote-audio) | [DiariZen-Base-s80](https://huggingface.co/BUT-FIT/diarizen-wavlm-base-s80-md) |[DiariZen-Large-s80](https://huggingface.co/BUT-FIT/diarizen-wavlm-large-s80-md) | [DiariZen-Large-s80-v2](https://huggingface.co/BUT-FIT/diarizen-wavlm-large-s80-md-v2)
|:---------------|:-----------:|:-----------:|:-----------:| :-----------:|
| AMI-SDM           | 22.4      | 15.8 | 14.0 | 13.9 |
| AISHELL-4     | 12.2      | 10.7 | 9.8 | 10.1 |
| AliMeeting far    | 24.4      | 14.1 | 12.5 | **10.8** | 
| NOTSOFAR-1    | -      | 20.3 | 17.9 | **16.7** |
| MSDWild       | 25.3      | 17.4 | 15.6 | 15.8 |
| DIHARD3 full      | 21.7      | 15.9 | 14.5 | 14.5 |
| RAMC          | 22.2      | 11.4 | 11.0 | 11.0 |
| VoxConverse   | 11.3      | 9.7 | 9.2 | 9.1 |

## Updates
2025-12-09: Updated benchmarks with DiariZen-Large-s80-v2.

2025-06-03: Uploaded structured pruning recipes, released new pre-trained models, and updated multiple benchmark results.

## Citations
If you found this work helpful, please consider citing
```
@inproceedings{han2025leveraging,
  title={Leveraging self-supervised learning for speaker diarization},
  author={Han, Jiangyu and Landini, Federico and Rohdin, Johan and Silnova, Anna and Diez, Mireia and Burget, Luk{\'a}{\v{s}}},
  booktitle={Proc. ICASSP},
  year={2025}
}

@article{han2025fine,
  title={Fine-tune Before Structured Pruning: Towards Compact and Accurate Self-Supervised Models for Speaker Diarization},
  author={Han, Jiangyu and Landini, Federico and Rohdin, Johan and Silnova, Anna and Diez, Mireia and Cernocky, Jan and Burget, Lukas},
  journal={arXiv preprint arXiv:2505.24111},
  year={2025}
}

@article{han2025efficient,
  title={Efficient and Generalizable Speaker Diarization via Structured Pruning of Self-Supervised Models},
  author={Han, Jiangyu and P{\'a}lka, Petr and Delcroix, Marc and Landini, Federico and Rohdin, Johan and Cernock{\`y}, Jan and Burget, Luk{\'a}{\v{s}}},
  journal={arXiv preprint arXiv:2506.18623},
  year={2025}
}

```


## License
- The **code** in this repository is licensed under the [MIT license](https://github.com/BUTSpeechFIT/DiariZen/blob/main/LICENSE).
- The **pre-trained model weights** are released strictly for **research and non-commercial use only**, in accordance with the licenses of the datasets used for training. Commercial use of the model weights is prohibited. See [MODEL_LICENSE](https://github.com/BUTSpeechFIT/DiariZen/blob/main/MODEL_LICENSE) for details.

## Contact
If you have any comment or question, please contact ihan@fit.vut.cz
