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
- Diarization Error Rate (DER) is evaluated **without** applying a collar.
- **No domain adaptation** is applied to any individual dataset.
- All experiments use the **same clustering hyperparameters** across datasets.

| Dataset       | [Pyannote v3.1](https://github.com/pyannote/pyannote-audio) | [DiariZen-Base-s80](https://huggingface.co/BUT-FIT/diarizen-wavlm-base-s80-md) |[DiariZen-Large-s80](https://huggingface.co/BUT-FIT/diarizen-wavlm-large-s80-md) |
|:---------------|:-----------:|:-----------:|:-----------:|
| AMI-SDM           | 22.4      | 15.8 | 14.0 |
| AISHELL-4*     | 12.2      | 10.7 | 9.8 |
| AliMeeting far    | 24.4      | 14.1 | 12.5 | 
| NOTSOFAR-1    | -      | 20.3 |   17.9 |
| MSDWild       | 25.3      | 17.4 | 15.6 |
| DIHARD3 full      | 21.7      | 15.9 | 14.5 |
| RAMC          | 22.2      | 11.4 | 11.0 |
| VoxConverse   | 11.3      | 9.7 | 9.2 |
\* AISHELL-4 was converted to mono using `sox in.wav -c 1 out.wav`

## Updates
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

@inproceedings{han2025finetunestructuredpruningcompact,
  title={Fine-tune Before Structured Pruning: Towards Compact and Accurate Self-Supervised Models for Speaker Diarization},
  author={Jiangyu Han and Federico Landini and Johan Rohdin and Anna Silnova and Mireia Diez and Jan Cernocky and Lukas Burget},
  booktitle={Proc. INTERSPEECH},
  year={2025}
}

```


## License
This repository under the [MIT license](https://github.com/BUTSpeechFIT/DiariZen/blob/main/LICENSE).

## Contact
If you have any comment or question, please contact ihan@fit.vut.cz
