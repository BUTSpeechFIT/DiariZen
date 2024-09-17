## DiariZen
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
```

## Usage

- Leveraging Self-Supervised Learning for Speaker Diarization, submitted to ICASSP 2025 [[paper]](https://arxiv.org/pdf/2409.09408) [[recipe]](https://github.com/BUTSpeechFIT/DiariZen/tree/main/recipes/diar_ssl)


## License
This repository under the [MIT license](https://github.com/BUTSpeechFIT/DiariZen/blob/main/LICENSE).

## Contact
If you have any comment or question, please contact ihan@fit.vut.cz
