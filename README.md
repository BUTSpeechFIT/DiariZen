## DiariZen
DiariZen is a toolkit for speaker diarization. The model training and inference are driven by [AudioZen](https://github.com/haoxiangsnr/spiking-fullsubnet) and [Pyannote 3.1](https://huggingface.co/pyannote/speaker-diarization-3.1), respectively. 


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
```
Paper: Leveraging Self-Supervised Learning for Speaker Diarization
Recipes: cd recipes/diar_ssl && bash -i run_stage.sh
```

## License
This repository under the MIT license.

## Contact
If you have any comment or question, please contact ihan@fit.vut.cz
