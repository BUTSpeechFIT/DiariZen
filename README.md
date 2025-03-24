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

# install dscore
git submodule init
git submodule update
```

## Datasets
We use **SDM (first channel from the first far-field microphone array)** data from public [AMI](https://github.com/pyannote/AMI-diarization-setup/tree/main/pyannote), [AISHELL-4](https://www.openslr.org/111/), and [AliMeeting](https://openslr.org/119/) for model training and evaluation. Please download these datasets firstly. Our data partition is [here](https://github.com/BUTSpeechFIT/DiariZen/tree/main/recipes/diar_ssl/data/AMI_AliMeeting_AISHELL4).

## Usage
- download [WavLM Base+ model](https://github.com/microsoft/unilm/blob/master/wavlm/README.md)
- download [ResNet34-LM model](https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM)
- modify the path of used dataset and configuration file
- `cd recipes/diar_ssl && bash -i run_stage.sh`


## Pre-trained 
- Our pre-trained checkpoints and the estimated rttm files can be found [here](https://1drv.ms/f/s!Al8zHxdaFGuCi1W9tTb7TGcy1b_a?e=8pcjK0). The local experimental path has been anonymized. To use the pre-trained models, please check the `diar_ssl/run_stage.sh`.
- In case you have trouble reproducing our experiments, we also provide the [intermediate results](https://1drv.ms/f/s!Al8zHxdaFGuCi078zX3bYwaTsaE3?e=moycQ7) of `EN2002a`, an AMI test recording,  during inference for debugging.   
- Our model also supports for [Hugging Face](https://huggingface.co/BUT-FIT/diarizen-meeting-base) 🤗. Please check `example/run_example.py`.

## Results (SDM)
We aim to make the whole pipeline as simple as possible. Therefore, for the results below: 
- we **did not** use any simulated data
- we **did not** apply advanced learning scheduler strategies
- we **did not** perform further domain adaptation to each dataset 
- all experiments share the **same hyper-parameters** for clustering
``` 
collar=0s                           
--------------------------------------------------------------
System         Features       AMI   AISHELL-4   AliMeeting         
--------------------------------------------------------------
Pyannote3       SincNet       21.1     13.9       22.8

Proposed         Fbank        19.7     12.5       21.0
              WavLM-frozen    17.0     11.7       19.9
              WavLM-updated   15.4     11.7       17.6
--------------------------------------------------------------

collar=0.25s 
--------------------------------------------------------------
System         Features       AMI   AISHELL-4   AliMeeting         
--------------------------------------------------------------
Pyannote3       SincNet       13.7     7.7       13.6

Proposed         Fbank        12.9     6.9       12.6
              WavLM-frozen    10.9     6.1       12.0
              WavLM-updated    9.8     5.9       10.2
--------------------------------------------------------------
Note:
The results above are different from our ICASSP submission. 
We made a few updates to experimental numbers but the conclusions in our paper are as same as the original ones.
```

## Citation
If you found this work helpful, please consider citing:
J. Han, F. Landini, J. Rohdin, A. Silnova, M. Diez, and L. Burget, [Leveraging Self-Supervised Learning for Speaker Diarization](https://arxiv.org/pdf/2409.09408), in Proc. ICASSP. IEEE, 2025, pp. 1–5.
```
@inproceedings{han2025leveraging,
  title={Leveraging self-supervised learning for speaker diarization},
  author={Han, Jiangyu and Landini, Federico and Rohdin, Johan and Silnova, Anna and Diez, Mireia and Burget, Luk{\'a}{\v{s}}},
  booktitle={ICASSP 2025-2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={1--5},
  year={2025},
  organization={IEEE}
}
```


## License
This repository under the [MIT license](https://github.com/BUTSpeechFIT/DiariZen/blob/main/LICENSE).

## Contact
If you have any comment or question, please contact ihan@fit.vut.cz
