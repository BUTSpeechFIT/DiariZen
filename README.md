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
We use **SDM (first channel from the first far-field microphone array)** data from public [AMI](https://github.com/pyannote/AMI-diarization-setup/tree/main/pyannote), [AISHELL-4](https://www.openslr.org/111/), and [AliMeeting](https://openslr.org/119/) for model training and evaluation. Please download these datasets firstly. Our [data partition](https://github.com/BUTSpeechFIT/DiariZen/tree/main/recipes/diar_ssl/data/AMI_AliMeeting_AISHELL4) is also provided.

## Usage
- download [WavLM Base+ model](https://github.com/microsoft/unilm/blob/master/wavlm/README.md)
- download [ResNet34-LM model](https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM)
- modify the path of used dataset and configuration file
- `cd recipes/diar_ssl && bash -i run_stage.sh`

## Pre-trained 
- our pre-trained checkpoints and the estimated rttm files can be found [here](https://1drv.ms/f/s!Al8zHxdaFGuCiyQNBeav1eEB1Uiv?e=wsBhVU). The local experimental path has been anonymized. To use the pre-trained models, please check the `diar_ssl/run_stage.sh`.
- in case you have trouble reproducing our experiments, we also provide the [intermediate results](https://onedrive.live.com/?authkey=%21APzNfdtjBpOxoTc&id=826B145A171F335F%211486&cid=826B145A171F335F) of `EN2002a`, an AMI test recording,  during inference for debugging.   

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
              WavLM-updated    9.8     5.9       10.0
--------------------------------------------------------------
Note:
The results above are different from our ICASSP submission. 
We made a few updates to experimental numbers but the conclusions in our paper are as same as the original ones.
```

## Citation
If you found this work helpful, please consider citing:
J. Han, F. Landini, J. Rohdin, A. Silnova, M. Diez, and L. Burget, [Leveraging Self-Supervised Learning for Speaker Diarization](https://arxiv.org/pdf/2409.09408), arXiv preprint arXiv:2409.09408, 2024.
```
@article{han2024leveragingselfsupervisedlearningspeaker,
      title={Leveraging Self-Supervised Learning for Speaker Diarization}, 
      author={Jiangyu Han and Federico Landini and Johan Rohdin and Anna Silnova and Mireia Diez and Lukas Burget},
      journal={arXiv preprint arXiv:2409.09408},
      year={2024}
}
```


## License
This repository under the [MIT license](https://github.com/BUTSpeechFIT/DiariZen/blob/main/LICENSE).

## Contact
If you have any comment or question, please contact ihan@fit.vut.cz
