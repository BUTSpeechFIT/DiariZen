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

## Usage
- download [WavLM Base+ model](https://github.com/microsoft/unilm/blob/master/wavlm/README.md)
- download [ResNet34-LM model](https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM)
- modify the path of used dataset and configuration file
- `cd recipes/diar_ssl && bash -i run_stage.sh`

## Pre-trained 
Our pre-trained checkpoints and the estimated rttm files can be found [here](https://1drv.ms/f/s!Al8zHxdaFGuCiyQNBeav1eEB1Uiv?e=wsBhVU). The local experimental path has been anonymized. To use the pre-trained models, please check the `diar_ssl/run_stage.sh`.

## Results
```
collar=0s                           
--------------------------------------------------------------
System         Features       AMI   AISHELL-4   AliMeeting         
--------------------------------------------------------------
Pyannote3       SincNet       21.3     13.7       23.7

Proposed         Fbank        19.8     12.2       21.0
              WavLM-frozen    17.0     11.6       18.7
              WavLM-updated   15.4     11.5       16.0
--------------------------------------------------------------

collar=0.25s 
--------------------------------------------------------------
System         Features       AMI   AISHELL-4   AliMeeting         
--------------------------------------------------------------
Pyannote3       SincNet       13.8     7.5       14.5

Proposed         Fbank        13.1     6.6       12.4
              WavLM-frozen    10.9     5.9       10.8
              WavLM-updated    9.9     5.7        8.2
--------------------------------------------------------------
Note:
The results above are different from our ICASSP submission. 
We made a few updates to experimental numbers on AMI/AISHELL-4/AliMeeting datasets. 
The conclusions in our paper are as same as the original ones.
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
