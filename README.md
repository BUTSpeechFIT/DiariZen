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
Pyannote3       SincNet       22.5     14.4       25.1

Proposed         Fbank        21.7     13.0       22.7
              WavLM-frozen    19.1     12.5       20.3
              WavLM-updated   17.5     12.4       18.3
--------------------------------------------------------------

collar=0.25s 
--------------------------------------------------------------
System         Features       AMI   AISHELL-4   AliMeeting         
--------------------------------------------------------------
Pyannote3       SincNet       14.6     8.1       15.4

Proposed         Fbank        14.4     7.2       13.5
              WavLM-frozen    12.5     6.6       11.8
              WavLM-updated   11.3     6.4       8.9
--------------------------------------------------------------
```

## Citation
If you found this work helpful, please consider citing
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
