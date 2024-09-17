## Usage
- download [WavLM Base+ model](https://github.com/microsoft/unilm/blob/master/wavlm/README.md)
- download [ResNet34-LM model](https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM)
- modify the path of used dataset and configuration file
- run training, inference, and evaluation in one script. `bash -i run_stage.sh`

## Pre-trained
Due to the limit our model size, it's hard to upload the pre-trained checkpoints to an icloud space. However, the estimated RTTM files are provided for a reference. See `pt_rttms` for more information.

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