from diarizen.pipelines.inference import DiariZenPipeline
from huggingface_hub import snapshot_download, hf_hub_download
from pathlib import Path
import toml
import os
import sys

if len(sys.argv) == 5:
   data_dir_path=sys.argv[1]
   wav_file_name =sys.argv[2]
   out_rttm_dir = sys.argv[3]
   config_parse_path = Path(sys.argv[4])
else:
   print(" Run the file as follows \n \t\t python3 inference.py data_dir_path wav_file_name out_rttm_dir config_parse_path")
   sys.exit(0)
   

out_rttm_dir = Path(out_rttm_dir)  
score = True

if not os.path.exists(out_rttm_dir):
    os.makedirs(out_rttm_dir)

repo_id = "BUT-FIT/diarizen-wavlm-base-s80-md"
cache_dir=None

diarizen_hub = snapshot_download(
            repo_id=repo_id,
            cache_dir=cache_dir,
            local_files_only=cache_dir is not None)

embedding_model = hf_hub_download(
            repo_id="pyannote/wespeaker-voxceleb-resnet34-LM",
            filename="pytorch_model.bin",
            cache_dir=cache_dir,
            local_files_only=cache_dir is not None
        )

# load pre-trained model and save RTTM result
diar_pipeline = DiariZenPipeline(Path(diarizen_hub).expanduser().absolute(), embedding_model, config_parse = toml.load(config_parse_path.as_posix()))

for rec in os.listdir(os.path.join(data_dir_path, wav_file_name) ):
    rec_path = os.path.join(data_dir_path,wav_file_name, rec)
    
    # apply diarization pipeline
    sess_name=rec[:rec.index('.')]
    diar_results = diar_pipeline(rec_path, sess_name=sess_name)
    rttm_out = os.path.join(out_rttm_dir, sess_name + ".rttm") #'min'+str(min_spk) + '_max' + str(max_spk) + '_' + sess_name + ".rttm")
    with open(rttm_out, "w") as f:
        f.write(diar_results.to_rttm())