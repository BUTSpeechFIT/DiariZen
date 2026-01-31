#!/bin/bash

# Licensed under the MIT license.
# Copyright 2024 Brno University of Technology (author: Jiangyu Han, ihan@fit.vut.cz)

set -eu
ulimit -n 2048

# general setup
stage=1
recipe_root=/YOUR_PATH/DiariZen/recipes/diar_ssl_mc
exp_root=$recipe_root/exp
conf_dir=$recipe_root/conf

# training setup
use_dual_opt=true  
train_conf=$conf_dir/channel_att_hidden_256_ft_base_8sec.toml

conf_name=`ls $train_conf | awk -F '/' '{print $NF}' | awk -F '.' '{print $1}'`

# inference setup
dtype=test
data_dir=$recipe_root/data/AMI_AIS_ALI_NSF_CHiME7
seg_duration=8

# clustering setup
clustering_method=VBxClustering
ahc_threshold=0.6
Fa=0.06
Fb=0.9
infer_affix=_vbx_thres_${ahc_threshold}_Fa_${Fa}_Fb_${Fb}

avg_ckpt_num=5
val_metric=Loss   # Loss or DER
val_mode=best   # [prev, best, center]  

# scoring setup
REF_DIR=$data_dir
dscore_dir=/YOUR_PATH/DiariZen/dscore


# =======================================
# =======================================
if [ $stage -le 1 ]; then
    if (! $use_dual_opt); then
        echo "stage1: use single-opt for model training..."
        conda activate diarizen && CUDA_VISIBLE_DEVICES="0,1" accelerate launch \
            --num_processes 2 --main_process_port 1134 \
            run_single_opt.py -C $train_conf -M validate
    else
        echo "stage1: use dual-opt for model training..."
        conda activate diarizen && accelerate launch \
            --num_processes 4 --main_process_port 1134 \
            run_dual_opt.py -C $train_conf -M train 
    fi
fi

diarization_dir=$exp_root/$conf_name    # can be replaced by our pre-trained models, e.g. diarization_dir=/YOUR_PATH/checkpoints/wavlm_updated_conformer
config_dir=`ls $diarization_dir/*.toml | sort -r | head -n 1`
diarizen_hub=/YOUR_PATH/huggingface_hub/diarizen-wavlm-large-s80-md
embedding_model=/YOUR_PATH/pretrained/pyannote3/wespeaker-voxceleb-resnet34-LM/pytorch_model.bin     # it's necessary to have "pyannote" in your directory pat

if [ $stage -le 2 ]; then
    echo "stage2: model inference..."
    export CUDA_VISIBLE_DEVICES=0

    train_log=`du -h $diarization_dir/*.log | sort -rh | head -n 1 | awk '{print $NF}'`
    cat $train_log | grep 'Loss/DER' | awk -F ']:' '{print $NF}' > $diarization_dir/val_metric_summary.lst

    for dset in AMI AISHELL4 AliMeeting CHiME7 NOTSOFAR1; do
        conda activate diarizen && python infer_avg.py -C $config_dir \
            -i ${data_dir}/${dtype}/${dset}/wav.scp \
            -o ${diarization_dir}/infer$infer_affix/metric_${val_metric}_${val_mode}/avg_ckpt${avg_ckpt_num}/${dtype}/${dset} \
            --embedding_model $embedding_model \
            --avg_ckpt_num $avg_ckpt_num \
            --val_metric $val_metric \
            --val_mode $val_mode \
            --val_metric_summary $diarization_dir/val_metric_summary.lst \
            --seg_duration $seg_duration \
            --diarizen_hub $diarizen_hub \
            --clustering_method $clustering_method \
            --ahc_threshold $ahc_threshold \
            --Fa $Fa \
            --Fb $Fb 

        echo "stage3: scoring..."
        SYS_DIR=${diarization_dir}/infer$infer_affix/metric_${val_metric}_${val_mode}/avg_ckpt${avg_ckpt_num}
        OUT_DIR=${SYS_DIR}/${dtype}/${dset}
        for collar in 0 0.25; do
            conda activate diarizen && python ${dscore_dir}/score.py \
                -r ${REF_DIR}/${dtype}/${dset}/rttm \
                -s $OUT_DIR/*.rttm --collar ${collar} \
                > $OUT_DIR/result_collar${collar}
        done
    done
fi
