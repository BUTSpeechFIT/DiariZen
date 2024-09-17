#!/bin/bash

# Licensed under the MIT license.
# Copyright 2024 Brno University of Technology (author: Jiangyu Han, ihan@fit.vut.cz)

set -eu
ulimit -n 2048

# general setup
stage=1
recipe_root=/YOUR_PATH/DiariZen/recipes/diar_ssl
exp_root=$recipe_root/exp
conf_dir=$recipe_root/conf

# training setup
use_dual_opt=true  # true for wavlm_updated_conformer.toml; false for the others
train_conf=$conf_dir/wavlm_updated_conformer.toml
# train_conf=$conf_dir/wavlm_frozen_conformer.toml
# train_conf=$conf_dir/fbank_conformer.toml
# train_conf=$conf_dir/pyannote_baseline.toml

conf_name=`ls $train_conf | awk -F '/' '{print $NF}' | awk -F '.' '{print $1}'`

# inference setup
dtype=test
data_dir=$recipe_root/data/AMI_AliMeeting_AISHELL4

pyan_merge_closer=0.5
pyan_max_length_merged=50
pyan_inf_max_batch=32

cluster_threshold=0.70
segmentation_step=0.25
infer_affix=_segmentation_step_${segmentation_step}_AHC_thres_${cluster_threshold}_pyan_max_length_merged${pyan_max_length_merged}

avg_ckpt_num=5
val_metric=Loss   # Loss or DER
val_mode=prev   # [prev, best, center]  

# scoring setup
collar=0
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
        conda activate diarizen && CUDA_VISIBLE_DEVICES="0,1,2,3" accelerate launch \
            --num_processes 4 --main_process_port 1134 \
            run_dual_opt.py -C $train_conf -M train
    fi
fi

diarization_dir=$exp_root/$conf_name    # can be replaced by our pre-trained models, e.g. diarization_dir=/YOUR_PATH/checkpoints/wavlm_updated_conformer
config_dir=`ls $diarization_dir/*.toml | sort -r | head -n 1`
segmentation_model=$diarization_dir/checkpoints/best/pytorch_model.bin
embedding_model=/YOUR_PATH/wespeaker-voxceleb-resnet34-LM/pytorch_model.bin

if [ $stage -le 2 ]; then
    echo "stage2: model inference..."
    export CUDA_VISIBLE_DEVICES=0

    train_log=`du -h $diarization_dir/*.log | sort -rh | head -n 1 | awk '{print $NF}'`
    cat $train_log | grep 'Loss/DER' | awk -F ']:' '{print $NF}' > $diarization_dir/val_metric_summary.lst

    for dset in AMI AliMeeting AISHELL4; do
        conda activate diarizen && python infer_avg.py -C $config_dir \
            -i ${data_dir}/${dtype}/${dset}/wav.scp \
            -o ${diarization_dir}/infer$infer_affix/metric_${val_metric}_${val_mode}/avg_ckpt${avg_ckpt_num}/${dtype}/${dset} \
            -u ${data_dir}/${dtype}/${dset}/all.uem \
            --avg_ckpt_num $avg_ckpt_num \
            --val_metric $val_metric \
            --val_mode $val_mode \
            --val_metric_summary $diarization_dir/val_metric_summary.lst \
            --segmentation_step $segmentation_step \
            --cluster_threshold $cluster_threshold \
            --segmentation_model $segmentation_model \
            --embedding_model $embedding_model \
            --merge_closer $pyan_merge_closer \
            --max_length_merged $pyan_max_length_merged \
            --batch_size $pyan_inf_max_batch

        echo "stage3: scoring..."
        SYS_DIR=${diarization_dir}/infer$infer_affix/metric_${val_metric}_${val_mode}/avg_ckpt${avg_ckpt_num}
        OUT_DIR=${SYS_DIR}/${dtype}/${dset}
        conda activate diarizen && python ${dscore_dir}/score.py \
            -r ${REF_DIR}/${dtype}/${dset}/rttm \
            -s $OUT_DIR/*.rttm --collar ${collar} \
            > $OUT_DIR/result_collar${collar}
    done
fi
