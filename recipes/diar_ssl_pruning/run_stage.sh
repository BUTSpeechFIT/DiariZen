#!/bin/bash

# Licensed under the MIT license.
# Copyright 2025 Brno University of Technology (author: Jiangyu Han, ihan@fit.vut.cz)

set -eu
ulimit -n 2048

# general setup
stage=1
recipe_root=/YOUR_PATH/DiariZen/recipes/diar_ssl_pruning
exp_root=$recipe_root/exp
conf_dir=$recipe_root/conf

dual_opt_common_conf=$conf_dir/dual_opt_common.toml

# training setup
train_conf=$conf_dir/s80_base.toml
# train_conf=$conf_dir/s80_large.toml

conf_name=`ls $train_conf | awk -F '/' '{print $NF}' | awk -F '.' '{print $1}'`
diarization_dir=$exp_root/$conf_name

# inference setup
dtype=test
data_dir=$recipe_root/data/AMI_AliMeeting_AISHELL4
seg_duration=8

# clustering setup
clustering_method=AgglomerativeClustering
ahc_threshold=0.70
min_cluster_size=30
infer_affix=_${clustering_method}_seg${seg_duration}_thres_${ahc_threshold}_mcs_${min_cluster_size}

avg_ckpt_num=5
val_metric=Loss   # Loss or DER
val_mode=best   # [prev, best, center]  

# scoring setup
collar=0
REF_DIR=$data_dir
dscore_dir=/YOUR_PATH/DiariZen/dscore

# =======================================
# =======================================
if [ $stage -le 1 ]; then
    echo "stage1-1: distillation and pruning..."
    conda activate diarizen && CUDA_VISIBLE_DEVICES="0,1,2,3" accelerate launch \
        --num_processes 4 --main_process_port 1134 \
        run_distill_prune.py -C $train_conf -M train 

    echo "stage1-2: model pruning..."
    config_dir=`ls $diarization_dir/*.toml | sort -r | head -n 1`
    train_log=`du -h $diarization_dir/*.log | sort -rh | head -n 1 | awk '{print $NF}'`
    cat $train_log | grep 'Validation Loss | Loss_distill' | awk -F ']:' '{print $NF}' > $diarization_dir/val_metric_summary.lst
    conda activate diarizen && python apply_pruning.py -C $config_dir \
            -o ${diarization_dir}/pruned \
            --mode prune \
            --avg_ckpt_num $avg_ckpt_num \
            --val_metric_summary $diarization_dir/val_metric_summary.lst 
fi

further_distill_dir=$exp_root/${conf_name}_further_distill
if [ $stage -le 2 ]; then
    echo "stage2-1: further distill..."
    conf_further_distill=$conf_dir/${conf_name}_further_distill.toml
    cp $train_conf $conf_further_distill
    conda activate diarizen && CUDA_VISIBLE_DEVICES="0,1,2,3" accelerate launch \
        --num_processes 4 --main_process_port 1134 \
        run_distill_prune.py -C $conf_further_distill -M train \
            --batch_size 16 \
            --max_epochs 20 \
            --pruned_ckpt_path ${diarization_dir}/pruned/pytorch_model.bin 

    echo "stage2-2: save the further distilled model..."
    further_distill_config_dir=`ls $further_distill_dir/*.toml | sort -r | head -n 1`
    train_log=`du -h $further_distill_dir/*.log | sort -rh | head -n 1 | awk '{print $NF}'`
    cat $train_log | grep 'Validation Loss | Loss_distill' | awk -F ']:' '{print $NF}' > $further_distill_dir/val_metric_summary.lst
    conda activate diarizen && python apply_pruning.py -C $further_distill_config_dir \
            -o ${further_distill_dir}/avg_student_ckpt \
            --mode extract \
            --avg_ckpt_num $avg_ckpt_num \
            --val_metric_summary $further_distill_dir/val_metric_summary.lst  
fi

conf_continued_training=$conf_dir/${conf_name}_further_distill_diar_training.toml
if [ $stage -le 3 ]; then
    echo "stage3: diarization training..."
    cp $dual_opt_common_conf $conf_continued_training
    conda activate diarizen && CUDA_VISIBLE_DEVICES="0,1,2,3" accelerate launch \
        --num_processes 4 --main_process_port 1134 \
        run_dual_opt_continued_training.py -C $conf_continued_training -M train \
            --pruned_ckpt_path ${further_distill_dir}/avg_student_ckpt/pytorch_model.bin 
fi

conf_name_continued_training=`ls $conf_continued_training | awk -F '/' '{print $NF}' | awk -F '.' '{print $1}'`
diarization_dir_continued_training=$exp_root/$conf_name_continued_training
config_dir_continued_training=`ls $diarization_dir_continued_training/*.toml | sort -r | head -n 1`
embedding_model=/YOUR_PATH/pretrained/pyannote3/wespeaker-voxceleb-resnet34-LM/pytorch_model.bin     # it's necessary to have "pyannote" in your directory path
if [ $stage -le 4 ]; then
    echo "stage4-1: model inference..."
    export CUDA_VISIBLE_DEVICES=0

    train_log=`du -h $diarization_dir_continued_training/*.log | sort -rh | head -n 1 | awk '{print $NF}'`
    cat $train_log | grep 'Loss/DER' | awk -F ']:' '{print $NF}' > $diarization_dir_continued_training/val_metric_summary.lst

    for dset in AMI AliMeeting AISHELL4; do
        conda activate diarizen && python infer_avg.py -C $config_dir_continued_training \
            -i ${data_dir}/${dtype}/${dset}/wav.scp \
            -o ${diarization_dir_continued_training}/infer$infer_affix/metric_${val_metric}_${val_mode}/avg_ckpt${avg_ckpt_num}/${dtype}/${dset} \
            --embedding_model $embedding_model \
            --avg_ckpt_num $avg_ckpt_num \
            --val_metric $val_metric \
            --val_mode $val_mode \
            --val_metric_summary $diarization_dir_continued_training/val_metric_summary.lst \
            --seg_duration $seg_duration \
            --clustering_method $clustering_method \
            --ahc_threshold $ahc_threshold \
            --min_cluster_size $min_cluster_size 

        echo "stage4-2: scoring..."
        SYS_DIR=${diarization_dir_continued_training}/infer$infer_affix/metric_${val_metric}_${val_mode}/avg_ckpt${avg_ckpt_num}
        OUT_DIR=${SYS_DIR}/${dtype}/${dset}
        conda activate diarizen && python ${dscore_dir}/score.py \
            -r ${REF_DIR}/${dtype}/${dset}/rttm \
            -s $OUT_DIR/*.rttm --collar ${collar} \
            > $OUT_DIR/result_collar${collar}
    done
fi
