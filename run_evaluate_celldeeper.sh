#!/bin/bash
#BSUB -q i2c2_gpu
#BSUB -J fcs_eval
#BSUB -o logs/result%J.o
#BSUB -e logs/result%J.e
#BSUB -gpu "num=10"

source activate covid19

CUDA_VISIBLE_DEVICES=6 python run_class_finetuning.py \
    --eval \
    --data_path ~/covid19/data/vaccine_covid_cytof_deeper_with_labels/ \
    --finetune ~/covid19/MAE/ckpts/vit_base_patch16_224_pretrained0.75_celldeeper_fold_0_checkpoint-best.pth \
    --fold 0 \
    --task cell-level \
    --nb_classes 46 \
    --batch_size 768

CUDA_VISIBLE_DEVICES=6 python run_class_finetuning.py \
    --eval \
    --data_path ~/covid19/data/vaccine_covid_cytof_deeper_with_labels/ \
    --finetune ~/covid19/MAE/ckpts/vit_base_patch16_224_pretrained0.5_celldeeper_fold_0_checkpoint-best.pth \
    --fold 0 \
    --task cell-level \
    --nb_classes 46 \
    --batch_size 768

CUDA_VISIBLE_DEVICES=6 python run_class_finetuning.py \
    --eval \
    --data_path ~/covid19/data/vaccine_covid_cytof_deeper_with_labels/ \
    --finetune ~/covid19/MAE/ckpts/vit_base_patch16_224_pretrained0.25_celldeeper_fold_0_checkpoint-best.pth \
    --fold 0 \
    --task cell-level \
    --nb_classes 46 \
    --batch_size 768
