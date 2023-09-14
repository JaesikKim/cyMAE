#!/bin/bash
#BSUB -q i2c2_gpu
#BSUB -J covid19_fcs
#BSUB -o logs/covid19_fcs%J.o
#BSUB -e logs/covid19_fcs%J.e
#BSUB -gpu "num=10"

source activate covid19

CUDA_VISIBLE_DEVICES=7 python run_class_finetuning.py \
    --fold 0 \
    --task cell-level \
    --save_ckpt_freq 5 \
    --lr 5e-5 \
    --min_lr 1e-7 \
    --nb_classes 46 \
    --epochs 100 \
    --warmup_epochs 2 \
    --batch_size 768 \
    --finetune ~/covid19/MAE/ckpts/pretrain_mae_base_patch16_224_0.75_checkpoint-150.pth \
    --data_path ~/covid19/data/vaccine_covid_cytof_deeper_with_labels/

# CUDA_VISIBLE_DEVICES=1 python run_class_finetuning.py \
#     --fold 0 \
#     --task cell-level \
#     --save_ckpt_freq 1 \
#     --lr 1e-5 \
#     --nb_classes 46 \
#     --epochs 5 \
#     --warmup_epochs 2 \
#     --batch_size 768 \
#     --finetune ~/covid19/MAE/ckpts/pretrain_mae_base_patch16_224_0.75_checkpoint-150.pth \
#     --freeze


#CUDA_VISIBLE_DEVICES=6,7,8,9 OMP_NUM_THREADS=1 CUDA_LAUNCH_BLOCKING=1 torchrun --nproc_per_node=4 run_class_finetuning.py \
#    --finetune /home/jaesik/covid19/MAE/ckpts/pretrain_mae_base_patch16_224_0.75_checkpoint-150.pth \
#    --fold 0 \
#    --task cell-level \
#    --save_ckpt_freq 5 \
#    --nb_classes 13 \
#    --epochs 10 \
#    --warmup_epochs 5 \
#    --batch_size 768
    #--resume 
 
#CUDA_VISIBLE_DEVICES=9 python run_class_finetuning.py \
#    --finetune /home/jaesik/covid19/MAE/ckpts/pretrain_mae_base_patch16_224_0.75_checkpoint-150.pth \
#    --fold 0 \
#    --task cell-level \
#    --save_ckpt_freq 5 \
#    --nb_classes 13 \
#    --epochs 10 \
#    --warmup_epochs 5 \
#    --batch_size 768
