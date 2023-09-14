#!/bin/bash
#BSUB -q i2c2_gpu
#BSUB -J covid19_fcs
#BSUB -o logs/covid19_fcs%J.o
#BSUB -e logs/covid19_fcs%J.e
#BSUB -gpu "num=9"

source activate covid19

CUDA_VISIBLE_DEVICES=1,2,3 OMP_NUM_THREADS=1 CUDA_LAUNCH_BLOCKING=1 torchrun --nproc_per_node=3 --master_port=25642 run_mae_pretraining.py \
    --epochs 250 \
    --warmup_epochs 5 \
    --batch_size 768 \
    --mask_ratio 0.5 \
    --lr 1.5e-6 \
    --is_ZIGloss \
    --resume /home/jaesik/covid19/MAE/ckpts/pretrain_mae_base_patch16_224_0.75_ZIGloss_checkpoint-90.pth

#CUDA_VISIBLE_DEVICES=7 python run_mae_pretraining.py \
#    --epochs 50 \
#    --warmup_epochs 5 \
#    --batch_size 8 \
#    --mask_ratio 0.5 \
#    --is_ZIGloss
