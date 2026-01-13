#!/bin/bash

# Define parameters
DATASET="reallad"
CLASS_NAME="pcb_reallad"
GPU_ID=2

# Define k-shot and seed values
K_SHOTS=(4) # 16 32 64
SEEDS=(111 222 333)

for SEED in "${SEEDS[@]}"; do
  for K_SHOT in "${K_SHOTS[@]}"; do
    echo "Running training with k-shot=$K_SHOT and seed=$SEED"
    python train_cls.py --dataset $DATASET --k-shot $K_SHOT --class_name $CLASS_NAME --seed $SEED --gpu-id $GPU_ID --bank 16 --root-dir ./all_logs/cls_lambda1_0.1 --lambda1 0.1
  done
done


for SEED in "${SEEDS[@]}"; do
  for K_SHOT in "${K_SHOTS[@]}"; do
    echo "Running training with k-shot=$K_SHOT and seed=$SEED"
    python train_seg.py --dataset $DATASET --k-shot $K_SHOT --class_name $CLASS_NAME --seed $SEED --gpu-id $GPU_ID --bank 16 --root-dir ./all_logs/seg_lambda1_0.1 --lambda1 0.1
  done
done