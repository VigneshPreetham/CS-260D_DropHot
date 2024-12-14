#!/bin/bash

MODELS=("resnet50")
DATASETS=("CIFAR10")
DATA_RATIOS=("0.1" "0.2" "0.3" "0.4" "0.5")
PRUNE_RATIOS=("0.0" "0.2" "0.4" "0.6" "0.8")

for MODEL in "${MODELS[@]}"; do
    for DATASET in "${DATASETS[@]}"; do
        for DATA_RATIO in "${DATA_RATIOS[@]}"; do
            for PRUNE_RATIO in "${PRUNE_RATIOS[@]}"; do
                python run_experiment.py \
                --model $MODEL \
                --dataset $DATASET \
                --data_ratio $DATA_RATIO \
                --prune_ratio $PRUNE_RATIO \
                --epochs 10 \
                --post_epochs 10
            done
        done
    done
done
