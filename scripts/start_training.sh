#!/bin/bash
cd /mnt/data/sri/wag/scripts/training
source ../venv/bin/activate

CHECKPOINT_PATH="../output/models/wag-copywriter/checkpoint-500"

if [ -d "$CHECKPOINT_PATH" ]; then
    echo "Resuming from checkpoint: $CHECKPOINT_PATH"
    nohup python train.py --config config.yaml --resume "$CHECKPOINT_PATH" > ../output/training.log 2>&1 &
else
    echo "No checkpoint found, starting fresh training"
    nohup python train.py --config config.yaml > ../output/training.log 2>&1 &
fi

echo "Training started with PID: $!"
