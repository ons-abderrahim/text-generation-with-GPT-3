#!/usr/bin/env bash
# run_training.sh — launch fine-tuning with optional multi-GPU support

set -euo pipefail

CONFIG=${1:-"configs/train_gpt2.yaml"}
NUM_GPUS=${2:-1}

echo "=============================================="
echo " GPT Fine-Tuning Launcher"
echo "=============================================="
echo " Config:   $CONFIG"
echo " GPUs:     $NUM_GPUS"
echo "=============================================="

# Activate virtual environment if present
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

# Create output directories
mkdir -p outputs/logs outputs/checkpoints outputs/generated_texts

if [ "$NUM_GPUS" -gt 1 ]; then
    echo "Launching multi-GPU training with accelerate..."
    accelerate launch \
        --num_processes="$NUM_GPUS" \
        --mixed_precision=fp16 \
        train.py --config "$CONFIG"
else
    echo "Launching single-GPU / CPU training..."
    python train.py --config "$CONFIG"
fi

echo ""
echo "Training complete! Checkpoints saved to outputs/checkpoints/"
echo "View TensorBoard logs: tensorboard --logdir outputs/logs/"
