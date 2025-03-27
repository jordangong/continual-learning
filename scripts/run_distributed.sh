#!/bin/bash

# Script to run distributed training on multiple GPUs

# Default values
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
CONFIG="config"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --gpus)
      NUM_GPUS="$2"
      shift
      shift
      ;;
    --config)
      CONFIG="$2"
      shift
      shift
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

echo "Running distributed training with $NUM_GPUS GPUs"
echo "Using config: $CONFIG"

# Run the training script with distributed flag
python -m src.main distributed.enabled=true distributed.world_size="$NUM_GPUS" hydra.config_name="$CONFIG"
