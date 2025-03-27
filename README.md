# Continual Learning Framework

A PyTorch-based framework for continual learning research, focusing on pretrained foundation models like Vision Transformers (ViT), CLIP, and DINO.

## Features

- **Class Incremental Learning**: Configurable steps with customizable class distribution
- **Foundation Models**: Support for pretrained models via `timm` and `openclip`
- **Multiple Datasets**: CIFAR-100, CUB-200, ImageNet variants, ObjectNet, OmniBench, and VTAB
- **Automatic Mixed Precision**: Support for FP16 and BF16 training for faster iterations
- **Flexible Training**: Configurable optimizers, schedulers, and hyperparameters
- **Experiment Tracking**: Integration with TensorBoard and Weights & Biases
- **Multi-GPU Support**: Efficient distributed training with PyTorch's Distributed Data Parallel (DDP)
- **Debug Mode**: Fast development iterations with configurable debug settings

## Project Structure

```plaintext
continual-learning/
├── src/                    # Source code
│   ├── data/               # Dataset handling and continual learning datasets
│   ├── models/             # Model definitions and classifier heads
│   ├── trainers/           # Training logic with continual learning strategies
│   ├── utils/              # Utility functions for logging, config, etc.
│   └── config/             # Configuration files using Hydra
├── outputs/                # All outputs including:
│   ├── logs/               # Training logs
│   ├── checkpoints/        # Model checkpoints
│   ├── plots/              # Performance plots
│   └── wandb/              # Weights & Biases logs
├── cache/                  # Cache directory for model weights
├── data/                   # Dataset files
├── scripts/                # Utility scripts for training and evaluation
├── requirements.txt        # Dependencies
└── README.md               # This file
```

## Getting Started

1. Install dependencies:

```bash
pip install -r requirements.txt
```

1. Configure your experiment in `src/config/`

1. Run training:

```bash
# Single GPU training
python -m src.main

# Multi-GPU training
python -m src.main distributed.enabled=true distributed.world_size=<num_gpus>

# Or use the provided script
./scripts/run_distributed.sh --gpus <num_gpus> --config <config_name>
```

## Configuration

The framework uses [Hydra](https://hydra.cc/) for configuration management. Example configurations can be found in the `src/config/` directory.

### Main Configuration Options

```yaml
dataset:
  name: cifar100                # Dataset name
  input_size: 224               # Input image size
  num_classes: 100              # Total number of classes

model:
  name: vit_base_patch16_224    # Model architecture
  source: timm                  # Model source (timm, openclip)
  pretrained: true              # Use pretrained weights
  freeze_backbone: false        # Freeze backbone weights

continual:
  num_steps: 10                 # Number of continual learning steps
  classes_per_step: 10          # Classes per step
  memory_size: 2000             # Memory buffer size for rehearsal
  strategy: finetune            # CL strategy (finetune, ewc, replay, etc.)

training:
  num_epochs: 100               # Max epochs per step
  batch_size: 64                # Batch size
  mixed_precision:              # Automatic mixed precision settings
    enabled: true
    dtype: auto                 # auto, bfloat16, or float16
```

### Distributed Training

For multi-GPU training, the framework uses PyTorch's Distributed Data Parallel (DDP):

```yaml
distributed:
  enabled: true                 # Enable distributed training
  world_size: 4                 # Number of GPUs to use
```

Command line override:

```bash
python -m src.main distributed.enabled=true distributed.world_size=4
```

### Debug Mode

Debug mode provides enhanced verbosity and faster training iterations for development:

```yaml
debug:
  enabled: true                 # Enable debug mode
  verbose: true                 # Enable verbose logging
  fast_dev_run: true            # Run only a few batches for quick testing
  limit_train_batches: 0.1      # Use only 10% of training data
  limit_val_batches: 0.1        # Use only 10% of validation data
  log_every_n_steps: 1          # Log metrics every step
```

Example usage:

```bash
# Quick test run
python -m src.main debug.enabled=true debug.fast_dev_run=true

# Limited data run
python -m src.main debug.enabled=true debug.limit_train_batches=0.1
```

## Logging and Visualization

Training progress is logged to both TensorBoard and Weights & Biases. The framework automatically generates accuracy and forgetting curves for each continual learning experiment.

## Citation

If you use this framework in your research, please cite:

```bibtex
@misc{continual-learning-framework,
  author = {Your Name},
  title = {Continual Learning Framework},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/jordangong/continual-learning}
}
```
