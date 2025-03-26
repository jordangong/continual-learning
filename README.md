# Continual Learning Research Framework

A PyTorch-based framework for continual learning research, focusing on pretrained models like ViTs, CLIP, and DINO.

## Features

- Class incremental learning setup with configurable steps
- Support for pretrained models via `timm` and `openclip`
- Flexible dataset handling for continual learning scenarios
- Configurable optimizers, schedulers, and hyperparameters
- Integration with TensorBoard and Weights & Biases for experiment tracking
- Multi-GPU support using PyTorch's Distributed Data Parallel (DDP)

## Project Structure

```
continual-learning/
├── src/                    # Source code
│   ├── data/               # Dataset handling
│   ├── models/             # Model definitions
│   ├── trainers/           # Training logic
│   ├── utils/              # Utility functions
│   └── config/             # Configuration files
├── logs/                   # Training logs
├── checkpoints/            # Model checkpoints
├── scripts/                # Utility scripts
├── requirements.txt        # Dependencies
└── README.md               # This file
```

## Getting Started

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Configure your experiment in `src/config/`

3. Run training:
   ```
   # Single GPU training
   python -m src.main
   
   # Multi-GPU training
   python -m src.main distributed.enabled=true distributed.world_size=<num_gpus>
   ```

## Configuration

The framework uses Hydra for configuration management. Example configurations can be found in the `src/config/` directory.

### Distributed Training

Distributed training can be enabled in two ways:

1. Command line arguments: `distributed.enabled=true distributed.world_size=<num_gpus>`
2. Configuration: Set `distributed.enabled: true` and `distributed.world_size: <num_gpus>` in the config file

The framework uses PyTorch's Distributed Data Parallel (DDP) for efficient multi-GPU training.

### Debug Mode

Debug mode provides enhanced verbosity and faster training iterations for development purposes. It can be enabled in two ways:

1. Command line arguments: `debug.enabled=true`
2. Configuration: Set `debug.enabled: true` in the config file

Debug mode options:

```yaml
debug:
  enabled: false  # Set to true to enable debug mode
  verbose: true   # Enable verbose logging
  fast_dev_run: false  # Run only a few batches for quick testing
  limit_train_batches: 1.0  # Fraction of training batches to use (1.0 = all)
  limit_val_batches: 1.0    # Fraction of validation batches to use (1.0 = all)
  log_every_n_steps: 1      # Log metrics every N steps
```

Example usage:

```bash
# Enable debug mode with fast dev run
python -m src.main debug.enabled=true debug.fast_dev_run=true

# Enable debug mode with limited batches
python -m src.main debug.enabled=true debug.limit_train_batches=0.1 debug.limit_val_batches=0.1
```

## Logging

Training progress is logged to both TensorBoard and Weights & Biases.
