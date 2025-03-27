import os
from typing import Any, Dict, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import wandb
from torch import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.utils.metrics import forgetting


class ContinualTrainer:
    """Trainer for continual learning."""

    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        device: torch.device,
        local_rank: int = -1,
    ):
        """
        Args:
            model: Model to train
            config: Configuration dictionary
            device: Device to train on
            local_rank: Local rank for distributed training (-1 for non-distributed)
        """
        self.model = model
        self.config = config
        self.device = device
        self.local_rank = local_rank
        self.distributed = local_rank != -1

        # Move model to device
        self.model = self.model.to(self.device)

        # Wrap model with DDP if distributed training is enabled
        if self.distributed:
            self.model = DDP(
                self.model, device_ids=[local_rank], output_device=local_rank
            )

        # Training configuration
        self.training_config = config["training"]
        self.continual_config = config["continual"]

        # Debug configuration
        self.debug_config = config.get("debug", {"enabled": False})

        # Mixed precision training setup
        self.mixed_precision_enabled = self.training_config.get(
            "mixed_precision", {}
        ).get("enabled", False)
        mixed_precision_dtype = self.training_config.get("mixed_precision", {}).get(
            "dtype", "auto"
        )

        # Determine device type for mixed precision
        self.device_type = "cuda" if "cuda" in self.device.type else "cpu"

        # Determine the appropriate dtype for mixed precision
        if self.mixed_precision_enabled:
            # CPU only supports bfloat16 for mixed precision
            if self.device_type == "cpu":
                if mixed_precision_dtype in ["auto", "bfloat16"]:
                    # There's no direct way to check CPU BF16 support in PyTorch
                    # We'll try to create a small tensor and convert it to BF16
                    # If it works, we'll assume BF16 is supported
                    try:
                        _ = torch.zeros(1, device="cpu").to(torch.bfloat16)
                        self.mixed_precision_dtype = torch.bfloat16
                        print("Using bfloat16 mixed precision training on CPU")
                    except Exception as e:
                        print(
                            f"Warning: bfloat16 not supported on this CPU. Disabling mixed precision. Error: {e}"
                        )
                        self.mixed_precision_enabled = False
                else:
                    print(
                        f"Warning: {mixed_precision_dtype} not supported on CPU. Disabling mixed precision."
                    )
                    self.mixed_precision_enabled = False
            # CUDA device handling
            elif self.device_type == "cuda":
                if mixed_precision_dtype == "auto":
                    # Use bfloat16 if supported, otherwise fall back to float16
                    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                        self.mixed_precision_dtype = torch.bfloat16
                        print("Using bfloat16 mixed precision training on CUDA")
                    else:
                        self.mixed_precision_dtype = torch.float16
                        print(
                            "bfloat16 not supported on this GPU, falling back to float16 mixed precision training"
                        )
                elif mixed_precision_dtype == "bfloat16":
                    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                        self.mixed_precision_dtype = torch.bfloat16
                        print("Using bfloat16 mixed precision training on CUDA")
                    else:
                        print(
                            "Warning: bfloat16 requested but not supported by GPU. Disabling mixed precision."
                        )
                        self.mixed_precision_enabled = False
                elif mixed_precision_dtype == "float16":
                    self.mixed_precision_dtype = torch.float16
                    print("Using float16 mixed precision training on CUDA")
                else:
                    print(
                        f"Warning: Unknown mixed precision dtype '{mixed_precision_dtype}'. Disabling mixed precision."
                    )
                    self.mixed_precision_enabled = False
            else:
                print(
                    f"Warning: Mixed precision not supported on device type '{self.device_type}'. Disabling mixed precision."
                )
                self.mixed_precision_enabled = False

            # Initialize GradScaler for float16 (not needed for bfloat16)
            if (
                self.mixed_precision_enabled
                and self.mixed_precision_dtype == torch.float16
                and self.device_type == "cuda"
            ):
                self.scaler = amp.GradScaler()
            else:
                self.scaler = None

        # Setup optimizer
        self.optimizer = self._setup_optimizer()

        # Scheduler will be set up in train_step for each continual learning step
        self.scheduler = None

        # Setup criterion
        self.criterion = nn.CrossEntropyLoss()

        # Setup logging
        self.logging_config = config["logging"]
        self.setup_logging()

        # Output directories
        self.output_root = config["paths"]["output_root"]
        self.checkpoint_dir = config["paths"]["checkpoint_dir"]

        # Ensure output directories exist
        os.makedirs(self.output_root, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Metrics tracking
        self.metrics = {
            "accuracy": [],  # Accuracy after each step
            "forgetting": [],  # Forgetting after each step
        }

        # For EWC (if used)
        self.ewc_data = None

    def _setup_optimizer(self) -> torch.optim.Optimizer:
        """Setup optimizer based on configuration."""
        optimizer_config = self.config["optimizer"]
        optimizer_name = optimizer_config["name"].lower()

        if optimizer_name == "adam":
            return optim.Adam(
                self.model.parameters(),
                lr=optimizer_config["lr"],
                weight_decay=optimizer_config["weight_decay"],
                betas=optimizer_config.get("betas", (0.9, 0.999)),
            )
        elif optimizer_name == "sgd":
            return optim.SGD(
                self.model.parameters(),
                lr=optimizer_config["lr"],
                momentum=optimizer_config.get("momentum", 0.9),
                weight_decay=optimizer_config["weight_decay"],
                nesterov=optimizer_config.get("nesterov", False),
            )
        elif optimizer_name == "adamw":
            return optim.AdamW(
                self.model.parameters(),
                lr=optimizer_config["lr"],
                weight_decay=optimizer_config["weight_decay"],
                betas=optimizer_config.get("betas", (0.9, 0.999)),
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    def _setup_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Setup scheduler based on configuration.

        Returns:
            Configured learning rate scheduler
        """
        scheduler_config = self.config["scheduler"]
        scheduler_name = scheduler_config["name"].lower()
        use_global_scheduler = scheduler_config.get("global_scheduler", False)

        if scheduler_name == "cosine":
            # Check if warmup is enabled
            warmup_epochs = scheduler_config.get("warmup_epochs", 0)
            warmup_start_lr = scheduler_config.get("warmup_start_lr", 0.00001)
            min_lr = scheduler_config.get("min_lr", 0)

            # Calculate T_max based on whether we're using a global scheduler
            if use_global_scheduler:
                # For global scheduler: total epochs across all steps
                total_steps = self.continual_config["num_steps"]
                epochs_per_step = self.training_config["num_epochs"]
                total_epochs = total_steps * epochs_per_step

                # Create base cosine scheduler
                cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=total_epochs
                    - warmup_epochs,  # Adjust T_max to account for warmup
                    eta_min=min_lr,
                )
            else:
                # For per-step scheduler: just the epochs for this step
                total_epochs = self.training_config["num_epochs"]

                # Create base cosine scheduler
                cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=total_epochs
                    - warmup_epochs,  # Adjust T_max to account for warmup
                    eta_min=min_lr,
                )

            # If warmup is enabled, use a sequential scheduler
            if warmup_epochs > 0:
                # Get initial learning rate from optimizer
                initial_lr = self.optimizer.param_groups[0]["lr"]

                # Create linear warmup scheduler
                warmup_scheduler = optim.lr_scheduler.LinearLR(
                    self.optimizer,
                    start_factor=warmup_start_lr / initial_lr,
                    end_factor=1.0,
                    total_iters=warmup_epochs,
                )

                # Combine warmup and cosine schedulers
                return optim.lr_scheduler.SequentialLR(
                    self.optimizer,
                    schedulers=[warmup_scheduler, cosine_scheduler],
                    milestones=[warmup_epochs],
                )
            else:
                # If no warmup, just return the cosine scheduler
                return cosine_scheduler
        elif scheduler_name == "step":
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=scheduler_config["step_size"],
                gamma=scheduler_config["gamma"],
            )
        elif scheduler_name == "multistep":
            return optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                milestones=scheduler_config["milestones"],
                gamma=scheduler_config["gamma"],
            )
        elif scheduler_name == "none":
            return None
        else:
            raise ValueError(f"Unsupported scheduler: {scheduler_name}")

    def setup_logging(self):
        """Setup logging based on configuration."""
        # Create experiment-specific log directory
        experiment_name = self.config["experiment"]["name"]
        self.log_dir = os.path.join(self.config["paths"]["log_dir"], experiment_name)
        os.makedirs(self.log_dir, exist_ok=True)

        # Only setup logging on the main process if distributed
        if not self.distributed or (self.distributed and self.local_rank == 0):
            # Setup TensorBoard
            if self.logging_config["tensorboard"]:
                self.writer = SummaryWriter(log_dir=self.log_dir)
            else:
                self.writer = None

            # Setup Weights & Biases
            if self.logging_config["wandb"]:
                # Create wandb directory if it doesn't exist
                wandb_dir = self.config["paths"]["wandb_dir"]
                if wandb_dir:
                    os.makedirs(wandb_dir, exist_ok=True)

                # Initialize wandb with experiment name from config
                wandb.init(
                    project=self.logging_config["wandb_project"],
                    entity=self.logging_config["wandb_entity"],
                    config=self.config,
                    dir=wandb_dir,
                    name=self.config["experiment"]["name"],
                )
        else:
            self.writer = None

    def _limit_batches(self, data_loader: DataLoader, num_batches: int) -> DataLoader:
        """
        Limit the number of batches in a data loader.

        Args:
            data_loader: Data loader to limit
            num_batches: Number of batches to keep

        Returns:
            Limited data loader
        """
        # Create a subset of the dataset with only the first num_batches * batch_size samples
        batch_size = data_loader.batch_size
        subset_size = min(num_batches * batch_size, len(data_loader.dataset))

        # Create a subset of the dataset
        indices = list(range(subset_size))
        subset = torch.utils.data.Subset(data_loader.dataset, indices)

        # Create a new data loader with the subset
        limited_loader = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=data_loader.shuffle if hasattr(data_loader, "shuffle") else False,
            num_workers=(
                data_loader.num_workers if hasattr(data_loader, "num_workers") else 0
            ),
            persistent_workers=(
                data_loader.persistent_workers
                if hasattr(data_loader, "persistent_workers")
                else False
            ),
            pin_memory=(
                data_loader.pin_memory if hasattr(data_loader, "pin_memory") else False
            ),
            sampler=None,  # We're using a subset, so we don't need a sampler
            drop_last=(
                data_loader.drop_last if hasattr(data_loader, "drop_last") else False
            ),
        )

        return limited_loader

    def train_step(
        self, step: int, train_loader: DataLoader, test_loader: DataLoader
    ) -> Dict[str, float]:
        """
        Train for one continual learning step.

        Args:
            step: Current continual learning step
            train_loader: Training data loader
            test_loader: Test data loader

        Returns:
            Dictionary of metrics
        """
        debug_enabled = self.debug_config.get("enabled", False)
        debug_prefix = "[DEBUG] " if debug_enabled else ""
        print(
            f"{debug_prefix}=== Training Step {step + 1}/{self.continual_config['num_steps']} ==="
        )

        # Apply debug settings if enabled
        if debug_enabled:
            # Print debug information
            if not self.distributed or (self.distributed and self.local_rank == 0):
                print("\n[DEBUG MODE ENABLED]")
                print(f"Debug settings: {self.debug_config}")

            # Limit the number of batches if fast_dev_run is enabled
            if self.debug_config.get("fast_dev_run", False):
                # Only run a single epoch with a few batches
                num_epochs = 1
                # Limit the number of batches
                train_loader = self._limit_batches(train_loader, 3)  # Just 3 batches
                test_loader = self._limit_batches(test_loader, 3)  # Just 3 batches
                if not self.distributed or (self.distributed and self.local_rank == 0):
                    print("[DEBUG] Fast dev run: Running only 1 epoch with 3 batches")
            else:
                # Apply batch limits if specified
                train_limit = self.debug_config.get("limit_train_batches", 1.0)
                val_limit = self.debug_config.get("limit_val_batches", 1.0)

                if train_limit < 1.0:
                    train_loader = self._limit_batches(
                        train_loader, int(len(train_loader) * train_limit)
                    )
                    if not self.distributed or (
                        self.distributed and self.local_rank == 0
                    ):
                        print(
                            f"[DEBUG] Limited training to {train_limit * 100}% of batches ({len(train_loader)} batches)"
                        )

                if val_limit < 1.0:
                    test_loader = self._limit_batches(
                        test_loader, int(len(test_loader) * val_limit)
                    )
                    if not self.distributed or (
                        self.distributed and self.local_rank == 0
                    ):
                        print(
                            f"[DEBUG] Limited validation to {val_limit * 100}% of batches ({len(test_loader)} batches)"
                        )

                num_epochs = self.training_config["num_epochs"]
        else:
            num_epochs = self.training_config["num_epochs"]

        # Get scheduler configuration
        scheduler_config = self.config["scheduler"]
        use_global_scheduler = scheduler_config.get("global_scheduler", False)

        # For per-step scheduler, reset both optimizer and scheduler
        if not use_global_scheduler:
            # Reset optimizer for this step
            self.optimizer = self._setup_optimizer()

            # Setup a fresh scheduler for this continual learning step
            self.scheduler = self._setup_scheduler()

            if not self.distributed or (self.distributed and self.local_rank == 0):
                print(
                    f"Initialized new optimizer and scheduler for step {step + 1}/{self.continual_config['num_steps']}"
                )
        else:
            # For global scheduler, only initialize once or update with current step
            if self.scheduler is None:
                self.scheduler = self._setup_scheduler()
                if not self.distributed or (self.distributed and self.local_rank == 0):
                    print("Initialized global scheduler across all steps")

        # Training loop
        best_acc = 0.0
        best_epoch = 0
        patience_counter = 0

        for epoch in range(num_epochs):
            # Train for one epoch
            train_loss, train_acc = self._train_epoch(train_loader, step, epoch)

            # Evaluate if needed
            if (epoch + 1) % self.training_config["eval_every"] == 0:
                test_loss, test_acc = self._evaluate(test_loader)

                # Log metrics
                self._log_metrics(
                    step=step,
                    epoch=epoch,
                    train_loss=train_loss,
                    train_acc=train_acc,
                    test_loss=test_loss,
                    test_acc=test_acc,
                )

                # Save checkpoint if best
                if test_acc > best_acc:
                    best_acc = test_acc
                    best_epoch = epoch
                    patience_counter = 0
                    self._save_checkpoint(step, epoch, best=True)
                else:
                    patience_counter += 1

                # Early stopping
                if patience_counter >= self.training_config["early_stopping_patience"]:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break
            else:  # Log training metrics
                self._log_metrics(
                    step=step,
                    epoch=epoch,
                    train_loss=train_loss,
                    train_acc=train_acc,
                )

            # Step scheduler
            if self.scheduler is not None:
                self.scheduler.step()

            # Save checkpoint if needed
            if (epoch + 1) % self.training_config["save_every"] == 0:
                self._save_checkpoint(step, epoch)

        print(f"Best accuracy: {best_acc:.4f} at epoch {best_epoch + 1}")

        # Load best checkpoint
        self._load_checkpoint(step, best=True)

        # Final evaluation
        _, final_acc = self._evaluate(test_loader)

        # Update accuracy metrics first
        self.metrics["accuracy"].append(final_acc)

        # Then calculate forgetting using updated metrics
        if step > 0:
            fgt = forgetting(self.metrics["accuracy"], step)
            self.metrics["forgetting"].append(fgt)

        # Store EWC data if needed
        if (
            self.continual_config["strategy"] == "ewc"
            and step < self.continual_config["num_steps"] - 1
        ):
            self._compute_ewc_data(train_loader)

        return {
            "accuracy": final_acc,
            "forgetting": self.metrics["forgetting"][-1] if step > 0 else 0.0,
        }

    def _train_epoch(
        self, train_loader: DataLoader, step: int, epoch: int
    ) -> Tuple[float, float]:
        """
        Train for one epoch.

        Args:
            train_loader: Training data loader
            step: Current continual learning step
            epoch: Current epoch

        Returns:
            Tuple of (train_loss, train_accuracy)
        """
        self.model.train()

        total_loss = 0.0
        correct = 0
        total = 0

        # Only show progress bar on main process if distributed
        should_show_pbar = not self.distributed or (
            self.distributed and self.local_rank == 0
        )

        # Set description based on debug mode
        debug_enabled = self.debug_config.get("enabled", False)
        debug_verbose = debug_enabled and self.debug_config.get("verbose", False)
        desc = "Training [DEBUG]" if debug_enabled else "Training"

        train_iter = tqdm(train_loader, desc=desc) if should_show_pbar else train_loader

        for inputs, targets in train_iter:
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Mixed precision training
            if self.mixed_precision_enabled:
                # Use autocast context manager for mixed precision forward pass
                with amp.autocast(
                    device_type=self.device_type, dtype=self.mixed_precision_dtype
                ):
                    # Forward pass
                    outputs = self.model(inputs)

                    # Compute loss
                    loss = self.criterion(outputs, targets)

                    # Add EWC loss if needed
                    if (
                        self.continual_config["strategy"] == "ewc"
                        and step > 0
                        and self.ewc_data is not None
                    ):
                        ewc_lambda = self.continual_config.get("ewc_lambda", 1.0)
                        ewc_loss = self._compute_ewc_loss()
                        loss += ewc_lambda * ewc_loss

                # For float16, use the gradient scaler
                if self.mixed_precision_dtype == torch.float16:
                    # Scale loss and do backward pass
                    self.scaler.scale(loss).backward()

                    # Unscale gradients and call optimizer.step()
                    self.scaler.step(self.optimizer)

                    # Update the scaler for next iteration
                    self.scaler.update()
                else:
                    # For bfloat16, regular backward pass is stable
                    loss.backward()
                    self.optimizer.step()
            else:
                # Standard precision training
                # Forward pass
                outputs = self.model(inputs)

                # Compute loss
                loss = self.criterion(outputs, targets)

                # Add EWC loss if needed
                if (
                    self.continual_config["strategy"] == "ewc"
                    and step > 0
                    and self.ewc_data is not None
                ):
                    ewc_lambda = self.continual_config.get("ewc_lambda", 1.0)
                    ewc_loss = self._compute_ewc_loss()
                    loss += ewc_lambda * ewc_loss

                # Backward pass
                loss.backward()

                # Update weights
                self.optimizer.step()

            # Update metrics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # Update progress bar if showing
            if should_show_pbar:
                debug_enabled = self.debug_config.get("enabled", False)
                debug_verbose = debug_enabled and self.debug_config.get(
                    "verbose", False
                )

                postfix = {
                    "loss": total_loss / (train_iter.n + 1),
                    "acc": 100.0 * correct / total,
                    "epoch": epoch + 1,
                }

                # Add more detailed info in verbose debug mode
                if debug_verbose:
                    batch_size = inputs.size(0)
                    postfix.update(
                        {
                            "batch_size": batch_size,
                            "lr": self.optimizer.param_groups[0]["lr"],
                            "step": step,
                        }
                    )

                train_iter.set_postfix(postfix)

        # Synchronize metrics across processes if distributed
        if self.distributed:
            # Sum up metrics from all processes
            metrics = torch.tensor(
                [total_loss, correct, total], dtype=torch.float32, device=self.device
            )
            dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
            total_loss, correct, total = metrics.tolist()

        train_loss = total_loss / len(train_loader)
        train_acc = 100.0 * correct / total

        return train_loss, train_acc

    def _evaluate(self, test_loader: DataLoader) -> Tuple[float, float]:
        """
        Evaluate the model.

        Args:
            test_loader: Test data loader

        Returns:
            Tuple of (test_loss, test_accuracy)
        """
        self.model.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        # Only show progress bar on main process if distributed
        should_show_pbar = not self.distributed or (
            self.distributed and self.local_rank == 0
        )

        # Set description based on debug mode
        debug_enabled = self.debug_config.get("enabled", False)
        desc = "Evaluating [DEBUG]" if debug_enabled else "Evaluating"

        with torch.no_grad():
            test_iter = (
                tqdm(test_loader, desc=desc) if should_show_pbar else test_loader
            )
            for inputs, targets in test_iter:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                # Use mixed precision for evaluation if enabled
                if self.mixed_precision_enabled:
                    with amp.autocast(
                        device_type=self.device_type, dtype=self.mixed_precision_dtype
                    ):
                        # Forward pass
                        outputs = self.model(inputs)

                        # Compute loss
                        loss = self.criterion(outputs, targets)
                else:
                    # Forward pass
                    outputs = self.model(inputs)

                    # Compute loss
                    loss = self.criterion(outputs, targets)

                # Update metrics
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                # Update progress bar if showing
                if should_show_pbar:
                    test_iter.set_postfix(
                        {
                            "loss": total_loss / (test_iter.n + 1),
                            "acc": 100.0 * correct / total,
                        }
                    )

        # Synchronize metrics across processes if distributed
        if self.distributed:
            # Sum up metrics from all processes
            metrics = torch.tensor(
                [total_loss, correct, total], dtype=torch.float32, device=self.device
            )
            dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
            total_loss, correct, total = metrics.tolist()

        test_loss = total_loss / len(test_loader)
        test_acc = 100.0 * correct / total

        return test_loss, test_acc

    def _log_metrics(
        self,
        step: int,
        epoch: int,
        train_loss: float,
        train_acc: float,
        test_loss: Optional[float] = None,
        test_acc: Optional[float] = None,
    ):
        """Log metrics to TensorBoard and Weights & Biases."""
        # Only log on the main process if distributed
        if self.distributed and self.local_rank != 0:
            return

        # Check if we should log based on debug settings
        debug_enabled = self.debug_config.get("enabled", False)
        log_every_n_steps = (
            self.debug_config.get("log_every_n_steps", 1) if debug_enabled else 1
        )

        if epoch % log_every_n_steps != 0:
            return

        # Print metrics
        debug_prefix = "[DEBUG] " if debug_enabled else ""
        train_metrics = f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%"
        if test_loss is not None and test_acc is not None:
            train_metrics += f", Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%"
        print(f"{debug_prefix}Epoch {epoch + 1}: {train_metrics}")

        # Log to TensorBoard
        if self.writer is not None:
            self.writer.add_scalar(f"step_{step}/train_loss", train_loss, epoch)
            self.writer.add_scalar(f"step_{step}/train_acc", train_acc, epoch)
            if test_loss is not None and test_acc is not None:
                self.writer.add_scalar(f"step_{step}/test_loss", test_loss, epoch)
                self.writer.add_scalar(f"step_{step}/test_acc", test_acc, epoch)

            # Log learning rate
            lr = self.optimizer.param_groups[0]["lr"]
            self.writer.add_scalar("learning_rate", lr, epoch)
            self.writer.add_scalar("step", step, epoch)
            self.writer.add_scalar("epoch", epoch, epoch)

        # Log to Weights & Biases
        if self.logging_config["wandb"]:
            log_data = {
                f"step_{step}/train_loss": train_loss,
                f"step_{step}/train_acc": train_acc,
                "learning_rate": self.optimizer.param_groups[0]["lr"],
                "step": step,
                "epoch": epoch,
            }
            if test_loss is not None and test_acc is not None:
                log_data.update({
                    f"step_{step}/test_loss": test_loss,
                    f"step_{step}/test_acc": test_acc,
                })

            wandb.log(log_data)

    def _save_checkpoint(self, step: int, epoch: int, best: bool = False):
        """
        Save checkpoint.

        Args:
            step: Current continual learning step
            epoch: Current epoch
            best: Whether this is the best checkpoint
        """
        # Only save checkpoint on the main process if distributed
        if self.distributed and self.local_rank != 0:
            return

        # Get model state dict (handle DDP wrapper if present)
        if self.distributed:
            model_state = self.model.module.state_dict()
        else:
            model_state = self.model.state_dict()

        checkpoint = {
            "model": model_state,
            "optimizer": self.optimizer.state_dict(),
            "scheduler": (
                self.scheduler.state_dict() if self.scheduler is not None else None
            ),
            "step": step,
            "epoch": epoch,
            "metrics": self.metrics,
            "config": self.config,
        }

        # Create a descriptive filename with key hyperparameters
        dataset_name = self.config["dataset"]["name"]
        model_name = self.config["model"]["name"]
        strategy = self.config["continual"]["strategy"]

        if best:
            checkpoint_path = os.path.join(
                self.checkpoint_dir,
                f"{dataset_name}_{model_name}_{strategy}_step{step}_best.pth",
            )
        else:
            checkpoint_path = os.path.join(
                self.checkpoint_dir,
                f"{dataset_name}_{model_name}_{strategy}_step{step}_epoch{epoch}.pth",
            )

        # Save checkpoint with _use_new_zipfile_serialization=True for better compatibility
        torch.save(checkpoint, checkpoint_path, _use_new_zipfile_serialization=True)

    def _load_checkpoint(self, step: int, best: bool = True):
        """
        Load checkpoint.

        Args:
            step: Continual learning step to load
            best: Whether to load the best checkpoint
        """
        # Create the same descriptive filename format used in _save_checkpoint
        dataset_name = self.config["dataset"]["name"]
        model_name = self.config["model"]["name"]
        strategy = self.config["continual"]["strategy"]

        if best:
            checkpoint_path = os.path.join(
                self.checkpoint_dir,
                f"{dataset_name}_{model_name}_{strategy}_step{step}_best.pth",
            )
        else:
            raise ValueError("Loading non-best checkpoints not implemented")

        if not os.path.exists(checkpoint_path):
            if not self.distributed or (self.distributed and self.local_rank == 0):
                print(f"Checkpoint {checkpoint_path} not found")
            return

        # Load checkpoint
        # Set weights_only=False to handle PyTorch 2.6+ compatibility
        checkpoint = torch.load(
            checkpoint_path, map_location=self.device, weights_only=False
        )

        # Load model state dict (handle DDP wrapper if present)
        if self.distributed:
            self.model.module.load_state_dict(checkpoint["model"])
        else:
            self.model.load_state_dict(checkpoint["model"])

        scheduler_config = self.config["scheduler"]
        use_global_scheduler = scheduler_config.get("global_scheduler", False)
        if not use_global_scheduler and best:
            self.optimizer.load_state_dict(checkpoint["optimizer"])

            if self.scheduler is not None and checkpoint["scheduler"] is not None:
                self.scheduler.load_state_dict(checkpoint["scheduler"])

    def _compute_ewc_data(self, train_loader: DataLoader):
        """
        Compute EWC data (Fisher information matrix and parameter values).

        Args:
            train_loader: Training data loader
        """
        # Store current parameter values
        model_to_use = self.model.module if self.distributed else self.model
        params = {
            n: p.clone().detach()
            for n, p in model_to_use.named_parameters()
            if p.requires_grad
        }

        # Initialize Fisher information matrix
        fisher = {n: torch.zeros_like(p) for n, p in params.items()}

        # Compute Fisher information matrix
        self.model.eval()

        # Only show progress bar on main process if distributed
        should_show_pbar = not self.distributed or (
            self.distributed and self.local_rank == 0
        )
        data_iter = (
            tqdm(train_loader, desc="Computing EWC data")
            if should_show_pbar
            else train_loader
        )

        for inputs, targets in data_iter:
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()

            # Accumulate Fisher information
            for n, p in model_to_use.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[n] += p.grad.pow(2).detach()

        # Synchronize Fisher information across processes if distributed
        if self.distributed:
            for n in fisher.keys():
                dist.all_reduce(fisher[n], op=dist.ReduceOp.SUM)

        # Normalize Fisher information
        for n in fisher.keys():
            fisher[n] /= len(train_loader)

        # Store EWC data
        self.ewc_data = {"fisher": fisher, "params": params}

    def _compute_ewc_loss(self) -> torch.Tensor:
        """
        Compute EWC loss.

        Returns:
            EWC loss
        """
        if self.ewc_data is None:
            return torch.tensor(0.0, device=self.device)

        loss = torch.tensor(0.0, device=self.device)

        # Get the correct model reference (module if distributed)
        model_to_use = self.model.module if self.distributed else self.model

        for n, p in model_to_use.named_parameters():
            if n in self.ewc_data["fisher"] and p.requires_grad:
                # Compute squared distance
                loss += (
                    self.ewc_data["fisher"][n] * (p - self.ewc_data["params"][n]).pow(2)
                ).sum()

        return loss

    # Note: Memory sample retrieval is handled by the DataModule class
