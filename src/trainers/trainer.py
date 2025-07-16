import glob
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

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
        data_module=None,
    ):
        """
        Args:
            model: Model to train
            config: Configuration dictionary
            device: Device to train on
            local_rank: Local rank for distributed training (-1 for non-distributed)
            data_module: Data module for accessing test transforms
        """
        self.model = model
        self.config = config
        self.device = device
        self.local_rank = local_rank
        self.data_module = data_module
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

        # Class logit masking configuration
        self.mask_logits = self.continual_config.get("mask_logits", False)

        # Entropy prediction configuration
        self.entropy_prediction = self.continual_config.get("entropy_prediction", False)

        # Class tracking for logit masking and calibration
        self.current_task_classes = None  # Classes in current step

        # Debug configuration
        self.debug_config = config.get("debug", {"enabled": False})

        # Classifier calibration configuration
        self.calibration_config = self.continual_config.get("calibration", {"enabled": False})
        self.calibration_enabled = self.calibration_config.get("enabled", False)
        self.calibration_classifier_as_prototype = self.calibration_config.get("classifier_as_prototype", False)
        self.calibration_eval_with_calibration = self.calibration_config.get("eval_with_calibration", False)
        self.calibration_method = self.calibration_config.get("method", "rigid")  # rigid, affine, nonlinear
        self.calibration_reg_weight = self.calibration_config.get("regularization_weight", 0.01)

        # Storage for historical prototypes and checkpoint paths for calibration
        self.historical_prototypes = {}  # Format: {step: {class_idx: prototype}}
        self.historical_checkpoint_paths = {}  # Format: {step: checkpoint_path}

        # Gradient clipping configuration
        self.gradient_clipping_enabled = self.training_config.get(
            "gradient_clipping", {}
        ).get("enabled", False)
        self.gradient_clipping_max_norm = self.training_config.get(
            "gradient_clipping", {}
        ).get("max_norm", 1.0)

        # Mixed precision training setup
        self.mixed_precision_enabled = self.training_config.get(
            "mixed_precision", {}
        ).get("enabled", False)
        self.mixed_precision_eval = self.training_config.get("mixed_precision", {}).get(
            "eval", False
        )
        mixed_precision_dtype = self.training_config.get("mixed_precision", {}).get(
            "dtype", "auto"
        )

        # Determine device type for mixed precision
        self.device_type = "cuda" if "cuda" in self.device.type else "cpu"

        # Determine the appropriate dtype for mixed precision
        if self.mixed_precision_enabled or self.mixed_precision_eval:
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
                        self.mixed_precision_eval = False
                else:
                    print(
                        f"Warning: {mixed_precision_dtype} not supported on CPU. Disabling mixed precision."
                    )
                    self.mixed_precision_enabled = False
                    self.mixed_precision_eval = False
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
                        self.mixed_precision_eval = False
                elif mixed_precision_dtype == "float16":
                    self.mixed_precision_dtype = torch.float16
                    print("Using float16 mixed precision training on CUDA")
                else:
                    print(
                        f"Warning: Unknown mixed precision dtype '{mixed_precision_dtype}'. Disabling mixed precision."
                    )
                    self.mixed_precision_enabled = False
                    self.mixed_precision_eval = False
            else:
                print(
                    f"Warning: Mixed precision not supported on device type '{self.device_type}'. Disabling mixed precision."
                )
                self.mixed_precision_enabled = False
                self.mixed_precision_eval = False

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

        # For EMA (if used)
        self.ema_config = self.continual_config.get("ema", {})
        self.ema_enabled = self.continual_config["strategy"] == "ema"
        self.ema_momentum = self.ema_config.get("momentum", 0.999)
        self.ema_eval_with_teacher = self.ema_config.get("eval_with_teacher", False)
        self.ema_refresh_interval = self.ema_config.get("refresh_interval", None)
        self.ema_refresh_at_step_start = self.ema_config.get(
            "refresh_at_step_start", True
        )
        self.ema_skip_names = self.ema_config.get(
            "skip_names", ["classifier", "head", "fc"]
        )
        self.ema_momentum_overrides = self.ema_config.get("momentum_overrides", {})
        self.ema_teacher_model = None  # Will be initialized at the start of each step

        # For prototypical classifier (if used)
        self.prototypical_config = self.continual_config.get("prototypical", {})
        self.prototypical_enabled = self.config["model"]["classifier"]["type"] == "prototypical"
        self.prototypical_replace_classifiers = self.prototypical_config.get(
            "replace_classifiers", False
        )

        # Gradient norm accumulation for epoch-wise logging
        self.grad_norm_accumulator = {}  # Dict to accumulate gradient norms by parameter name
        self.grad_norm_batch_count = 0  # Count of batches for averaging

    def _setup_optimizer(self) -> torch.optim.Optimizer:
        """Setup optimizer based on configuration."""
        optimizer_config = self.config["optimizer"]
        optimizer_name = optimizer_config["name"].lower()
        lr = optimizer_config["lr"]
        weight_decay = optimizer_config["weight_decay"]

        # Get the correct model reference (module if distributed)
        model_to_use = self.model.module if self.distributed else self.model

        # Get backbone, prompt, and temperature learning rate multipliers from optimizer config
        backbone_lr_multiplier = optimizer_config.get("backbone_lr_multiplier", 1.0)
        prompt_lr_multiplier = optimizer_config.get("prompt_lr_multiplier", 1.0)
        temperature_lr_multiplier = optimizer_config.get("temperature_lr_multiplier", 1.0)

        # Separate parameters by type to handle weight decay properly
        backbone_params = []
        prompt_params = []
        classifier_params = []
        temperature_params = []
        other_params = []

        # Separate all parameters by type
        for name, param in model_to_use.named_parameters():
            if "backbone" in name:
                backbone_params.append(param)
            elif "prompt" in name:
                prompt_params.append(param)
            elif "classifier" in name:
                # Exclude classifier from weight decay to prevent previous task weights from changing
                classifier_params.append(param)
            elif "temperature" in name:
                # Temperature parameters with separate learning rate
                temperature_params.append(param)
            else:
                other_params.append(param)

        # Create parameter groups with different settings
        param_groups = []

        # Backbone parameters with optional learning rate multiplier
        if backbone_params:
            param_groups.append(
                {
                    "params": backbone_params,
                    "lr": lr * backbone_lr_multiplier,
                    "weight_decay": weight_decay,
                }
            )

        # Prompt parameters with optional learning rate multiplier
        if prompt_params:
            param_groups.append(
                {
                    "params": prompt_params,
                    "lr": lr * prompt_lr_multiplier,
                    "weight_decay": weight_decay,
                }
            )

        # Classifier parameters WITHOUT weight decay
        if classifier_params:
            param_groups.append(
                {
                    "params": classifier_params,
                    "lr": lr,
                    "weight_decay": 0.0,  # No weight decay for classifier
                }
            )

        # Temperature parameters with optional learning rate multiplier and no weight decay
        if temperature_params:
            param_groups.append(
                {
                    "params": temperature_params,
                    "lr": lr * temperature_lr_multiplier,
                    "weight_decay": 0.0,  # No weight decay for temperature parameters
                }
            )

        # Other parameters with standard settings
        if other_params:
            param_groups.append(
                {"params": other_params, "lr": lr, "weight_decay": weight_decay}
            )

        if optimizer_name == "adam":
            return optim.Adam(
                param_groups,
                lr=lr,
                weight_decay=weight_decay,
                betas=optimizer_config.get("betas", (0.9, 0.999)),
            )
        elif optimizer_name == "sgd":
            return optim.SGD(
                param_groups,
                lr=lr,
                momentum=optimizer_config.get("momentum", 0.9),
                weight_decay=weight_decay,
                nesterov=optimizer_config.get("nesterov", False),
            )
        elif optimizer_name == "adamw":
            return optim.AdamW(
                param_groups,
                lr=lr,
                weight_decay=weight_decay,
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
                # Use last group, in case of multiplied backbone learning rate in the first group
                initial_lr = self.optimizer.param_groups[-1]["lr"]

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

                if self.debug_config.get("enabled", False):
                    job_type = "debug"
                elif self.config.get("eval_only", False):
                    job_type = "eval"
                else:
                    job_type = "train"
                # Initialize wandb with experiment name from config
                wandb.init(
                    project=self.logging_config["wandb_project"],
                    entity=self.logging_config["wandb_entity"],
                    config=self.config,
                    dir=wandb_dir,
                    name=self.config["experiment"]["name"],
                    job_type=job_type,
                )
        else:
            self.writer = None

    def train_step(
        self,
        step: int,
        all_step_classes: List[List[int]],
        train_loader: DataLoader,
        test_loader: DataLoader,
    ) -> Dict[str, float]:
        """
        Train for one continual learning step.

        Args:
            step: Current continual learning step
            all_step_classes: Classes in all steps
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

        # Reset frequency tracking for L2P-style diversity regularization at the start of each new task
        if hasattr(self.model, "reset_frequency_tracking"):
            self.model.reset_frequency_tracking()
            print(f"{debug_prefix}Reset frequency tracking for step {step + 1}")

        # Set current task classes for logit masking and calibration
        if self.mask_logits or self.calibration_enabled:
            self.current_task_classes = all_step_classes[step]

        # Initialize EMA teacher model at the start of each step (conditionally)
        if self.ema_enabled:
            # Always initialize for the first step, or if refresh_at_step_start is enabled
            if step == 0 or self.ema_refresh_at_step_start:
                self._initialize_ema_teacher()
                if step == 0:
                    print(f"{debug_prefix}Initialized EMA teacher for first step")
                else:
                    print(f"{debug_prefix}Refreshed EMA teacher at step start {step + 1}")
            else:
                print(f"{debug_prefix}Skipped EMA teacher refresh at step start {step + 1} (using refresh_interval instead)")

        # Apply debug settings if enabled
        if debug_enabled and self.debug_config.get("fast_dev_run", False):
            # Only run a single epoch with a few batches in fast_dev_run mode
            self.training_config["num_epochs"] = num_epochs = 1
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
                print(f"Initialized new optimizer and scheduler for step {step + 1}/{self.continual_config['num_steps']}")
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

            # Refresh EMA teacher if needed (at specified interval)
            if (
                self.ema_enabled
                and self.ema_refresh_interval is not None
                and (epoch + 1) % self.ema_refresh_interval == 0
            ):
                self._initialize_ema_teacher()
                print(f"{debug_prefix}Refreshed EMA teacher at epoch {epoch + 1}")

            # Evaluate if needed
            if (epoch + 1) % self.training_config["eval_every"] == 0:
                # Save current classifier state before evaluation modifications
                original_classifier_state = self._save_classifier_state()

                # Handle prototype replacement and/or calibration during training evaluation
                try:
                    # Apply prototype replacement if enabled
                    if self.prototypical_enabled and self.prototypical_replace_classifiers:
                        print(f"{debug_prefix}Applying temporary prototype replacement for training evaluation")
                        self._replace_classifier_with_prototypes(train_loader)

                    # Handle evaluation with teacher model if enabled
                    if self.ema_enabled and self.ema_eval_with_teacher:
                        # Save current student parameters before replacement
                        self._save_student_parameters()
                        # Temporarily replace student with teacher
                        self._replace_student_with_teacher()
                        # Handle calibration during training evaluation
                        if self.calibration_enabled and self.calibration_eval_with_calibration and step > 0:
                            print(f"{debug_prefix}Applying temporary calibration for training evaluation")
                            # Apply calibration to previous classifiers for evaluation
                            self.calibrate_previous_classifiers(step, train_loader, all_step_classes)
                        test_loss, test_acc = self._evaluate(test_loader)
                        # Restore original student parameters for continued training
                        self._restore_student_parameters()
                    else:
                        # Handle calibration during training evaluation
                        if self.calibration_enabled and self.calibration_eval_with_calibration and step > 0:
                            print(f"{debug_prefix}Applying temporary calibration for training evaluation")
                            # Apply calibration to previous classifiers for evaluation
                            self.calibrate_previous_classifiers(step, train_loader, all_step_classes)
                        test_loss, test_acc = self._evaluate(test_loader)
                finally:
                    # Always restore original classifier state after evaluation
                    self._restore_classifier_state(original_classifier_state)
                    print(f"{debug_prefix}Restored original classifier weights after evaluation")

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

        # Load best checkpoint first (includes teacher model if eval_with_teacher=True)
        self._load_checkpoint(step, best=True)

        # Handle evaluation with teacher model
        if self.ema_enabled and self.ema_eval_with_teacher:
            # Replace student with saved teacher AFTER loading checkpoint
            # This uses the teacher from the same training iteration as the best checkpoint
            self._replace_student_with_teacher()
            print("Using EMA teacher parameters from best checkpoint for evaluation")

        # Replace classifier with prototypes if using prototypical classifier
        if self.prototypical_enabled and self.prototypical_replace_classifiers:
            self._replace_classifier_with_prototypes(train_loader)

        # Apply classifier calibration if enabled
        if self.calibration_enabled:
            self.calibrate_previous_classifiers(step, train_loader, all_step_classes)

        # Final evaluation
        _, final_acc = self._evaluate(test_loader)

        eval_model_type = (
            "teacher"
            if (self.ema_enabled and self.ema_eval_with_teacher)
            else "student"
        )
        print(f"Final evaluation with {eval_model_type} model: {final_acc:.4f}")

        # Update accuracy metrics first
        self.metrics["accuracy"].append(final_acc)

        # Then calculate forgetting using updated metrics
        if step > 0:
            fgt = forgetting(self.metrics["accuracy"], step)
            self.metrics["forgetting"].append(fgt)

        # Replace student parameters with EMA teacher at the end of the step
        # (Skip if already done for teacher evaluation)
        if self.ema_enabled and not self.ema_eval_with_teacher:
            self._replace_student_with_teacher()
            print(f"Replaced student parameters with EMA teacher for step {step + 1}")

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

        # Reset gradient norm accumulator at the start of each epoch
        self._reset_gradient_norm_accumulator()

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

                    # Apply logit masking if enabled
                    outputs = self._apply_logit_masking(outputs, targets)

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

                    # Add prompt tuning auxiliary losses if needed
                    if self.continual_config["strategy"] == "prompt_tuning":
                        aux_losses = self._get_prompt_auxiliary_losses()
                        for aux_name, aux_loss in aux_losses.items():
                            if aux_name == "diversity_loss":
                                diversity_weight = self.continual_config.get(
                                    "prompt_tuning", {}
                                ).get("diversity_weight", 0.01)
                                loss += diversity_weight * aux_loss
                            elif aux_name == "similarity_loss":
                                similarity_weight = self.continual_config.get(
                                    "prompt_tuning", {}
                                ).get("similarity_weight", 0.01)
                                loss += similarity_weight * aux_loss

                # For float16, use the gradient scaler
                if self.mixed_precision_dtype == torch.float16:
                    # Scale loss and do backward pass
                    self.scaler.scale(loss).backward()

                    # Unscale gradients before clipping
                    if self.gradient_clipping_enabled:
                        self.scaler.unscale_(self.optimizer)

                        # Accumulate gradient norms after unscaling, before clipping
                        self._accumulate_gradient_norms()

                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.gradient_clipping_max_norm,
                        )
                    else:
                        # Unscale and accumulate gradient norms even without clipping
                        self.scaler.unscale_(self.optimizer)
                        self._accumulate_gradient_norms()

                    # Call optimizer.step() with scaler
                    self.scaler.step(self.optimizer)

                    # Update EMA teacher after optimizer step
                    if self.ema_enabled:
                        self._update_ema_teacher()

                    # Update the scaler for next iteration
                    self.scaler.update()
                else:
                    # For bfloat16, regular backward pass is stable
                    loss.backward()

                    # Accumulate gradient norms after backward, before clipping
                    self._accumulate_gradient_norms()

                    # Apply gradient clipping if enabled
                    if self.gradient_clipping_enabled:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.gradient_clipping_max_norm,
                        )

                    self.optimizer.step()

                    # Update EMA teacher after optimizer step
                    if self.ema_enabled:
                        self._update_ema_teacher()
            else:
                # Standard precision training
                # Forward pass
                outputs = self.model(inputs)

                # Apply logit masking if enabled
                outputs = self._apply_logit_masking(outputs, targets)

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

                # Add prompt tuning auxiliary losses if needed
                if self.continual_config["strategy"] == "prompt_tuning":
                    aux_losses = self._get_prompt_auxiliary_losses()
                    for aux_name, aux_loss in aux_losses.items():
                        if aux_name == "diversity_loss":
                            diversity_weight = self.continual_config.get(
                                "prompt_tuning", {}
                            ).get("diversity_weight", 0.01)
                            loss += diversity_weight * aux_loss
                        elif aux_name == "similarity_loss":
                            similarity_weight = self.continual_config.get(
                                "prompt_tuning", {}
                            ).get("similarity_weight", 0.01)
                            loss += similarity_weight * aux_loss

                # Backward pass
                loss.backward()

                # Accumulate gradient norms after backward, before clipping
                self._accumulate_gradient_norms()

                # Apply gradient clipping if enabled
                if self.gradient_clipping_enabled:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.gradient_clipping_max_norm
                    )

                # Update weights
                self.optimizer.step()

                # Update EMA teacher after optimizer step
                if self.ema_enabled:
                    self._update_ema_teacher()

            # Update metrics
            total_loss += loss.item()
            if self.entropy_prediction:
                predicted = self._entropy_based_prediction(outputs)
            else:
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
                }

                # Add more detailed info in verbose debug mode
                if debug_verbose:
                    batch_size = inputs.size(0)
                    lr_stats = {
                        f"lr_{i}": param_group["lr"]
                        for i, param_group in enumerate(self.optimizer.param_groups)
                    }
                    postfix.update(
                        {
                            "batch_size": batch_size,
                            **lr_stats,
                            "step": step,
                            "epoch": epoch + 1,
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
                if self.mixed_precision_eval:
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
                total_loss += loss.item() * inputs.size(0)
                if self.entropy_prediction:
                    predicted = self._entropy_based_prediction(outputs)
                else:
                    _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                # Update progress bar if showing
                if should_show_pbar:
                    test_iter.set_postfix(
                        {
                            "loss": total_loss / total,
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

        test_loss = total_loss / total
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
        global_epoch = step * self.training_config["num_epochs"] + epoch
        if self.writer is not None:
            self.writer.add_scalar("global/train_loss", train_loss, global_epoch)
            self.writer.add_scalar(f"step_{step}/train_loss", train_loss, global_epoch)
            self.writer.add_scalar(f"step_{step}/train_acc", train_acc, global_epoch)
            if test_loss is not None and test_acc is not None:
                self.writer.add_scalar("global/test_loss", test_loss, global_epoch)
                self.writer.add_scalar(
                    f"step_{step}/test_loss", test_loss, global_epoch
                )
                self.writer.add_scalar(f"step_{step}/test_acc", test_acc, global_epoch)

            # Log learning rate
            for i, param_group in enumerate(self.optimizer.param_groups):
                self.writer.add_scalar(
                    f"global/learning_rate_{i}", param_group["lr"], global_epoch
                )
            self.writer.add_scalar("global/step", step, global_epoch)
            self.writer.add_scalar("global/epoch", epoch, global_epoch)

        # Log to Weights & Biases
        if self.logging_config["wandb"]:
            lr_data = {
                f"global/learning_rate_{i}": param_group["lr"]
                for i, param_group in enumerate(self.optimizer.param_groups)
            }
            log_data = {
                "global/train_loss": train_loss,
                f"step_{step}/train_loss": train_loss,
                f"step_{step}/train_acc": train_acc,
                **lr_data,
                "global/step": step,
                "global/epoch": epoch,
            }
            if test_loss is not None and test_acc is not None:
                log_data.update(
                    {
                        "global/test_loss": test_loss,
                        f"step_{step}/test_loss": test_loss,
                        f"step_{step}/test_acc": test_acc,
                    }
                )

            wandb.log(log_data, step=global_epoch)

        # Log gradient norms if enabled
        if self.logging_config.get("log_grad_norms", False):
            self._log_gradient_norms(step, epoch)

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

        # Save teacher model state if using EMA
        if self.ema_enabled and self.ema_teacher_model is not None:
            if self.distributed:
                teacher_state = self.ema_teacher_model.module.state_dict()
            else:
                teacher_state = self.ema_teacher_model.state_dict()
            checkpoint["ema_teacher"] = teacher_state

        # Get experiment name (includes timestamp)
        experiment_name = self.config["experiment"]["name"]

        # Create experiment-specific checkpoint directory
        experiment_checkpoint_dir = os.path.join(self.checkpoint_dir, experiment_name)
        os.makedirs(experiment_checkpoint_dir, exist_ok=True)

        if best:
            checkpoint_path = os.path.join(
                experiment_checkpoint_dir,
                f"step{step}_best.pth",
            )
        else:
            checkpoint_path = os.path.join(
                experiment_checkpoint_dir,
                f"step{step}_epoch{epoch}.pth",
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
        # Get experiment name (includes timestamp)
        experiment_name = self.config["experiment"]["name"]

        # Use experiment-specific checkpoint directory
        experiment_checkpoint_dir = os.path.join(self.checkpoint_dir, experiment_name)

        if not os.path.exists(experiment_checkpoint_dir):
            if not self.distributed or (self.distributed and self.local_rank == 0):
                print(f"Checkpoint directory {experiment_checkpoint_dir} not found")
            return

        if best:
            checkpoint_path = os.path.join(
                experiment_checkpoint_dir,
                f"step{step}_best.pth",
            )
        else:
            # Load the latest epoch checkpoint
            checkpoint_files = glob.glob(
                os.path.join(
                    experiment_checkpoint_dir,
                    f"step{step}_epoch*.pth",
                )
            )
            if not checkpoint_files:
                if not self.distributed or (self.distributed and self.local_rank == 0):
                    print(f"No checkpoint found for step {step}")
                return

            # Sort by epoch number and get the latest
            checkpoint_files.sort(
                key=lambda x: int(x.split("_epoch")[-1].split(".")[0]), reverse=True
            )
            checkpoint_path = checkpoint_files[0]

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

        # Load teacher model state if available and needed
        if (
            self.ema_enabled
            and "ema_teacher" in checkpoint
            and self.ema_teacher_model is not None
        ):
            if self.distributed:
                self.ema_teacher_model.module.load_state_dict(checkpoint["ema_teacher"])
            else:
                self.ema_teacher_model.load_state_dict(checkpoint["ema_teacher"])
            print(f"Loaded EMA teacher model from checkpoint for step {step}")

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

        # Set description based on debug mode
        debug_enabled = self.debug_config.get("enabled", False)
        desc = "Computing EWC data [DEBUG]" if debug_enabled else "Computing EWC data"

        data_iter = tqdm(train_loader, desc=desc) if should_show_pbar else train_loader

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

    def _entropy_based_prediction(self, outputs: torch.Tensor) -> torch.Tensor:
        """
        Make predictions based on entropy of class probabilities

        Instead of simply taking the max logit, this method:
        1. Calculates class probabilities for each step
        2. Computes entropy for each step's probability distribution
        3. Selects the step with lowest entropy (highest confidence)
        4. Returns the class with highest probability from that step

        Args:
            outputs: Model output logits [batch_size, num_classes]

        Returns:
            Predicted class indices [batch_size]
        """
        batch_size = outputs.size(0)
        num_classes = outputs.size(1)
        num_steps = self.continual_config["num_steps"]
        classes_per_step = self.continual_config["classes_per_step"]

        # Handle case where num_classes is not divisible by classes_per_step
        # by padding the outputs tensor to make it divisible
        if num_classes % classes_per_step != 0:
            # Calculate padding needed
            padding_size = classes_per_step - (num_classes % classes_per_step)
            # Pad with negative infinity to ensure they don't affect softmax
            padding = torch.full(
                (batch_size, padding_size), -torch.inf, device=outputs.device
            )
            # Pad the outputs tensor
            padded_outputs = torch.cat([outputs, padding], dim=1)
        else:
            padded_outputs = outputs

        # Reshape the outputs to [batch_size, num_steps, classes_per_step]
        # This creates a 3D tensor where each slice along dim=1 represents a step's logits
        reshaped_outputs = padded_outputs.view(batch_size, num_steps, classes_per_step)

        # Calculate softmax probabilities for each step (along classes_per_step dimension)
        step_probs = reshaped_outputs.softmax(dim=2)
        # [batch_size, num_steps, classes_per_step]

        # Calculate entropy for each step: -sum(p * log(p))
        # Add small epsilon to avoid log(0)
        log_probs = torch.log(step_probs + 1e-10)
        entropies = -torch.sum(step_probs * log_probs, dim=2)  # [batch_size, num_steps]

        # Find step with minimum entropy (maximum confidence) for each sample
        min_entropy_steps = entropies.argmin(dim=1)  # [batch_size]

        # Create a mask to select the logits from the step with minimum entropy for each sample
        # First create indices for the batch dimension
        batch_indices = torch.arange(batch_size, device=outputs.device)

        # Select the logits from the minimum entropy step for each sample
        # This gives us [batch_size, classes_per_step]
        selected_step_logits = reshaped_outputs[batch_indices, min_entropy_steps]

        # Find the class with maximum probability within the selected step
        max_class_indices = selected_step_logits.argmax(dim=1)  # [batch_size]

        # Convert to the actual class indices in the original output space
        predictions = min_entropy_steps * classes_per_step + max_class_indices

        # Ensure predictions don't exceed the original number of classes
        predictions = torch.clamp(predictions, max=num_classes - 1)

        return predictions

    def _apply_logit_masking(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply logit masking to restrict predictions to current task classes.

        Args:
            outputs: Model output logits [batch_size, num_classes]
            targets: Target labels [batch_size]

        Returns:
            Masked logits [batch_size, num_classes]
        """
        if not self.mask_logits or self.current_task_classes is None:
            return outputs

        # Create a mask for replay samples
        replay_mask = torch.zeros_like(outputs, dtype=torch.bool)

        # For each target in the batch, check if it's from replay
        for i, target in enumerate(targets):
            target_class = target.item()
            if target_class not in self.current_task_classes:
                # This is a replay sample, don't mask this sample
                replay_mask[i, :] = True

        # Create a mask for current task classes
        class_mask = torch.zeros_like(outputs, dtype=torch.bool)
        for cls in self.current_task_classes:
            class_mask[:, cls] = True

        # Combine masks: keep current task classes and allowed classes for replay samples
        combined_mask = class_mask | replay_mask

        # Apply negative infinity to mask out non-current task logits
        masked_outputs = outputs.clone()
        masked_outputs[~combined_mask] = -torch.inf

        return masked_outputs

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

    def _get_prompt_auxiliary_losses(self) -> Dict[str, torch.Tensor]:
        """
        Compute auxiliary losses for prompt tuning.

        Returns:
            Dictionary of auxiliary losses
        """
        aux_losses = {}

        # Get the correct model reference (module if distributed)
        model_to_use = self.model.module if self.distributed else self.model

        # Check if the model has prompt tuning auxiliary losses
        if hasattr(model_to_use, "get_auxiliary_losses"):
            aux_losses.update(model_to_use.get_auxiliary_losses())

        return aux_losses

    def _initialize_ema_teacher(self):
        """
        Initialize EMA teacher model as a deep copy of the student model.
        Creates a complete copy with all parameters.
        """
        if not self.ema_enabled:
            return

        # Clean up previous teacher model to avoid memory leaks
        if self.ema_teacher_model is not None:
            # Move to CPU first to free GPU memory
            self.ema_teacher_model.cpu()
            # Delete the model
            del self.ema_teacher_model
            # Force GPU memory cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Create a deep copy of the model for the teacher
        import copy

        # Get the correct model reference (module if distributed)
        student_model = self.model.module if self.distributed else self.model

        # Create teacher model as deep copy
        self.ema_teacher_model = copy.deepcopy(student_model)

        # Move teacher to same device
        self.ema_teacher_model = self.ema_teacher_model.to(self.device)

        # Set teacher to eval mode (it won't be trained directly)
        self.ema_teacher_model.eval()

        # Disable gradients for teacher model to save memory
        for param in self.ema_teacher_model.parameters():
            param.requires_grad = False

        print(f"Initialized EMA teacher model with momentum {self.ema_momentum}")

    def _update_ema_teacher(self):
        """
        Update EMA teacher model parameters with student model parameters.
        Updates parameters based on skip_names configuration.
        """
        if not self.ema_enabled or self.ema_teacher_model is None:
            return

        # Get the correct model reference (module if distributed)
        student_model = self.model.module if self.distributed else self.model

        # Update teacher parameters with EMA
        with torch.no_grad():
            for (name, student_param), (_, teacher_param) in zip(
                student_model.named_parameters(),
                self.ema_teacher_model.named_parameters(),
            ):
                # Skip parameters based on configurable skip names
                if any(skip_name in name.lower() for skip_name in self.ema_skip_names):
                    continue

                # Check for momentum override for this parameter
                momentum = self.ema_momentum
                for (
                    override_name,
                    override_momentum,
                ) in self.ema_momentum_overrides.items():
                    if override_name in name.lower():
                        momentum = override_momentum
                        break

                # EMA update: teacher = momentum * teacher + (1 - momentum) * student
                teacher_param.data.mul_(momentum).add_(
                    student_param.data, alpha=1 - momentum
                )

    def _replace_student_with_teacher(self):
        """
        Replace student parameters with teacher parameters based on skip configuration.
        Only replaces parameters not in the skip_names list.
        """
        if not self.ema_enabled or self.ema_teacher_model is None:
            return

        # Get the correct model reference (module if distributed)
        student_model = self.model.module if self.distributed else self.model

        # Copy teacher parameters to student (respecting skip_names)
        with torch.no_grad():
            for (name, student_param), (_, teacher_param) in zip(
                student_model.named_parameters(),
                self.ema_teacher_model.named_parameters(),
            ):
                # Skip parameters based on configurable skip names
                if any(skip_name in name.lower() for skip_name in self.ema_skip_names):
                    continue

                # Copy teacher parameter to student
                student_param.data.copy_(teacher_param.data)

        print(
            "Replaced student parameters with EMA teacher parameters (respecting skip_names)"
        )

    def _save_student_parameters(self):
        """
        Save current student parameters for later restoration (respecting skip_names).
        Used for temporary teacher evaluation during training.
        """
        if not self.ema_enabled:
            return

        # Get the correct model reference (module if distributed)
        student_model = self.model.module if self.distributed else self.model

        # Save parameters (skip those in skip_names)
        self.student_parameters_backup = {}

        for name, param in student_model.named_parameters():
            # Only save parameters not in skip_names
            if not any(skip_name in name.lower() for skip_name in self.ema_skip_names):
                self.student_parameters_backup[name] = param.data.clone()

    def _restore_student_parameters(self) -> None:
        """
        Restore previously saved student parameters (respecting skip_names).
        Used after temporary teacher evaluation during training.
        """
        if not self.ema_enabled or not hasattr(self, "student_parameters_backup"):
            return

        # Get the correct model reference (module if distributed)
        student_model = self.model.module if self.distributed else self.model

        # Restore saved parameters
        with torch.no_grad():
            for name, param in student_model.named_parameters():
                # Only restore saved parameters
                if name in self.student_parameters_backup:
                    param.data.copy_(self.student_parameters_backup[name])

        # Clear backup to save memory
        del self.student_parameters_backup

    def _save_classifier_state(self) -> Dict[str, torch.Tensor]:
        """
        Save current classifier weights for temporary calibration during evaluation.
        
        Returns:
            Dictionary mapping classifier parameter names to tensor values
        """
        model = self.model.module if hasattr(self.model, "module") else self.model
        classifier_state = {}
        
        # Save classifier weights (including prototypes)
        for name, param in model.named_parameters():
            if "classifier" in name:
                classifier_state[name] = param.data.clone()
        
        return classifier_state

    def _restore_classifier_state(self, classifier_state: Dict[str, torch.Tensor]) -> None:
        """
        Restore classifier weights from saved state.
        
        Args:
            classifier_state: Dictionary mapping classifier parameter names to tensor values
        """
        model = self.model.module if hasattr(self.model, "module") else self.model
        
        for name, param in model.named_parameters():
            if name in classifier_state:
                param.data.copy_(classifier_state[name])

    def _store_task_prototypes(
        self,
        step: int,
        task_prototypes: Dict[int, torch.Tensor],
    ):
        """
        Store prototypes and checkpoint path for current task to enable future calibration.

        Args:
            step: Current continual learning step
            task_prototypes: Prototypes extracted with current step backbone
        """
        if not self.calibration_enabled:
            return

        # Store prototypes
        self.historical_prototypes[step] = {cid: proto.clone().detach() for cid, proto in task_prototypes.items()}

        # Store checkpoint path for loading backbone state when needed
        experiment_name = self.config["experiment"]["name"]
        experiment_checkpoint_dir = os.path.join(self.checkpoint_dir, experiment_name)
        checkpoint_path = os.path.join(experiment_checkpoint_dir, f"step{step}_best.pth")
        self.historical_checkpoint_paths[step] = checkpoint_path

        if not self.distributed or (self.distributed and self.local_rank == 0):
            print(f"Stored prototypes and checkpoint path for step {step} for future calibration")

    def _compute_rigid_transform(
        self, source: torch.Tensor, target: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute rigid transformation (rotation + translation) from source to target prototypes.

        Args:
            source: Source prototypes [num_classes, features]
            target: Target prototypes [num_classes, features]

        Returns:
            Dict containing rotation matrix and translation vector
        """
        # Center the point sets
        source_mean = source.mean(dim=0, keepdim=True)
        target_mean = target.mean(dim=0, keepdim=True)

        source_centered = source - source_mean
        target_centered = target - target_mean

        # Compute rotation using SVD (Procrustes analysis)
        # H = source_centered.T @ target_centered
        H = torch.matmul(source_centered.T, target_centered)
        U, S, Vt = torch.svd(H)

        # Compute rotation matrix
        R = torch.matmul(Vt.T, U.T)

        # Ensure proper rotation (det(R) = 1)
        if torch.det(R) < 0:
            Vt[-1, :] *= -1
            R = torch.matmul(Vt.T, U.T)

        # Compute translation
        t = target_mean - torch.matmul(source_mean, R.T)

        return {"rotation": R, "translation": t.squeeze(0)}

    def _compute_affine_transform(
        self, source: torch.Tensor, target: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute affine transformation (linear map + translation) from source to target prototypes.

        Args:
            source: Source prototypes [num_classes, features]
            target: Target prototypes [num_classes, features]

        Returns:
            Dict containing transformation matrix and translation vector
        """
        # Add bias term for affine transformation
        # Solve: target = source @ A + b
        # This is equivalent to: target = [source, 1] @ [A; b]

        num_classes, num_features = source.shape

        # Add column of ones for bias term
        source_aug = torch.cat(
            [source, torch.ones(num_classes, 1, device=source.device)], dim=1
        )

        # Solve least squares: source_aug @ transform = target
        # transform = (source_aug.T @ source_aug)^-1 @ source_aug.T @ target
        try:
            transform = torch.linalg.solve(
                torch.matmul(source_aug.T, source_aug),
                torch.matmul(source_aug.T, target),
            )
        except torch.linalg.LinAlgError:
            # Fallback to pseudo-inverse if singular
            transform = torch.linalg.pinv(source_aug) @ target

        # Extract linear transformation and translation
        linear_transform = transform[:-1, :]
        translation = transform[-1, :]

        return {"linear": linear_transform, "translation": translation}

    def _compute_nonlinear_transform(
        self, source: torch.Tensor, target: torch.Tensor
    ) -> Callable:
        """
        Compute nonlinear transformation using neural network from source to target prototypes.

        Args:
            source: Source prototypes [num_classes, features]
            target: Target prototypes [num_classes, features]

        Returns:
            Trained transformation function
        """
        num_features = source.shape[1]

        # Create a simple MLP for nonlinear transformation
        class TransformNet(nn.Module):
            def __init__(self, input_dim, hidden_dim=None):
                super().__init__()
                if hidden_dim is None:
                    hidden_dim = max(input_dim // 2, 64)

                self.net = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, input_dim),
                )

            def forward(self, x):
                return self.net(x)

        # Initialize and train the transformation network
        transform_net = TransformNet(num_features).to(source.device)
        optimizer = torch.optim.Adam(transform_net.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        # Training loop
        transform_net.train()
        for epoch in range(100):  # Quick training
            optimizer.zero_grad()
            pred = transform_net(source)
            loss = criterion(pred, target)

            # Add regularization to prevent overfitting
            reg_loss = 0
            for param in transform_net.parameters():
                reg_loss += param.norm(2)
            loss += self.calibration_reg_weight * reg_loss

            loss.backward()
            optimizer.step()

            if epoch % 20 == 0 and (not self.distributed or (self.distributed and self.local_rank == 0)):
                print(f"Calibration training epoch {epoch}, loss: {loss.item():.6f}")

        transform_net.eval()
        return transform_net

    def _apply_calibration_transform(
        self, prototypes: torch.Tensor, transform_params: Dict, method: str
    ) -> torch.Tensor:
        """
        Apply calibration transformation to prototypes.

        Args:
            prototypes: Prototypes to transform [num_classes, features]
            transform_params: Transformation parameters
            method: Transformation method ('rigid', 'affine', 'nonlinear')

        Returns:
            Transformed prototypes
        """
        if method == "rigid":
            # Apply rigid transformation: R @ p + t
            return (
                torch.matmul(prototypes, transform_params["rotation"].T)
                + transform_params["translation"]
            )

        elif method == "affine":
            # Apply affine transformation: p @ A + b
            return (
                torch.matmul(prototypes, transform_params["linear"])
                + transform_params["translation"]
            )

        elif method == "nonlinear":
            # Apply nonlinear transformation using trained network
            with torch.no_grad():
                return transform_params(prototypes)

        else:
            raise ValueError(f"Unknown calibration method: {method}")

    def _calibrate_previous_classifiers(
        self, current_step: int, train_loader: DataLoader, all_step_classes: List[List[int]]
    ):
        """
        Calibrate all previous task classifiers by computing a transformation
        from current task prototypes (with previous backbone) to current task prototypes
        (with current backbone) and applying the transformation to previous task prototypes.

        This measures backbone drift and helps maintain alignment between previous
        task classifiers and the evolved backbone after training on new tasks.

        Args:
            current_step: The current continual learning step
            train_loader: Training data loader for current task
            all_step_classes: List of lists of classes seen so far
        """
        if not self.calibration_enabled or current_step <= 0:
            return

        if not self.distributed or (self.distributed and self.local_rank == 0):
            print(f"\n=== Calibrating previous classifiers at step {current_step} ===")

        # Get model reference
        model_to_use = self.model.module if self.distributed else self.model

        # Check if we have a prototypical classifier
        if not hasattr(model_to_use, "classifier") or not hasattr(
            model_to_use.classifier, "classifier"
        ):
            print("Warning: No prototypical classifier found for calibration")
            return

        classifier = model_to_use.classifier.classifier
        if not hasattr(classifier, "prototypes"):
            print("Warning: Classifier does not have prototypes for calibration")
            return

        # Get current task prototypes
        if current_step not in self.historical_prototypes:
            print(f"Warning: No historical prototypes found for step {current_step}")
            return

        current_task_prototypes = self.historical_prototypes[current_step]

        # We need at least one previous step to compute calibration transform
        if current_step <= 0:
            return

        # Use most recent previous step's backbone to compute current task prototypes
        # This measures how the current task prototypes changed due to backbone evolution
        reference_step = current_step - 1
        if reference_step not in self.historical_checkpoint_paths:
            print(f"Warning: No reference checkpoint found for step {reference_step}")
            return

        # Compute current task prototypes using previous backbone from checkpoint
        previous_checkpoint_path = self.historical_checkpoint_paths[reference_step]
        reference_task_prototypes = self._extract_prototypes_with_backbone(
            train_loader, previous_checkpoint_path
        )

        # Convert prototypes to tensors
        reference_task_prototypes = torch.stack(tuple(reference_task_prototypes.values()))
        current_task_prototypes = torch.stack(tuple(current_task_prototypes.values()))

        # Compute transformation from current task prototypes (with previous backbone) to current task prototypes (with current backbone)
        # This measures the backbone drift for the current task
        if self.calibration_method == "rigid":
            transform_params = self._compute_rigid_transform(
                reference_task_prototypes, current_task_prototypes
            )
        elif self.calibration_method == "affine":
            transform_params = self._compute_affine_transform(
                reference_task_prototypes, current_task_prototypes
            )
        elif self.calibration_method == "nonlinear":
            transform_params = self._compute_nonlinear_transform(
                reference_task_prototypes, current_task_prototypes
            )
        else:
            raise ValueError(f"Unknown calibration method: {self.calibration_method}")

        # Apply calibration to all previous task classifiers
        num_calibrated = 0
        with torch.no_grad():
            for step in range(current_step):
                if step in self.historical_prototypes:
                    # Get the classes for this step
                    class_indices = all_step_classes[step]

                    # Get the actual prototypes used by this previous task (after its training)
                    original_prototypes = self.historical_prototypes[step]
                    original_prototypes = torch.stack(tuple(original_prototypes.values()))

                    # Apply calibration transform
                    calibrated_prototypes = self._apply_calibration_transform(
                        original_prototypes, transform_params, self.calibration_method
                    )

                    # Update the classifier prototypes
                    classifier.prototypes.data[class_indices] = calibrated_prototypes

                    num_calibrated += 1

                    if not self.distributed or (self.distributed and self.local_rank == 0):
                        print(f"Calibrated prototypes for step {step}")

        if not self.distributed or (self.distributed and self.local_rank == 0):
            print(f"Successfully calibrated {num_calibrated} previous task classifiers using {self.calibration_method} method")

    def _extract_prototypes_with_backbone(
        self, train_loader: DataLoader, checkpoint_path: str=None
    ) -> Dict[int, torch.Tensor]:
        """
        Compute current task prototypes using backbone from checkpoint.
        If checkpoint_path is None, use current backbone.
        Uses test transforms for clean, non-augmented features.

        Args:
            train_loader: Training data loader for current task
            checkpoint_path: Path to checkpoint

        Returns:
            Prototypes computed with backbone from checkpoint
        """
        model_to_use = self.model.module if self.distributed else self.model

        # Load checkpoint
        if checkpoint_path is not None:
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            previous_model_state = checkpoint["model"]

        # Use test transforms for clean prototype extraction
        if self.data_module is None:
            print(
                "Warning: data_module not available for test transforms, "
                "using training transforms for prototype extraction"
            )
            temp_loader = train_loader
        else:
            # Initialize prototypes from training data but with test transforms
            # Create a temporary data loader with test transforms for prototype initialization
            # This ensures consistent feature extraction without augmentations
            temp_dataset = []
            dataset = train_loader.dataset
            # Handle nested datasets (e.g., Subsets, Concatenated datasets)
            while True:
                if hasattr(dataset, "dataset"):
                    dataset = dataset.dataset
                elif hasattr(dataset, "datasets"):
                    dataset = dataset.datasets
                else:
                    if isinstance(dataset, list):
                        temp_dataset.extend(dataset)
                    else:
                        temp_dataset.append(dataset)
                    break

            original_transforms = []
            for dataset in temp_dataset:
                assert hasattr(dataset, "transform"), (
                    f"Dataset {dataset} does not have a transform attribute"
                )
                original_transforms.append(dataset.transform)
                dataset.transform = self.data_module.test_transform

            # Create temporary loader with test batch size for efficiency
            temp_loader = torch.utils.data.DataLoader(
                train_loader.dataset,
                batch_size=self.training_config["eval_batch_size"],  # Use larger eval batch size
                shuffle=False,  # No need to shuffle for prototype extraction
                num_workers=train_loader.num_workers,
                pin_memory=train_loader.pin_memory,
            )

        try:
            # Temporarily replace current backbone with backbone from checkpoint
            if checkpoint_path is not None:
                current_backbone_state = {}
                for name, param in model_to_use.named_parameters():
                    if "backbone" in name or "encoder" in name:
                        current_backbone_state[name] = param.clone().detach()
                        if name in previous_model_state:
                            param.data.copy_(previous_model_state[name])

            # Initialize lists for features and labels
            features_list = []
            labels_list = []

            # Compute prototypes with backbone from checkpoint
            model_to_use.eval()

            # Add progress bar with tqdm
            data_iter = tqdm(temp_loader, desc="Extracting features for prototypes")

            with torch.no_grad():
                for inputs, targets in data_iter:
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)

                    # Filter to current task classes only
                    task_mask = torch.isin(
                        targets,
                        torch.tensor(list(self.current_task_classes), device=self.device),
                    )
                    if not task_mask.any():
                        continue

                    inputs = inputs[task_mask]
                    targets = targets[task_mask]

                    # Extract features with backbone from checkpoint
                    features = model_to_use.forward_features(inputs)

                    features_list.append(features.cpu())
                    labels_list.append(targets.cpu())

            # Compute prototypes
            prototypes = self._compute_prototypes_from_features(
                features_list, labels_list
            )

            # Restore current backbone state if using checkpoint
            if checkpoint_path is not None:
                # Restore current backbone state
                for name, param in model_to_use.named_parameters():
                    if name in current_backbone_state:
                        param.data.copy_(current_backbone_state[name])

            return prototypes

        finally:
            # Restore original transforms
            if (
                self.data_module is not None
                and original_transforms
            ):
                for dataset, original in zip(temp_dataset, original_transforms):
                    assert hasattr(dataset, "transform"), (
                        f"Dataset {dataset} does not have a transform attribute"
                    )
                    dataset.transform = original

    def _compute_prototypes_from_features(
        self,
        features_list: List[torch.Tensor],
        labels_list: List[torch.Tensor],
    ) -> Dict[int, torch.Tensor]:
        """
        Compute prototypes from features and labels.

        Args:
            features_list: List of feature tensors
            labels_list: List of label tensors

        Returns:
            Prototypes dict with class indices as keys and prototypes as values
        """
        # Concatenate all features and labels
        all_features = torch.cat(features_list, dim=0).to(self.device)
        all_labels = torch.cat(labels_list, dim=0).to(self.device)

        # Compute prototypes for each class
        prototypes = {}
        for class_idx in self.current_task_classes:
            class_mask = all_labels == class_idx
            if class_mask.any():
                class_features = all_features[class_mask]
                prototypes[class_idx] = class_features.mean(dim=0)

        return prototypes

    def _replace_classifier_with_prototypes(self, train_loader: DataLoader):
        """
        Replace classifier weights with computed prototypes from current task.
        Uses the _extract_prototypes_with_backbone method to compute prototypes
        with the current backbone and replaces the classifier prototypes.
        
        Args:
            train_loader: Training data loader for current task
        """
        debug_enabled = self.debug_config.get("enabled", False)
        debug_prefix = "[DEBUG] " if debug_enabled else ""

        print(f"{debug_prefix}Starting classifier replacement with prototypes")

        # Extract prototypes using current backbone
        prototypes = self._extract_prototypes_with_backbone(train_loader)

        if not prototypes:
            print(f"Warning: No prototypes computed for current task classes {self.current_task_classes}")
            return

        # Get the model (handle distributed training)
        model_to_use = self.model.module if self.distributed else self.model

        # Check if the model has a prototypical classifier
        if not hasattr(model_to_use.classifier, 'classifier') or not hasattr(model_to_use.classifier.classifier, 'prototypes'):
            print("Warning: Model does not have a prototypical classifier, skipping classifier replacement")
            return

        # Replace classifier prototypes with computed prototypes
        classifier = model_to_use.classifier.classifier
        num_replaced = 0

        for class_idx, prototype in prototypes.items():
            if class_idx < classifier.prototypes.shape[0]:
                # Replace the prototype for this class
                classifier.prototypes.data[class_idx] = prototype.to(classifier.prototypes.device)
                num_replaced += 1
                if debug_enabled:
                    print(f"{debug_prefix}Replaced prototype for class {class_idx}")
            else:
                print(f"Warning: Class index {class_idx} is out of bounds for classifier prototypes")

        if not self.distributed or (self.distributed and self.local_rank == 0):
            print(f"Replaced {num_replaced} classifier prototypes with computed prototypes")

    def calibrate_previous_classifiers(
        self,
        step: int,
        train_loader: DataLoader,
        all_step_classes: List[List[int]],
    ):
        debug_enabled = self.debug_config.get("enabled", False)
        debug_prefix = "[DEBUG] " if debug_enabled else ""
        print(f"{debug_prefix}Starting classifier calibration for step {step + 1}")

        # Extract prototypes for current task
        if self.calibration_classifier_as_prototype:
            model = self.model.module if self.distributed else self.model
            all_prototypes = model.classifier.classifier.prototypes.data.clone().detach()
            current_prototypes = {cid: all_prototypes[cid] for cid in self.current_task_classes}
        else:
            current_prototypes = self._extract_prototypes_with_backbone(train_loader)

        # Store prototypes for this step
        self._store_task_prototypes(step, current_prototypes)

        # Calibrate previous task classifiers if this is not the first step
        if step > 0:
            self._calibrate_previous_classifiers(step, train_loader, all_step_classes)
            print(f"{debug_prefix}Calibrated previous task classifiers for step {step + 1}")
        else:
            print(f"{debug_prefix}No previous task classifiers to calibrate for step {step + 1}")

        print(f"{debug_prefix}Completed classifier calibration for step {step + 1}")

    def _reset_gradient_norm_accumulator(self):
        """
        Reset gradient norm accumulator at the start of each epoch.
        """
        self.grad_norm_accumulator = {}
        self.grad_norm_batch_count = 0

    def _accumulate_gradient_norms(self):
        """
        Accumulate gradient norms for the current batch.
        Should be called after backward() and unscaling (if using mixed precision),
        but before gradient clipping and optimizer.step().
        """
        if not self.logging_config.get("log_grad_norms", False):
            return

        model_to_use = self.model.module if self.distributed else self.model

        # Accumulate gradient norms for each parameter
        for name, param in model_to_use.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.data.norm(2).item()
                clean_name = name.replace("module.", "")

                if clean_name not in self.grad_norm_accumulator:
                    self.grad_norm_accumulator[clean_name] = 0.0

                self.grad_norm_accumulator[clean_name] += grad_norm

        self.grad_norm_batch_count += 1

    def _log_gradient_norms(self, step: int, epoch: int):
        """
        Log accumulated gradient norms for each parameter layer to tensorboard and wandb.
        This logs the average gradient norms across all batches in the epoch.

        Args:
            step: Current continual learning step
            epoch: Current epoch
        """
        # Only log on main process in distributed training
        if self.distributed and self.local_rank != 0:
            return

        # Skip if no gradients were accumulated (shouldn't happen if logging is enabled)
        if self.grad_norm_batch_count == 0:
            return

        # Compute global step for logging (same as other logging methods)
        global_step = step * self.training_config["num_epochs"] + epoch

        # Log average gradient norm for each parameter layer
        for clean_name, accumulated_norm in self.grad_norm_accumulator.items():
            # Compute average gradient norm across all batches in this epoch
            avg_grad_norm = accumulated_norm / self.grad_norm_batch_count

            # Log to tensorboard
            if self.writer is not None:
                self.writer.add_scalar(
                    f"grad_norms/{clean_name}", avg_grad_norm, global_step
                )

            # Log to wandb
            if self.logging_config.get("wandb", False):
                wandb.log({f"grad_norms/{clean_name}": avg_grad_norm}, step=global_step)
