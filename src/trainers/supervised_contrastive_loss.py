"""
Supervised contrastive losses for CLIP/SigLIP fine-tuning.

These losses extend the standard CLIP/SigLIP contrastive losses to handle
supervised learning scenarios where multiple samples in a batch may belong
to the same class. In standard CLIP/SigLIP, all non-diagonal pairs are treated
as negatives, but in supervised learning, same-class samples should be positives.
"""

from typing import Optional

import torch
import torch.nn.functional as F

try:
    from open_clip.loss import ClipLoss, SigLipLoss, gather_features

    CLIP_LOSSES_AVAILABLE = True
except ImportError:
    CLIP_LOSSES_AVAILABLE = False
    ClipLoss = object  # Fallback for type hints
    SigLipLoss = object
    gather_features = None


class SupervisedClipLoss(ClipLoss):
    """
    Supervised CLIP Loss that treats same-class samples as positive pairs.

    Unlike standard ClipLoss which only treats diagonal (i, i) pairs as positives,
    this loss treats all pairs (i, j) where labels[i] == labels[j] as positives.

    This is more appropriate for supervised fine-tuning where batches contain
    multiple samples from the same class.
    
    Supports multi-GPU training with proper label gathering.
    """

    def __init__(
        self,
        local_loss=False,
        gather_with_grad=False,
        cache_labels=False,
        rank=0,
        world_size=1,
        use_horovod=False,
    ):
        super().__init__(
            local_loss=local_loss,
            gather_with_grad=gather_with_grad,
            cache_labels=cache_labels,
            rank=rank,
            world_size=world_size,
            use_horovod=use_horovod,
        )

    def _gather_labels(self, labels: torch.Tensor) -> torch.Tensor:
        """Gather labels from all GPUs."""
        if self.use_horovod:
            import horovod.torch as hvd
            return hvd.allgather(labels)
        else:
            all_labels = [torch.zeros_like(labels) for _ in range(self.world_size)]
            torch.distributed.all_gather(all_labels, labels)
            return torch.cat(all_labels, dim=0)

    def _compute_logits_and_masks(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
        labels: torch.Tensor,
        logit_scale: torch.Tensor,
    ):
        """Compute logits and label masks for supervised loss."""
        if self.world_size > 1:
            # Gather features from all GPUs
            all_image_features, all_text_features = gather_features(
                image_features,
                text_features,
                local_loss=self.local_loss,
                gather_with_grad=self.gather_with_grad,
                rank=self.rank,
                world_size=self.world_size,
                use_horovod=self.use_horovod,
            )
            all_labels = self._gather_labels(labels)

            if self.local_loss:
                # Local features, all gathered features
                logits_per_image = logit_scale * image_features @ all_text_features.T
                logits_per_text = logit_scale * text_features @ all_image_features.T
                # Label masks: local vs all (same for both directions in supervised mode)
                label_mask_i2t = labels.unsqueeze(1) == all_labels.unsqueeze(0)
                label_mask_t2i = label_mask_i2t
            else:
                # All gathered features
                logits_per_image = logit_scale * all_image_features @ all_text_features.T
                logits_per_text = logits_per_image.T
                # Label masks: all vs all (symmetric)
                label_mask_i2t = all_labels.unsqueeze(1) == all_labels.unsqueeze(0)
                label_mask_t2i = label_mask_i2t.T
        else:
            # Single GPU
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logit_scale * text_features @ image_features.T
            # Label masks (symmetric)
            label_mask_i2t = labels.unsqueeze(1) == labels.unsqueeze(0)
            label_mask_t2i = label_mask_i2t.T

        return logits_per_image, logits_per_text, label_mask_i2t, label_mask_t2i

    def _supervised_contrastive_loss(
        self, logits: torch.Tensor, label_mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute supervised contrastive loss for one direction (i2t or t2i).
        
        Loss = -log(sum(exp(positive logits)) / sum(exp(all logits)))
        """
        # Mask out negative pairs
        logits_masked = logits.clone()
        logits_masked[~label_mask] = float("-inf")

        # Log-sum-exp over positives and all
        log_sum_exp_pos = torch.logsumexp(logits_masked, dim=1)
        log_sum_exp_all = torch.logsumexp(logits, dim=1)

        return (log_sum_exp_all - log_sum_exp_pos).mean()

    def forward(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
        logit_scale: torch.Tensor,
        logit_bias: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_dict: bool = False,
    ):
        """
        Forward pass with optional supervised ground truth.

        Args:
            image_features: Image features [batch_size, feature_dim]
            text_features: Text features [batch_size, feature_dim]
            logit_scale: Temperature parameter (scalar or tensor)
            logit_bias: Optional logit bias
            labels: Optional class labels [batch_size] for supervised learning
            output_dict: Whether to return dict or scalar

        Returns:
            Loss value or dict containing loss
        """
        if labels is not None:
            # Supervised mode: compute logits and label masks
            logits_per_image, logits_per_text, label_mask_i2t, label_mask_t2i = (
                self._compute_logits_and_masks(
                    image_features, text_features, labels, logit_scale
                )
            )

            # Add logit bias if present
            if logit_bias is not None:
                logits_per_image = logits_per_image + logit_bias
                logits_per_text = logits_per_text + logit_bias

            # Compute supervised contrastive loss for both directions
            loss_i2t = self._supervised_contrastive_loss(logits_per_image, label_mask_i2t)
            loss_t2i = self._supervised_contrastive_loss(logits_per_text, label_mask_t2i)

            total_loss = (loss_i2t + loss_t2i) / 2
            return {"contrastive_loss": total_loss} if output_dict else total_loss
        else:
            # Standard CLIP loss (diagonal positives only)
            return super().forward(
                image_features,
                text_features,
                logit_scale,
                logit_bias=logit_bias,
                output_dict=output_dict,
            )


class SupervisedSigLipLoss(SigLipLoss):
    """
    Supervised SigLIP Loss that treats same-class samples as positive pairs.

    Unlike standard SigLipLoss which only treats diagonal (i, i) pairs as positives,
    this loss treats all pairs (i, j) where labels[i] == labels[j] as positives.
    
    Supports multi-GPU training. Note: Only 'gather' dist_impl is supported for supervised mode.
    Other dist_impl modes ('bidir', 'shift', 'reduce') fall back to standard SigLIP loss.
    """

    def __init__(
        self,
        cache_labels: bool = False,
        rank: int = 0,
        world_size: int = 1,
        dist_impl: Optional[str] = None,
    ):
        super().__init__(
            cache_labels=cache_labels,
            rank=rank,
            world_size=world_size,
            dist_impl=dist_impl,
        )

    def _gather_labels(self, labels: torch.Tensor) -> torch.Tensor:
        """Gather labels from all GPUs."""
        all_labels = [torch.zeros_like(labels) for _ in range(self.world_size)]
        torch.distributed.all_gather(all_labels, labels)
        return torch.cat(all_labels, dim=0)

    def _gather_features(self, features: torch.Tensor) -> torch.Tensor:
        """Gather features from all GPUs."""
        all_features = [torch.zeros_like(features) for _ in range(self.world_size)]
        torch.distributed.all_gather(all_features, features)
        return torch.cat(all_features, dim=0)

    def _compute_logits_and_masks(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
        labels: torch.Tensor,
        logit_scale: torch.Tensor,
        logit_bias: Optional[torch.Tensor],
    ):
        """Compute logits and label masks for supervised loss."""
        # Gather features and labels from all GPUs if multi-GPU
        if self.world_size > 1:
            image_features = self._gather_features(image_features)
            text_features = self._gather_features(text_features)
            labels = self._gather_labels(labels)

        # Compute logits and label masks (same logic for single/multi-GPU)
        logits = self.get_logits(image_features, text_features, logit_scale, logit_bias)
        label_mask = labels.unsqueeze(1) == labels.unsqueeze(0)

        return logits, label_mask

    def _supervised_siglip_loss(
        self, logits: torch.Tensor, label_mask: torch.Tensor, batch_size: int
    ) -> torch.Tensor:
        """Compute supervised SigLIP loss.
        
        Loss = -log(sigmoid(ground_truth * logits))
        where ground_truth = +1 for same-class pairs, -1 for different-class pairs
        """
        # Convert label mask to SigLIP format: +1 for positives, -1 for negatives
        ground_truth = torch.where(
            label_mask,
            torch.ones_like(logits),
            -torch.ones_like(logits),
        )

        # SigLIP loss: -log(sigmoid(ground_truth * logits))
        loss = -F.logsigmoid(ground_truth * logits).sum() / batch_size
        return loss

    def forward(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
        logit_scale: torch.Tensor,
        logit_bias: Optional[torch.Tensor],
        labels: Optional[torch.Tensor] = None,
        output_dict: bool = False,
    ):
        """
        Forward pass with optional supervised ground truth.

        Args:
            image_features: Image features [batch_size, feature_dim]
            text_features: Text features [batch_size, feature_dim]
            logit_scale: Temperature parameter
            logit_bias: Optional logit bias
            labels: Optional class labels [batch_size] for supervised learning
            output_dict: Whether to return dict or scalar

        Returns:
            Loss value or dict containing loss
        """
        if labels is not None:
            # Supervised mode
            # Note: For multi-GPU, only 'gather' dist_impl is supported
            if self.world_size > 1 and self.dist_impl != "gather":
                print(
                    f"Warning: Supervised SigLIP loss with dist_impl='{self.dist_impl}' "
                    f"not supported. Falling back to standard SigLIP loss. "
                    f"Use dist_impl='gather' for supervised mode."
                )
                return super().forward(
                    image_features,
                    text_features,
                    logit_scale,
                    logit_bias=logit_bias,
                    output_dict=output_dict,
                )
            else:
                # Compute logits and label masks
                logits, label_mask = self._compute_logits_and_masks(
                    image_features, text_features, labels, logit_scale, logit_bias
                )

                # Compute supervised SigLIP loss
                batch_size = (
                    image_features.shape[0] * self.world_size
                    if self.world_size > 1
                    else image_features.shape[0]
                )
                loss = self._supervised_siglip_loss(logits, label_mask, batch_size)
                return {"contrastive_loss": loss} if output_dict else loss
        else:
            # Standard SigLIP loss (diagonal positives only)
            return super().forward(
                image_features,
                text_features,
                logit_scale,
                logit_bias=logit_bias,
                output_dict=output_dict,
            )
