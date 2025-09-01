import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict


class Loss:
    """Base class for all losses."""

    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class LossRegistry:
    """Registry for managing loss functions."""

    _losses: dict[str, type[Loss]] = {}

    @classmethod
    def register(cls, name: str):
        """Decorator to register a loss function."""

        def wrapper(loss_cls):
            cls._losses[name] = loss_cls
            return loss_cls

        return wrapper

    @classmethod
    def get(cls, name: str) -> type[Loss]:
        """Get a loss function by name."""
        if name not in cls._losses:
            raise ValueError(
                f"Loss {name} not found in registry. Available losses: {list(cls._losses.keys())}"
            )
        return cls._losses[name]

    @classmethod
    def create(cls, name: str, **kwargs) -> Loss:
        """Create a loss instance by name with given parameters."""
        loss_cls = cls.get(name)
        return loss_cls(**kwargs)


@LossRegistry.register("mse")
class MSELoss(Loss):
    def __call__(self, outputs, targets):
        return F.mse_loss(outputs.view(-1), targets)


@LossRegistry.register("huber")
class HuberLoss(Loss):
    def __init__(self, delta: float = 0.1):
        self.delta = delta

    def __call__(self, outputs, targets):
        return F.huber_loss(outputs.view(-1), targets, delta=self.delta)


@LossRegistry.register("l1")
class L1Loss(Loss):
    def __call__(self, outputs, targets):
        return F.l1_loss(outputs.view(-1), targets)


@LossRegistry.register("rmse")
class RMSELoss(Loss):
    def __call__(self, outputs, targets):
        return torch.sqrt(F.mse_loss(outputs.view(-1), targets))


@LossRegistry.register("bce_logit")
class BCEWithLogitsLoss(Loss):
    def __init__(self, weight=None):
        if weight is not None and isinstance(weight, torch.Tensor):
            weight = weight.to(weight.device)
        self.criterion = nn.BCEWithLogitsLoss(weight=weight)

    def __call__(self, outputs, targets):
        if outputs.dim() > 1 and outputs.size(1) == 2:
            outputs = outputs[:, 1]  # Select the second item for binary classification
        elif outputs.dim() > 1 and outputs.size(1) == 1:
            outputs = outputs.squeeze()  # Squeeze the dimension

        # Convert targets to float type
        targets = targets.float()

        return self.criterion(outputs, targets)


@LossRegistry.register("ce")
class CrossEntropyLoss(Loss):
    def __init__(self, weight=None):
        self.criterion = nn.CrossEntropyLoss(weight=weight)

    def __call__(self, outputs, targets):
        outputs = outputs.squeeze()
        targets = targets.squeeze().long()
        return self.criterion(outputs, targets)


@LossRegistry.register("multiclass_focal")
class MultiClassFocalLoss(Loss):
    def __init__(self, gamma: float = 2.0):
        self.gamma = gamma
        self.ce_loss = nn.CrossEntropyLoss(reduction="none")

    def __call__(self, pred, target):
        """Compute Multi-class Focal Loss"""
        ce_loss = self.ce_loss(pred, target)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


@LossRegistry.register("binary_focal")
class BinaryFocalLoss(Loss):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        self.alpha = alpha
        self.gamma = gamma

    def __call__(self, pred, target):
        """Compute Binary Focal Loss"""
        pred = pred.squeeze()
        target = target.float()

        bce_loss = self.bce_loss(pred, target)
        probs = torch.sigmoid(pred)
        pt = torch.where(target == 1, probs, 1 - probs)
        focal_weight = (1 - pt) ** self.gamma

        if self.alpha is not None:
            alpha_weight = torch.where(target == 1, self.alpha, 1 - self.alpha)
            focal_weight = focal_weight * alpha_weight

        return (focal_weight * bce_loss).mean()


@LossRegistry.register("multi_head")
class MultiHeadLoss(nn.Module):
    def __init__(
        self,
        head_structure: dict[str, int],
        loss_structure: dict[str, str] = None,
        alpha: float = 0.25,
        gamma: float = 2.0,
        head_weights: dict[str, float] = None,
        loss_weights: dict[str, torch.Tensor] = None,
        head_task: dict[str, str] = None,
        task: str = "classification",
    ):
        """
        Initialize Multi-Head Loss with support for both standard and focal loss.

        Args:
            head_structure (dict): Dictionary mapping head names to number of classes
            loss_structure (dict): Dictionary mapping head names to loss function names
            alpha (float): Weighting factor for focal loss (used only for focal losses)
            gamma (float): Focusing parameter for focal loss (used only for focal losses)
            head_weights (dict): Optional weights for each head in the final loss computation
            loss_weights (dict): Optional dictionary mapping head names to loss weights tensor
            head_task (dict): Optional dictionary mapping head names to task type ('classification' or 'regression')
            task (str): Default task type if head_task is not specified
        """
        super().__init__()
        self.head_structure = head_structure
        self.head_weights = head_weights or {head: 1.0 for head in head_structure.keys()}
        self.loss_weights = loss_weights or {}
        self.head_task = head_task or {head: task for head in head_structure.keys()}

        # Initialize loss functions for each head
        if loss_structure is None:
            # Default to BCE for binary and CE for multi-class
            self.loss_fns = {}
            for head, num_classes in head_structure.items():
                weight = self.loss_weights.get(head, None)
                if num_classes == 1:
                    self.loss_fns[head] = LossRegistry.create("bce_logit", weight=weight)
                else:
                    self.loss_fns[head] = LossRegistry.create("ce", weight=weight)
        else:
            # Create loss functions based on specified structure
            self.loss_fns = {}
            for head, loss_name in loss_structure.items():
                weight = self.loss_weights.get(head, None)
                if loss_name == "binary_focal":
                    self.loss_fns[head] = LossRegistry.create(loss_name, alpha=alpha, gamma=gamma)
                elif loss_name == "multiclass_focal":
                    self.loss_fns[head] = LossRegistry.create(loss_name, gamma=gamma)
                elif loss_name in ["bce_logit", "ce"]:
                    self.loss_fns[head] = LossRegistry.create(loss_name, weight=weight)
                else:
                    self.loss_fns[head] = LossRegistry.create(loss_name)

    def forward(
        self, outputs: dict[str, torch.Tensor], targets: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Calculate the combined loss for all heads.

        Args:
            outputs (dict): Dictionary of model outputs for each head
            targets (dict): Dictionary of target values for each head

        Returns:
            total_loss (torch.Tensor): Combined loss from all heads
            losses (dict): Individual losses per head for logging
        """
        losses = {}
        total_loss = 0.0

        for head_name in self.head_structure.keys():
            # Compute loss using the appropriate loss function
            head_loss = self.loss_fns[head_name](outputs[head_name], targets[head_name])

            # Apply head-specific weight
            head_weight = self.head_weights[head_name]
            losses[head_name] = head_loss
            total_loss += head_weight * head_loss

        return total_loss, losses


@LossRegistry.register("masked_video_modeling")
class MaskedVideoModelingLoss(Loss):
    """
    Loss for Masked Video Modeling pretraining.
    
    This loss wraps the MaskedVideoModeling module and handles the forward pass.
    """
    
    def __init__(
        self,
        hidden_size: int = 768,
        decoder_hidden_size: int = 384,
        decoder_layers: int = 4,
        decoder_heads: int = 6,
        mask_ratio: float = 0.75,
        mask_type: str = "random",
        use_patch_projection: bool = True,
        patch_size: int = 16,
        num_frames: int = 16,
        image_size: int = 224,
    ):
        from orion.models.masked_video_modeling import MaskedVideoModeling
        
        self.mvm_module = MaskedVideoModeling(
            hidden_size=hidden_size,
            decoder_hidden_size=decoder_hidden_size,
            decoder_layers=decoder_layers,
            decoder_heads=decoder_heads,
            mask_ratio=mask_ratio,
            mask_type=mask_type,
            use_patch_projection=use_patch_projection,
            patch_size=patch_size,
            num_frames=num_frames,
            image_size=image_size,
        )
    
    def __call__(
        self, 
        features: torch.Tensor,
        videos: Optional[torch.Tensor] = None,
        encoder_model: Optional[nn.Module] = None,
        mask_ratio: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute MVM loss.
        
        Args:
            features: Encoder features [B, N, D] or None if using videos
            videos: Original videos [B, C, T, H, W] for patch-level reconstruction
            encoder_model: Encoder model for feature extraction
            mask_ratio: Optional override for mask ratio
            
        Returns:
            Dictionary with loss and additional outputs
        """
        if videos is not None:
            # Use videos for reconstruction target
            outputs = self.mvm_module(videos, encoder_model, mask_ratio)
        else:
            # Use features directly
            outputs = self.mvm_module(features, encoder_model, mask_ratio)
        
        return outputs
