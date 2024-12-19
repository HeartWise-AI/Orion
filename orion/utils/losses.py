import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Type


class Loss(object):
    """Base class for all losses."""
    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class LossRegistry:
    """Registry for managing loss functions."""
    _losses: Dict[str, Type[Loss]] = {}

    @classmethod
    def register(cls, name: str):
        """Decorator to register a loss function."""
        def wrapper(loss_cls):
            cls._losses[name] = loss_cls
            return loss_cls
        return wrapper

    @classmethod
    def get(cls, name: str) -> Type[Loss]:
        """Get a loss function by name."""
        if name not in cls._losses:
            raise ValueError(f"Loss {name} not found in registry. Available losses: {list(cls._losses.keys())}")
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
        self.criterion = nn.BCEWithLogitsLoss(weight=weight)

    def __call__(self, outputs, targets):
        if outputs.dim() > 1 and outputs.size(1) == 2:
            outputs = outputs[:, 1]  # Select the second item for binary classification
        elif outputs.dim() > 1 and outputs.size(1) == 1:
            outputs = outputs.squeeze()  # Squeeze the dimension
        return self.criterion(outputs, targets)


@LossRegistry.register("ce")
class CrossEntropyLoss(Loss):
    def __init__(self, weight=None):
        self.criterion = nn.CrossEntropyLoss(weight=weight)

    def __call__(self, outputs, targets):
        outputs = outputs.squeeze()
        targets = targets.squeeze().long()
        return self.criterion(outputs, targets)


class MultiHeadLoss(nn.Module):
    def __init__(
        self, 
        head_structure: Dict[str, int], 
        loss_type: str = "standard",  # "standard" or "focal"
        alpha: float = 0.25, 
        gamma: float = 2.0,
        head_weights: Dict[str, float] = None
    ):
        """
        Initialize Multi-Head Loss with support for both standard and focal loss.
        
        Args:
            head_structure (dict): Dictionary mapping head names to number of classes
            loss_type (str): Type of loss to use - "standard" (BCE/CE) or "focal"
            alpha (float): Weighting factor for focal loss
            gamma (float): Focusing parameter for focal loss
            head_weights (dict): Optional weights for each head in the final loss computation
        """
        super().__init__()
        self.head_structure = head_structure
        self.loss_type = loss_type
        self.alpha = alpha
        self.gamma = gamma
        self.head_weights = head_weights or {head: 1.0 for head in head_structure.keys()}
        
        # Initialize standard loss functions
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
        
    def binary_focal_loss(self, pred, target):
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
    
    def multi_class_focal_loss(self, pred, target):
        """Compute Multi-class Focal Loss"""
        ce_loss = self.ce_loss(pred, target)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()
    
    def binary_standard_loss(self, pred, target):
        """Compute standard binary cross entropy loss"""
        pred = pred.squeeze()
        target = target.float()
        return self.bce_loss(pred, target).mean()
    
    def multi_class_standard_loss(self, pred, target):
        """Compute standard cross entropy loss"""
        return self.ce_loss(pred, target).mean()
    
    def compute_head_loss(self, pred, target, num_classes):
        """
        Compute loss for a single head based on number of classes and loss type.
        """
        if num_classes == 1:  # Binary classification
            if self.loss_type == "focal":
                return self.binary_focal_loss(pred, target)
            else:
                return self.binary_standard_loss(pred, target)
        else:  # Multi-class classification
            if self.loss_type == "focal":
                return self.multi_class_focal_loss(pred, target)
            else:
                return self.multi_class_standard_loss(pred, target)
    
    def forward(
        self, 
        outputs: Dict[str, torch.Tensor], 
        targets: Dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
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
        
        for head_name, num_classes in self.head_structure.items():
            head_loss = self.compute_head_loss(
                outputs[head_name], 
                targets[head_name],
                num_classes
            )
            
            # Apply head-specific weight
            head_weight = self.head_weights[head_name]
            losses[head_name] = head_loss
            total_loss += head_weight * head_loss
        
        return total_loss, losses