import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class CombinedLoss(nn.Module):
    """Combined CrossEntropy + Dice loss with class weighting.

    CE handles pixel-level classification with class imbalance weights.
    Dice handles region-level overlap, crucial for small class 1 features.
    """

    def __init__(self, class_weights: list, ce_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

        weight_tensor = torch.tensor(class_weights, dtype=torch.float32)
        self.ce_loss = nn.CrossEntropyLoss(weight=weight_tensor)
        self.dice_loss = smp.losses.DiceLoss(mode="multiclass", from_logits=True)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> dict:
        """
        Args:
            pred: (B, C, H, W) logits
            target: (B, H, W) class indices
        Returns:
            dict with total loss and components
        """
        ce = self.ce_loss(pred, target)
        dice = self.dice_loss(pred, target)
        total = self.ce_weight * ce + self.dice_weight * dice

        return {
            "loss": total,
            "ce_loss": ce.detach(),
            "dice_loss": dice.detach(),
        }

    def to(self, device):
        """Move loss function to device (needed for CE class weights)."""
        super().to(device)
        self.ce_loss = self.ce_loss.to(device)
        return self
