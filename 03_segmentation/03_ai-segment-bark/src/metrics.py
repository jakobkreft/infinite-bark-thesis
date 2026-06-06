import numpy as np
import torch


class SegmentationMetrics:
    """Accumulates confusion matrix and computes segmentation metrics.

    Supports incremental updates for sliding-window evaluation.
    """

    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    def reset(self):
        self.confusion_matrix.fill(0)

    def update(self, pred: np.ndarray, target: np.ndarray):
        """Update confusion matrix with a prediction-target pair.

        Args:
            pred: (H, W) predicted class indices
            target: (H, W) ground truth class indices
        """
        assert pred.shape == target.shape, f"Shape mismatch: {pred.shape} vs {target.shape}"
        valid = (target >= 0) & (target < self.num_classes)
        label = self.num_classes * target[valid].astype(np.int64) + pred[valid].astype(np.int64)
        count = np.bincount(label, minlength=self.num_classes ** 2)
        self.confusion_matrix += count.reshape(self.num_classes, self.num_classes)

    def compute(self) -> dict:
        """Compute all metrics from accumulated confusion matrix."""
        cm = self.confusion_matrix.astype(np.float64)

        # Per-class IoU
        intersection = np.diag(cm)
        union = cm.sum(axis=1) + cm.sum(axis=0) - intersection
        iou = np.where(union > 0, intersection / union, 0.0)

        # Per-class Dice (F1)
        dice = np.where(union > 0, 2 * intersection / (cm.sum(axis=1) + cm.sum(axis=0)), 0.0)

        # Per-class precision and recall
        pred_totals = cm.sum(axis=0)
        true_totals = cm.sum(axis=1)
        precision = np.where(pred_totals > 0, intersection / pred_totals, 0.0)
        recall = np.where(true_totals > 0, intersection / true_totals, 0.0)

        # Overall pixel accuracy
        total = cm.sum()
        pixel_acc = intersection.sum() / total if total > 0 else 0.0

        # Frequency-weighted IoU
        freq = true_totals / total if total > 0 else np.zeros(self.num_classes)
        fw_iou = (freq * iou).sum()

        # Mean IoU (over classes that exist in ground truth)
        valid_classes = true_totals > 0
        mean_iou = iou[valid_classes].mean() if valid_classes.any() else 0.0
        mean_dice = dice[valid_classes].mean() if valid_classes.any() else 0.0

        return {
            "per_class_iou": iou,
            "per_class_dice": dice,
            "per_class_precision": precision,
            "per_class_recall": recall,
            "mean_iou": mean_iou,
            "mean_dice": mean_dice,
            "pixel_accuracy": pixel_acc,
            "fw_iou": fw_iou,
            "confusion_matrix": self.confusion_matrix.copy(),
        }


def compute_per_image_metrics(pred: np.ndarray, target: np.ndarray, num_classes: int) -> dict:
    """Compute metrics for a single image."""
    metrics = SegmentationMetrics(num_classes)
    metrics.update(pred, target)
    result = metrics.compute()

    # Add binary flags for class presence
    for c in range(num_classes):
        result[f"has_class{c}_gt"] = bool((target == c).any())
        result[f"has_class{c}_pred"] = bool((pred == c).any())

    return result
