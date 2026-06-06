import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from matplotlib.colors import ListedColormap


# Consistent color scheme matching existing dataset visualizations
CLASS_COLORS = np.array([[0, 0, 0], [0, 255, 255], [255, 165, 0]], dtype=np.uint8)
CLASS_COLORS_FLOAT = CLASS_COLORS / 255.0
CLASS_NAMES = ["Background", "Slepice / odrezane veje", "Mehanske poskodbe / odlusceno"]


def create_mask_overlay(image: np.ndarray, mask: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Create semi-transparent mask overlay on image.

    Args:
        image: (H, W, 3) RGB image, uint8
        mask: (H, W) class indices
        alpha: overlay opacity
    """
    overlay = image.copy().astype(np.float32)
    for cls_id in range(1, len(CLASS_COLORS)):
        cls_mask = mask == cls_id
        if cls_mask.any():
            color = CLASS_COLORS[cls_id].astype(np.float32)
            overlay[cls_mask] = overlay[cls_mask] * (1 - alpha) + color * alpha
    return overlay.astype(np.uint8)


def create_difference_map(gt_mask: np.ndarray, pred_mask: np.ndarray) -> np.ndarray:
    """Create difference map: green=correct, red=false positive, blue=false negative.

    Returns (H, W, 3) RGB image.
    """
    h, w = gt_mask.shape
    diff = np.zeros((h, w, 3), dtype=np.uint8)

    correct = gt_mask == pred_mask
    diff[correct] = [0, 128, 0]  # Green for correct

    # False positives: predicted a class but GT is different
    for cls_id in range(1, len(CLASS_COLORS)):
        fp = (pred_mask == cls_id) & (gt_mask != cls_id)
        diff[fp] = [255, 0, 0]  # Red

    # False negatives: GT has a class but prediction missed it
    for cls_id in range(1, len(CLASS_COLORS)):
        fn = (gt_mask == cls_id) & (pred_mask != cls_id)
        diff[fn] = [0, 0, 255]  # Blue

    return diff


def plot_prediction_overlay(
    image: np.ndarray,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    stem: str,
    save_path: str,
    metrics: dict = None,
):
    """Generate side-by-side GT vs prediction overlay with difference map."""
    gt_overlay = create_mask_overlay(image, gt_mask)
    pred_overlay = create_mask_overlay(image, pred_mask)
    diff_map = create_difference_map(gt_mask, pred_mask)

    fig, axes = plt.subplots(1, 3, figsize=(24, 8))

    axes[0].imshow(gt_overlay)
    axes[0].set_title("Ground Truth", fontsize=14)
    axes[0].axis("off")

    axes[1].imshow(pred_overlay)
    title = "Prediction"
    if metrics:
        miou = metrics.get("mean_iou", 0)
        title += f" (mIoU: {miou:.3f})"
    axes[1].set_title(title, fontsize=14)
    axes[1].axis("off")

    axes[2].imshow(diff_map)
    axes[2].set_title("Difference (R=FP, B=FN, G=correct)", fontsize=14)
    axes[2].axis("off")

    # Legend
    patches = [mpatches.Patch(color=c, label=n) for c, n in zip(CLASS_COLORS_FLOAT[1:], CLASS_NAMES[1:])]
    fig.legend(handles=patches, loc="lower center", ncol=2, fontsize=12, frameon=True)

    plt.suptitle(stem, fontsize=16, fontweight="bold")
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_training_curves(history: dict, save_path: str):
    """Plot training curves: loss, per-class IoU, mean metrics, LR."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Train/Val Loss
    ax = axes[0, 0]
    if "train_loss" in history and history["train_loss"]:
        ax.plot(history["epoch"], history["train_loss"], label="Train Loss", linewidth=2)
    if "val_loss" in history and history["val_loss"]:
        val_epochs = history.get("val_epoch", history["epoch"])
        ax.plot(val_epochs, history["val_loss"], label="Val Loss", linewidth=2, marker="o", markersize=4)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Per-class IoU
    ax = axes[0, 1]
    val_epochs = history.get("val_epoch", [])
    for i, name in enumerate(CLASS_NAMES):
        key = f"val_iou_class{i}"
        if key in history:
            ax.plot(val_epochs, history[key], label=name, color=CLASS_COLORS_FLOAT[i] if i > 0 else "gray",
                    linewidth=2, marker="o", markersize=4)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("IoU")
    ax.set_title("Per-Class IoU (Validation)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    # 3. Mean IoU + Dice
    ax = axes[1, 0]
    if "val_mean_iou" in history:
        ax.plot(val_epochs, history["val_mean_iou"], label="Mean IoU", linewidth=2, marker="o", markersize=4)
    if "val_mean_dice" in history:
        ax.plot(val_epochs, history["val_mean_dice"], label="Mean Dice", linewidth=2, marker="s", markersize=4)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.set_title("Mean IoU & Dice (Validation)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    # 4. Learning Rate
    ax = axes[1, 1]
    if "lr" in history:
        ax.plot(history["epoch"], history["lr"], linewidth=2, color="tab:red")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule")
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    plt.suptitle("Training Overview", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix(cm: np.ndarray, save_path: str, normalize: bool = True):
    """Plot confusion matrix (optionally normalized)."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Raw counts
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    axes[0].set_title("Confusion Matrix (Raw Counts)")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")

    # Normalized (row-wise)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.where(row_sums > 0, cm.astype(float) / row_sums, 0)
    sns.heatmap(cm_norm, annot=True, fmt=".3f", cmap="Blues", ax=axes[1],
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, vmin=0, vmax=1)
    axes[1].set_title("Confusion Matrix (Row-Normalized)")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_class1_analysis(per_image_results: list, save_path: str):
    """Scatter plot: class 1 GT percentage vs class 1 IoU per image."""
    gt_pcts = []
    ious = []
    stems = []

    for r in per_image_results:
        gt_pct = (r["confusion_matrix"][1, :].sum() /
                  r["confusion_matrix"].sum() * 100) if r["confusion_matrix"].sum() > 0 else 0
        gt_pcts.append(gt_pct)
        ious.append(r["per_class_iou"][1])
        stems.append(r.get("stem", ""))

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Scatter
    ax = axes[0]
    sc = ax.scatter(gt_pcts, ious, c=ious, cmap="RdYlGn", vmin=0, vmax=1, s=60, edgecolors="black", linewidth=0.5)
    ax.set_xlabel("Class 1 GT Percentage (%)", fontsize=12)
    ax.set_ylabel("Class 1 IoU", fontsize=12)
    ax.set_title("Class 1: GT Coverage vs IoU", fontsize=14)
    ax.set_ylim(-0.05, 1.05)
    plt.colorbar(sc, ax=ax, label="IoU")
    ax.grid(True, alpha=0.3)

    # Histogram of class 1 IoU
    ax = axes[1]
    valid_ious = [iou for iou, pct in zip(ious, gt_pcts) if pct > 0]
    if valid_ious:
        ax.hist(valid_ious, bins=20, range=(0, 1), color="teal", edgecolor="black", alpha=0.7)
    ax.set_xlabel("Class 1 IoU", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(f"Class 1 IoU Distribution (n={len(valid_ious)} images with class 1)", fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_best_worst_cases(
    images: list,
    gt_masks: list,
    pred_masks: list,
    mean_ious: list,
    stems: list,
    save_path: str,
    n: int = 4,
):
    """Grid showing the n best and n worst predictions by mean IoU."""
    sorted_idx = np.argsort(mean_ious)
    worst_idx = sorted_idx[:n]
    best_idx = sorted_idx[-n:][::-1]

    fig, axes = plt.subplots(2 * n, 3, figsize=(24, 8 * n))

    for row, idx in enumerate(list(best_idx) + list(worst_idx)):
        gt_ov = create_mask_overlay(images[idx], gt_masks[idx])
        pred_ov = create_mask_overlay(images[idx], pred_masks[idx])
        diff = create_difference_map(gt_masks[idx], pred_masks[idx])

        label = "BEST" if row < n else "WORST"
        axes[row, 0].imshow(gt_ov)
        axes[row, 0].set_title(f"{label} #{row % n + 1}: {stems[idx]} - GT", fontsize=11)
        axes[row, 0].axis("off")

        axes[row, 1].imshow(pred_ov)
        axes[row, 1].set_title(f"Pred (mIoU: {mean_ious[idx]:.3f})", fontsize=11)
        axes[row, 1].axis("off")

        axes[row, 2].imshow(diff)
        axes[row, 2].set_title("Diff (R=FP, B=FN)", fontsize=11)
        axes[row, 2].axis("off")

    plt.suptitle("Best & Worst Predictions by Mean IoU", fontsize=16, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_performance_vs_distribution(per_image_results: list, save_path: str):
    """Per-image class distribution alongside IoU scores."""
    n = len(per_image_results)
    if n == 0:
        return

    stems = [r["stem"] for r in per_image_results]
    ious = np.array([r["per_class_iou"] for r in per_image_results])

    # Compute class pixel fractions from confusion matrix
    class_fracs = []
    for r in per_image_results:
        cm = r["confusion_matrix"]
        total = cm.sum()
        fracs = cm.sum(axis=1) / total if total > 0 else np.zeros(len(CLASS_NAMES))
        class_fracs.append(fracs)
    class_fracs = np.array(class_fracs)

    # Sort by mean IoU
    sort_idx = np.argsort(ious.mean(axis=1))
    stems = [stems[i] for i in sort_idx]
    ious = ious[sort_idx]
    class_fracs = class_fracs[sort_idx]

    fig, axes = plt.subplots(2, 1, figsize=(max(16, n * 0.5), 12), sharex=True)

    # Stacked bar for class distribution
    ax = axes[0]
    x = np.arange(n)
    bottom = np.zeros(n)
    for c in range(len(CLASS_NAMES)):
        color = CLASS_COLORS_FLOAT[c] if c > 0 else [0.7, 0.7, 0.7]
        ax.bar(x, class_fracs[:, c], bottom=bottom, label=CLASS_NAMES[c], color=color, width=0.8)
        bottom += class_fracs[:, c]
    ax.set_ylabel("Class Fraction")
    ax.set_title("Class Distribution per Image (sorted by mIoU)")
    ax.legend(fontsize=9)

    # Grouped bar for IoU
    ax = axes[1]
    width = 0.25
    for c in range(len(CLASS_NAMES)):
        color = CLASS_COLORS_FLOAT[c] if c > 0 else [0.7, 0.7, 0.7]
        ax.bar(x + c * width - width, ious[:, c], width=width, label=CLASS_NAMES[c], color=color, alpha=0.8)
    ax.set_ylabel("IoU")
    ax.set_title("Per-Class IoU per Image")
    ax.set_xticks(x)
    ax.set_xticklabels(stems, rotation=90, fontsize=7)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
