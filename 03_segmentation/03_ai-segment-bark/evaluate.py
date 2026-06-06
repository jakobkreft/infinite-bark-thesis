"""
Bark Segmentation Evaluation Script

Loads the best checkpoint, runs sliding window inference on the test set,
computes all metrics, and generates research-quality visualizations.

Usage:
    conda activate bark-seg
    python evaluate.py --config configs/default.yaml
    python evaluate.py --config configs/default.yaml --checkpoint outputs/exp/checkpoints/best_mean_iou.pth --split test
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from src.utils import load_config, set_seed, get_output_dir
from src.dataset import BarkSegmentationDataset, get_val_transforms
from src.model import create_model
from src.metrics import SegmentationMetrics, compute_per_image_metrics
from src.sliding_window import sliding_window_inference
from src.visualization import (
    plot_prediction_overlay,
    plot_confusion_matrix,
    plot_class1_analysis,
    plot_best_worst_cases,
    plot_performance_vs_distribution,
    create_mask_overlay,
    CLASS_NAMES,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate bark segmentation model")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint (default: best_mean_iou.pth)")
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    parser.add_argument("--save_predictions", action="store_true", default=True)
    return parser.parse_args()


def evaluate(config, checkpoint_path, split, save_predictions):
    set_seed(config["experiment"]["seed"])
    device = torch.device(config["experiment"]["device"])
    output_dir = get_output_dir(config)
    ds_root = config["dataset"]["root"]
    num_classes = config["dataset"]["num_classes"]
    crop_size = config["training"]["crop_size"]
    stride = config["validation"]["stride"] if split == "val" else config["inference"]["stride"]

    print(f"Evaluating on {split} set with stride {stride}...")
    print(f"Device: {device}")

    # Model
    model = create_model(config)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    ckpt_epoch = ckpt.get("epoch", "?")
    ckpt_metrics = ckpt.get("metrics", {})
    print(f"Loaded checkpoint from epoch {ckpt_epoch}")
    if ckpt_metrics:
        print(f"  Checkpoint val mIoU: {ckpt_metrics.get('mean_iou', 'N/A'):.4f}")

    # Dataset
    dataset = BarkSegmentationDataset(
        image_dir=os.path.join(ds_root, split, "images"),
        mask_dir=os.path.join(ds_root, split, "masks"),
        transform=get_val_transforms(),
        mode="val",
    )
    print(f"Dataset: {len(dataset)} images")

    # Evaluate
    global_metrics = SegmentationMetrics(num_classes)
    per_image_results = []

    # For best/worst visualization
    all_images = []
    all_gt_masks = []
    all_pred_masks = []
    all_stems = []
    all_mean_ious = []

    for idx in tqdm(range(len(dataset)), desc=f"Evaluating {split}"):
        image_tensor, mask_tensor, stem, orig_size = dataset[idx]

        # Sliding window inference
        pred_mask, prob_map = sliding_window_inference(
            model, image_tensor, crop_size=crop_size, stride=stride,
            num_classes=num_classes, device=device, batch_size=8,
        )

        gt_mask = mask_tensor.numpy()
        h, w = gt_mask.shape
        pred_mask = pred_mask[:h, :w]

        # Global metrics
        global_metrics.update(pred_mask, gt_mask)

        # Per-image metrics
        img_metrics = compute_per_image_metrics(pred_mask, gt_mask, num_classes)
        img_metrics["stem"] = stem
        per_image_results.append(img_metrics)

        # Load original image for visualizations
        img_path = dataset.image_paths[idx]
        orig_image = np.array(Image.open(img_path).convert("RGB"))

        all_images.append(orig_image)
        all_gt_masks.append(gt_mask)
        all_pred_masks.append(pred_mask)
        all_stems.append(stem)
        all_mean_ious.append(img_metrics["mean_iou"])

        # Save per-image prediction overlay
        if save_predictions:
            pred_dir = os.path.join(output_dir, "predictions", split)
            os.makedirs(pred_dir, exist_ok=True)
            plot_prediction_overlay(
                orig_image, gt_mask, pred_mask, stem,
                os.path.join(pred_dir, f"{stem}_overlay.png"),
                metrics=img_metrics,
            )

            # Save raw predicted mask
            mask_img = Image.fromarray(pred_mask)
            mask_img.save(os.path.join(pred_dir, f"{stem}_pred.png"))

    # Compute global metrics
    results = global_metrics.compute()

    # Print results table
    print("\n" + "=" * 70)
    print(f"  {split.upper()} SET RESULTS")
    print("=" * 70)
    print(f"  {'Class':<35} {'IoU':>8} {'Dice':>8} {'Prec':>8} {'Recall':>8}")
    print("-" * 70)
    for c in range(num_classes):
        print(
            f"  {CLASS_NAMES[c]:<35} "
            f"{results['per_class_iou'][c]:>8.4f} "
            f"{results['per_class_dice'][c]:>8.4f} "
            f"{results['per_class_precision'][c]:>8.4f} "
            f"{results['per_class_recall'][c]:>8.4f}"
        )
    print("-" * 70)
    print(f"  {'Mean':<35} {results['mean_iou']:>8.4f} {results['mean_dice']:>8.4f}")
    print(f"  {'Pixel Accuracy':<35} {results['pixel_accuracy']:>8.4f}")
    print(f"  {'Freq-weighted IoU':<35} {results['fw_iou']:>8.4f}")
    print("=" * 70)

    # Save results JSON
    results_json = {
        "split": split,
        "checkpoint": checkpoint_path,
        "epoch": ckpt_epoch,
        "num_images": len(dataset),
        "mean_iou": float(results["mean_iou"]),
        "mean_dice": float(results["mean_dice"]),
        "pixel_accuracy": float(results["pixel_accuracy"]),
        "fw_iou": float(results["fw_iou"]),
        "per_class": {},
    }
    for c in range(num_classes):
        results_json["per_class"][CLASS_NAMES[c]] = {
            "iou": float(results["per_class_iou"][c]),
            "dice": float(results["per_class_dice"][c]),
            "precision": float(results["per_class_precision"][c]),
            "recall": float(results["per_class_recall"][c]),
        }

    results_path = os.path.join(output_dir, "results", f"{split}_results.json")
    with open(results_path, "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Save per-image metrics CSV
    rows = []
    for r in per_image_results:
        row = {"stem": r["stem"]}
        for c in range(num_classes):
            row[f"iou_{c}"] = r["per_class_iou"][c]
            row[f"dice_{c}"] = r["per_class_dice"][c]
            row[f"precision_{c}"] = r["per_class_precision"][c]
            row[f"recall_{c}"] = r["per_class_recall"][c]
        row["mean_iou"] = r["mean_iou"]
        row["mean_dice"] = r["mean_dice"]
        row["has_class1_gt"] = r["has_class1_gt"]
        row["has_class1_pred"] = r["has_class1_pred"]
        rows.append(row)

    csv_path = os.path.join(output_dir, "results", f"{split}_per_image_metrics.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"Per-image metrics saved to {csv_path}")

    # Generate visualizations
    plots_dir = os.path.join(output_dir, "plots")

    print("\nGenerating visualizations...")

    # Confusion matrix
    plot_confusion_matrix(
        results["confusion_matrix"],
        os.path.join(plots_dir, f"{split}_confusion_matrix.png"),
    )

    # Class 1 analysis
    plot_class1_analysis(
        per_image_results,
        os.path.join(plots_dir, f"{split}_class1_analysis.png"),
    )

    # Performance vs distribution
    plot_performance_vs_distribution(
        per_image_results,
        os.path.join(plots_dir, f"{split}_performance_vs_distribution.png"),
    )

    # Best/worst cases
    if len(all_images) >= 8:
        plot_best_worst_cases(
            all_images, all_gt_masks, all_pred_masks, all_mean_ious, all_stems,
            os.path.join(plots_dir, f"{split}_best_worst_cases.png"),
            n=4,
        )
    elif len(all_images) >= 2:
        n = len(all_images) // 2
        plot_best_worst_cases(
            all_images, all_gt_masks, all_pred_masks, all_mean_ious, all_stems,
            os.path.join(plots_dir, f"{split}_best_worst_cases.png"),
            n=n,
        )

    print("Evaluation complete!")


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)
    output_dir = get_output_dir(config)

    checkpoint = args.checkpoint
    if checkpoint is None:
        checkpoint = os.path.join(output_dir, "checkpoints", "best_mean_iou.pth")

    if not os.path.exists(checkpoint):
        print(f"Error: Checkpoint not found: {checkpoint}")
        print("Train the model first with: python train.py")
        exit(1)

    evaluate(config, checkpoint, args.split, args.save_predictions)
