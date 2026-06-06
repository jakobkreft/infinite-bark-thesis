"""
Bark Segmentation Training Script

Usage:
    conda activate bark-seg
    python train.py --config configs/default.yaml
    python train.py --config configs/default.yaml --override training.epochs=50 experiment.name=quick_test
"""

import argparse
import json
import os
import time
import csv

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image

from src.utils import load_config, set_seed, setup_logging, get_output_dir, save_checkpoint
from src.dataset import BarkSegmentationDataset, get_train_transforms, get_val_transforms
from src.model import create_model, count_parameters
from src.losses import CombinedLoss
from src.metrics import SegmentationMetrics
from src.sliding_window import sliding_window_inference
from src.visualization import plot_training_curves, plot_prediction_overlay, create_mask_overlay


def parse_args():
    parser = argparse.ArgumentParser(description="Train bark segmentation model")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--override", nargs="*", help="Override config values: key=value")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    return parser.parse_args()


def apply_overrides(config, overrides):
    """Apply command-line overrides like training.epochs=50."""
    if not overrides:
        return config
    for override in overrides:
        key, value = override.split("=", 1)
        parts = key.split(".")
        d = config
        for part in parts[:-1]:
            d = d[part]
        # Try to cast to original type
        orig = d.get(parts[-1])
        if isinstance(orig, bool):
            value = value.lower() in ("true", "1", "yes")
        elif isinstance(orig, int):
            value = int(value)
        elif isinstance(orig, float):
            value = float(value)
        d[parts[-1]] = value
    return config


def validate_epoch(model, val_dataset, config, device, logger):
    """Run sliding window validation on full-resolution images."""
    model.eval()
    num_classes = config["dataset"]["num_classes"]
    crop_size = config["training"]["crop_size"]
    stride = config["validation"]["stride"]

    metrics = SegmentationMetrics(num_classes)

    for idx in range(len(val_dataset)):
        image, mask, stem, orig_size = val_dataset[idx]

        pred_mask, _ = sliding_window_inference(
            model, image, crop_size=crop_size, stride=stride,
            num_classes=num_classes, device=device, batch_size=8,
        )

        gt_mask = mask.numpy()
        # Ensure same size (should match since we don't resize)
        h, w = gt_mask.shape
        pred_mask = pred_mask[:h, :w]

        metrics.update(pred_mask, gt_mask)

    return metrics.compute()


def train(config):
    # Setup
    set_seed(config["experiment"]["seed"])
    output_dir = get_output_dir(config)
    logger = setup_logging(output_dir)
    device = torch.device(config["experiment"]["device"])

    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Device: {device}")

    # Save config
    with open(os.path.join(output_dir, "config.yaml"), "w") as f:
        import yaml
        yaml.dump(config, f, default_flow_style=False)

    # Wandb
    if config["experiment"]["wandb_enabled"]:
        import wandb
        wandb.init(project=config["experiment"]["wandb_project"],
                   name=config["experiment"]["name"], config=config)

    # Dataset
    ds_root = config["dataset"]["root"]
    crop_size = config["training"]["crop_size"]

    train_transform = get_train_transforms(crop_size)
    val_transform = get_val_transforms()

    train_dataset = BarkSegmentationDataset(
        image_dir=os.path.join(ds_root, "train", "images"),
        mask_dir=os.path.join(ds_root, "train", "masks"),
        transform=train_transform,
        mode="train",
        crop_size=crop_size,
        crops_per_image=config["training"]["crops_per_image"],
        class_aware_ratio=config["training"]["class_aware_ratio"],
    )
    val_dataset = BarkSegmentationDataset(
        image_dir=os.path.join(ds_root, "val", "images"),
        mask_dir=os.path.join(ds_root, "val", "masks"),
        transform=val_transform,
        mode="val",
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"]["num_workers"],
        pin_memory=True,
        persistent_workers=True if config["training"]["num_workers"] > 0 else False,
        drop_last=True,
    )

    logger.info(f"Train: {len(train_dataset)} crops ({len(train_dataset) // config['training']['crops_per_image']} images x {config['training']['crops_per_image']} crops)")
    logger.info(f"Val: {len(val_dataset)} images")

    # Model
    model = create_model(config)
    params = count_parameters(model)
    logger.info(f"Model: {config['model']['architecture']}-{config['model']['encoder']}")
    logger.info(f"Parameters: {params['total']:,} total, {params['trainable']:,} trainable")
    model = model.to(device)

    # Loss
    criterion = CombinedLoss(
        class_weights=config["loss"]["class_weights"],
        ce_weight=config["loss"]["ce_weight"],
        dice_weight=config["loss"]["dice_weight"],
    ).to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["optimizer"]["lr"],
        weight_decay=config["optimizer"]["weight_decay"],
    )

    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=config["scheduler"]["T_0"],
        T_mult=config["scheduler"]["T_mult"],
        eta_min=config["scheduler"]["eta_min"],
    )

    # Mixed precision
    use_amp = config["experiment"]["mixed_precision"]
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    # Training history
    history = {
        "epoch": [], "train_loss": [], "lr": [],
        "val_epoch": [], "val_mean_iou": [], "val_mean_dice": [],
    }
    for c in range(config["dataset"]["num_classes"]):
        history[f"val_iou_class{c}"] = []

    # Early stopping
    best_mean_iou = 0.0
    best_class1_iou = 0.0
    patience_counter = 0
    patience = config["early_stopping"]["patience"]

    # Resume
    start_epoch = 0
    if hasattr(args, 'resume') and args.resume:
        from src.utils import load_checkpoint
        start_epoch, prev_metrics = load_checkpoint(args.resume, model, optimizer, scheduler, device)
        logger.info(f"Resumed from epoch {start_epoch}")

    # Training loop
    num_epochs = config["training"]["epochs"]
    eval_every = config["validation"]["eval_every"]
    logger.info(f"Starting training for {num_epochs} epochs...")

    for epoch in range(start_epoch, num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_ce = 0.0
        epoch_dice = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
        for images, masks in pbar:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            if use_amp:
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    outputs = model(images)
                    loss_dict = criterion(outputs, masks)
                scaler.scale(loss_dict["loss"]).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss_dict = criterion(outputs, masks)
                loss_dict["loss"].backward()
                optimizer.step()

            epoch_loss += loss_dict["loss"].item()
            epoch_ce += loss_dict["ce_loss"].item()
            epoch_dice += loss_dict["dice_loss"].item()
            num_batches += 1

            pbar.set_postfix(loss=f"{loss_dict['loss'].item():.4f}")

        scheduler.step()

        avg_loss = epoch_loss / num_batches
        avg_ce = epoch_ce / num_batches
        avg_dice = epoch_dice / num_batches
        current_lr = optimizer.param_groups[0]["lr"]

        history["epoch"].append(epoch + 1)
        history["train_loss"].append(avg_loss)
        history["lr"].append(current_lr)

        logger.info(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Loss: {avg_loss:.4f} (CE: {avg_ce:.4f}, Dice: {avg_dice:.4f}) | "
            f"LR: {current_lr:.2e}"
        )

        # Validation
        if (epoch + 1) % eval_every == 0 or epoch == num_epochs - 1:
            logger.info("Running validation (sliding window)...")
            val_start = time.time()
            val_metrics = validate_epoch(model, val_dataset, config, device, logger)
            val_time = time.time() - val_start

            mean_iou = val_metrics["mean_iou"]
            mean_dice = val_metrics["mean_dice"]
            class_ious = val_metrics["per_class_iou"]

            history["val_epoch"].append(epoch + 1)
            history["val_mean_iou"].append(mean_iou)
            history["val_mean_dice"].append(mean_dice)
            for c in range(config["dataset"]["num_classes"]):
                history[f"val_iou_class{c}"].append(class_ious[c])

            logger.info(
                f"  Val mIoU: {mean_iou:.4f} | mDice: {mean_dice:.4f} | "
                f"IoU: [{class_ious[0]:.3f}, {class_ious[1]:.3f}, {class_ious[2]:.3f}] | "
                f"Time: {val_time:.1f}s"
            )

            # Wandb logging
            if config["experiment"]["wandb_enabled"]:
                log_dict = {
                    "train/loss": avg_loss, "train/ce_loss": avg_ce, "train/dice_loss": avg_dice,
                    "train/lr": current_lr,
                    "val/mean_iou": mean_iou, "val/mean_dice": mean_dice,
                    "val/pixel_accuracy": val_metrics["pixel_accuracy"],
                }
                for c, name in enumerate(config["dataset"]["class_names"]):
                    log_dict[f"val/iou_{name}"] = class_ious[c]
                wandb.log(log_dict, step=epoch + 1)

            # Save best model (mean IoU)
            if mean_iou > best_mean_iou:
                best_mean_iou = mean_iou
                patience_counter = 0
                save_checkpoint(
                    model, optimizer, scheduler, epoch + 1,
                    val_metrics, os.path.join(output_dir, "checkpoints", "best_mean_iou.pth"),
                )
                logger.info(f"  -> New best mean IoU: {best_mean_iou:.4f}")
            else:
                patience_counter += eval_every

            # Save best model (class 1 IoU)
            if class_ious[1] > best_class1_iou:
                best_class1_iou = class_ious[1]
                save_checkpoint(
                    model, optimizer, scheduler, epoch + 1,
                    val_metrics, os.path.join(output_dir, "checkpoints", "best_class1_iou.pth"),
                )
                logger.info(f"  -> New best class 1 IoU: {best_class1_iou:.4f}")

            # Early stopping
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1} (patience={patience})")
                break

        # Save last checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch + 1,
                {}, os.path.join(output_dir, "checkpoints", "last.pth"),
            )

    # Final save
    save_checkpoint(
        model, optimizer, scheduler, epoch + 1,
        {}, os.path.join(output_dir, "checkpoints", "last.pth"),
    )

    # Save training history
    with open(os.path.join(output_dir, "results", "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    # Plot training curves
    plot_training_curves(history, os.path.join(output_dir, "plots", "training_curves.png"))
    logger.info(f"Training complete. Best mIoU: {best_mean_iou:.4f}, Best class1 IoU: {best_class1_iou:.4f}")

    if config["experiment"]["wandb_enabled"]:
        wandb.finish()


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)
    config = apply_overrides(config, args.override)
    train(config)
