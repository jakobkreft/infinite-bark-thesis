"""
Bark Segmentation Inference Script

Runs sliding window inference on unlabeled images and saves predicted masks + overlays.

Usage:
    conda activate bark-seg
    python inference.py --config configs/default.yaml
    python inference.py --config configs/default.yaml --input_dir path/to/images --output_dir path/to/output
"""

import argparse
import os

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.utils import load_config, get_output_dir
from src.model import create_model
from src.sliding_window import sliding_window_inference
from src.visualization import create_mask_overlay, CLASS_COLORS, CLASS_NAMES


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference on unlabeled images")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--input_dir", type=str, default=None,
                        help="Input image directory (default: dataset/inference/images)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: outputs/<exp>/inference)")
    parser.add_argument("--save_overlay", action="store_true", default=True)
    parser.add_argument("--save_probabilities", action="store_true", default=False)
    return parser.parse_args()


def discover_images(image_dir):
    """Find all images in directory."""
    extensions = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
    images = []
    for f in sorted(os.listdir(image_dir)):
        if os.path.splitext(f)[1].lower() in extensions:
            images.append(os.path.join(image_dir, f))
    return images


def main():
    args = parse_args()
    config = load_config(args.config)
    device = torch.device(config["experiment"]["device"])
    output_dir_base = get_output_dir(config)

    # Paths
    input_dir = args.input_dir or os.path.join(config["dataset"]["root"], "inference", "images")
    output_dir = args.output_dir or os.path.join(output_dir_base, "inference")
    checkpoint = args.checkpoint or os.path.join(output_dir_base, "checkpoints", "best_mean_iou.pth")

    os.makedirs(os.path.join(output_dir, "masks"), exist_ok=True)
    if args.save_overlay:
        os.makedirs(os.path.join(output_dir, "overlays"), exist_ok=True)
    if args.save_probabilities:
        os.makedirs(os.path.join(output_dir, "probabilities"), exist_ok=True)

    # Model
    print(f"Loading model from {checkpoint}...")
    model = create_model(config)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    print(f"Model loaded (epoch {ckpt.get('epoch', '?')})")

    # Transform
    transform = A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    # Discover images
    image_paths = discover_images(input_dir)
    print(f"Found {len(image_paths)} images in {input_dir}")

    num_classes = config["dataset"]["num_classes"]
    crop_size = config["training"]["crop_size"]
    stride = config["inference"]["stride"]
    use_gaussian = config["inference"]["use_gaussian_weights"]

    for img_path in tqdm(image_paths, desc="Inference"):
        stem = os.path.splitext(os.path.basename(img_path))[0]

        # Load and transform
        orig_image = np.array(Image.open(img_path).convert("RGB"))
        transformed = transform(image=orig_image)
        image_tensor = transformed["image"]

        # Sliding window inference
        pred_mask, prob_map = sliding_window_inference(
            model, image_tensor, crop_size=crop_size, stride=stride,
            num_classes=num_classes, device=device,
            use_gaussian=use_gaussian, batch_size=8,
        )

        # Crop to original size
        h, w = orig_image.shape[:2]
        pred_mask = pred_mask[:h, :w]

        # Save mask (class indices 0, 1, 2)
        mask_img = Image.fromarray(pred_mask)
        mask_img.save(os.path.join(output_dir, "masks", f"{stem}.png"))

        # Save overlay
        if args.save_overlay:
            overlay = create_mask_overlay(orig_image, pred_mask, alpha=0.4)
            overlay_img = Image.fromarray(overlay)
            overlay_img.save(os.path.join(output_dir, "overlays", f"{stem}_overlay.jpg"), quality=90)

        # Save probability maps
        if args.save_probabilities:
            prob_map = prob_map[:, :h, :w]
            np.save(os.path.join(output_dir, "probabilities", f"{stem}_probs.npy"), prob_map.astype(np.float16))

        # Free GPU memory periodically
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\nInference complete! Results saved to {output_dir}")
    print(f"  Masks: {output_dir}/masks/")
    if args.save_overlay:
        print(f"  Overlays: {output_dir}/overlays/")
    if args.save_probabilities:
        print(f"  Probabilities: {output_dir}/probabilities/")


if __name__ == "__main__":
    main()
