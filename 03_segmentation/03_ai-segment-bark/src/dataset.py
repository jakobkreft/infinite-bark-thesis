import os
import glob
import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(crop_size: int) -> A.Compose:
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1, scale_limit=0.2, rotate_limit=30,
            border_mode=cv2.BORDER_REFLECT, p=0.5,
        ),
        A.ElasticTransform(alpha=120, sigma=6.0, p=0.2),
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.5),
        A.GaussianBlur(blur_limit=(3, 7), p=0.2),
        A.GaussNoise(std_range=(0.01, 0.05), p=0.2),
        A.CLAHE(clip_limit=4.0, p=0.3),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def get_val_transforms() -> A.Compose:
    return A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


class BarkSegmentationDataset(Dataset):
    """Dataset for bark segmentation with class-aware random cropping.

    Training mode: returns random 512x512 crops from full-res images.
    Val/Test mode: returns full images for sliding window inference.
    """

    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        transform: A.Compose,
        mode: str = "train",
        crop_size: int = 512,
        crops_per_image: int = 8,
        class_aware_ratio: float = 0.5,
    ):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.mode = mode
        self.crop_size = crop_size
        self.crops_per_image = crops_per_image
        self.class_aware_ratio = class_aware_ratio

        # Discover image-mask pairs from filesystem
        self.image_paths = sorted(
            glob.glob(os.path.join(image_dir, "*.jpg"))
            + glob.glob(os.path.join(image_dir, "*.JPG"))
            + glob.glob(os.path.join(image_dir, "*.png"))
        )
        self.stems = [os.path.splitext(os.path.basename(p))[0] for p in self.image_paths]
        self.mask_paths = [os.path.join(mask_dir, f"{stem}.png") for stem in self.stems]

        # Verify all masks exist
        valid = []
        for i, mp in enumerate(self.mask_paths):
            if os.path.exists(mp):
                valid.append(i)
            else:
                print(f"Warning: mask not found for {self.stems[i]}, skipping")
        self.image_paths = [self.image_paths[i] for i in valid]
        self.mask_paths = [self.mask_paths[i] for i in valid]
        self.stems = [self.stems[i] for i in valid]

        # Pre-compute class 1 locations for class-aware sampling
        self.class1_coords = {}
        if mode == "train":
            self._precompute_class1_locations()

    def _precompute_class1_locations(self):
        """Pre-compute coarse grid of class 1 pixel locations per image."""
        print("Pre-computing class 1 locations for class-aware sampling...")
        for idx, mask_path in enumerate(self.mask_paths):
            mask = np.array(Image.open(mask_path))
            ys, xs = np.where(mask == 1)
            if len(ys) > 0:
                # Subsample to max 500 coordinates for efficiency
                if len(ys) > 500:
                    indices = np.random.choice(len(ys), 500, replace=False)
                    ys, xs = ys[indices], xs[indices]
                self.class1_coords[idx] = np.stack([ys, xs], axis=1)
        n_with = len(self.class1_coords)
        print(f"  {n_with}/{len(self.mask_paths)} images contain class 1 pixels")

    def __len__(self):
        if self.mode == "train":
            return len(self.image_paths) * self.crops_per_image
        return len(self.image_paths)

    def _random_crop(self, image, mask, class_aware=False, img_idx=0):
        """Extract a random crop, optionally centered on a class 1 pixel."""
        h, w = image.shape[:2]
        cs = self.crop_size

        if class_aware and img_idx in self.class1_coords:
            coords = self.class1_coords[img_idx]
            coord = coords[np.random.randint(len(coords))]
            cy, cx = coord

            # Add random jitter (+/- crop_size/4)
            jitter = cs // 4
            cy += np.random.randint(-jitter, jitter + 1)
            cx += np.random.randint(-jitter, jitter + 1)

            # Compute crop bounds, clamp to image
            y1 = max(0, min(cy - cs // 2, h - cs))
            x1 = max(0, min(cx - cs // 2, w - cs))
        else:
            y1 = np.random.randint(0, max(1, h - cs))
            x1 = np.random.randint(0, max(1, w - cs))

        y2 = y1 + cs
        x2 = x1 + cs

        # Handle images smaller than crop size (pad with reflect)
        if h < cs or w < cs:
            pad_h = max(0, cs - h)
            pad_w = max(0, cs - w)
            image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
            mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode="reflect")
            y1, x1 = 0, 0
            y2, x2 = cs, cs

        return image[y1:y2, x1:x2], mask[y1:y2, x1:x2]

    def __getitem__(self, idx):
        if self.mode == "train":
            img_idx = idx // self.crops_per_image

            image = np.array(Image.open(self.image_paths[img_idx]).convert("RGB"))
            mask = np.array(Image.open(self.mask_paths[img_idx]))

            # Decide class-aware vs random crop
            use_class_aware = random.random() < self.class_aware_ratio
            image, mask = self._random_crop(image, mask, class_aware=use_class_aware, img_idx=img_idx)

            transformed = self.transform(image=image, mask=mask)
            return transformed["image"], transformed["mask"].long()
        else:
            # Val/Test: return full image + mask + metadata
            image = np.array(Image.open(self.image_paths[idx]).convert("RGB"))
            mask = np.array(Image.open(self.mask_paths[idx]))

            transformed = self.transform(image=image, mask=mask)
            return (
                transformed["image"],
                transformed["mask"].long(),
                self.stems[idx],
                torch.tensor(image.shape[:2]),  # original H, W
            )


class InferenceDataset(Dataset):
    """Dataset for inference on unlabeled images."""

    def __init__(self, image_dir: str, transform: A.Compose):
        self.image_dir = image_dir
        self.transform = transform
        self.image_paths = sorted(
            glob.glob(os.path.join(image_dir, "*.jpg"))
            + glob.glob(os.path.join(image_dir, "*.JPG"))
            + glob.glob(os.path.join(image_dir, "*.png"))
        )
        self.stems = [os.path.splitext(os.path.basename(p))[0] for p in self.image_paths]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = np.array(Image.open(self.image_paths[idx]).convert("RGB"))
        transformed = self.transform(image=image)
        return (
            transformed["image"],
            self.stems[idx],
            torch.tensor(image.shape[:2]),
        )


if __name__ == "__main__":
    """Visualize sample training batches to verify augmentation."""
    import matplotlib.pyplot as plt
    from src.utils import load_config

    config = load_config("configs/default.yaml")
    ds_root = config["dataset"]["root"]
    crop_size = config["training"]["crop_size"]
    class_colors = np.array(config["dataset"]["class_colors"], dtype=np.uint8)

    transform = get_train_transforms(crop_size)
    dataset = BarkSegmentationDataset(
        image_dir=os.path.join(ds_root, "train", "images"),
        mask_dir=os.path.join(ds_root, "train", "masks"),
        transform=transform,
        mode="train",
        crop_size=crop_size,
    )

    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    for i in range(8):
        img, mask = dataset[np.random.randint(len(dataset))]
        img_np = img.permute(1, 2, 0).numpy() * std + mean
        img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
        mask_np = mask.numpy()

        # Create colored mask overlay
        overlay = img_np.copy().astype(np.float32)
        for cls_id in range(1, 3):
            cls_mask = mask_np == cls_id
            if cls_mask.any():
                color = class_colors[cls_id].astype(np.float32)
                overlay[cls_mask] = overlay[cls_mask] * 0.6 + color * 0.4

        row = i // 2
        col = (i % 2) * 2
        axes[row, col].imshow(img_np)
        axes[row, col].set_title(f"Image {i}")
        axes[row, col].axis("off")
        axes[row, col + 1].imshow(overlay.astype(np.uint8))
        axes[row, col + 1].set_title(f"Overlay {i}")
        axes[row, col + 1].axis("off")

    plt.suptitle("Training Augmentation Samples", fontsize=16)
    plt.tight_layout()
    os.makedirs("outputs", exist_ok=True)
    plt.savefig("outputs/augmentation_samples.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved augmentation samples to outputs/augmentation_samples.png")
