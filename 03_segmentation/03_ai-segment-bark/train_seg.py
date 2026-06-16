"""
Train a semantic segmentation network on the bark patches dataset.

Pipeline:
  1. 5-fold cross-validation on the train folder -> robust mean/std metrics.
  2. Train a final model on (almost) all train data, small val split for
     best-checkpoint selection.
  3. Evaluate the final model on the held-out test folder.
  4. Save thesis-ready plots: training curves, CV summary, confusion matrix,
     per-class IoU bars, qualitative prediction grid. Metrics also saved as JSON.

Dataset layout (same folder as this script):
  diffinfinite-bark_512-train/  bark_XXXX.jpg + bark_XXXX_mask.png
  diffinfinite-bark_512-test/   bark_XXXX.jpg + bark_XXXX_mask.png
Mask values: 0=bark, 1=knot, 2=defect.

Requirements:
  pip install torch torchvision segmentation-models-pytorch albumentations \
              opencv-python matplotlib scikit-learn tqdm
"""

import json
import random
from pathlib import Path

import albumentations as A
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

# ===========================================================================
# CONFIG
# ===========================================================================
SCRIPT_DIR = Path(__file__).resolve().parent

TRAIN_DIR = SCRIPT_DIR / "diffinfinite-bark_512-train"
TEST_DIR = SCRIPT_DIR / "diffinfinite-bark_512-test"
OUTPUT_DIR = SCRIPT_DIR / "segmentation_results"

CLASSES = {0: "bark", 1: "knot", 2: "defect"}
NUM_CLASSES = len(CLASSES)
CLASS_COLORS = {0: (60, 60, 60), 1: (0, 255, 255), 2: (255, 128, 0)}  # RGB for plots

ARCHITECTURE = "Unet"            # smp architecture: Unet / UnetPlusPlus / DeepLabV3Plus / FPN
ENCODER = "resnet34"             # pretrained encoder
ENCODER_WEIGHTS = "imagenet"

IMG_SIZE = 512
BATCH_SIZE = 8                   # lower to 4 if GPU memory is tight
NUM_WORKERS = 4
EPOCHS = 40
LR = 3e-4
WEIGHT_DECAY = 1e-4

N_FOLDS = 5
FINAL_VAL_FRACTION = 0.1         # val split for best-checkpoint in final training
SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda"
# ===========================================================================


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class BarkDataset(Dataset):
    def __init__(self, folder: Path, transform=None):
        self.images = sorted(p for p in folder.glob("bark_*.jpg"))
        self.images = [p for p in self.images
                       if (folder / f"{p.stem}_mask.png").exists()]
        self.folder = folder
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        img_path = self.images[i]
        mask_path = self.folder / f"{img_path.stem}_mask.png"

        img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if self.transform:
            out = self.transform(image=img, mask=mask)
            img, mask = out["image"], out["mask"]

        return img, mask.long()


def get_transforms(train: bool):
    norm = [A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()]
    if not train:
        return A.Compose([A.Resize(IMG_SIZE, IMG_SIZE)] + norm)
    return A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15,
                           border_mode=cv2.BORDER_REFLECT_101, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20,
                             val_shift_limit=10, p=0.3),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.GaussNoise(var_limit=(5.0, 30.0), p=0.2),
    ] + norm)


# ---------------------------------------------------------------------------
# Loss / metrics
# ---------------------------------------------------------------------------
def compute_class_weights(dataset: BarkDataset) -> torch.Tensor:
    """Inverse-log-frequency class weights from mask pixel counts."""
    counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    for img_path in tqdm(dataset.images, desc="Class weights", leave=False):
        mask = cv2.imread(str(dataset.folder / f"{img_path.stem}_mask.png"),
                          cv2.IMREAD_GRAYSCALE)
        for c in range(NUM_CLASSES):
            counts[c] += int((mask == c).sum())
    freq = counts / max(counts.sum(), 1)
    weights = 1.0 / np.log(1.02 + freq)
    weights = weights / weights.sum() * NUM_CLASSES
    print(f"  pixel freq: {dict(zip(CLASSES.values(), np.round(freq, 5)))}")
    print(f"  CE weights: {dict(zip(CLASSES.values(), np.round(weights, 3)))}")
    return torch.tensor(weights, dtype=torch.float32)


class ComboLoss(nn.Module):
    """Weighted CrossEntropy + Dice loss."""
    def __init__(self, class_weights):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=class_weights)
        self.dice = smp.losses.DiceLoss(mode="multiclass")

    def forward(self, logits, target):
        return self.ce(logits, target) + self.dice(logits, target)


class ConfusionAccumulator:
    """Accumulates a pixel-level confusion matrix; derives IoU/Dice/accuracy."""
    def __init__(self):
        self.cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        p = pred.flatten().cpu().numpy()
        t = target.flatten().cpu().numpy()
        idx = NUM_CLASSES * t + p
        self.cm += np.bincount(idx, minlength=NUM_CLASSES ** 2).reshape(NUM_CLASSES, NUM_CLASSES)

    def metrics(self) -> dict:
        cm = self.cm.astype(np.float64)
        tp = np.diag(cm)
        fp = cm.sum(0) - tp
        fn = cm.sum(1) - tp
        iou = tp / np.maximum(tp + fp + fn, 1)
        dice = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
        acc = tp.sum() / max(cm.sum(), 1)
        return {
            "pixel_accuracy": float(acc),
            "mIoU": float(iou.mean()),
            "mDice": float(dice.mean()),
            "per_class_IoU": {CLASSES[c]: float(iou[c]) for c in range(NUM_CLASSES)},
            "per_class_Dice": {CLASSES[c]: float(dice[c]) for c in range(NUM_CLASSES)},
            "confusion_matrix": self.cm.tolist(),
        }


# ---------------------------------------------------------------------------
# Train / eval loops
# ---------------------------------------------------------------------------
def build_model():
    return smp.create_model(ARCHITECTURE, encoder_name=ENCODER,
                            encoder_weights=ENCODER_WEIGHTS,
                            in_channels=3, classes=NUM_CLASSES).to(DEVICE)


def run_epoch(model, loader, criterion, optimizer=None, scaler=None, desc=""):
    train = optimizer is not None
    model.train() if train else model.eval()
    total_loss, n = 0.0, 0
    conf = ConfusionAccumulator()

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for imgs, masks in tqdm(loader, desc=desc, leave=False):
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)

            with torch.autocast(device_type=DEVICE.type, enabled=USE_AMP):
                logits = model(imgs)
                loss = criterion(logits, masks)

            if train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            total_loss += loss.item() * imgs.size(0)
            n += imgs.size(0)
            conf.update(logits.argmax(1), masks)

    m = conf.metrics()
    m["loss"] = total_loss / max(n, 1)
    return m


def train_model(train_ds, val_ds, class_weights, tag: str, ckpt_path: Path):
    """Train one model; returns history and best-val-mIoU metrics."""
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)

    model = build_model()
    criterion = ComboLoss(class_weights.to(DEVICE))
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = torch.amp.GradScaler(enabled=USE_AMP)

    history = {"train_loss": [], "val_loss": [], "train_mIoU": [], "val_mIoU": [],
               "train_acc": [], "val_acc": []}
    best = {"mIoU": -1.0}

    for epoch in range(1, EPOCHS + 1):
        tr = run_epoch(model, train_loader, criterion, optimizer, scaler,
                       desc=f"{tag} ep{epoch}/{EPOCHS} train")
        va = run_epoch(model, val_loader, criterion, desc=f"{tag} ep{epoch}/{EPOCHS} val")
        scheduler.step()

        history["train_loss"].append(tr["loss"]);  history["val_loss"].append(va["loss"])
        history["train_mIoU"].append(tr["mIoU"]);  history["val_mIoU"].append(va["mIoU"])
        history["train_acc"].append(tr["pixel_accuracy"]); history["val_acc"].append(va["pixel_accuracy"])

        if va["mIoU"] > best["mIoU"]:
            best = va
            torch.save(model.state_dict(), ckpt_path)

        tqdm.write(f"[{tag}] ep {epoch:02d}  "
                   f"train loss {tr['loss']:.4f} mIoU {tr['mIoU']:.4f} | "
                   f"val loss {va['loss']:.4f} mIoU {va['mIoU']:.4f}"
                   f"{'  *best*' if va['mIoU'] == best['mIoU'] else ''}")

    return history, best


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_history(history, title, path):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    ep = range(1, len(history["train_loss"]) + 1)

    axes[0].plot(ep, history["train_loss"], label="train")
    axes[0].plot(ep, history["val_loss"], label="val")
    axes[0].set_title("Loss"); axes[0].set_xlabel("epoch")

    axes[1].plot(ep, history["train_mIoU"], label="train")
    axes[1].plot(ep, history["val_mIoU"], label="val")
    axes[1].set_title("mIoU"); axes[1].set_xlabel("epoch")

    axes[2].plot(ep, history["train_acc"], label="train")
    axes[2].plot(ep, history["val_acc"], label="val")
    axes[2].set_title("Pixel accuracy"); axes[2].set_xlabel("epoch")

    for ax in axes:
        ax.legend(); ax.grid(alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_cv_summary(fold_metrics, path):
    names = list(CLASSES.values())
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    mious = [m["mIoU"] for m in fold_metrics]
    axes[0].bar(range(1, len(mious) + 1), mious, color="steelblue")
    axes[0].axhline(np.mean(mious), color="crimson", ls="--",
                    label=f"mean {np.mean(mious):.3f} ± {np.std(mious):.3f}")
    axes[0].set_xlabel("fold"); axes[0].set_ylabel("val mIoU")
    axes[0].set_title("Cross-validation mIoU per fold"); axes[0].legend(); axes[0].grid(alpha=0.3, axis="y")

    per_class = np.array([[m["per_class_IoU"][c] for c in names] for m in fold_metrics])
    x = np.arange(len(names))
    axes[1].bar(x, per_class.mean(0), yerr=per_class.std(0), capsize=5,
                color=[np.array(CLASS_COLORS[i]) / 255 for i in range(NUM_CLASSES)])
    axes[1].set_xticks(x); axes[1].set_xticklabels(names)
    axes[1].set_ylabel("IoU"); axes[1].set_title("Per-class IoU (CV mean ± std)")
    axes[1].grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_confusion_matrix(cm, path, title="Confusion matrix (test, row-normalized)"):
    cm = np.array(cm, dtype=np.float64)
    cm_norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    names = list(CLASSES.values())

    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(NUM_CLASSES)); ax.set_xticklabels(names)
    ax.set_yticks(range(NUM_CLASSES)); ax.set_yticklabels(names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Ground truth")
    ax.set_title(title)
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(j, i, f"{cm_norm[i, j]:.2f}",
                    ha="center", va="center",
                    color="white" if cm_norm[i, j] > 0.5 else "black")
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for c, col in CLASS_COLORS.items():
        out[mask == c] = col
    return out


def plot_qualitative(model, dataset: BarkDataset, path, n_samples=6):
    model.eval()
    rng = random.Random(SEED)
    indices = rng.sample(range(len(dataset)), min(n_samples, len(dataset)))
    tf = get_transforms(train=False)

    fig, axes = plt.subplots(len(indices), 3, figsize=(12, 4 * len(indices)))
    if len(indices) == 1:
        axes = axes[None, :]

    for row, i in enumerate(indices):
        img_path = dataset.images[i]
        img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        gt = cv2.imread(str(dataset.folder / f"{img_path.stem}_mask.png"),
                        cv2.IMREAD_GRAYSCALE)

        t = tf(image=img, mask=gt)
        with torch.no_grad():
            logits = model(t["image"].unsqueeze(0).to(DEVICE))
        pred = logits.argmax(1)[0].cpu().numpy().astype(np.uint8)
        pred = cv2.resize(pred, (img.shape[1], img.shape[0]),
                          interpolation=cv2.INTER_NEAREST)

        axes[row, 0].imshow(img);                axes[row, 0].set_title(img_path.stem)
        axes[row, 1].imshow(colorize_mask(gt));  axes[row, 1].set_title("Ground truth")
        axes[row, 2].imshow(colorize_mask(pred)); axes[row, 2].set_title("Prediction")
        for ax in axes[row]:
            ax.axis("off")

    handles = [plt.Rectangle((0, 0), 1, 1, color=np.array(CLASS_COLORS[c]) / 255)
               for c in range(NUM_CLASSES)]
    fig.legend(handles, list(CLASSES.values()), loc="lower center",
               ncol=NUM_CLASSES, frameon=False)
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    set_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Device: {DEVICE}\nModel:  {ARCHITECTURE} + {ENCODER} ({ENCODER_WEIGHTS})\n")

    full_train = BarkDataset(TRAIN_DIR)            # untransformed; subsets get transforms
    test_ds = BarkDataset(TEST_DIR, get_transforms(train=False))
    print(f"Train pairs: {len(full_train)}  |  Test pairs: {len(test_ds)}\n")
    if len(full_train) == 0:
        print(f"No data found in {TRAIN_DIR}"); return

    print("Computing class weights from train masks...")
    class_weights = compute_class_weights(full_train)

    # Two dataset views over the same files, different transforms
    train_view = BarkDataset(TRAIN_DIR, get_transforms(train=True))
    val_view = BarkDataset(TRAIN_DIR, get_transforms(train=False))

    # ---------------- 5-fold cross-validation ----------------
    print(f"\n===== {N_FOLDS}-fold cross-validation =====")
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_metrics, fold_histories = [], []

    for fold, (tr_idx, va_idx) in enumerate(kf.split(range(len(full_train))), start=1):
        tag = f"fold{fold}"
        hist, best = train_model(Subset(train_view, tr_idx.tolist()),
                                 Subset(val_view, va_idx.tolist()),
                                 class_weights, tag,
                                 OUTPUT_DIR / f"model_{tag}.pt")
        fold_metrics.append(best)
        fold_histories.append(hist)
        plot_history(hist, f"Fold {fold} training curves",
                     OUTPUT_DIR / f"curves_{tag}.png")
        print(f"  fold {fold}: best val mIoU = {best['mIoU']:.4f}")

    cv_mious = [m["mIoU"] for m in fold_metrics]
    print(f"\nCV mIoU: {np.mean(cv_mious):.4f} ± {np.std(cv_mious):.4f}")
    plot_cv_summary(fold_metrics, OUTPUT_DIR / "cv_summary.png")

    # ---------------- final model on (almost) full train ----------------
    print("\n===== Final model =====")
    n = len(full_train)
    idx = list(range(n))
    random.Random(SEED).shuffle(idx)
    n_val = max(1, int(n * FINAL_VAL_FRACTION))
    val_idx, tr_idx = idx[:n_val], idx[n_val:]

    final_ckpt = OUTPUT_DIR / "model_final.pt"
    final_hist, final_best = train_model(Subset(train_view, tr_idx),
                                         Subset(val_view, val_idx),
                                         class_weights, "final", final_ckpt)
    plot_history(final_hist, "Final model training curves",
                 OUTPUT_DIR / "curves_final.png")

    # ---------------- test evaluation ----------------
    print("\n===== Test evaluation =====")
    model = build_model()
    model.load_state_dict(torch.load(final_ckpt, map_location=DEVICE))
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True)
    criterion = ComboLoss(class_weights.to(DEVICE))
    test_metrics = run_epoch(model, test_loader, criterion, desc="test")

    print(f"Test  loss {test_metrics['loss']:.4f}  "
          f"mIoU {test_metrics['mIoU']:.4f}  "
          f"mDice {test_metrics['mDice']:.4f}  "
          f"acc {test_metrics['pixel_accuracy']:.4f}")
    for c in CLASSES.values():
        print(f"  {c:8s} IoU {test_metrics['per_class_IoU'][c]:.4f}  "
              f"Dice {test_metrics['per_class_Dice'][c]:.4f}")

    plot_confusion_matrix(test_metrics["confusion_matrix"],
                          OUTPUT_DIR / "confusion_matrix_test.png")
    plot_qualitative(model, BarkDataset(TEST_DIR),
                     OUTPUT_DIR / "qualitative_test.png", n_samples=6)

    # ---------------- save all metrics ----------------
    results = {
        "config": {"architecture": ARCHITECTURE, "encoder": ENCODER,
                   "epochs": EPOCHS, "batch_size": BATCH_SIZE, "lr": LR,
                   "n_folds": N_FOLDS, "seed": SEED,
                   "train_size": len(full_train), "test_size": len(test_ds)},
        "cv": {"fold_val_metrics": fold_metrics,
               "mIoU_mean": float(np.mean(cv_mious)),
               "mIoU_std": float(np.std(cv_mious))},
        "final_val_metrics": final_best,
        "test_metrics": test_metrics,
    }
    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nAll outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
