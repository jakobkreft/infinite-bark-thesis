# Bark Segmentation Pipeline

Semantic segmentation of bark damage using DeepLabV3+ (ResNet50). Classifies pixels into 3 classes:
- **Class 0** - Background (intact bark)
- **Class 1** - Pruning wounds (cyan)
- **Class 2** - Mechanical damage (orange)

## Setup

The conda environment `bark-seg` is already created. If you need to recreate it:

```bash
conda create -n bark-seg python=3.11 -y
conda activate bark-seg
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install segmentation-models-pytorch albumentations
pip install matplotlib seaborn scikit-learn pandas
pip install tqdm opencv-python-headless pyyaml wandb tensorboard
```

## How to Run

**Important:** Use `conda run -n bark-seg` to prefix all commands. This avoids activation issues in PowerShell.

### 1. Analyze Dataset

```
conda run -n bark-seg python analyze_dataset.py
```

Generates figures in `outputs/dataset_analysis/`: class distributions, resolution stats, sample overlays.

### 2. Train

```
conda run -n bark-seg python train.py --config configs/default.yaml
```

Trains DeepLabV3+ with:
- 512x512 random crops from full-res images
- Class-aware sampling (50% of crops centered on rare class 1)
- Combined CrossEntropy + Dice loss with class weights
- Mixed precision, cosine LR schedule, early stopping

Outputs to `outputs/<experiment_name>/`: checkpoints, training curves, logs.

Override config values from command line:

```
conda run -n bark-seg python train.py --config configs/default.yaml --override training.epochs=20 training.batch_size=4
```

### 3. Evaluate

```
conda run -n bark-seg python evaluate.py --config configs/default.yaml --split test
```

Runs sliding window inference on test set. Generates:
- Per-class IoU, Dice, precision, recall
- Confusion matrix
- Per-image prediction overlays (GT vs prediction vs difference)
- Best/worst case analysis
- Class 1 failure analysis

### 4. Inference (unlabeled images)

```
conda run -n bark-seg python inference.py --config configs/default.yaml
```

Segments all images in `dataset_step2_smooth/inference/images/`. Saves masks and overlays to `outputs/<experiment_name>/inference/`.

## Project Structure

```
configs/default.yaml     - All hyperparameters
src/dataset.py           - Dataset with class-aware cropping + augmentation
src/model.py             - Model factory (smp)
src/losses.py            - Combined CE + Dice loss
src/metrics.py           - IoU, Dice, confusion matrix
src/sliding_window.py    - Sliding window inference with Gaussian blending
src/visualization.py     - All plotting functions
train.py                 - Training script
evaluate.py              - Test evaluation + research visualizations
inference.py             - Inference on unlabeled images
analyze_dataset.py       - Dataset analysis figures
```

## Config

Edit `configs/default.yaml` to change architecture, hyperparameters, paths. Key settings:

| Setting | Default | Description |
|---------|---------|-------------|
| `model.architecture` | DeepLabV3Plus | Also supports Unet, UnetPlusPlus, FPN, PSPNet |
| `model.encoder` | resnet50 | Any smp encoder (efficientnet-b4, mit_b2, etc.) |
| `training.crop_size` | 512 | Patch size for training |
| `training.epochs` | 50 | Max epochs (early stopping at patience 20) |
| `loss.class_weights` | [1, 15, 3] | CE weights for class imbalance |
