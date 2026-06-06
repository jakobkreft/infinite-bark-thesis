import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple


def create_gaussian_window(size: int, sigma: float = None) -> np.ndarray:
    """Create a 2D Gaussian weighting window for blending overlapping patches."""
    if sigma is None:
        sigma = size / 4.0
    coords = np.arange(size, dtype=np.float64) - (size - 1) / 2.0
    g = np.exp(-(coords ** 2) / (2 * sigma ** 2))
    window = np.outer(g, g)
    window /= window.max()
    return window


def sliding_window_inference(
    model: torch.nn.Module,
    image: torch.Tensor,
    crop_size: int = 512,
    stride: int = 256,
    num_classes: int = 3,
    device: str = "cuda",
    use_gaussian: bool = True,
    batch_size: int = 8,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run sliding window inference on a single full-resolution image.

    Args:
        model: segmentation model in eval mode
        image: (C, H, W) normalized tensor
        crop_size: patch size
        stride: step between patches
        num_classes: number of output classes
        device: cuda device
        use_gaussian: use Gaussian weighting for blending
        batch_size: number of patches to process at once

    Returns:
        pred_mask: (H, W) predicted class indices
        prob_map: (num_classes, H, W) softmax probability map
    """
    model.eval()
    C, H, W = image.shape

    # Create weight window
    if use_gaussian:
        weight_window = create_gaussian_window(crop_size)
    else:
        weight_window = np.ones((crop_size, crop_size))
    weight_window = torch.tensor(weight_window, dtype=torch.float32, device=device)

    # Pad image if needed so we cover all pixels
    pad_h = max(0, crop_size - H)
    pad_w = max(0, crop_size - W)
    if pad_h > 0 or pad_w > 0:
        image = F.pad(image, (0, pad_w, 0, pad_h), mode="reflect")

    _, H_pad, W_pad = image.shape

    # Generate patch positions
    y_positions = list(range(0, H_pad - crop_size + 1, stride))
    x_positions = list(range(0, W_pad - crop_size + 1, stride))
    # Ensure we cover the last row/column
    if y_positions[-1] + crop_size < H_pad:
        y_positions.append(H_pad - crop_size)
    if x_positions[-1] + crop_size < W_pad:
        x_positions.append(W_pad - crop_size)

    # Accumulation buffers
    prob_acc = torch.zeros((num_classes, H_pad, W_pad), dtype=torch.float32, device=device)
    weight_acc = torch.zeros((H_pad, W_pad), dtype=torch.float32, device=device)

    # Collect all patches and positions
    all_patches = []
    all_positions = []
    for y in y_positions:
        for x in x_positions:
            patch = image[:, y:y + crop_size, x:x + crop_size]
            all_patches.append(patch)
            all_positions.append((y, x))

    # Process in batches
    with torch.no_grad():
        for i in range(0, len(all_patches), batch_size):
            batch = torch.stack(all_patches[i:i + batch_size]).to(device)
            outputs = model(batch)
            probs = F.softmax(outputs, dim=1)

            for j, (y, x) in enumerate(all_positions[i:i + batch_size]):
                prob_acc[:, y:y + crop_size, x:x + crop_size] += probs[j] * weight_window
                weight_acc[y:y + crop_size, x:x + crop_size] += weight_window

    # Average probabilities
    weight_acc = torch.clamp(weight_acc, min=1e-8)
    prob_map = prob_acc / weight_acc.unsqueeze(0)

    # Crop back to original size
    prob_map = prob_map[:, :H, :W]

    # Predict
    pred_mask = prob_map.argmax(dim=0).cpu().numpy().astype(np.uint8)
    prob_map_np = prob_map.cpu().numpy()

    return pred_mask, prob_map_np
