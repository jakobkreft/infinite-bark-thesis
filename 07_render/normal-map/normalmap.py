#!/usr/bin/env python3
"""
Generate a tangent-space normal map from an image.

The input image is converted to grayscale and treated as a heightmap
(brighter = higher). The output is an RGB normal map in the OpenGL
convention (+Y up), which is what Blender's Normal Map node expects
by default.

Usage:
    python normal_map.py input.png output.png
    python normal_map.py input.jpg output.png --strength 3.0 --blur 1.5

Requirements:
    pip install pillow numpy
"""

import argparse
import numpy as np
from PIL import Image, ImageFilter


def sobel_gradients(height: np.ndarray):
    """Compute X and Y gradients with a 3x3 Sobel filter.

    Sobel averages over a small neighborhood, which gives smoother and
    less noisy normals than a plain central-difference gradient.
    """
    padded = np.pad(height, 1, mode="edge")

    # Sobel-X: detects horizontal slope (dh/dx)
    gx = (
        -1 * padded[:-2, :-2] + 1 * padded[:-2, 2:]
        + -2 * padded[1:-1, :-2] + 2 * padded[1:-1, 2:]
        + -1 * padded[2:,  :-2] + 1 * padded[2:,  2:]
    ) / 8.0

    # Sobel-Y: detects vertical slope (dh/dy)
    gy = (
        -1 * padded[:-2, :-2] + -2 * padded[:-2, 1:-1] + -1 * padded[:-2, 2:]
        +  1 * padded[2:,  :-2] +  2 * padded[2:,  1:-1] +  1 * padded[2:,  2:]
    ) / 8.0

    return gx, gy


def image_to_normal_map(
    input_path: str,
    output_path: str,
    strength: float = 2.0,
    blur: float = 0.0,
    invert_height: bool = False,
    flip_green: bool = False,
):
    # Load and convert to a grayscale heightmap in [0, 1]
    img = Image.open(input_path).convert("L")
    if blur > 0:
        img = img.filter(ImageFilter.GaussianBlur(radius=blur))

    height = np.asarray(img, dtype=np.float32) / 255.0
    if invert_height:
        height = 1.0 - height

    # Compute slopes
    gx, gy = sobel_gradients(height)
    gx *= strength
    gy *= strength

    # Build normal vectors. For a heightmap, the surface normal at each
    # pixel is normalize(-dh/dx, -dh/dy, 1).
    nx = -gx
    ny = -gy
    nz = np.ones_like(height)

    length = np.sqrt(nx * nx + ny * ny + nz * nz)
    nx /= length
    ny /= length
    nz /= length

    # OpenGL convention: +Y is up. DirectX flips the green channel.
    if flip_green:
        ny = -ny

    # Encode [-1, 1] -> [0, 255]
    r = np.clip((nx * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8)
    g = np.clip((ny * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8)
    b = np.clip((nz * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8)

    normal_map = np.stack([r, g, b], axis=-1)
    Image.fromarray(normal_map, mode="RGB").save(output_path)
    print(f"Saved normal map: {output_path}  ({normal_map.shape[1]}x{normal_map.shape[0]})")


def main():
    parser = argparse.ArgumentParser(description="Generate a normal map from an image.")
    parser.add_argument("input", help="Input image path (any format Pillow can read)")
    parser.add_argument("output", help="Output normal map path (PNG recommended)")
    parser.add_argument("-s", "--strength", type=float, default=2.0,
                        help="Bumpiness multiplier (default: 2.0). Higher = more pronounced relief.")
    parser.add_argument("-b", "--blur", type=float, default=0.0,
                        help="Gaussian blur radius applied before gradient (default: 0). "
                             "Useful to suppress high-frequency noise on photos.")
    parser.add_argument("--invert", action="store_true",
                        help="Treat darker pixels as higher (default: brighter = higher).")
    parser.add_argument("--directx", action="store_true",
                        help="Use DirectX convention (flip green). Default is OpenGL (Blender).")
    args = parser.parse_args()

    image_to_normal_map(
        args.input,
        args.output,
        strength=args.strength,
        blur=args.blur,
        invert_height=args.invert,
        flip_green=args.directx,
    )


if __name__ == "__main__":
    main()
