import cv2
import numpy as np
import os
import glob
import random

def draw_line(image, vx, vy, x, y, color=(0, 255, 0), thickness=2):
    """Draws a line on the image given a vector and a point."""
    h, w = image.shape[:2]
    # Calculate two points on the line to draw it across the image
    # Line equation: P = P0 + t * V
    # We need to find t such that the points are outside the image bounds
    
    # Let's just pick a large t
    t = max(h, w) * 2
    
    p1 = (int(x - t * vx), int(y - t * vy))
    p2 = (int(x + t * vx), int(y + t * vy))
    
    cv2.line(image, p1, p2, color, thickness)

def visualize_random_mask():
    base_dir = "dataset_big"
    masks_dir = os.path.join(base_dir, "masks")
    output_viz_dir = os.path.join(base_dir, "masks_viz")
    
    os.makedirs(output_viz_dir, exist_ok=True)
    
    mask_files = sorted(glob.glob(os.path.join(masks_dir, "*.png")))
    
    if not mask_files:
        print("No masks found.")
        return

    # Pick a random mask
    mask_path = random.choice(mask_files)
    mask_filename = os.path.basename(mask_path)
    print(f"Visualizing {mask_filename}...")
    
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    if mask is None:
        print(f"Failed to load mask {mask_path}")
        return

    # Convert to color for drawing colored lines
    mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    # Threshold mask to ensure binary
    _, thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # Find all white pixels
    # Optimized approach using vectorized operations
    h, w = thresh.shape
    mask_bool = thresh > 0
    
    # Check which columns have data
    cols_with_data = np.any(mask_bool, axis=0)
    
    if not np.any(cols_with_data):
        print(f"No white pixels in mask {mask_filename}")
        return

    # Top edge: first True from top
    top_ys = np.argmax(mask_bool, axis=0)
    
    # Bottom edge: first True from bottom
    # We flip the mask upside down
    bot_ys_flipped = np.argmax(mask_bool[::-1, :], axis=0)
    bot_ys = h - 1 - bot_ys_flipped
    
    # Filter valid columns
    valid_xs = np.where(cols_with_data)[0]
    
    valid_top_ys = top_ys[valid_xs]
    valid_bot_ys = bot_ys[valid_xs]
    
    # Construct points
    top_points = np.stack((valid_xs, valid_top_ys), axis=-1).astype(np.int32)
    bottom_points = np.stack((valid_xs, valid_bot_ys), axis=-1).astype(np.int32)

    # Fit lines
    if len(top_points) < 2 or len(bottom_points) < 2:
         print(f"Not enough points to fit lines for {mask_filename}")
         return

    [vx_top, vy_top, x_top, y_top] = cv2.fitLine(top_points, cv2.DIST_L2, 0, 0.01, 0.01)
    [vx_bot, vy_bot, x_bot, y_bot] = cv2.fitLine(bottom_points, cv2.DIST_L2, 0, 0.01, 0.01)

    # Draw lines
    # Top line in Green
    draw_line(mask_color, vx_top, vy_top, x_top, y_top, color=(0, 255, 0), thickness=5)
    
    # Bottom line in Blue
    draw_line(mask_color, vx_bot, vy_bot, x_bot, y_bot, color=(255, 0, 0), thickness=5)
    
    # Save
    output_path = os.path.join(output_viz_dir, f"viz_{mask_filename}")
    cv2.imwrite(output_path, mask_color)
    print(f"Saved visualization to {output_path}")

if __name__ == "__main__":
    visualize_random_mask()
