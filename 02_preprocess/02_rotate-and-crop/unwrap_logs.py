import cv2
import numpy as np
import os
import glob
import argparse
import math

def unwrap_log(image, radius_factor=1.0, distance_factor=10.0):
    """
    Unwraps a log image.
    
    Args:
        image: Input image (H, W, C) or (H, W).
        radius_factor: Multiplier for the radius relative to half-height. 
                       1.0 means radius = height / 2.
        distance_factor: Distance of camera from log center, in units of radius.
                         Must be > 1.0.
    
    Returns:
        Unwrapped image.
    """
    h, w = image.shape[:2]
    
    # Radius of the log
    r = (h / 2.0) * radius_factor
    
    # Distance from camera to center of log
    d = r * distance_factor
    
    if d <= r:
        # Camera inside or on surface, invalid for this projection model
        # Fallback to orthographic (large distance)
        d = r * 1000.0

    # We assume the input image covers the projection of the log.
    # The max y coordinate in input is h/2 (centered).
    # We need to find the max angle theta that corresponds to h/2.
    # y_proj = (d - r) * (r * sin(theta)) / (d - r * cos(theta))
    # Let's invert this to find theta for a given y_proj.
    # y_p * (d - r * cos(theta)) = (d - r) * r * sin(theta)
    # y_p * d - y_p * r * cos(theta) = K * sin(theta), where K = (d-r)*r
    # y_p * d = K * sin(theta) + y_p * r * cos(theta)
    # This is of form C = A sin(theta) + B cos(theta) = sqrt(A^2+B^2) sin(theta + alpha)
    # A = K, B = y_p * r
    # We can solve for theta numerically or analytically.
    # But actually, we need the inverse map for remap:
    # For each pixel (x, y_new) in output, find (x, y_old) in input.
    
    # Output height corresponds to arc length.
    # We need to determine the range of theta covered by the input image height.
    # Let's solve for theta_max at y_proj = h/2.
    # Using the equation above.
    
    def get_y_proj(theta, r, d):
        # Perspective projection formula
        # Assuming image plane is at the closest surface point (z = r - d? No, camera at -d, surface at -r to r?)
        # Let's stick to: Camera at (0, -d). Cylinder center (0, 0).
        # Point P: (r sin theta, r cos theta).
        # Vector CP: (r sin theta, r cos theta + d).
        # Image plane at y = -d + (d-r) = -r (tangent to front face).
        # Intersection of CP with y = -r line?
        # Wait, coordinate system:
        # Z axis is depth?
        # Let's say Camera is at Z = -d. Cylinder axis at Z = 0.
        # Image plane at Z = -r (front of cylinder).
        # Point P: (r sin theta, r cos theta). Z is r cos theta. Y is r sin theta.
        # Wait, cylinder along X. Cross section in Y-Z.
        # P = (x, r sin theta, r cos theta).
        # Camera at (0, 0, -d).
        # Image plane at Z = -r.
        # Projected Y: y_proj = y_P * (d - r) / (d + z_P)
        # y_proj = r sin theta * (d - r) / (d + r cos theta)
        
        denom = d + r * math.cos(theta)
        if abs(denom) < 1e-6: return 0
        return r * math.sin(theta) * (d - r) / denom

    # Find theta_max such that get_y_proj(theta_max) = h/2
    # Since function is monotonic for theta in [0, pi/2), we can binary search or just solve.
    # Let's use binary search for simplicity and robustness.
    
    target_y = h / 2.0
    low = 0
    high = math.pi / 2.0 - 0.01 # Slightly less than 90 degrees
    theta_max = 0
    
    for _ in range(20):
        mid = (low + high) / 2.0
        y_val = get_y_proj(mid, r, d)
        if y_val < target_y:
            low = mid
        else:
            high = mid
    theta_max = high
    
    # Calculate output height
    # Arc length = r * theta
    # Total height = 2 * r * theta_max
    h_new = int(2 * r * theta_max)
    
    # Create map
    map_x = np.zeros((h_new, w), dtype=np.float32)
    map_y = np.zeros((h_new, w), dtype=np.float32)
    
    # Populate map_x (identity along x)
    # map_x[y, x] = x
    # We can use np.tile or meshgrid
    xs = np.arange(w, dtype=np.float32)
    map_x[:] = xs
    
    # Populate map_y
    # For each y_new, calculate theta, then y_old.
    # y_new ranges from 0 to h_new. Center is h_new / 2.
    ys_new = np.arange(h_new, dtype=np.float32)
    thetas = (ys_new - h_new / 2.0) / r
    
    # Vectorized y_proj calculation
    # y_proj = r * sin(theta) * (d - r) / (d + r * cos(theta))
    sin_thetas = np.sin(thetas)
    cos_thetas = np.cos(thetas)
    
    ys_old_centered = r * sin_thetas * (d - r) / (d + r * cos_thetas)
    ys_old = ys_old_centered + h / 2.0
    
    # map_y needs to be (h_new, w). The values are constant along rows.
    # Broadcast to shape
    map_y = np.tile(ys_old[:, np.newaxis], (1, w)).astype(np.float32)
    
    # Remap
    unwrapped = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    
    return unwrapped

def main():
    parser = argparse.ArgumentParser(description="Unwrap tree log images.")
    parser.add_argument("--radius", type=float, default=1.0, help="Radius factor (multiplier of half-height).")
    parser.add_argument("--distance", type=float, default=10.0, help="Distance factor (multiplier of radius).")
    args = parser.parse_args()

    base_dir = "dataset_big"
    images_dir = os.path.join(base_dir, "images-crop")
    masks_dir = os.path.join(base_dir, "masks-crop")
    
    output_images_dir = os.path.join(base_dir, "images-unwrap")
    output_masks_dir = os.path.join(base_dir, "masks-unwrap")

    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_masks_dir, exist_ok=True)

    image_files = sorted(glob.glob(os.path.join(images_dir, "*")))
    print(f"Found {len(image_files)} images.", flush=True)
    
    count = 0
    for image_path in image_files:
        basename = os.path.splitext(os.path.basename(image_path))[0]
        
        # Find corresponding mask
        mask_path = os.path.join(masks_dir, basename + ".png")
        if not os.path.exists(mask_path):
             # Try other extensions
            possible_exts = [".jpg", ".JPG", ".jpeg"]
            found = False
            for ext in possible_exts:
                if os.path.exists(os.path.join(masks_dir, basename + ext)):
                    mask_path = os.path.join(masks_dir, basename + ext)
                    found = True
                    break
            if not found:
                print(f"Skipping {basename}: Mask not found", flush=True)
                continue

        img = cv2.imread(image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None or mask is None:
            print(f"Failed to load {basename}", flush=True)
            continue
            
        unwrapped_img = unwrap_log(img, args.radius, args.distance)
        unwrapped_mask = unwrap_log(mask, args.radius, args.distance)
        
        cv2.imwrite(os.path.join(output_images_dir, basename + ".JPG"), unwrapped_img)
        cv2.imwrite(os.path.join(output_masks_dir, basename + ".png"), unwrapped_mask)
        
        count += 1
        if count % 10 == 0:
            print(f"Processed {count}/{len(image_files)}", flush=True)

if __name__ == "__main__":
    main()
