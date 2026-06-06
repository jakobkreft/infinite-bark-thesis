import cv2
import numpy as np
import os
import glob
import math

def rotate_image_with_matrix(image, angle, center=None):
    """Rotates an image and returns the rotated image and the rotation matrix."""
    (h, w) = image.shape[:2]

    if center is None:
        center = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Calculate new bounding box dimensions to avoid cropping
    abs_cos = abs(M[0, 0])
    abs_sin = abs(M[0, 1])

    bound_w = int(h * abs_sin + w * abs_cos)
    bound_h = int(h * abs_cos + w * abs_sin)

    # Adjust the rotation matrix to take into account translation
    M[0, 2] += bound_w / 2 - center[0]
    M[1, 2] += bound_h / 2 - center[1]

    rotated = cv2.warpAffine(image, M, (bound_w, bound_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    return rotated, M

def process_pair(image_path, mask_path, output_image_dir, output_mask_dir):
    """Processes a single image-mask pair."""
    filename = os.path.basename(image_path)
    mask_filename = os.path.basename(mask_path)
    
    if not os.path.exists(mask_path):
        print(f"Mask not found for {filename}: {mask_path}", flush=True)
        return

    img = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if img is None or mask is None:
        print(f"Failed to load {filename} or its mask.", flush=True)
        return

    # Threshold mask to ensure binary
    _, thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # Find all white pixels
    # Optimized approach using vectorized operations
    h, w = thresh.shape
    mask_bool = thresh > 0
    
    # Check which columns have data
    cols_with_data = np.any(mask_bool, axis=0)
    
    if not np.any(cols_with_data):
        print(f"No white pixels in mask {mask_filename}", flush=True)
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
         print(f"Not enough points to fit lines for {filename}", flush=True)
         return

    [vx_top, vy_top, x_top, y_top] = cv2.fitLine(top_points, cv2.DIST_L2, 0, 0.01, 0.01)
    [vx_bot, vy_bot, x_bot, y_bot] = cv2.fitLine(bottom_points, cv2.DIST_L2, 0, 0.01, 0.01)

    # Calculate angles in degrees
    angle_top = math.degrees(math.atan2(vy_top, vx_top))
    angle_bot = math.degrees(math.atan2(vy_bot, vx_bot))

    # Average angle
    avg_angle = (angle_top + angle_bot) / 2.0
    
    rotation_angle = avg_angle

    # Rotate
    rotated_img, M = rotate_image_with_matrix(img, rotation_angle)
    rotated_mask, _ = rotate_image_with_matrix(mask, rotation_angle)

    # Calculate crop boundaries
    # Transform the points on the lines to the rotated coordinate system
    # Point is (x, y). M is 2x3.
    # P_new = M * P_old
    
    # x_top, y_top are arrays/lists from fitLine, extract scalar values
    x_top_val = x_top[0]
    y_top_val = y_top[0]
    x_bot_val = x_bot[0]
    y_bot_val = y_bot[0]

    # Top point
    pt_top = np.array([x_top_val, y_top_val, 1.0])
    new_pt_top = M.dot(pt_top)
    new_y_top = new_pt_top[1]

    # Bottom point
    pt_bot = np.array([x_bot_val, y_bot_val, 1.0])
    new_pt_bot = M.dot(pt_bot)
    new_y_bot = new_pt_bot[1]

    # Determine crop range
    # Ensure y_start < y_end
    y_start = int(min(new_y_top, new_y_bot))
    y_end = int(max(new_y_top, new_y_bot))
    
    # Clamp to image bounds
    h_rot, w_rot = rotated_img.shape[:2]
    y_start = max(0, y_start)
    y_end = min(h_rot, y_end)
    
    if y_end <= y_start:
        print(f"Invalid crop for {filename}: {y_start} to {y_end}", flush=True)
        return

    # Calculate X crop to remove black borders
    h_orig, w_orig = img.shape[:2]
    # Corners of original image
    corners = np.array([
        [0, 0],
        [w_orig, 0],
        [w_orig, h_orig],
        [0, h_orig]
    ], dtype=np.float32)
    
    # Transform corners
    # M is 2x3, we need to add 1s
    corners_homog = np.hstack((corners, np.ones((4, 1))))
    corners_rot = M.dot(corners_homog.T).T # (4, 2)
    
    # Define edges of the rotated rectangle
    edges = [
        (corners_rot[0], corners_rot[1]),
        (corners_rot[1], corners_rot[2]),
        (corners_rot[2], corners_rot[3]),
        (corners_rot[3], corners_rot[0])
    ]
    
    left_bounds = []
    right_bounds = []
    
    # Center of rotated image
    cx_rot = rotated_img.shape[1] / 2.0
    
    # Check intersections with y_start and y_end
    for y_line in [y_start, y_end]:
        for p1, p2 in edges:
            # Check if line intersects with horizontal line y = y_line
            # Segment p1-p2
            if (p1[1] <= y_line <= p2[1]) or (p2[1] <= y_line <= p1[1]):
                # Avoid division by zero
                if abs(p2[1] - p1[1]) > 1e-5:
                    t = (y_line - p1[1]) / (p2[1] - p1[1])
                    x_inter = p1[0] + t * (p2[0] - p1[0])
                    
                    if x_inter < cx_rot:
                        left_bounds.append(x_inter)
                    else:
                        right_bounds.append(x_inter)
    
    # Also check for vertices inside the y-range
    for p in corners_rot:
        if y_start <= p[1] <= y_end:
            if p[0] < cx_rot:
                left_bounds.append(p[0])
            else:
                right_bounds.append(p[0])
                
    # Determine crop range
    if not left_bounds:
        x_start = 0
    else:
        x_start = int(max(left_bounds))
        
    if not right_bounds:
        x_end = rotated_img.shape[1]
    else:
        x_end = int(min(right_bounds))
        
    # Clamp
    x_start = max(0, x_start)
    x_end = min(rotated_img.shape[1], x_end)
    
    if x_end <= x_start:
         print(f"Invalid x-crop for {filename}: {x_start} to {x_end}", flush=True)
         return

    # Crop
    cropped_img = rotated_img[y_start:y_end, x_start:x_end]
    cropped_mask = rotated_mask[y_start:y_end, x_start:x_end]

    # Save
    cv2.imwrite(os.path.join(output_image_dir, filename), cropped_img)
    cv2.imwrite(os.path.join(output_mask_dir, mask_filename), cropped_mask)

def main():
    base_dir = "dataset_big"
    images_dir = os.path.join(base_dir, "images")
    masks_dir = os.path.join(base_dir, "masks")
    
    output_images_dir = os.path.join(base_dir, "images-crop")
    output_masks_dir = os.path.join(base_dir, "masks-crop")

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

        process_pair(image_path, mask_path, output_images_dir, output_masks_dir)
        count += 1
        if count % 10 == 0:
            print(f"Processed {count}/{len(image_files)}", flush=True)

if __name__ == "__main__":
    main()
