import cv2
import numpy as np
import os
import glob
import math

def rotate_image(image, angle, center=None):
    """Rotates an image by a given angle around a center."""
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
    return rotated

def process_pair(image_path, mask_path, output_image_dir, output_mask_dir):
    """Processes a single image-mask pair."""
    filename = os.path.basename(image_path)
    mask_filename = os.path.basename(mask_path) # Should be same basename usually, but let's handle if different extensions or something
    
    # Check if mask exists
    if not os.path.exists(mask_path):
        print(f"Mask not found for {filename}: {mask_path}")
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
    # fitLine expects points in (N, 1, 2) or (N, 2) format? 
    # cv2.fitLine docs say: "Input vector of 2D or 3D points, stored in std::vector<> or Mat."
    # Numpy array of shape (N, 2) usually works.
    top_points = np.stack((valid_xs, valid_top_ys), axis=-1).astype(np.int32)
    bottom_points = np.stack((valid_xs, valid_bot_ys), axis=-1).astype(np.int32)

    # Fit lines
    # cv2.fitLine returns [vx, vy, x, y] (normalized vector and a point on the line)
    # Angle can be calculated from vx, vy.
    
    if len(top_points) < 2 or len(bottom_points) < 2:
         print(f"Not enough points to fit lines for {filename}")
         return

    [vx_top, vy_top, x_top, y_top] = cv2.fitLine(top_points, cv2.DIST_L2, 0, 0.01, 0.01)
    [vx_bot, vy_bot, x_bot, y_bot] = cv2.fitLine(bottom_points, cv2.DIST_L2, 0, 0.01, 0.01)

    # Calculate angles in degrees
    angle_top = math.degrees(math.atan2(vy_top, vx_top))
    angle_bot = math.degrees(math.atan2(vy_bot, vx_bot))

    # Average angle
    avg_angle = (angle_top + angle_bot) / 2.0
    
    # We want to rotate so the line becomes horizontal (angle 0).
    # If the line is at angle `avg_angle`, we need to rotate by `-avg_angle`? 
    # Or is it just `avg_angle`?
    # atan2 returns angle relative to x-axis. 
    # If line is going down (positive slope in image coords), angle is positive.
    # To make it horizontal, we rotate counter-clockwise by -angle?
    # cv2.warpAffine rotation: positive angle is counter-clockwise.
    # If line has positive slope (y increases as x increases), it looks like \ (no, y is down).
    # Normal math: y up. Image: y down.
    # Let's stick to: we want to rotate by `avg_angle` to align it? 
    # Actually, if the log is tilted 10 degrees, we want to rotate -10 degrees to straighten it.
    # Let's try rotating by `avg_angle` first. Wait.
    # If line is 10 degrees (slight slope down), we want to rotate it back to 0.
    # So we rotate by -10 degrees?
    # Let's assume we rotate by `avg_angle` and see. 
    # Actually, `fitLine` gives vector. `atan2(vy, vx)` gives angle of that vector.
    # If the log is horizontal, angle is 0.
    # If log is tilted, say 45 degrees. We want to rotate by -45 degrees.
    # So rotation angle should be `-avg_angle`.
    
    rotation_angle = avg_angle

    # Rotate
    rotated_img = rotate_image(img, rotation_angle)
    rotated_mask = rotate_image(mask, rotation_angle)

    # Save
    cv2.imwrite(os.path.join(output_image_dir, filename), rotated_img)
    cv2.imwrite(os.path.join(output_mask_dir, mask_filename), rotated_mask)
    
    print(f"Processed {filename}: Top Angle={angle_top:.2f}, Bot Angle={angle_bot:.2f}, Rot Angle={rotation_angle:.2f}", flush=True)

def main():
    base_dir = "dataset_big"
    images_dir = os.path.join(base_dir, "images")
    masks_dir = os.path.join(base_dir, "masks")
    
    output_images_dir = os.path.join(base_dir, "images-rot")
    output_masks_dir = os.path.join(base_dir, "masks-rot")

    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_masks_dir, exist_ok=True)

    # Get list of images
    # Assuming png or jpg. The list_dir showed .png for masks and .JPG for images?
    # Let's check the list_dir output again.
    # Masks: DSC03814.png
    # Images: DSC03814.JPG
    # So extensions differ.
    
    image_files = sorted(glob.glob(os.path.join(images_dir, "*")))
    print(f"Found {len(image_files)} images.", flush=True)
    
    count = 0
    for image_path in image_files:
        print(f"Processing {image_path}...", flush=True)
        basename = os.path.splitext(os.path.basename(image_path))[0]
        
        # Find corresponding mask
        # Mask has .png extension based on previous ls
        mask_path = os.path.join(masks_dir, basename + ".png")
        
        if not os.path.exists(mask_path):
            # Try other extensions just in case
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
