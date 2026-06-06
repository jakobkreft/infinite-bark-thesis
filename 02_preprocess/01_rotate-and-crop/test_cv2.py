import cv2
import os
import glob
import sys

print("Python version:", sys.version)
print("OpenCV version:", cv2.__version__)

base_dir = "dataset_big"
images_dir = os.path.join(base_dir, "images")
masks_dir = os.path.join(base_dir, "masks")

print(f"Images dir: {images_dir}, exists: {os.path.exists(images_dir)}")
print(f"Masks dir: {masks_dir}, exists: {os.path.exists(masks_dir)}")

image_files = sorted(glob.glob(os.path.join(images_dir, "*")))
print(f"Found {len(image_files)} images.")

if len(image_files) > 0:
    img_path = image_files[0]
    print(f"First image: {img_path}")
    img = cv2.imread(img_path)
    if img is None:
        print("Failed to load image.")
    else:
        print(f"Image loaded, shape: {img.shape}")

    basename = os.path.splitext(os.path.basename(img_path))[0]
    mask_path = os.path.join(masks_dir, basename + ".png")
    print(f"Expected mask path: {mask_path}")
    
    if os.path.exists(mask_path):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print("Failed to load mask.")
        else:
            print(f"Mask loaded, shape: {mask.shape}")
    else:
        print("Mask file does not exist.")
