import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np

def load_model(model_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Same model definition as training
    model = models.segmentation.deeplabv3_resnet50(weights=None) # No weights needed, loading state_dict
    model.classifier[4] = nn.Conv2d(256, 2, kernel_size=(1, 1), stride=(1, 1))
    
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    model.to(device)
    model.eval()
    return model, device

def inference(model, device, image_path, output_path, viz_path=None):
    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    original_size = image.size
    
    # Same transforms as training
    target_size = (512, 512)
    preprocess = transforms.Compose([
        transforms.Resize(target_size, interpolation=Image.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    input_tensor = preprocess(image).unsqueeze(0) # Add batch dimension
    input_tensor = input_tensor.to(device)
    
    with torch.no_grad():
        output = model(input_tensor)['out'][0]
        output_predictions = output.argmax(0)
    
    # Convert to numpy and resize back to original size
    mask = output_predictions.byte().cpu().numpy()
    mask_image = Image.fromarray(mask * 255) # Scale to 0-255 for visibility
    mask_image = mask_image.resize(original_size, Image.NEAREST)
    
    mask_image.save(output_path)
    print(f"Saved mask to {output_path}")
    
    # Create visualization if path provided
    if viz_path:
        # Convert image and mask to numpy
        img_np = np.array(image)
        mask_np = np.array(mask_image)
        
        # Create colored overlay
        overlay = img_np.copy().astype(np.float32)
        
        # Green for mask=255 (class 1)
        green_mask = mask_np == 255
        overlay[green_mask] = overlay[green_mask] * 0.6 + np.array([0, 255, 0]) * 0.4
        
        # Red for mask=0 (class 0)
        red_mask = mask_np == 0
        overlay[red_mask] = overlay[red_mask] * 0.6 + np.array([255, 0, 0]) * 0.4
        
        viz_image = Image.fromarray(overlay.astype(np.uint8))
        viz_image.save(viz_path)
        print(f"Saved visualization to {viz_path}")

def main():
    model_path = 'best_model.pth'
    input_dir = 'dataset_big/images'
    output_dir = 'dataset_big/masks'
    viz_dir = 'dataset_big/viz'
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(viz_dir):
        os.makedirs(viz_dir)
        
    model, device = load_model(model_path)
    
    images = [f for f in os.listdir(input_dir) if f.endswith(('.jpg', '.JPG', '.png'))]
    print(f"Found {len(images)} images to process.")
    
    for img_name in images:
        img_path = os.path.join(input_dir, img_name)
        mask_name = os.path.splitext(img_name)[0] + '.png'
        out_path = os.path.join(output_dir, mask_name)
        viz_path = os.path.join(viz_dir, mask_name)
        
        inference(model, device, img_path, out_path, viz_path)

if __name__ == "__main__":
    main()
