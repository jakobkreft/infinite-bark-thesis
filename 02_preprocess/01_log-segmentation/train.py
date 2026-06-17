import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import numpy as np

class SegmentationDataset(Dataset):
    def __init__(self, root_dir, image_folder="images", mask_folder="masks", transforms=None):
        self.root_dir = root_dir
        self.transforms = transforms
        self.image_dir = os.path.join(root_dir, image_folder)
        self.mask_dir = os.path.join(root_dir, mask_folder)
        
        # List all images
        self.images = sorted([f for f in os.listdir(self.image_dir) if f.endswith(('.jpg', '.JPG', '.png'))])
        
        # Verify masks exist
        self.masks = []
        for img_name in self.images:
            # Assume mask has same name but maybe different extension (png)
            base_name = os.path.splitext(img_name)[0]
            mask_name = base_name + ".png"
            if not os.path.exists(os.path.join(self.mask_dir, mask_name)):
                 # Try same extension
                 mask_name = img_name
            
            if os.path.exists(os.path.join(self.mask_dir, mask_name)):
                self.masks.append(mask_name)
            else:
                print(f"Warning: Mask not found for {img_name}")
                # Remove image from list if mask missing
                self.images.remove(img_name)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])
        
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L") # Grayscale
        
        if self.transforms:
            # Apply transforms to both image and mask
            # Note: Random transforms need to be applied identically. 
            # For simplicity in this script, we'll do basic resizing and normalization manually or use a helper if needed.
            # But torchvision transforms are separate for image/mask usually unless using v2.
            # Here we will just resize to a fixed size.
            pass

        # Resize
        target_size = (512, 512)
        image = image.resize(target_size, Image.BILINEAR)
        mask = mask.resize(target_size, Image.NEAREST)
        
        # Convert to tensor
        image = transforms.ToTensor()(image)
        mask = np.array(mask)
        mask = torch.from_numpy(mask).long()
        
        # Normalize mask to 0 and 1
        # User said 0 and 255.
        mask = mask // 255
        
        # Normalize image
        image = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image)
        
        return image, mask

def train_model(model, dataloaders, criterion, optimizer, num_epochs=25):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = model.to(device)
    
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)
        
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
                
            running_loss = 0.0
            running_corrects = 0
            total_pixels = 0
            
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)['out']
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                total_pixels += torch.numel(labels)
                
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / total_pixels
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), 'best_model.pth')
                
    print(f'Best val Acc: {best_acc:4f}')
    return model

def main():
    data_dir = 'dataset_small'
    
    # Datasets
    image_datasets = {x: SegmentationDataset(os.path.join(data_dir, x)) for x in ['train', 'test']}
    
    # Dataloaders
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=0) for x in ['train', 'test']}
    
    # Model
    model = models.segmentation.deeplabv3_resnet50(weights='DEFAULT')
    # Replace the classifier head for 2 classes
    model.classifier[4] = nn.Conv2d(256, 2, kernel_size=(1, 1), stride=(1, 1))
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    train_model(model, dataloaders, criterion, optimizer, num_epochs=20)

if __name__ == "__main__":
    main()
