# Bark Segmentation Pipeline

Semantic segmentation of bark damage using unet or DeepLabV3+ (ResNet50). Classifies pixels into 3 classes:
- **Class 0** - Background (intact bark)
- **Class 1** - Pruning wounds (cyan)
- **Class 2** - Mechanical damage (orange)

it runs using 5 fold cross validation. then run final train version.

## Run
"python train_seg.py"