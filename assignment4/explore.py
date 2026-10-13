import os
import numpy as np
import matplotlib.pyplot as plt
from data import get_raw_datasets, get_loaders

os.makedirs("figures", exist_ok=True)

# 1. Explore Dataset (Part 1.2)
train_dataset, test_dataset = get_raw_datasets()
print(f"Training samples: {len(train_dataset)}")
print(f"Testing samples: {len(test_dataset)}")

image, (label, mask) = train_dataset[0]
print(f"First Image Size (W, H): {image.size}")
print(f"First Image Classification Label (0=Cat, 1=Dog): {label}")
print(f"First Mask Size (W, H): {mask.size}")
print(f"First Mask Unique Values: {np.unique(np.array(mask))}")

# 2. Visualize Sample Images (Part 1.3)
fig, axes = plt.subplots(3, 2, figsize=(6, 9))
for i in range(3):
    img, (_, msk) = train_dataset[i]
    axes[i, 0].imshow(img)
    axes[i, 0].set_title(f"Image {i} (Label: {label})")
    axes[i, 0].axis('off')
    
    axes[i, 1].imshow(np.array(msk), vmin=0, vmax=2, cmap='viridis')
    axes[i, 1].set_title(f"Mask {i} Unique: {np.unique(np.array(msk))}")
    axes[i, 1].axis('off')

plt.tight_layout()
plt.savefig("figures/sample_images.png")
plt.close()
print("Saved sample visualization to figures/sample_images.png")

# 3. Inspect Processed Batch (Part 1.4)
train_loader, test_loader = get_loaders(batch_size=16)
images, (labels, masks) = next(iter(train_loader))
print("Processed Batch Shapes:")
print(f"  images: {images.shape}")
print(f"  labels: {labels.shape}")
print(f"  masks:  {masks.shape}")
