import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image
import os
import glob
import random

# Image size
width = 1600
height = 1200

class UnsupervisedHandClassificationNetwork(nn.Module):
    def __init__(self, input_channels=3, out_channels=5, stride=1, padding=0):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=out_channels, kernel_size=(4,3), padding="same")
        self.pool1 = nn.MaxPool2d(kernel_size=(4,3)) # from 1600x1200 to 400x400
        self.conv2 = nn.Conv2d(out_channels, 8, kernel_size=2, padding="same")
        self.pool2 = nn.MaxPool2d(kernel_size=2) # from 400x400 to 200x200
        self.conv3 = nn.Conv2d(in_channels=8, out_channels=12, kernel_size=2, padding="same")
        self.pool3 = nn.MaxPool2d(kernel_size=2) # from 200x200 to 100x100
        self.conv4 = nn.Conv2d(in_channels=12, out_channels=16, kernel_size=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2) # from 100x100 to 50x50

    def forward(self, x):
        x = self.pool1(F.elu(self.conv1(x)))
        x = self.pool2(F.elu(self.conv2(x)))
        x = self.pool3(F.elu(self.conv3(x)))
        x = self.pool4(F.elu(self.conv4(x)))

        return x
    
class UnsupervisedImageDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.image_paths = glob.glob(os.path.join(folder_path, "*.[jp][pn]g")) 
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        # Open image and convert to RGB (drops alpha channels if they exist)
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
            
        return image, img_path
    
    # Setup device (Use GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
# Initialize model and move it to the device
model = UnsupervisedHandClassificationNetwork().to(device)
model.eval() # CRITICAL: Puts model in inference mode (disables dropout, etc.)

# Define transforms: Resize (Height, Width) and convert to Tensor
transform = transforms.Compose([
    transforms.Resize((1200, 1600)), 
    transforms.ToTensor(),
    # Optional but recommended: Normalize your images based on standard ImageNet stats
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
])

# Initialize Dataset and DataLoader
# Replace "path/to/your/11k_images" with your actual folder path
# 1. Initialize your full dataset
dataset = UnsupervisedImageDataset(folder_path="/Users/stevenzinn/Desktop/Projects/Hand Interpretation/Hands/", transform=transform)
    
# 2. Figure out how many images make up "half"
half_length = 1000

# --- OPTION A: Grab a RANDOM half (Recommended) ---
# This is usually better so you get a diverse sample of your data
all_indices = list(range(len(dataset)))
random.shuffle(all_indices)
subset_indices = all_indices[:half_length]

# 3. Create the Subset
half_dataset = Subset(dataset, subset_indices)

# 4. Pass the SUBSET into the DataLoader, not the full dataset
dataloader = DataLoader(half_dataset, batch_size=16, shuffle=False, num_workers=0)

print(f"Starting extraction for {len(dataset)} images...")

# --- 4. The Extraction Loop ---
all_features = []

# torch.no_grad() tells PyTorch not to calculate gradients, saving massive amounts of memory
with torch.no_grad():
    for images, paths in dataloader:
        # Move images to GPU/CPU
        images = images.to(device)
        
        # Pass through the model
        features = model(images) 
        
        # features is currently shaped [Batch_Size, 64, 50, 50]
        # Move back to CPU to free up GPU memory, and append to our list
        all_features.append(features.cpu())
        
        print(f"Processed batch of {images.size(0)}. Output shape: {features.shape}")

# Concatenate all batches into one massive tensor
# Shape will be: [11000, 64, 50, 50]
final_feature_tensor = torch.cat(all_features, dim=0) 
print(f"Extraction complete! Final tensor shape: {final_feature_tensor.shape}")

import matplotlib.pyplot as plt
import numpy as np

# 1. Grab a single image's features
# We select index 0. The shape goes from [11000, 64, 50, 50] -> [64, 50, 50]
single_image_features = final_feature_tensor[0]

# Convert it to a NumPy array for Matplotlib
features_np = single_image_features.numpy()

# ==========================================
# Visualization 1: Overall Activation Heatmap
# ==========================================
# Average across the channel dimension (dim=0)
# This results in a single 50x50 map
activation_map = np.mean(features_np, axis=0)

plt.figure(figsize=(6, 6))
plt.title("Overall Activation Heatmap (Averaged across all channels)")
# 'viridis' is a great colormap for heatmaps
plt.imshow(activation_map, cmap='viridis') 
plt.colorbar(label='Activation Intensity')
plt.axis('off')
plt.show()

# ==========================================
# Visualization 2: Grid of Individual Channels
# ==========================================
# Let's look at the first 16 channels in a 4x4 grid
fig, axes = plt.subplots(4, 4, figsize=(10, 10))
fig.suptitle("First 16 Individual Feature Channels", fontsize=16)

for i, ax in enumerate(axes.flat):
    # Plot channel 'i' (which is a 50x50 map)
    ax.imshow(features_np[i], cmap='gray')
    ax.set_title(f"Channel {i}")
    ax.axis('off')

plt.tight_layout()
plt.show()

import cv2

