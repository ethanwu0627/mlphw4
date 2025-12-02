import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import os


class TanhScaled(nn.Module):
    def forward(self, x):
        return 1.7159 * torch.tanh((2.0 / 3.0) * x)

def load_digit_codes():
    
    # Path to the digits dataset
    data_root = os.path.join(os.path.dirname(__file__), "digits updated")
    
    # Initialize accumulator for mean images per digit
    digit_sums = {i: [] for i in range(10)}
    
    # Load and binarize images for each digit
    for digit in range(10):
        digit_dir = os.path.join(data_root, str(digit))
        
        # Use first 100 images per digit (or all if less than 100)
        image_files = sorted(os.listdir(digit_dir))[:100]
        
        for img_file in image_files:
            if img_file.startswith('.'):  # Skip hidden files
                continue
                
            img_path = os.path.join(digit_dir, img_file)
            
            try:
                # Load image
                img = Image.open(img_path).convert('L')
                
                # Binarize: pixels < 128 -> 0, else -> 255
                img_array = np.array(img, dtype=np.float32)
                img_binarized = np.where(img_array < 128, 0, 255)
                
                digit_sums[digit].append(img_binarized)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
                continue
    
    # Compute mean and resize to 12x7
    mu = torch.zeros(10, 84)
    resize_transform = transforms.Resize((12, 7), antialias=True)
    
    for digit in range(10):
        if len(digit_sums[digit]) == 0:
            print(f"Warning: No images found for digit {digit}")
            continue
            
        # Stack and compute mean of binarized images
        stacked = np.stack(digit_sums[digit])
        mean_img = np.mean(stacked, axis=0)
        
        # Convert to tensor and add channel dimension
        mean_tensor = torch.from_numpy(mean_img).float().unsqueeze(0)
        
        # Resize to 12x7
        resized = resize_transform(mean_tensor)
        
        # Normalize: (1.0 - pixel/255) * 1.275 - 0.1
        normalized = (1.0 - resized / 255.0) * 1.275 - 0.1
        
        # Flatten to 84-dimensional vector
        mu[digit] = normalized.flatten()
    
    return mu

class LeNet5(nn.Module):
    def __init__(self, mu_codes):
        super().__init__()

        self.tanh = TanhScaled()

        # C1: Conv (1x32x32 to 6x28x28)
        self.C1 = nn.Conv2d(1, 6, kernel_size = 5)

        # S2: Subsampling (AvgPool + learnable a,b)
        self.S2_pool = nn.AvgPool2d(kernel_size = 2, stride = 2)
        self.S2_a = nn.Parameter(torch.ones(6))
        self.S2_b = nn.Parameter(torch.zeros(6))

        # C3: Conv (6x14x14 to 16x10x10)
        self.C3 = nn.Conv2d(6, 16, kernel_size=5)

        # S4: Subsampling (AvgPool + learnable a,b)
        self.S4_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.S4_a = nn.Parameter(torch.ones(16))
        self.S4_b = nn.Parameter(torch.zeros(16))

        # C5: Conv (16x5x5 to 120x1x1)
        self.C5 = nn.Conv2d(16, 120, kernel_size=5)

        # F6: Fully connected (120 to 84)
        self.F6 = nn.Linear(120, 84)

        # RBF output layer using DIGIT bitmap codes
        # mu_codes: shape (10, 84)
        self.mu = nn.Parameter(mu_codes, requires_grad=False)



    #  Forward pass: returns 10 squared Euclidean distances
    def forward(self, x):

        # C1
        x = self.tanh(self.C1(x))

        # S2
        x = self.S2_pool(x)
        x = self.tanh(self.S2_a.view(1,6,1,1) * x + self.S2_b.view(1,6,1,1))

        # C3
        x = self.tanh(self.C3(x))

        # S4
        x = self.S4_pool(x)
        x = self.tanh(self.S4_a.view(1,16,1,1) * x + self.S4_b.view(1,16,1,1))

        # C5
        x = self.tanh(self.C5(x))
        x = torch.flatten(x, start_dim=1)

        # F6
        h = self.tanh(self.F6(x))

        # RBF output: squared Euclidean distances to 10 code vectors
        # h: (batch, 84)
        # mu: (10, 84)
        # output distances: (batch, 10)
        D = torch.sum((h.unsqueeze(1) - self.mu.unsqueeze(0)) ** 2, dim=2)

        return D

def build_lenet5():
    mu_codes = load_digit_codes()
    return LeNet5(mu_codes)

def visualize_digit_codes(save_path='digit_bitmaps.png'):
    """
    Visualize the 12x7 bitmap prototypes for each digit (0-9).
    Saves a figure showing all 10 digit prototypes.
    """
    
    mu_codes = load_digit_codes()
    
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    fig.suptitle('LeNet5 RBF Layer: 12x7 Digit Prototypes', fontsize=16)
    
    for digit in range(10):
        row = digit // 5
        col = digit % 5
        ax = axes[row, col]
        
        # Reshape from 84-d vector to 12x7 image
        bitmap = mu_codes[digit].reshape(12, 7).numpy()
        
        # Display
        im = ax.imshow(bitmap, cmap='gray_r', interpolation='nearest')
        ax.set_title(f'Digit {digit}', fontsize=12)
        ax.axis('off')
        
        # Add colorbar to show value range
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Digit bitmaps saved to {save_path}")
    plt.show()
    
    return mu_codes

visualize_digit_codes()