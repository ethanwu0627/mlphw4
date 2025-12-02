import sys
sys.path.append('..')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from lenet1 import build_lenet5
from PIL import Image
from io import BytesIO
import pandas as pd

class MNISTParquet(Dataset):
    """MNIST dataset loaded from Hugging Face parquet files."""
    def __init__(self, split='train', transform=None):
        self.transform = transform
        
        # Load from parquet
        splits = {'train': 'mnist/train-00000-of-00001.parquet', 
                  'test': 'mnist/test-00000-of-00001.parquet'}
        self.df = pd.read_parquet(f"hf://datasets/ylecun/mnist/{splits[split]}")
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        label = row['label']
        
        # Get image bytes (handle both nested and flattened structures)
        try:
            image_bytes = row['image']['bytes']
        except (KeyError, TypeError):
            image_bytes = row['image.bytes']
        
        # Load image
        image = Image.open(BytesIO(image_bytes))
        image = torch.from_numpy(np.array(image)).unsqueeze(0).float()
        
        # Apply transform
        if self.transform:
            image = self.transform(image)
        
        return image, label

def rbf_loss(distances, target, j=0.1):
    """
    Custom RBF loss function from LeNet-5 paper equation (9).
    E(W) = (1/P) * sum[ y_Dp(Z^p, W) + log(e^-j + sum_i(e^-y_i(Z^p, W))) ]
    
    Args:
        distances: (batch, 10) - squared Euclidean distances from RBF layer (y_i values)
        target: (batch,) - true class labels (D^p is the correct class)
        j: penalty parameter
    
    Returns:
        loss: scalar loss value
    """
    batch_size = distances.shape[0]
    loss = 0.0
    
    for i in range(batch_size):
        correct_class = target[i].item()
        
        # First term: y_Dp(Z^p, W) - distance to correct class
        y_correct = distances[i, correct_class]
        
        # Second term: log(e^-j + sum_i(e^-y_i))
        # Sum over all classes
        sum_exp = torch.sum(torch.exp(-distances[i]))
        log_term = torch.log(torch.exp(torch.tensor(-j, device=distances.device)) + sum_exp)
        
        # Total loss for this sample
        loss += y_correct + log_term
    
    return loss / batch_size

def train_epoch(model, dataloader, optimizer, device):
    """Train for one epoch and return average loss and error rate."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        distances = model(images)
        
        # Compute loss
        loss = rbf_loss(distances, labels, j=0.1)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Compute accuracy (predicted class = minimum distance)
        predicted = torch.argmin(distances, dim=1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        total_loss += loss.item()
        
        if (batch_idx + 1) % 1000 == 0:
            print(f'  Batch {batch_idx + 1}/{len(dataloader)}, Loss: {loss.item():.4f}')
    
    avg_loss = total_loss / len(dataloader)
    error_rate = 1.0 - (correct / total)
    
    return avg_loss, error_rate

def evaluate(model, dataloader, device):
    """Evaluate model and return error rate."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            distances = model(images)
            predicted = torch.argmin(distances, dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    
    error_rate = 1.0 - (correct / total)
    return error_rate

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Hyperparameters
    batch_size = 1  # As specified in assignment
    learning_rate = 0.001  # c ≈ 0.001
    num_epochs = 20
    
    # Data preprocessing: pad 28x28 to 32x32
    transform = transforms.Pad(2, fill=0, padding_mode='constant')
    
    # Load datasets
    print('Loading datasets...')
    train_dataset = MNISTParquet(split='train', transform=transform)
    test_dataset = MNISTParquet(split='test', transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Build model
    print('Building LeNet5 model...')
    model = build_lenet5()
    model = model.to(device)
    
    # Optimizer: steepest gradient descent (SGD)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    
    # Track error rates for plotting
    train_errors = []
    test_errors = []
    
    # Training loop
    print(f'\nTraining for {num_epochs} epochs...')
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch + 1}/{num_epochs}')
        
        # Train
        train_loss, train_error = train_epoch(model, train_loader, optimizer, device)
        
        # Evaluate on test set
        test_error = evaluate(model, test_loader, device)
        
        # Store error rates
        train_errors.append(train_error)
        test_errors.append(test_error)
        
        print(f'Epoch {epoch + 1}: Train Error = {train_error:.4f}, Test Error = {test_error:.4f}')
    
    # Save model
    model_path = 'LeNet1.pth'
    torch.save(model, model_path)
    print(f'\nModel saved to {model_path}')
    
    # Plot error rates
    plt.figure(figsize=(10, 6))
    epochs_range = range(1, num_epochs + 1)
    plt.plot(epochs_range, train_errors, 'b-', label='Training Error', linewidth=2)
    plt.plot(epochs_range, test_errors, 'r-', label='Test Error', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Error Rate', fontsize=12)
    plt.title('LeNet5 Training and Test Error Rates', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('lenet1_error_rates.png', dpi=150)
    print('Error rate plot saved to lenet1_error_rates.png')
    plt.show()
    
    # Print final results
    print(f'\nFinal Results:')
    print(f'Train Error Rate at Epoch 20: {train_errors[-1]:.4f}')
    print(f'Test Error Rate at Epoch 20: {test_errors[-1]:.4f}')

if __name__ == '__main__':
    main()
