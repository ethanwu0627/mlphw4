import sys
sys.path.append('lenet1')

from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import seaborn as sns
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
        image_raw = np.array(image)  # Keep raw for visualization
        image = torch.from_numpy(image_raw).unsqueeze(0).float()
        
        # Apply transform
        if self.transform:
            image = self.transform(image)
        
        return image, label, image_raw

def test(dataloader, model, device):
    """
    Test the model and compute:
    1. Test accuracy
    2. Confusion matrix (10x10)
    3. Most confusing examples for each digit
    """
    model.eval()
    correct = 0
    total = 0
    
    # For confusion matrix
    confusion_matrix = np.zeros((10, 10), dtype=int)
    
    # Track misclassifications: digit -> list of (confidence, predicted_class, image, true_label, idx)
    misclassifications = {i: [] for i in range(10)}
    
    with torch.no_grad():
        for batch_idx, (images, labels, images_raw) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass - get distances
            distances = model(images)
            
            # Predicted class = minimum distance
            predicted = torch.argmin(distances, dim=1)
            
            # Update accuracy
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
            # Update confusion matrix
            for true_label, pred_label in zip(labels.cpu().numpy(), predicted.cpu().numpy()):
                confusion_matrix[true_label, pred_label] += 1
            
            # Track misclassifications with confidence (negative distance to predicted class)
            for i in range(len(labels)):
                true_label = labels[i].item()
                pred_label = predicted[i].item()
                
                if true_label != pred_label:
                    # Confidence = how small the distance is to the predicted class
                    # We want high confidence mistakes (very small distance to wrong class)
                    confidence = -distances[i, pred_label].item()
                    
                    misclassifications[true_label].append(
                        (confidence, pred_label, images_raw[i].numpy(), true_label, batch_idx)
                    )
    
    test_accuracy = correct / total
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Test error rate: {1 - test_accuracy:.4f}")
    
    return test_accuracy, confusion_matrix, misclassifications

def plot_confusion_matrix(confusion_matrix, save_path='lenet1_confusion_matrix.png'):
    """Plot and save confusion matrix as percentages."""
    # Normalize by row (true labels) to get percentages
    confusion_matrix_pct = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1, keepdims=True)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(confusion_matrix_pct, annot=True, fmt='.3f', cmap='Blues', 
                xticklabels=range(10), yticklabels=range(10),
                cbar_kws={'label': 'Percentage'}, vmin=0, vmax=1)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title('Confusion Matrix (10×10) - Normalized', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f'Confusion matrix saved to {save_path}')
    plt.show()

def plot_most_confusing_examples(misclassifications, save_path='lenet1_most_confusing.png'):
    """
    For each digit, identify the most confusing example.
    Most confusing = misclassified with highest confidence.
    """
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    fig.suptitle('Most Confusing Examples (Highest Confidence Misclassifications)', fontsize=14)
    
    for digit in range(10):
        row = digit // 5
        col = digit % 5
        ax = axes[row, col]
        
        if misclassifications[digit]:
            # Sort by confidence (descending) - most confident wrong prediction
            misclassifications[digit].sort(reverse=True)
            confidence, pred_class, image, true_label, idx = misclassifications[digit][0]
            
            ax.imshow(image, cmap='gray')
            ax.set_title(f'True: {digit}, Pred: {pred_class}\nConf: {-confidence:.2f}', fontsize=10)
            ax.axis('off')
            
            print(f'Digit {digit}: Most confusing example was misclassified as {pred_class} '
                  f'with confidence {-confidence:.2f}')
        else:
            ax.text(0.5, 0.5, f'No errors\nfor digit {digit}', 
                   ha='center', va='center', fontsize=10)
            ax.axis('off')
            print(f'Digit {digit}: No misclassifications!')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f'\nMost confusing examples saved to {save_path}')
    plt.show()

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}\n')
    
    # Data preprocessing: pad 28x28 to 32x32
    pad = transforms.Pad(2, fill=0, padding_mode='constant')
    
    # Load test dataset
    print('Loading test dataset...')
    mnist_test = MNISTParquet(split='test', transform=pad)
    test_dataloader = DataLoader(mnist_test, batch_size=1, shuffle=False)
    
    # Load trained model
    print('Loading trained model...')
    model = torch.load("LeNet1.pth", map_location=device, weights_only=False)
    model.eval()
    
    # Run evaluation
    print('\nEvaluating model...\n')
    test_accuracy, confusion_matrix, misclassifications = test(test_dataloader, model, device)
    
    # Plot confusion matrix
    print('\n' + '='*50)
    print('Generating confusion matrix...')
    plot_confusion_matrix(confusion_matrix)
    
    # Plot most confusing examples
    print('\n' + '='*50)
    print('Identifying most confusing examples...\n')
    plot_most_confusing_examples(misclassifications)
    
    print('\n' + '='*50)
    print('Evaluation complete!')
    print(f'Final Test Accuracy: {test_accuracy:.4f}')
    print(f'Final Test Error Rate: {1 - test_accuracy:.4f}')

if __name__ == "__main__":
    main()
