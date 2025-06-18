#!/usr/bin/env python3
"""
PyTorch I/O Intensive Training Script
Synthetic training loop with large dataset for baseline performance measurement
Performance monitoring now handled by eBPF agent
"""

import os
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

class SyntheticImageDataset(Dataset):
    """Synthetic dataset that simulates ImageNet-like data loading"""
    
    def __init__(self, num_samples=100000, image_size=224, num_classes=1000):
        """
        Initialize the SyntheticImageDataset with synthetic image data.

        Args:
            num_samples (int): Number of synthetic samples to generate. Default is 100,000.
            image_size (int): Size of the synthetic images (height and width). Default is 224.
            num_classes (int): Number of distinct classes for labels. Default is 1,000.

        Attributes:
            images (torch.Tensor): Tensor containing the synthetic image data with shape (num_samples, 3, image_size, image_size).
            labels (torch.Tensor): Tensor containing the labels for each image with shape (num_samples,).
        """

        self.num_samples = num_samples
        self.image_size = image_size
        self.num_classes = num_classes
        
        # Generate synthetic data
        print(f"Generating {num_samples} synthetic images...")
        self.images = torch.randn(num_samples, 3, image_size, image_size)
        self.labels = torch.randint(0, num_classes, (num_samples,))
        
    def __len__(self):
        """
        Returns the number of samples in the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        return self.num_samples
    
    def __getitem__(self, idx):
        # Simulate disk I/O by accessing data
        """
        Returns a tuple containing the synthetic image and label at the given index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: A tuple containing the image and label for the given index.
        """
        return self.images[idx], self.labels[idx]

class SimpleCNN(nn.Module):
    """Simple CNN for training"""
    
    def __init__(self, num_classes=1000):
        """
        Initialize the SimpleCNN model.

        Args:
            num_classes (int): The number of output classes for the classifier. Default is 1000.

        The model consists of:
        - A feature extraction part with three convolutional layers, each followed by a ReLU activation and max pooling.
        - A classifier that maps the extracted features to the specified number of classes.
        """

        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Linear(256, num_classes)
    
    def forward(self, x):
        """
        Forward pass of the SimpleCNN model.

        Args:
            x (torch.Tensor): The input tensor, of shape (B, C, H, W), where B is the batch size, C is the number of channels, H is the height, and W is the width.

        Returns:
            torch.Tensor: The output tensor, of shape (B, num_classes).

        Notes:
            - The input tensor is passed through the feature extraction part of the model, which consists of three convolutional layers, each followed by a ReLU activation and max pooling.
            - The output of the feature extraction part is then flattened and passed through the classifier, which consists of a single fully connected layer.
            - The output of the classifier is the final output of the model.
        """
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def main():
    # Configuration
    """
    Main entry point for PyTorch I/O Intensive Training (eBPF Monitored).

    This script trains a simple convolutional neural network on a synthetic dataset
    using PyTorch. The dataset is generated on the fly and consists of 50,000 samples.
    The model is trained for 5 epochs using the Adam optimizer with a learning rate of
    0.001. The dataset is loaded using a DataLoader with a batch size of 32 and 4
    workers.

    The script also prints out basic training metrics, such as the total wall-clock time,
    average epoch time, average batch time, data loader throughput, and total samples
    processed. The results are saved to a file named "training_results.json".

    Note that the I/O performance metrics are collected by an eBPF agent, which is
    launched separately. The agent logs the I/O metrics to the console, so you can
    check the agent logs to see the detailed I/O performance metrics.
    """
    config = {
        'batch_size': 32,
        'num_epochs': 5,
        'learning_rate': 0.001,
        'num_samples': 50000,  # Large dataset
        'num_workers': 4,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    print("=== PyTorch I/O Intensive Training (eBPF Monitored) ===")
    print(f"Device: {config['device']}")
    print(f"Dataset size: {config['num_samples']} samples")
    print(f"Batch size: {config['batch_size']}")
    print(f"Epochs: {config['num_epochs']}")
    print(f"DataLoader workers: {config['num_workers']}")
    print("Note: I/O performance monitoring handled by eBPF agent")
    print()
    
    # Create dataset and dataloader
    print("Creating synthetic dataset...")
    dataset = SyntheticImageDataset(num_samples=config['num_samples'])
    dataloader = DataLoader(
        dataset, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    # Create model
    model = SimpleCNN().to(config['device'])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # Training metrics
    training_metrics = {
        'epoch_times': [],
        'batch_times': [],
        'data_loading_times': [],
        'forward_times': [],
        'backward_times': []
    }
    
    # Training loop
    print("Starting training...")
    total_start_time = time.time()
    
    for epoch in range(config['num_epochs']):
        epoch_start_time = time.time()
        model.train()
        
        for batch_idx, (data, target) in enumerate(dataloader):
            batch_start_time = time.time()
            
            # Move data to device
            data = data.to(config['device'])
            target = target.to(config['device'])
            
            data_loading_time = time.time() - batch_start_time
            
            # Forward pass
            forward_start = time.time()
            output = model(data)
            forward_time = time.time() - forward_start
            
            # Backward pass
            backward_start = time.time()
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            backward_time = time.time() - backward_start
            
            # Record times
            batch_time = time.time() - batch_start_time
            training_metrics['batch_times'].append(batch_time)
            training_metrics['data_loading_times'].append(data_loading_time)
            training_metrics['forward_times'].append(forward_time)
            training_metrics['backward_times'].append(backward_time)
            
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1}/{config['num_epochs']}, "
                      f"Batch {batch_idx}/{len(dataloader)}, "
                      f"Loss: {loss.item():.4f}, "
                      f"Batch time: {batch_time:.3f}s")
        
        epoch_time = time.time() - epoch_start_time
        training_metrics['epoch_times'].append(epoch_time)
        print(f"Epoch {epoch+1} completed in {epoch_time:.2f}s")
    
    total_training_time = time.time() - total_start_time
    
    # Calculate basic metrics
    total_samples = config['num_samples'] * config['num_epochs']
    avg_data_loading_time = np.mean(training_metrics['data_loading_times'])
    data_loader_throughput = config['batch_size'] / avg_data_loading_time if avg_data_loading_time > 0 else 0
    
    # Compile results (basic training metrics only)
    results = {
        'config': config,
        'total_wall_clock_time': total_training_time,
        'avg_epoch_time': np.mean(training_metrics['epoch_times']),
        'avg_batch_time': np.mean(training_metrics['batch_times']),
        'avg_data_loading_time': avg_data_loading_time,
        'avg_forward_time': np.mean(training_metrics['forward_times']),
        'avg_backward_time': np.mean(training_metrics['backward_times']),
        'data_loader_throughput_samples_per_sec': data_loader_throughput,
        'total_samples_processed': total_samples,
        'note': 'I/O performance metrics collected by eBPF agent'
    }
    
    # Save results
    output_file = "training_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n=== TRAINING COMPLETED ===")
    print(f"Total wall-clock time: {total_training_time:.2f}s")
    print(f"Average epoch time: {np.mean(training_metrics['epoch_times']):.2f}s")
    print(f"Average batch time: {np.mean(training_metrics['batch_times']):.3f}s")
    print(f"Data loader throughput: {data_loader_throughput:.1f} samples/sec")
    print(f"\nResults saved to: {output_file}")
    print("Check eBPF agent logs for detailed I/O performance metrics")

if __name__ == "__main__":
    main()