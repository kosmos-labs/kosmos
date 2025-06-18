#!/usr/bin/env python3
"""
PyTorch I/O Intensive Training Script
Synthetic training loop with large dataset for baseline performance measurement
"""

import os
import time
import json
import psutil
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import threading
import subprocess
import GPUtil
from collections import defaultdict

class SyntheticImageDataset(Dataset):
    """Synthetic dataset that simulates ImageNet-like data loading"""
    
    def __init__(self, num_samples=100000, image_size=224, num_classes=1000):
        self.num_samples = num_samples
        self.image_size = image_size
        self.num_classes = num_classes
        
        # Generate synthetic data
        print(f"Generating {num_samples} synthetic images...")
        self.images = torch.randn(num_samples, 3, image_size, image_size)
        self.labels = torch.randint(0, num_classes, (num_samples,))
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Simulate disk I/O by accessing data
        return self.images[idx], self.labels[idx]

class SimpleCNN(nn.Module):
    """Simple CNN for training"""
    
    def __init__(self, num_classes=1000):
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
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class PerformanceMonitor:
    """Monitor system performance metrics"""
    
    def __init__(self):
        self.metrics = defaultdict(list)
        self.running = False
        self.monitor_thread = None
    
    def start_monitoring(self):
        """Start monitoring in background thread"""
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join()
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                # CPU and I/O metrics
                cpu_percent = psutil.cpu_percent(interval=1)
                io_counters = psutil.disk_io_counters()
                
                # GPU metrics
                gpu_util = 0
                try:
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        gpu_util = gpus[0].load * 100
                except:
                    pass
                
                # Memory metrics
                memory = psutil.virtual_memory()
                
                # Store metrics
                timestamp = time.time()
                self.metrics['timestamp'].append(timestamp)
                self.metrics['cpu_percent'].append(cpu_percent)
                self.metrics['gpu_util'].append(gpu_util)
                self.metrics['memory_percent'].append(memory.percent)
                self.metrics['disk_read_bytes'].append(io_counters.read_bytes if io_counters else 0)
                self.metrics['disk_write_bytes'].append(io_counters.write_bytes if io_counters else 0)
                
                time.sleep(1)
            except Exception as e:
                print(f"Monitoring error: {e}")
                time.sleep(1)
    
    def get_average_metrics(self):
        """Get average metrics over monitoring period"""
        if not self.metrics['timestamp']:
            return {}
        
        return {
            'avg_cpu_percent': np.mean(self.metrics['cpu_percent']),
            'avg_gpu_util': np.mean(self.metrics['gpu_util']),
            'avg_memory_percent': np.mean(self.metrics['memory_percent']),
            'total_disk_read_mb': (self.metrics['disk_read_bytes'][-1] - self.metrics['disk_read_bytes'][0]) / (1024*1024),
            'total_disk_write_mb': (self.metrics['disk_write_bytes'][-1] - self.metrics['disk_write_bytes'][0]) / (1024*1024),
            'monitoring_duration': self.metrics['timestamp'][-1] - self.metrics['timestamp'][0]
        }

def measure_disk_throughput():
    """Measure disk throughput using dd command"""
    try:
        # Create a temporary file for testing
        test_file = "/tmp/disk_test"
        result = subprocess.run([
            'dd', 'if=/dev/zero', f'of={test_file}', 'bs=1M', 'count=100', 'conv=fdatasync'
        ], capture_output=True, text=True, timeout=30)
        
        # Parse output to get throughput
        if result.returncode == 0:
            # Extract time from dd output
            lines = result.stderr.split('\n')
            for line in lines:
                if 'bytes transferred' in line:
                    # Parse the throughput info
                    return "Disk throughput test completed"
        
        return "Disk throughput test failed"
    except Exception as e:
        return f"Disk throughput test error: {e}"

def main():
    # Configuration
    config = {
        'batch_size': 32,
        'num_epochs': 5,
        'learning_rate': 0.001,
        'num_samples': 50000,  # Large dataset
        'num_workers': 4,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    print("=== PyTorch I/O Intensive Training Baseline ===")
    print(f"Device: {config['device']}")
    print(f"Dataset size: {config['num_samples']} samples")
    print(f"Batch size: {config['batch_size']}")
    print(f"Epochs: {config['num_epochs']}")
    print(f"DataLoader workers: {config['num_workers']}")
    print()
    
    # Initialize performance monitor
    monitor = PerformanceMonitor()
    
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
    
    # Start monitoring
    print("Starting performance monitoring...")
    monitor.start_monitoring()
    
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
    
    # Stop monitoring
    monitor.stop_monitoring()
    
    # Get performance metrics
    perf_metrics = monitor.get_average_metrics()
    
    # Measure disk throughput
    disk_throughput = measure_disk_throughput()
    
    # Calculate data loader throughput
    total_samples = config['num_samples'] * config['num_epochs']
    avg_data_loading_time = np.mean(training_metrics['data_loading_times'])
    data_loader_throughput = config['batch_size'] / avg_data_loading_time if avg_data_loading_time > 0 else 0
    
    # Compile results
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
        'performance_metrics': perf_metrics,
        'disk_throughput_test': disk_throughput
    }
    
    # Save results
    output_file = "baseline_training_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n=== TRAINING COMPLETED ===")
    print(f"Total wall-clock time: {total_training_time:.2f}s")
    print(f"Average epoch time: {np.mean(training_metrics['epoch_times']):.2f}s")
    print(f"Average batch time: {np.mean(training_metrics['batch_times']):.3f}s")
    print(f"Data loader throughput: {data_loader_throughput:.1f} samples/sec")
    print(f"Average GPU utilization: {perf_metrics.get('avg_gpu_util', 0):.1f}%")
    print(f"Average CPU utilization: {perf_metrics.get('avg_cpu_percent', 0):.1f}%")
    print(f"Total disk read: {perf_metrics.get('total_disk_read_mb', 0):.1f} MB")
    print(f"Total disk write: {perf_metrics.get('total_disk_write_mb', 0):.1f} MB")
    print(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    main()