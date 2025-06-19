#!/bin/bash

echo "Running PyTorch I/O Intensive Baseline Test..."

# Check if running in container or locally
if [ -f /.dockerenv ]; then
    echo "Running in container..."
    python3 train.py
else
    echo "Running locally..."
    # Install dependencies if needed
    pip3 install -r requirements.txt
    
    # Run the training script
    python3 train.py
fi

echo "Baseline test completed. Check baseline_training_results.json for results." 