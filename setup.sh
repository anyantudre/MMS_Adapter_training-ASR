#!/bin/bash

# Function to check for GPU availability
check_gpu() {
    gpu_info=$(nvidia-smi)
    if [[ $gpu_info == *"failed"* ]]; then
        echo "Not connected to a GPU"
    else
        echo -e "GPU Information:\n$gpu_info"
    fi
}

# Install required packages
pip install -r requirements.txt

# Check GPU availability
check_gpu


echo "Setup complete!"
