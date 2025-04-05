#!/bin/bash

# Set up environment
echo "Setting up MAK cloud training environment..."

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Check if Google Drive is mounted
if [ ! -d "/workspace/mak_gdrive" ]; then
    echo "Google Drive not mounted. Mounting..."
    rclone mount gdrive:/mak /workspace/mak_gdrive --vfs-cache-mode writes &
    sleep 5  # Wait for mount to complete
fi

# Create necessary directories in Google Drive
mkdir -p /workspace/mak_gdrive/logs
mkdir -p /workspace/mak_gdrive/chroma_db
mkdir -p /workspace/mak_gdrive/models
mkdir -p /workspace/mak_gdrive/scores

# Start GPU monitoring in background
echo "Starting GPU monitoring..."
nvidia-smi dmon -i 0 -s mu -d 5 -o TD > /workspace/mak_gdrive/logs/gpu_usage.log &

# Start training with model scoring
echo "Starting MAK training with model scoring..."
python src/auto_trainer.py

# Generate training report
echo "Generating training report..."
python -c "
import json
from datetime import datetime
import torch

# Get GPU info
gpu_info = {
    'name': torch.cuda.get_device_name(0),
    'memory_total': torch.cuda.get_device_properties(0).total_memory / 1024**3,
    'memory_allocated': torch.cuda.memory_allocated(0) / 1024**3,
    'memory_reserved': torch.cuda.memory_reserved(0) / 1024**3
}

# Create report
report = {
    'timestamp': datetime.now().isoformat(),
    'gpu_info': gpu_info,
    'training_status': 'running',
    'last_checkpoint': datetime.now().isoformat()
}

# Save report
with open('/workspace/mak_gdrive/logs/training_report.md', 'w') as f:
    f.write('# MAK Training Report\n\n')
    f.write(f'## GPU Information\n')
    f.write(f'- Device: {gpu_info["name"]}\n')
    f.write(f'- Total Memory: {gpu_info["memory_total"]:.2f} GB\n')
    f.write(f'- Allocated Memory: {gpu_info["memory_allocated"]:.2f} GB\n')
    f.write(f'- Reserved Memory: {gpu_info["memory_reserved"]:.2f} GB\n\n')
    f.write(f'## Training Status\n')
    f.write(f'- Status: {report["training_status"]}\n')
    f.write(f'- Last Checkpoint: {report["last_checkpoint"]}\n')
"

# Keep script running
while true; do
    sleep 60
done 