#!/bin/bash

# Set up environment
echo "Setting up MAK training environment..."

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Check if Google Drive is mounted
if [ ! -d "C:/mak_gdrive" ]; then
    echo "Google Drive not mounted. Mounting..."
    rclone mount gdrive:/mak C:/mak_gdrive --vfs-cache-mode writes &
    sleep 5  # Wait for mount to complete
fi

# Create necessary directories in Google Drive
mkdir -p C:/mak_gdrive/logs
mkdir -p C:/mak_gdrive/chroma_db
mkdir -p C:/mak_gdrive/models

# Start training with GPU monitoring
echo "Starting MAK training..."
python src/auto_trainer.py

# Keep script running
while true; do
    sleep 60
done 