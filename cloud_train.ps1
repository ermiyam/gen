# Set up environment
Write-Host "Setting up MAK cloud training environment..."

# Check disk space
$diskSpace = Get-PSDrive C | Select-Object -ExpandProperty Free
if ($diskSpace -lt 10GB) {
    Write-Host "Error: Not enough disk space. Need at least 10GB free space."
    Write-Host "Current free space: $($diskSpace/1GB) GB"
    exit 1
}

# Check if Google Drive is mounted
if (-not (Test-Path "C:\mak_gdrive")) {
    Write-Host "Google Drive not mounted. Mounting..."
    Start-Process -FilePath "C:\Users\ermiy\Downloads\rclone-v1.69.1-windows-amd64\rclone-v1.69.1-windows-amd64\rclone.exe" -ArgumentList "mount gdrive:/mak C:\mak_gdrive --vfs-cache-mode writes" -NoNewWindow
    Start-Sleep -Seconds 5  # Wait for mount to complete
}

# Create necessary directories in Google Drive
New-Item -ItemType Directory -Force -Path "C:\mak_gdrive\logs"
New-Item -ItemType Directory -Force -Path "C:\mak_gdrive\chroma_db"
New-Item -ItemType Directory -Force -Path "C:\mak_gdrive\models"
New-Item -ItemType Directory -Force -Path "C:\mak_gdrive\scores"

# Start GPU monitoring in background
Write-Host "Starting GPU monitoring..."
Start-Process -FilePath "nvidia-smi" -ArgumentList "dmon -i 0 -s mu -d 5 -o TD" -RedirectStandardOutput "C:\mak_gdrive\logs\gpu_usage.log" -NoNewWindow

# Install/update required packages with specific versions
Write-Host "Installing required packages..."

# First, uninstall existing packages to avoid conflicts
pip uninstall -y torch torchvision transformers accelerate bitsandbytes huggingface-hub

# Install packages in correct order with specific versions
pip install --upgrade huggingface-hub==0.21.0
pip install --upgrade torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
pip install --upgrade transformers==4.35.0
pip install --upgrade accelerate==0.24.1
pip install --upgrade bitsandbytes==0.41.1
pip install --upgrade datasets==2.14.5

# Start training with model scoring
Write-Host "Starting MAK training with model scoring..."
python src/auto_trainer.py

# Generate training report
Write-Host "Generating training report..."
$reportScript = @"
import json
from datetime import datetime
import torch

try:
    # Get GPU info
    gpu_info = {
        'name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU available',
        'memory_total': torch.cuda.get_device_properties(0).total_memory / 1024**3 if torch.cuda.is_available() else 0,
        'memory_allocated': torch.cuda.memory_allocated(0) / 1024**3 if torch.cuda.is_available() else 0,
        'memory_reserved': torch.cuda.memory_reserved(0) / 1024**3 if torch.cuda.is_available() else 0
    }
except Exception as e:
    gpu_info = {
        'name': 'No GPU available',
        'memory_total': 0,
        'memory_allocated': 0,
        'memory_reserved': 0
    }

# Create report
report = {
    'timestamp': datetime.now().isoformat(),
    'gpu_info': gpu_info,
    'training_status': 'running',
    'last_checkpoint': datetime.now().isoformat()
}

# Save report
with open(r'C:\mak_gdrive\logs\training_report.md', 'w') as f:
    f.write('# MAK Training Report\n\n')
    f.write('## GPU Information\n')
    f.write(f'- Device: {gpu_info["name"]}\n')
    f.write(f'- Total Memory: {gpu_info["memory_total"]:.2f} GB\n')
    f.write(f'- Allocated Memory: {gpu_info["memory_allocated"]:.2f} GB\n')
    f.write(f'- Reserved Memory: {gpu_info["memory_reserved"]:.2f} GB\n\n')
    f.write('## Training Status\n')
    f.write(f'- Status: {report["training_status"]}\n')
    f.write(f'- Last Checkpoint: {report["last_checkpoint"]}\n')
"@

python -c $reportScript

# Keep script running
while ($true) {
    Start-Sleep -Seconds 60
} 