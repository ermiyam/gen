# Set up environment
Write-Host "Setting up MAK training environment..."

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

# Start training
Write-Host "Starting MAK training..."
python src/auto_trainer.py

# Keep script running
while ($true) {
    Start-Sleep -Seconds 60
} 