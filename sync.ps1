# Azure DevOps Sync Script
Write-Host "Starting Azure DevOps sync process..." -ForegroundColor Cyan

# Pull latest changes
Write-Host "Pulling latest from Azure DevOps..." -ForegroundColor Yellow
git pull

# Add all changes
Write-Host "Adding new changes..." -ForegroundColor Yellow
git add .

# Commit with timestamp
Write-Host "Committing changes..." -ForegroundColor Yellow
$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
git commit -m "Auto-sync update: $timestamp"

# Push changes
Write-Host "Pushing to Azure DevOps..." -ForegroundColor Yellow
git push

Write-Host "Sync complete!" -ForegroundColor Green 