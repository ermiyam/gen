# Create scheduled task for daily training
$action = New-ScheduledTaskAction -Execute "python" -Argument "src/learn.py" -WorkingDirectory (Get-Location)
$trigger = New-ScheduledTaskTrigger -Daily -At 2AM
$principal = New-ScheduledTaskPrincipal -UserId "SYSTEM" -LogonType ServiceAccount -RunLevel Highest
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -DontStopOnIdleEnd -RestartInterval (New-TimeSpan -Minutes 1) -RestartCount 3

# Register the task
Register-ScheduledTask -TaskName "MAK Daily Training" -Action $action -Trigger $trigger -Principal $principal -Settings $settings -Description "Runs MAK training daily at 2 AM"

Write-Host "✅ Scheduled task created successfully!"
Write-Host "📅 MAK will train daily at 2 AM"
Write-Host "📊 Training metrics will be saved to logs/"
Write-Host "💾 Model versions will be saved to models/v1, v2, etc." 