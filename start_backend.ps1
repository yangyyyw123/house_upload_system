$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$pythonExe = Join-Path $projectRoot ".venv311\Scripts\python.exe"
$appPath = Join-Path $projectRoot "backend\app.py"

if (-not (Test-Path $pythonExe)) {
    Write-Error "Missing interpreter: $pythonExe"
}

$existingBackendProcesses = Get-CimInstance Win32_Process |
    Where-Object { $_.Name -match '^python.*\.exe$' -and $_.CommandLine -like "*$appPath*" }

foreach ($process in $existingBackendProcesses) {
    Write-Host "Stopping existing backend process $($process.ProcessId): $($process.ExecutablePath)"
    Stop-Process -Id $process.ProcessId -Force
}

& $pythonExe -c "import cv2, torch, torchvision, segmentation_models_pytorch, flask, flask_sqlalchemy; print('Environment OK')"
if ($LASTEXITCODE -ne 0) {
    Write-Error "Dependency check failed in .venv311."
}

Write-Host "Starting backend with $pythonExe"
& $pythonExe $appPath
