$ErrorActionPreference = "Stop"

function Exit-WithFailure {
    param(
        [string]$Message,
        [int]$Code = 1
    )

    Write-Host ""
    Write-Host "Startup failed." -ForegroundColor Red
    Write-Host $Message
    Read-Host "Press Enter to exit"
    exit $Code
}

function Get-LocalIpv4Address {
    $getNetIPAddress = Get-Command Get-NetIPAddress -ErrorAction SilentlyContinue
    if (-not $getNetIPAddress) {
        return $null
    }

    $addresses = Get-NetIPAddress -AddressFamily IPv4 -ErrorAction SilentlyContinue |
        Where-Object {
            $_.IPAddress -ne "127.0.0.1" -and
            $_.IPAddress -notlike "169.254.*" -and
            $_.PrefixOrigin -ne "WellKnown"
        } |
        Sort-Object InterfaceMetric |
        Select-Object -ExpandProperty IPAddress -First 1

    return $addresses
}

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$pythonExe = Join-Path $projectRoot ".venv311\Scripts\python.exe"
$appPath = Join-Path $projectRoot "backend\app.py"
$port = 5000
$entryPath = "/platform"
$entryUrlSuffix = if ($entryPath -eq "/") { "/" } else { "$entryPath/" }
$localhostUrl = "http://127.0.0.1:$port$entryUrlSuffix"
$localIpAddress = Get-LocalIpv4Address
$lanUrl = if ($localIpAddress) { "http://$localIpAddress`:$port$entryUrlSuffix" } else { $null }

if (-not (Test-Path $pythonExe)) {
    Exit-WithFailure "Missing interpreter: $pythonExe"
}

if (-not $env:INFERENCE_DEVICE) {
    $env:INFERENCE_DEVICE = "auto"
}

$existingBackendProcesses = @()
try {
    $existingBackendProcesses = Get-CimInstance Win32_Process |
        Where-Object { $_.Name -match 'python.*\.exe$' -and $_.CommandLine -like "*$appPath*" }
} catch {
    Write-Warning "Unable to enumerate existing backend processes: $($_.Exception.Message)"
}

foreach ($process in $existingBackendProcesses) {
    Write-Host "Stopping existing backend process $($process.ProcessId): $($process.ExecutablePath)"
    Stop-Process -Id $process.ProcessId -Force -ErrorAction SilentlyContinue
}

Write-Host "INFERENCE_DEVICE=$env:INFERENCE_DEVICE"
Write-Host "Use 'cuda' to require GPU, 'cpu' to force CPU, or keep 'auto' to pick automatically."

$dependencyCheckScript = Join-Path $env:TEMP "house_upload_system_dependency_check.py"
$dependencyCheckLines = @(
    'import cv2, torch, torchvision, segmentation_models_pytorch, flask, flask_sqlalchemy'
    'print("Environment OK")'
    'print(f"torch={torch.__version__}")'
    'print(f"torch_cuda_build={torch.version.cuda or ''cpu-only''}")'
    'print(f"cuda_available={torch.cuda.is_available()}")'
    'print(f"device_count={torch.cuda.device_count()}")'
    'print(f"device_name={torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''N/A''}")'
)

Set-Content -LiteralPath $dependencyCheckScript -Value $dependencyCheckLines -Encoding UTF8
& $pythonExe $dependencyCheckScript
$dependencyExitCode = $LASTEXITCODE
Remove-Item -LiteralPath $dependencyCheckScript -Force -ErrorAction SilentlyContinue

if ($dependencyExitCode -ne 0) {
    Exit-WithFailure "Dependency check failed in .venv311."
}

Write-Host ""
Write-Host "Backend is starting. Keep this terminal open while using the web page." -ForegroundColor Cyan
Write-Host "Copy this URL into the browser on this computer:" -ForegroundColor Cyan
Write-Host $localhostUrl -ForegroundColor Green
if ($lanUrl) {
    Write-Host "If you open it from another device on the same network, use:" -ForegroundColor Cyan
    Write-Host $lanUrl -ForegroundColor Green
}
Write-Host "Press Ctrl+C in this terminal when you want to stop the backend." -ForegroundColor Yellow
Write-Host ""

Set-Location $projectRoot
$env:HOUSE_UPLOAD_HOST = "0.0.0.0"
$env:HOUSE_UPLOAD_PORT = "$port"
$env:HOUSE_UPLOAD_ENTRY_PATH = $entryPath
& $pythonExe $appPath

if ($LASTEXITCODE -ne 0) {
    Exit-WithFailure "Backend stopped with exit code $LASTEXITCODE." $LASTEXITCODE
}
