$scriptPath = Join-Path $PSScriptRoot "scripts\\windows\\start_backend.ps1"
& $scriptPath @args
exit $LASTEXITCODE
