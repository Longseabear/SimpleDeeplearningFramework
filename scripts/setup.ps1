param()

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"

Push-Location $projectRoot

if (-not (Test-Path $venvPython)) {
    Write-Host "Creating virtual environment at .venv"
    python -m venv (Join-Path $projectRoot ".venv")
}

Write-Host "Installing Python packages from requirements.txt"
& $venvPython -m pip install --upgrade pip
& $venvPython -m pip install -r (Join-Path $projectRoot "requirements.txt")

Write-Host "Setup completed. Python executable:"
& $venvPython -c "import sys; print(sys.executable)"

Pop-Location
