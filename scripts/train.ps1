param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Overrides
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $venvPython)) {
    throw "Virtual environment not found. Run .\scripts\setup.ps1 first."
}

Push-Location $projectRoot

Write-Host "Preparing MNIST PNG files and metadata"
& $venvPython (Join-Path $projectRoot "scripts\prepare_data.py")

Write-Host "Starting training"
& $venvPython (Join-Path $projectRoot "train.py") @Overrides

Pop-Location
