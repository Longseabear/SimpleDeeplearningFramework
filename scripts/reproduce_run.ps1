param(
    [Parameter(Mandatory = $true)]
    [string]$RunDir,

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Overrides
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $venvPython)) {
    throw "Virtual environment not found. Run .\scripts\setup.ps1 first."
}

if (-not (Test-Path $RunDir)) {
    throw "Run directory not found: $RunDir"
}

Push-Location $projectRoot

Write-Host "Reproducing run from saved Hydra config"
& $venvPython (Join-Path $projectRoot "train.py") --config-path $RunDir --config-name config @Overrides

Pop-Location
