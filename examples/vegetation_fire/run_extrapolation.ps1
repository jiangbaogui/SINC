[CmdletBinding()]
param(
    [switch]$SkipTraining,
    [ValidateSet("cuda", "cuda:0")]
    [string]$Device = "cuda"
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$modelPath = Join-Path $repoRoot "outputs\vegetation_fire\extrapolation\sinc_weights.gndc"

function Invoke-Python {
    param([Parameter(Mandatory = $true)][string[]]$Arguments)
    & python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed with exit code $LASTEXITCODE."
    }
}

Push-Location $repoRoot
try {
    if (-not $SkipTraining) {
        Write-Host "[1/2] Training the chronologically truncated baseline..." -ForegroundColor Cyan
        Invoke-Python @(
            "-m", "sinc.train",
            "--config", "configs/vegetation_fire_extrapolation.yaml"
        )
    }
    else {
        Write-Host "[1/2] Training skipped; using the existing extrapolation model." -ForegroundColor Yellow
    }

    if (-not (Test-Path -LiteralPath $modelPath)) {
        throw "Model not found: $modelPath. Run without -SkipTraining first."
    }

    Write-Host "[2/2] Evaluating the 22 held-out pre-event observations..." -ForegroundColor Cyan
    Invoke-Python @(
        "examples/vegetation_fire/evaluate_extrapolation.py",
        "--model", $modelPath,
        "--history-dir", "data/Vegetation Fire/train",
        "--after-date", "2020-11-05",
        "--through-date", "2021-07-08",
        "--output", "outputs/vegetation_fire/extrapolation/metrics.csv",
        "--device", $Device
    )
}
finally {
    Pop-Location
}
