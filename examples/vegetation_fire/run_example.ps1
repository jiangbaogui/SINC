[CmdletBinding()]
param(
    [switch]$SkipTraining,
    [switch]$ExportVector,
    [ValidateSet("cuda", "cuda:0")]
    [string]$Device = "cuda",
    [ValidateRange(0.0001, 0.1)]
    [double]$Alpha = 0.01
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$modelPath = Join-Path $repoRoot "outputs\vegetation_fire\sinc_weights.gndc"
$calibratedModelPath = Join-Path $repoRoot "outputs\vegetation_fire\sinc_calibrated.gndc"
$profilePath = Join-Path $repoRoot "outputs\vegetation_fire\thresholds.json"
$detectionDir = Join-Path $repoRoot "outputs\vegetation_fire\detection"

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
        Write-Host "[1/3] Training the full pre-event SINC baseline..." -ForegroundColor Cyan
        Invoke-Python @("-m", "sinc.train", "--config", "configs/vegetation_fire.yaml")
    }
    else {
        Write-Host "[1/3] Training skipped; using the existing model." -ForegroundColor Yellow
    }

    if (-not (Test-Path -LiteralPath $modelPath)) {
        throw "Model not found: $modelPath. Run without -SkipTraining first."
    }

    Write-Host "[2/3] Calibrating GNDC-conditioned empirical-null thresholds..." -ForegroundColor Cyan
    Invoke-Python @(
        "-m", "sinc.calibrate",
        "--model", $modelPath,
        "--normal-dir", "data/Vegetation Fire/train",
        "--output", $profilePath,
        "--calibrated-model", $calibratedModelPath,
        "--alpha", $Alpha.ToString([Globalization.CultureInfo]::InvariantCulture),
        "--device", $Device,
        "--seed", "42"
    )

    Write-Host "[3/3] Detecting the target observation..." -ForegroundColor Cyan
    $detectArgs = @(
        "-m", "sinc.detect",
        "--model", $calibratedModelPath,
        "--input", "data/Vegetation Fire/target/S2_JZ_2021-09-11_1903.tif",
        "--output", $detectionDir,
        "--device", $Device,
        "--pixel-size-m", "20"
    )
    if ($ExportVector) {
        $detectArgs += "--export-vector"
    }
    Invoke-Python $detectArgs

    Write-Host "Completed. Results: $detectionDir" -ForegroundColor Green
}
finally {
    Pop-Location
}
