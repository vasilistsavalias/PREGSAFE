# run_pipeline.ps1
# Orchestrates the full GDM Pipeline (Folds + Final)

# Parse arguments manually for POSIX-style support (--smoke-test, --tune)
$SmokeTest = $false
$Tune = $false

foreach ($arg in $args) {
    switch ($arg) {
        "--smoke-test" { $SmokeTest = $true }
        "--tune" { $Tune = $true }
    }
}

$ArgsList = @()
if ($SmokeTest) {
    $ArgsList += "--smoke-test"
    Write-Host "--- RUNNING PIPELINE IN SMOKE TEST MODE ---" -ForegroundColor Yellow
}
if ($Tune) {
    $ArgsList += "--tune"
    Write-Host "--- TUNING ENABLED: Will run Optuna Tuning Stage for Folds ---" -ForegroundColor Magenta
} else {
    Write-Host "--- RUNNING PIPELINE IN FULL PRODUCTION MODE (Baseline Generation) ---" -ForegroundColor Green
}

# Set Project Root
$ProjectRoot = "$PSScriptRoot/.."
cd $ProjectRoot

# Ensure module is installed (editable mode)
pip install -e . > $null 2>&1

# Run Python Orchestrator
Write-Host "Starting main.py..." -ForegroundColor Cyan
python main.py $ArgsList

if ($LASTEXITCODE -ne 0) {
    Write-Host "Pipeline Failed!" -ForegroundColor Red
    exit 1
}

Write-Host "Pipeline Completed Successfully!" -ForegroundColor Green
