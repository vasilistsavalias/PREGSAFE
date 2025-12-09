# This script resets the project to a clean state by deleting all generated files.

$Force = $false
foreach ($arg in $args) {
    if ($arg -eq "--force") {
        $Force = $true
    }
}

Write-Host "====================================================="
Write-Host " GDM Project Cleaner (PowerShell)"
Write-Host "====================================================="

# --- Confirmation ---
if (-not $Force) {
    $response = Read-Host "This will permanently delete the /outputs and /logs directories. Are you sure? (y/n)"
    if ($response -ne 'y') {
        Write-Host "Operation cancelled."
        exit 1
    }
}

# --- Deletion ---
Write-Host "Deleting /outputs directory..."
if (Test-Path -Path "outputs") {
    Remove-Item -Recurse -Force "outputs"
}

Write-Host "Deleting /logs directory..."
if (Test-Path -Path "logs") {
    Remove-Item -Recurse -Force "logs"
}

Write-Host "Deleting temporary Python cache files..."
Get-ChildItem -Path . -Include "__pycache__" -Recurse -Directory | Remove-Item -Recurse -Force

Write-Host "====================================================="
Write-Host " Project has been cleaned."
Write-Host "====================================================="