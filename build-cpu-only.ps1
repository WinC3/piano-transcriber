# Build script for creating optimized PyInstaller distribution
# This creates a CPU-only version to drastically reduce file size

Write-Host "Creating CPU-only distribution build..." -ForegroundColor Green

# Clean previous builds
if (Test-Path "dist") { Remove-Item -Recurse -Force "dist" }
if (Test-Path "build") { Remove-Item -Recurse -Force "build" }

Write-Host "Creating temporary virtual environment for CPU-only build..." -ForegroundColor Yellow

# Create temporary environment for CPU-only build
python -m venv venv-cpu-build
& "venv-cpu-build\Scripts\Activate.ps1"

# Install CPU-only PyTorch and dependencies
pip install -r requirements-cpu.txt
pip install -e .

Write-Host "Building with PyInstaller..." -ForegroundColor Yellow

# Build with PyInstaller
pyinstaller --clean piano-transcriber-gui.spec

# Deactivate and cleanup
deactivate
Remove-Item -Recurse -Force "venv-cpu-build"

Write-Host "Build complete! Check dist/ folder for the application." -ForegroundColor Green
Write-Host "Expected size should be much smaller (~200-500MB instead of 2.5GB)" -ForegroundColor Cyan