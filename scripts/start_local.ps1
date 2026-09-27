param(
    [string]$Scene = "00000000-0000-4000-8000-000000000001",
    [int]$Port = 8001,
    [switch]$NoOpen,
    [switch]$NoSeed
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
$dbPath = Join-Path $projectRoot "data\spatial-memory-v2.db"

if (-not (Test-Path $venvPython)) {
    throw "Python virtual environment not found at $venvPython. Please run the setup first."
}

# Make bundled ffmpeg available to audio/video subprocesses.
$env:PATH = (Join-Path $projectRoot ".venv\Scripts") + ";" + $env:PATH
if (-not $env:RECONSTRUCTION_DEVICE) {
    $env:RECONSTRUCTION_DEVICE = & $venvPython -c "import torch; print('cuda' if torch.cuda.is_available() else 'cpu')"
    if ($LASTEXITCODE -ne 0) { throw "PyTorch failed to initialize." }
}
if ($env:RECONSTRUCTION_DEVICE -eq "cpu") {
    Write-Warning "No CUDA device selected. VGGT runs on CPU; use short videos with very few frames."
    $env:OMP_NUM_THREADS = "4"
    $env:MKL_NUM_THREADS = "4"
}

Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "   Spatial Memory V2 - Local Runner      " -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan

# Check port occupancy
$existing = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if ($existing) {
    Write-Warning "Port $Port is already in use by PID $($existing.OwningProcess)."
    Write-Host "Attempting to continue or use another port if needed..."
}

# Auto seed if database does not exist
if (-not $NoSeed) {
    if (-not (Test-Path $dbPath)) {
        Write-Host "[1/2] Seeding initial demo room and entities..." -ForegroundColor Yellow
        $env:PYTHONPATH = "src"
        & $venvPython (Join-Path $PSScriptRoot "seed_local_demo.py")
        if ($LASTEXITCODE -ne 0) {
            Write-Warning "Seed script returned exit code $LASTEXITCODE"
        }
    } else {
        Write-Host "[1/2] Existing database found: $dbPath" -ForegroundColor Green
    }
} else {
    Write-Host "[1/2] Seeding skipped."
}

Write-Host "[2/2] Starting Spatial Memory API on port $Port..." -ForegroundColor Yellow

$env:PYTHONPATH = "src"
$env:SPATIAL_DB = Join-Path $projectRoot "data\spatial-memory-v2.db"
$env:SCENE_ROOT = Join-Path $projectRoot "data\scenes-v2"
$env:SPATIAL_WEB_DIR = Join-Path $projectRoot "web"

$url = "http://127.0.0.1:$Port/"
if ($Scene) {
    $url = "http://127.0.0.1:$Port/?scene=$([uri]::EscapeDataString($Scene))"
}

# Open browser after a brief delay in background job
if (-not $NoOpen) {
    Start-Job -ScriptBlock {
        param($targetUrl, $targetPort)
        for ($i = 0; $i -lt 30; $i++) {
            Start-Sleep -Milliseconds 500
            try {
                $res = Invoke-RestMethod "http://127.0.0.1:$targetPort/health" -TimeoutSec 1 -ErrorAction SilentlyContinue
                if ($res.status -eq "ok") {
                    Start-Process $targetUrl
                    break
                }
            } catch {}
        }
    } -ArgumentList $url, $Port | Out-Null
}

Write-Host ""
Write-Host "Server running at: $url" -ForegroundColor Green
Write-Host "Press Ctrl+C to stop the server." -ForegroundColor Gray
Write-Host ""

& $venvPython -m uvicorn spatial_memory.api:app --host 127.0.0.1 --port $Port
