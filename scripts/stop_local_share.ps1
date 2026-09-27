param([string]$State = "test-assets/local-share-20260926.local")
$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$statePath = Join-Path $projectRoot $State
$shareState = Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json

# Verify each process before stopping it: Windows can reuse a saved PID.
foreach ($entry in @(
    @{ Kind = "tunnel"; Id = $shareState.tunnel_pid },
    @{ Kind = "gateway"; Id = $shareState.gateway_pid }
)) {
    if (-not $entry.Id) { continue }
    $process = Get-CimInstance Win32_Process -Filter "ProcessId = $([int]$entry.Id)"
    if (-not $process) { continue }
    if ($entry.Kind -eq "tunnel") {
        $expected = Join-Path $projectRoot "tools\cloudflared.exe"
        if ($process.ExecutablePath -ne $expected -or $process.CommandLine -notlike "*http://127.0.0.1:$($shareState.port)*") {
            throw "Saved tunnel PID now belongs to a different process. Nothing was stopped."
        }
    } else {
        if ($process.CommandLine -notlike "*scripts/share_local_app.py*" -or $process.CommandLine -notlike "*$State*") {
            throw "Saved gateway PID now belongs to a different process. Nothing was stopped."
        }
    }
    Stop-Process -Id ([int]$entry.Id) -ErrorAction Stop
    Write-Host "Stopped share $($entry.Kind)."
}
Write-Host "The original app on port 8001 and its saved data were kept."
