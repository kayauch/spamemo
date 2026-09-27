@echo off
cd /d "%~dp0"
echo Starting Spatial Memory V2...
powershell -ExecutionPolicy Bypass -File ".\scripts\start_local.ps1"
pause
