@echo off
REM ============================================================
REM Start both local servers for the SIH demo (Windows / cmd)
REM Window 1: remote inference (Spectra-AASIST3 on GPU, :8001)
REM Window 2: FastAPI backend (:8000)
REM ============================================================
cd /d C:\Users\Rohith\Downloads\SIH
start "SIH-GPU" cmd /k "start-gpu-server.bat"
start "SIH-BACKEND" cmd /k "start-backend.bat"
echo Servers launching in separate windows.
echo NOTE: if a window shows port-in-use, that server is already running - check /health.