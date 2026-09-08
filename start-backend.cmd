@echo off
cd /d "%~dp0"
set MOCK_MODE=true
".venv\Scripts\python.exe" -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
pause
