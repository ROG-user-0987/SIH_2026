@echo off
cd /d "C:\Users\Rohith\Downloads\SIH"
set MOCK_MODE=true
"C:\Users\Rohith\Downloads\SIH\backend\.venv\Scripts\python.exe" -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
pause