@echo off
REM FastAPI backend (port 8000) - talks to local GPU server on 8001
cd /d C:\Users\Rohith\Downloads\SIH
set "HF_HOME=C:\Users\Rohith\Downloads\SIH\weights"
set "HF_HUB_OFFLINE=1"
set "MOCK_MODE=false"
set "REMOTE_INFERENCE_URL=http://127.0.0.1:8001"
set "SPEAKERS_AUTOLOAD=true"
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
pause