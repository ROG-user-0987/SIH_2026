@echo off
REM GPU inference server (Spectra-AASIST3, port 8001)
cd /d C:\Users\Rohith\Downloads\SIH\backend\remote_server
set "HF_HOME=C:\Users\Rohith\Downloads\SIH\weights"
set "HF_HUB_OFFLINE=1"
set "TRANSFORMERS_OFFLINE=1"
set "MODEL_NAME=spectra"
set "MODEL_FP16=true"
set "PORT=8001"
python -m uvicorn server:app --host 0.0.0.0 --port 8001
pause