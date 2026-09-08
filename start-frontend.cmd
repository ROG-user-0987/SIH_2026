@echo off
cd /d "%~dp0frontend"
set NEXT_PUBLIC_WS_URL=ws://localhost:8000/ws/analyze
call npm run dev
pause
