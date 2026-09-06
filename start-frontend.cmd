@echo off
cd /d "C:\Users\Rohith\Downloads\SIH\frontend"
set NEXT_PUBLIC_WS_URL=ws://localhost:8000/ws/analyze
call npm run dev
pause