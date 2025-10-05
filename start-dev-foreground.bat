@echo off
cd server
call venv\Scripts\activate.bat
start /b python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
cd ..\ui
npm run dev
