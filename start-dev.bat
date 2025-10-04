@echo off
REM Development startup script for Local Agent Studio (Windows)

echo ========================================
echo Local Agent Studio - Development Mode
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.11 or higher
    pause
    exit /b 1
)

REM Check if Node.js is installed
node --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Node.js is not installed or not in PATH
    echo Please install Node.js 18 or higher
    pause
    exit /b 1
)

echo [1/5] Checking environment...

REM Check for .env file
if not exist ".env" (
    echo WARNING: .env file not found
    echo Creating .env from .env.example...
    if exist ".env.example" (
        copy ".env.example" ".env"
    ) else (
        echo OPENAI_API_KEY=your-api-key-here > .env
        echo Please edit .env and add your OpenAI API key
    )
)

echo [2/5] Setting up Python backend...
cd server

REM Create virtual environment if it doesn't exist
if not exist "venv" (
    echo Creating Python virtual environment...
    python -m venv venv
)

REM Activate virtual environment and install dependencies
call venv\Scripts\activate.bat
echo Installing Python dependencies...
pip install -e . >nul 2>&1

echo [3/5] Setting up Node.js frontend...
cd ..\ui

REM Install Node dependencies if needed
if not exist "node_modules" (
    echo Installing Node.js dependencies...
    call npm install
)

echo [4/5] Initializing database...
cd ..
if not exist "data" mkdir data

echo [5/5] Starting services...
echo.
echo Starting FastAPI backend on http://localhost:8000
echo Starting Next.js frontend on http://localhost:3000
echo.
echo Press Ctrl+C to stop all services
echo.

REM Start backend in new window
start "LAS Backend" cmd /k "cd server && venv\Scripts\activate.bat && python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000"

REM Wait a bit for backend to start
timeout /t 3 /nobreak >nul

REM Start frontend in new window
start "LAS Frontend" cmd /k "cd ui && npm run dev"

echo.
echo ========================================
echo Services started successfully!
echo Backend: http://localhost:8000
echo Frontend: http://localhost:3000
echo API Docs: http://localhost:8000/docs
echo ========================================
echo.
echo Close this window or press any key to exit
pause >nul
