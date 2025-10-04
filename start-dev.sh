#!/bin/bash
# Development startup script for Local Agent Studio (Linux/Mac)

set -e

echo "========================================"
echo "Local Agent Studio - Development Mode"
echo "========================================"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed"
    echo "Please install Python 3.11 or higher"
    exit 1
fi

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "ERROR: Node.js is not installed"
    echo "Please install Node.js 18 or higher"
    exit 1
fi

echo "[1/5] Checking environment..."

# Check for .env file
if [ ! -f ".env" ]; then
    echo "WARNING: .env file not found"
    echo "Creating .env from .env.example..."
    if [ -f ".env.example" ]; then
        cp ".env.example" ".env"
    else
        echo "OPENAI_API_KEY=your-api-key-here" > .env
        echo "Please edit .env and add your OpenAI API key"
    fi
fi

echo "[2/5] Setting up Python backend..."
cd server

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment and install dependencies
source venv/bin/activate
echo "Installing Python dependencies..."
pip install -e . > /dev/null 2>&1

echo "[3/5] Setting up Node.js frontend..."
cd ../ui

# Install Node dependencies if needed
if [ ! -d "node_modules" ]; then
    echo "Installing Node.js dependencies..."
    npm install
fi

echo "[4/5] Initializing database..."
cd ..
mkdir -p data

echo "[5/5] Starting services..."
echo ""
echo "Starting FastAPI backend on http://localhost:8000"
echo "Starting Next.js frontend on http://localhost:3000"
echo ""
echo "Press Ctrl+C to stop all services"
echo ""

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "Stopping services..."
    kill $BACKEND_PID $FRONTEND_PID 2>/dev/null
    exit 0
}

trap cleanup INT TERM

# Start backend
cd server
source venv/bin/activate
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

# Wait a bit for backend to start
sleep 3

# Start frontend
cd ../ui
npm run dev &
FRONTEND_PID=$!

echo ""
echo "========================================"
echo "Services started successfully!"
echo "Backend: http://localhost:8000"
echo "Frontend: http://localhost:3000"
echo "API Docs: http://localhost:8000/docs"
echo "========================================"
echo ""

# Wait for processes
wait
