#!/bin/bash

# Start Backend Server
echo "🚀 Starting Bike Rental Prediction Backend..."

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BACKEND_DIR="$SCRIPT_DIR/backend"

cd "$BACKEND_DIR"

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate venv
source venv/bin/activate

# Install dependencies
echo "📦 Installing backend dependencies..."
pip install -r requirements.txt

# Start the server
echo "🌐 Starting FastAPI server on http://localhost:8000"
echo "📚 API Docs available at http://localhost:8000/docs"
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
