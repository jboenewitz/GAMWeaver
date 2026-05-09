#!/bin/bash

# Start Backend Server
echo "🚀 Starting Bike Rental Prediction Backend..."

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BACKEND_DIR="$SCRIPT_DIR/backend"

cd "$BACKEND_DIR"

# Load a dedicated local env file when present.
if [ -f ".env.local" ]; then
    echo "🔧 Loading local environment from backend/.env.local"
    set -a
    # shellcheck disable=SC1091
    source ".env.local"
    set +a
elif [ -f ".env" ]; then
    echo "ℹ️  Found backend/.env but local startup does not auto-load it."
    echo "   Create backend/.env.local for local development or export vars before starting."
fi

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

if [ -z "${DATABASE_URL:-}" ]; then
    echo "ℹ️  DATABASE_URL not set. Using SQLite fallback at backend/igann_app.db"
fi

if [ -z "${SUPERADMIN_PASSWORD:-}" ]; then
    echo "⚠️  SUPERADMIN_PASSWORD is not set."
    echo "   Logging in as superadmin will return an internal server error until you set it."
fi

# Start the server
echo "🌐 Starting FastAPI server on http://localhost:8000"
echo "📚 API Docs available at http://localhost:8000/docs"
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
