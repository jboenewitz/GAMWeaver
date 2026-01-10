#!/bin/bash

# Start both backend and frontend servers
echo "🚀 Starting Bike Rental Prediction Application..."
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Start backend in background
echo "📦 Starting backend server..."
gnome-terminal --title="Backend Server" -- bash -c "cd '$SCRIPT_DIR' && ./start_backend.sh; exec bash" 2>/dev/null || \
xterm -title "Backend Server" -e "cd '$SCRIPT_DIR' && ./start_backend.sh; exec bash" 2>/dev/null || \
(cd "$SCRIPT_DIR" && ./start_backend.sh &)

# Wait a bit for backend to start
sleep 3

# Start frontend
echo "📦 Starting frontend server..."
gnome-terminal --title="Frontend Server" -- bash -c "cd '$SCRIPT_DIR' && ./start_frontend.sh; exec bash" 2>/dev/null || \
xterm -title "Frontend Server" -e "cd '$SCRIPT_DIR' && ./start_frontend.sh; exec bash" 2>/dev/null || \
(cd "$SCRIPT_DIR" && ./start_frontend.sh &)

echo ""
echo "✅ Application starting!"
echo "📊 Frontend: http://localhost:3000"
echo "🔧 Backend API: http://localhost:8000"
echo "📚 API Docs: http://localhost:8000/docs"
