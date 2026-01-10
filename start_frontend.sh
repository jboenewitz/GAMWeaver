#!/bin/bash

# Start Frontend Development Server
echo "🚀 Starting Bike Rental Prediction Frontend..."

cd "$(dirname "$0")/frontend"

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "📦 Installing frontend dependencies..."
    npm install
fi

# Start the development server
echo "🌐 Starting Vite dev server on http://localhost:3000"
npm run dev
