#!/bin/bash

# Frontend startup script for Federal Student Loan Assistant

echo "🏦 Federal Student Loan Assistant - Frontend"
echo "==========================================="
echo ""

# Check if backend is running
echo "🔍 Checking backend API health..."
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ Backend API is running on http://localhost:8000"
else
    echo "⚠️  Backend API is not running on http://localhost:8000"
    echo "   Please start the backend first:"
    echo "   cd ../src/backend && python simple_api.py"
    echo ""
fi

echo "🚀 Starting frontend server..."
echo "📱 Frontend will be available at: http://localhost:3000"
echo "👆 Click the link above or open in your browser"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Start the frontend server
python3 serve.py 3000