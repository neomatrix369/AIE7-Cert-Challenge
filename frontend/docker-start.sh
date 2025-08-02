#!/bin/bash

# Frontend Docker startup script for Federal Student Loan Assistant

echo "🏦 Federal Student Loan Assistant - Full Stack Docker Deployment"
echo "=============================================================="
echo ""

# Check if .env file exists
if [ ! -f "../.env" ]; then
    echo "⚠️  Warning: .env file not found in project root"
    echo "   Please create ../.env with required API keys:"
    echo "   - OPENAI_API_KEY"
    echo "   - TAVILY_API_KEY"
    echo "   - COHERE_API_KEY"
    echo ""
fi

echo "🚀 Starting full-stack deployment with Docker Compose..."
echo ""
echo "Services:"
echo "  📱 Frontend: http://localhost:3000"
echo "  🔧 Backend API: http://localhost:8000"
echo "  📖 API Docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop all services"
echo ""

# Start the services
docker-compose up --build