#!/bin/bash

echo "🎓 Setting up Student Loan RAG System..."
echo "=================================================="
echo ""

# Check if we're in the right directory
if [ ! -f "README.md" ]; then
    echo "❌ Please run this script from the project root directory"
    exit 1
fi

# Check for Docker and prefer it
if command -v docker &> /dev/null && [[ "$1" != "--manual" ]]; then
    echo "🐳 Docker detected - using service startup (faster and more reliable)"
    echo "💡 Use './setup.sh --manual' if you prefer manual setup"
    echo ""
    exec ./start-services.sh
fi

echo "🔧 Using manual setup..."
echo "⚠️  Note: Docker setup is recommended for easier management"
echo ""

# Check for required Python version
if command -v python --version &> /dev/null; then
    PYTHON_VERSION=$(python --version 2>&1 | cut -d' ' -f2)
    echo "🐍 Python version: $PYTHON_VERSION"
else
    echo "❌ Python not found. Please install Python 3.12+"
    exit 1
fi

# Check for uv
if ! command -v uv &> /dev/null; then
    echo "⚠️  uv not found. Installing uv for faster Python package management..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi

# Setup Python environment
echo "🔧 Setting up Python environment with uv..."
uv sync --frozen

echo ""
echo "=================================================="
echo ""

# Check for required environment variables
echo "🔍 Checking environment variables..."
if [ ! -f ".env" ]; then
    echo "⚠️  .env file not found. Creating from template..."
    cp .env-example .env
    echo "📝 Please edit .env file with your API keys before running services"
fi

# Check required API keys
source .env 2>/dev/null || true
MISSING_KEYS=""

if [ -z "$OPENAI_API_KEY" ]; then
    MISSING_KEYS="$MISSING_KEYS OPENAI_API_KEY"
fi

if [ -z "$COHERE_API_KEY" ]; then
    MISSING_KEYS="$MISSING_KEYS COHERE_API_KEY"
fi

if [ -z "$TAVILY_API_KEY" ]; then
    MISSING_KEYS="$MISSING_KEYS TAVILY_API_KEY"
fi

if [ -n "$MISSING_KEYS" ]; then
    echo "⚠️  Missing required API keys in .env:$MISSING_KEYS"
    echo "    Please add these keys to .env file before running the system"
fi

# Setup data directories
echo "📁 Setting up data directories..."
mkdir -p cache golden-masters metrics

# Install Jupyter for notebooks (if not already available)
if ! command -v jupyter &> /dev/null; then
    echo "📚 Installing Jupyter for notebook analysis..."
    uv add jupyter jupyterlab ipywidgets
fi

echo ""
echo "=================================================="
echo ""
echo "🎉 Manual setup complete! ✅"
echo ""
echo "⚠️  Remember to:"
echo "   1. Add your API keys to .env file"
echo "   2. Start Qdrant separately: docker run -p 6333:6333 qdrant/qdrant"
echo "   3. Run notebooks: uv run jupyter lab"
echo "   4. Run API server: cd src/backend && uv run python simple_api.py"
echo ""
echo "💡 For easier management, try Docker startup next time:"
echo "   ./start-services.sh"