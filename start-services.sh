#!/bin/bash

# Student Loan RAG System - Service Startup Script
# Start all services (Docker-based by default)

set -e  # Exit on any error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Show help
show_help() {
    echo -e "${BLUE}🚀 Student Loan RAG Services - Start Script${NC}"
    echo "=============================================="
    echo ""
    echo "Usage: ./start-services.sh [OPTIONS]"
    echo ""
    echo "What this script does:"
    echo "  1. ✅ Validates Docker/Compose installation"
    echo "  2. ✅ Checks for required API keys in .env"
    echo "  3. 🛑 Stops any existing containers (docker compose down)"
    echo "  4. 🧹 Cleans up dangling Docker images and build cache"
    echo "  5. 📥 Pulls latest external images (Qdrant)"
    echo "  6. 🔨 Builds custom services (backend, jupyter, frontend)"
    echo "  7. 🚀 Starts all services with health checks"
    echo ""
    echo "Options:"
    echo "  --no-frontend     Skip starting the frontend dashboard"
    echo "  --skip-cleanup    Skip Docker image/cache cleanup (faster restart)"
    echo "  --help, -h        Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./start-services.sh                    # Full startup with cleanup"
    echo "  ./start-services.sh --skip-cleanup     # Quick restart without cleanup"
    echo "  ./start-services.sh --no-frontend      # Start without frontend UI"
    echo ""
    echo "Note: This script ALWAYS stops existing containers and rebuilds."
    echo "      Data in volumes (Qdrant, cache, notebooks) is preserved."
    exit 0
}

# Parse arguments
SKIP_CLEANUP=false
SKIP_FRONTEND=false

for arg in "$@"; do
    case $arg in
        --help|-h)
            show_help
            ;;
        --skip-cleanup)
            SKIP_CLEANUP=true
            ;;
        --no-frontend)
            SKIP_FRONTEND=true
            ;;
        *)
            echo -e "${RED}❌ Unknown option: $arg${NC}"
            echo "Run './start-services.sh --help' for usage information"
            exit 1
            ;;
    esac
done

echo -e "${BLUE}🚀 Starting Student Loan RAG Services${NC}"
echo "=========================================="

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker is not installed. Please install Docker Desktop first.${NC}"
    echo "   Download from: https://www.docker.com/products/docker-desktop/"
    exit 1
fi

# Check for docker-compose or docker compose
if command -v docker-compose &> /dev/null; then
    DOCKER_COMPOSE_COMMAND="docker-compose"
elif command -v docker compose &> /dev/null; then
    DOCKER_COMPOSE_COMMAND="docker compose"
else
    echo -e "${RED}❌ Docker Compose is not installed. Please install Docker Compose first.${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Docker and Docker Compose are installed${NC}"

# Check if .env file exists
if [[ ! -f .env ]]; then
    echo -e "${YELLOW}⚠️  .env file not found. Creating from template...${NC}"
    cp .env-example .env
    echo -e "${YELLOW}📝 Please edit .env file with your API keys${NC}"
    echo ""
    echo "Required API keys:"
    echo "  - OPENAI_API_KEY (required for embeddings and chat)"
    echo "  - COHERE_API_KEY (required for reranking)"
    echo "  - TAVILY_API_KEY (required for web search)"
    echo "  - LANGCHAIN_API_KEY (optional for tracing)"
    echo ""
    read -p "Press Enter to continue after adding API keys to .env..."
fi

# Load environment variables
source .env

# Check for required API keys
MISSING_KEYS=""
if [ -z "$OPENAI_API_KEY" ]; then MISSING_KEYS="$MISSING_KEYS OPENAI_API_KEY"; fi
if [ -z "$COHERE_API_KEY" ]; then MISSING_KEYS="$MISSING_KEYS COHERE_API_KEY"; fi
if [ -z "$TAVILY_API_KEY" ]; then MISSING_KEYS="$MISSING_KEYS TAVILY_API_KEY"; fi

if [ -n "$MISSING_KEYS" ]; then
    echo -e "${RED}❌ Missing required API keys:$MISSING_KEYS${NC}"
    echo "   Please add these keys to .env file"
    exit 1
fi

echo -e "${GREEN}✅ Environment variables configured${NC}"

# Create necessary directories
mkdir -p cache golden-masters metrics

# Stop any existing containers
echo -e "${BLUE}🛑 Stopping any existing containers...${NC}"
$DOCKER_COMPOSE_COMMAND down --remove-orphans > /dev/null 2>&1 || true

# Clean up dangling images and build cache to free disk space
if [ "$SKIP_CLEANUP" = false ]; then
    echo -e "${BLUE}🧹 Cleaning up dangling images and build cache...${NC}"
    echo -e "${YELLOW}   (Use --skip-cleanup to skip this for faster restarts)${NC}"
    docker image prune -f > /dev/null 2>&1 || true
    docker builder prune -f > /dev/null 2>&1 || true
else
    echo -e "${YELLOW}⏭️  Skipping cleanup (--skip-cleanup specified)${NC}"
fi

# Pull only external images (like Qdrant), skip custom built images
echo -e "${BLUE}📥 Pulling external images...${NC}"
$DOCKER_COMPOSE_COMMAND pull qdrant || echo -e "${YELLOW}⚠️  Qdrant pull failed, will use cached or build${NC}"

# Build services
echo -e "${BLUE}🔨 Building custom services...${NC}"
$DOCKER_COMPOSE_COMMAND build --parallel

# Start services in order
echo -e "${BLUE}🚀 Starting services...${NC}"

# Start Qdrant first (foundation layer)
echo -e "${BLUE}📊 Starting Qdrant vector database...${NC}"
$DOCKER_COMPOSE_COMMAND up -d qdrant

# Wait for Qdrant to be healthy
echo -e "${YELLOW}⏳ Waiting for Qdrant to be ready...${NC}"
timeout=60
counter=0
until curl -f http://localhost:6333/ > /dev/null 2>&1; do
    sleep 2
    counter=$((counter + 2))
    if [ $counter -gt $timeout ]; then
        echo -e "${RED}❌ Qdrant failed to start within ${timeout}s${NC}"
        exit 1
    fi
    echo -n "."
done
echo -e " ${GREEN}✅ Ready!${NC}"

# Start backend services
echo -e "${BLUE}🤖 Starting backend API...${NC}"
$DOCKER_COMPOSE_COMMAND up -d backend

echo -e "${BLUE}📚 Starting Jupyter notebook server...${NC}"
$DOCKER_COMPOSE_COMMAND up -d jupyter

# Wait for backend to be healthy
echo -e "${YELLOW}⏳ Waiting for backend API to be ready...${NC}"
counter=0
until curl -f http://localhost:8000/health > /dev/null 2>&1; do
    sleep 3
    counter=$((counter + 3))
    if [ $counter -gt 90 ]; then
        echo -e "${RED}❌ Backend failed to start within 90s${NC}"
        $DOCKER_COMPOSE_COMMAND logs backend
        exit 1
    fi
    echo -n "."
done
echo -e " ${GREEN}✅ Ready!${NC}"

# Start frontend (optional)
if [ "$SKIP_FRONTEND" = false ]; then
    echo -e "${BLUE}🌐 Starting frontend dashboard...${NC}"
    $DOCKER_COMPOSE_COMMAND up -d frontend

    # Wait for frontend to be healthy
    echo -e "${YELLOW}⏳ Waiting for frontend to be ready...${NC}"
    counter=0
    until curl -f http://localhost:3000 > /dev/null 2>&1; do
        sleep 3
        counter=$((counter + 3))
        if [ $counter -gt 60 ]; then
            echo -e "${YELLOW}⚠️  Frontend taking longer than expected, continuing...${NC}"
            break
        fi
        echo -n "."
    done
    echo -e " ${GREEN}✅ Ready!${NC}"
else
    echo -e "${YELLOW}⏭️  Skipping frontend (--no-frontend specified)${NC}"
fi

echo ""
echo "=================================================="
echo -e "${GREEN}🎉 All services started successfully! ✅${NC}"
echo ""
echo "🌐 Services available at:"
echo "   📊 Qdrant:     http://localhost:6333/dashboard"
echo "   🤖 Backend:    http://localhost:8000"
echo "   📚 Jupyter:    http://localhost:8888"
echo "   📖 API Docs:   http://localhost:8000/docs"
if [ "$SKIP_FRONTEND" = false ]; then
    echo "   🎨 Frontend:   http://localhost:3000"
fi
echo ""
echo "🔧 Management commands:"
echo "   📋 View logs:       $DOCKER_COMPOSE_COMMAND logs -f [service]"
echo "   🛑 Stop services:   ./stop-services.sh"
echo "   🔄 Restart:         $DOCKER_COMPOSE_COMMAND restart [service]"
echo "   ❓ Help:           ./start-services.sh --help"
echo ""
echo "💡 Tips:"
echo "   - Use Jupyter for RAG experiments and analysis"
echo "   - Check API docs for endpoints and examples"
echo "   - Monitor Qdrant dashboard for vector operations"
echo "   - Use --skip-cleanup for faster restarts during development"