#!/bin/bash

# Student Loan RAG System - Service Shutdown Script
# Gracefully stop all Docker services

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}🛑 Stopping Student Loan RAG Services${NC}"
echo "========================================"

# Check for docker-compose or docker compose
if command -v docker-compose &> /dev/null; then
    DOCKER_COMPOSE_COMMAND="docker-compose"
elif command -v docker compose &> /dev/null; then
    DOCKER_COMPOSE_COMMAND="docker compose"
else
    echo "❌ Docker Compose is not installed."
    exit 1
fi

# Graceful shutdown
echo -e "${YELLOW}⏳ Stopping services gracefully...${NC}"
$DOCKER_COMPOSE_COMMAND stop

# Remove containers (keeping volumes for data persistence)
if [ "$1" == "--remove" ]; then
    echo -e "${YELLOW}🗑️  Removing containers...${NC}"
    $DOCKER_COMPOSE_COMMAND down --remove-orphans
fi

# Remove volumes and data (destructive operation)
if [ "$1" == "--clean" ]; then
    echo -e "${YELLOW}⚠️  Removing containers and volumes (data will be lost)...${NC}"
    read -p "Are you sure? This will delete all vector database data and cache. (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        $DOCKER_COMPOSE_COMMAND down --remove-orphans --volumes
        echo -e "${YELLOW}🧹 Cleaning up unused Docker resources...${NC}"
        docker system prune -f
    else
        echo -e "${BLUE}Cancelled. Services stopped but data preserved.${NC}"
    fi
fi

echo ""
echo -e "${GREEN}✅ Services stopped successfully${NC}"
echo ""
echo "🔧 Available options:"
echo "   ./stop-services.sh          - Stop services (keep containers and data)"
echo "   ./stop-services.sh --remove - Stop and remove containers (keep data volumes)"  
echo "   ./stop-services.sh --clean  - Stop and remove everything (⚠️  destructive)"
echo ""
echo "🚀 To restart services:"
echo "   ./start-services.sh"