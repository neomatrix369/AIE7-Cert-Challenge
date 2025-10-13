#!/bin/bash

# Student Loan RAG System - Service Shutdown Script
# Gracefully stop all Docker services

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Show help
show_help() {
    echo -e "${BLUE}🛑 Student Loan RAG Services - Stop Script${NC}"
    echo "============================================"
    echo ""
    echo "Usage: ./stop-services.sh [OPTIONS]"
    echo ""
    echo "Options:"
    echo ""
    echo "  (no options)"
    echo "      What it does:"
    echo "        • Stops all running containers (docker compose stop)"
    echo "        • Cleans up dangling Docker images"
    echo "        • Cleans up Docker build cache"
    echo "        • Keeps containers (in stopped state)"
    echo "        • Keeps all data volumes (Qdrant, cache, notebooks)"
    echo ""
    echo "  --remove"
    echo "      What it does:"
    echo "        • Stops all running containers"
    echo "        • Removes containers (docker compose down)"
    echo "        • Cleans up dangling Docker images"
    echo "        • Keeps all data volumes (Qdrant, cache, notebooks)"
    echo "      Note: Containers will be recreated on next startup"
    echo ""
    echo "  --clean  ⚠️  DESTRUCTIVE"
    echo "      What it does:"
    echo "        • Stops all running containers"
    echo "        • Removes containers AND volumes (docker compose down --volumes)"
    echo "        • Removes ALL unused Docker images (aggressive cleanup)"
    echo "        • Removes ALL Docker build cache"
    echo "      WARNING: This DELETES all vector database data and cached evaluations!"
    echo ""
    echo "  --skip-cleanup"
    echo "      Can be combined with default or --remove modes"
    echo "      Skips Docker image and cache cleanup for faster operation"
    echo ""
    echo "  --help, -h"
    echo "      Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./stop-services.sh                    # Stop services, clean up images/cache"
    echo "  ./stop-services.sh --skip-cleanup     # Stop services, keep images/cache"
    echo "  ./stop-services.sh --remove           # Stop and remove containers"
    echo "  ./stop-services.sh --clean            # Remove everything (destructive)"
    echo ""
    exit 0
}

# Parse arguments
MODE="default"
SKIP_CLEANUP=false

for arg in "$@"; do
    case $arg in
        --help|-h)
            show_help
            ;;
        --remove)
            MODE="remove"
            ;;
        --clean)
            MODE="clean"
            ;;
        --skip-cleanup)
            SKIP_CLEANUP=true
            ;;
        *)
            echo -e "${RED}❌ Unknown option: $arg${NC}"
            echo "Run './stop-services.sh --help' for usage information"
            exit 1
            ;;
    esac
done

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

# Execute based on mode
case $MODE in
    default)
        echo -e "${BLUE}Mode: Default (stop services, keep containers and volumes)${NC}"
        echo -e "${YELLOW}⏳ Stopping services gracefully...${NC}"
        $DOCKER_COMPOSE_COMMAND stop

        if [ "$SKIP_CLEANUP" = false ]; then
            echo -e "${BLUE}🧹 Cleaning up dangling images and build cache...${NC}"
            echo -e "${YELLOW}   (Keeps: stopped containers, volumes, used images)${NC}"
            docker image prune -f > /dev/null 2>&1 || true
            docker builder prune -f > /dev/null 2>&1 || true
        else
            echo -e "${YELLOW}⏭️  Skipping cleanup (--skip-cleanup specified)${NC}"
        fi
        ;;

    remove)
        echo -e "${BLUE}Mode: Remove (stop and remove containers, keep volumes)${NC}"
        echo -e "${YELLOW}⏳ Stopping and removing containers...${NC}"
        $DOCKER_COMPOSE_COMMAND down --remove-orphans

        if [ "$SKIP_CLEANUP" = false ]; then
            echo -e "${BLUE}🧹 Cleaning up dangling images...${NC}"
            echo -e "${YELLOW}   (Keeps: volumes, used images)${NC}"
            docker image prune -f > /dev/null 2>&1 || true
        else
            echo -e "${YELLOW}⏭️  Skipping cleanup (--skip-cleanup specified)${NC}"
        fi
        ;;

    clean)
        echo -e "${RED}Mode: Clean - ⚠️  DESTRUCTIVE OPERATION${NC}"
        echo -e "${YELLOW}This will:"
        echo "  • Stop and remove all containers"
        echo "  • Delete ALL volumes (vector DB, cache, notebooks)"
        echo "  • Remove all unused Docker images"
        echo "  • Clear all Docker build cache${NC}"
        echo ""
        read -p "Are you sure? Type 'yes' to confirm: " -r
        echo
        if [[ $REPLY == "yes" ]]; then
            echo -e "${YELLOW}⏳ Removing containers and volumes...${NC}"
            $DOCKER_COMPOSE_COMMAND down --remove-orphans --volumes
            echo -e "${YELLOW}🧹 Cleaning up all unused Docker resources...${NC}"
            docker system prune -af > /dev/null 2>&1 || true
            docker builder prune -af > /dev/null 2>&1 || true
            echo -e "${RED}⚠️  All data deleted!${NC}"
        else
            echo -e "${BLUE}❌ Cancelled. Use './stop-services.sh' for safe shutdown.${NC}"
            exit 1
        fi
        ;;
esac

echo ""
echo -e "${GREEN}✅ Operation completed successfully${NC}"
echo ""

# Show what was preserved based on mode
case $MODE in
    default)
        echo "📦 What's preserved:"
        echo "   ✅ Stopped containers (can be restarted)"
        echo "   ✅ All data volumes (Qdrant, cache, notebooks)"
        echo "   ✅ Docker images (unless cleaned up)"
        ;;
    remove)
        echo "📦 What's preserved:"
        echo "   ✅ All data volumes (Qdrant, cache, notebooks)"
        echo "   ✅ Docker images (unless cleaned up)"
        echo "   ℹ️  Containers removed (will be recreated on restart)"
        ;;
    clean)
        echo "📦 What's preserved:"
        echo "   ❌ Nothing - full cleanup performed"
        ;;
esac

echo ""
echo "🔧 Available commands:"
echo "   ./start-services.sh              - Start all services"
echo "   ./stop-services.sh --help        - See all stop options"
echo "   docker compose ps                - Check container status"