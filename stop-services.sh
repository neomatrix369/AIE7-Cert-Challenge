#!/bin/bash

# Student Loan RAG System - Service Shutdown Script
# Gracefully stop all Docker services with interactive menu

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Parse command line arguments for non-interactive mode
MODE=""
NON_INTERACTIVE=false

for arg in "$@"; do
    case $arg in
        --help|-h)
            echo -e "${BLUE}🛑 Student Loan RAG Services - Stop Script${NC}"
            echo "============================================"
            echo ""
            echo "Usage: ./stop-services.sh [OPTIONS]"
            echo ""
            echo "Interactive Mode (default):"
            echo "  Run without options to see an interactive menu with 4 stop modes"
            echo ""
            echo "Non-Interactive Mode:"
            echo "  --mode=standard     Standard stop with cleanup (default)"
            echo "  --mode=quick        Quick pause without cleanup (fastest)"
            echo "  --mode=deep         Deep cleanup with container removal"
            echo "  --mode=nuclear      Nuclear reset - DELETES ALL DATA ⚠️"
            echo "  --non-interactive   Skip menu, use specified mode"
            echo ""
            echo "Legacy Flags (backward compatible):"
            echo "  --skip-cleanup      Maps to --mode=quick"
            echo "  --remove            Maps to --mode=deep"
            echo "  --clean             Maps to --mode=nuclear"
            echo ""
            echo "Examples:"
            echo "  ./stop-services.sh                      # Interactive menu"
            echo "  ./stop-services.sh --mode=standard      # Standard stop"
            echo "  ./stop-services.sh --mode=quick         # Quick pause"
            echo "  ./stop-services.sh --non-interactive    # Non-interactive standard"
            echo ""
            exit 0
            ;;
        --non-interactive)
            NON_INTERACTIVE=true
            ;;
        --mode=*)
            MODE="${arg#*=}"
            ;;
        --skip-cleanup)
            MODE="quick"
            NON_INTERACTIVE=true
            ;;
        --remove)
            MODE="deep"
            NON_INTERACTIVE=true
            ;;
        --clean)
            MODE="nuclear"
            NON_INTERACTIVE=true
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
echo ""

# Check for docker-compose or docker compose
if command -v docker-compose &> /dev/null; then
    DOCKER_COMPOSE_COMMAND="docker-compose"
elif command -v docker compose &> /dev/null; then
    DOCKER_COMPOSE_COMMAND="docker compose"
else
    echo -e "${RED}❌ Docker Compose is not installed.${NC}"
    exit 1
fi

# Show current status
echo -e "${BLUE}📦 Current container status:${NC}"
$DOCKER_COMPOSE_COMMAND ps
echo ""

# Show disk usage info
echo -e "${BLUE}💾 Docker disk usage:${NC}"
docker system df --format "table {{.Type}}\t{{.TotalCount}}\t{{.Size}}\t{{.Reclaimable}}" | head -5
echo ""

# Interactive menu if not in non-interactive mode
if [ "$NON_INTERACTIVE" = false ] && [ -z "$MODE" ]; then
    echo -e "${YELLOW}Choose stop method:${NC}"
    echo ""
    echo "1. 🛑 Standard stop (recommended for daily use)"
    echo "   • Stops all containers"
    echo "   • Cleans up: dangling images, build cache"
    echo "   • Preserves: stopped containers, volumes, used images"
    echo ""
    echo "2. ⏸️  Quick pause (fastest restart)"
    echo "   • Stops containers only"
    echo "   • No cleanup performed"
    echo "   • Next startup will be faster"
    echo ""
    echo "3. 🔧 Deep cleanup (reclaim disk space)"
    echo "   • Stops and removes containers"
    echo "   • Cleans up: dangling images, build cache"
    echo "   • Preserves: volumes (your data)"
    echo ""
    echo "4. 💣 Nuclear reset (⚠️  DATA LOSS WARNING)"
    echo "   • Removes containers AND volumes"
    echo "   • ⚠️  DELETES: Vector DB, cache, notebooks"
    echo "   • Use only when starting completely fresh"
    echo ""

    read -p "Enter choice (1-4) or press Enter for default [1]: " choice
    choice=${choice:-1}

    case $choice in
        1) MODE="standard" ;;
        2) MODE="quick" ;;
        3) MODE="deep" ;;
        4) MODE="nuclear" ;;
        *)
            echo -e "${RED}❌ Invalid choice. Using standard stop.${NC}"
            MODE="standard"
            ;;
    esac
    echo ""
fi

# Default to standard mode if still not set
MODE=${MODE:-standard}

# Execute based on mode
case $MODE in
    standard)
        echo -e "${BLUE}🛑 Mode: Standard stop${NC}"
        echo -e "${YELLOW}⏳ Stopping services gracefully...${NC}"
        $DOCKER_COMPOSE_COMMAND stop

        echo -e "${BLUE}🧹 Cleaning up dangling images and build cache...${NC}"
        echo -e "${YELLOW}   (Keeps: stopped containers, volumes, used images)${NC}"
        docker image prune -f > /dev/null 2>&1 || true
        docker builder prune -f > /dev/null 2>&1 || true

        echo ""
        echo "📦 What's preserved:"
        echo "   ✅ Stopped containers (can be restarted)"
        echo "   ✅ All data volumes (Qdrant, cache, notebooks)"
        echo "   ✅ Docker images (except dangling ones)"
        ;;

    quick)
        echo -e "${BLUE}⏸️  Mode: Quick pause${NC}"
        echo -e "${YELLOW}⏳ Stopping services gracefully...${NC}"
        $DOCKER_COMPOSE_COMMAND stop

        echo -e "${YELLOW}⏭️  Skipping cleanup for faster restart${NC}"

        echo ""
        echo "📦 What's preserved:"
        echo "   ✅ Stopped containers"
        echo "   ✅ All data volumes"
        echo "   ✅ All Docker images and build cache"
        ;;

    deep)
        echo -e "${BLUE}🔧 Mode: Deep cleanup${NC}"
        echo -e "${YELLOW}This will remove containers and clean up unused Docker resources.${NC}"

        if [ "$NON_INTERACTIVE" = false ]; then
            read -p "Continue? (y/N): " confirm_cleanup
            if [[ ! "$confirm_cleanup" =~ ^[Yy]$ ]]; then
                echo -e "${YELLOW}Cancelled. Falling back to standard stop...${NC}"
                MODE="standard"
                $DOCKER_COMPOSE_COMMAND stop
                docker image prune -f > /dev/null 2>&1 || true
                docker builder prune -f > /dev/null 2>&1 || true
                echo -e "${GREEN}✅ Standard stop completed${NC}"
                exit 0
            fi
        fi

        echo -e "${YELLOW}⏳ Stopping and removing containers...${NC}"
        $DOCKER_COMPOSE_COMMAND down --remove-orphans

        echo -e "${BLUE}🧹 Cleaning up dangling images...${NC}"
        echo -e "${YELLOW}   (Keeps: volumes, used images)${NC}"
        docker image prune -f > /dev/null 2>&1 || true

        echo ""
        echo "📦 What's preserved:"
        echo "   ✅ All data volumes (Qdrant, cache, notebooks)"
        echo "   ✅ Docker images (except dangling ones)"
        echo "   ℹ️  Containers removed (will be recreated on restart)"
        ;;

    nuclear)
        echo -e "${RED}💣 Mode: Nuclear reset - ⚠️  DESTRUCTIVE OPERATION${NC}"
        echo -e "${YELLOW}This will:"
        echo "  • Stop and remove all containers"
        echo "  • Delete ALL volumes (vector DB, cache, notebooks)"
        echo "  • Remove all unused Docker images"
        echo "  • Clear all Docker build cache${NC}"
        echo ""

        if [ "$NON_INTERACTIVE" = false ]; then
            read -p "Are you ABSOLUTELY SURE? Type 'DELETE ALL DATA' to confirm: " confirm_nuclear
            if [[ "$confirm_nuclear" != "DELETE ALL DATA" ]]; then
                echo -e "${GREEN}✅ Nuclear reset cancelled. Your data is safe.${NC}"
                echo -e "${YELLOW}Falling back to standard stop...${NC}"
                $DOCKER_COMPOSE_COMMAND stop
                docker image prune -f > /dev/null 2>&1 || true
                docker builder prune -f > /dev/null 2>&1 || true
                echo -e "${GREEN}✅ Standard stop completed${NC}"
                exit 0
            fi
        else
            echo -e "${RED}⚠️  Running in non-interactive mode - proceeding with nuclear reset!${NC}"
        fi

        echo -e "${YELLOW}⏳ Removing containers and volumes...${NC}"
        $DOCKER_COMPOSE_COMMAND down --remove-orphans --volumes

        echo -e "${YELLOW}🧹 Cleaning up all unused Docker resources...${NC}"
        docker system prune -af > /dev/null 2>&1 || true
        docker builder prune -af > /dev/null 2>&1 || true

        echo -e "${RED}⚠️  All data deleted!${NC}"

        echo ""
        echo "📦 What's preserved:"
        echo "   ❌ Nothing - full cleanup performed"
        ;;

    *)
        echo -e "${RED}❌ Invalid mode: $MODE${NC}"
        exit 1
        ;;
esac

echo ""
echo -e "${GREEN}✅ Operation completed successfully${NC}"
echo ""

# Show final status
echo -e "${BLUE}📊 Final container status:${NC}"
$DOCKER_COMPOSE_COMMAND ps
echo ""

echo -e "${BLUE}💡 What's next?${NC}"
echo "   🚀 Start services:      ./start-services.sh"
echo "   🔍 Check status:        docker compose ps"
echo "   ❓ Help:               ./stop-services.sh --help"
echo ""
