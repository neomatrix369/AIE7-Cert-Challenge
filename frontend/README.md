# Federal Student Loan Assistant - Frontend

A simple HTML/CSS/JavaScript chat interface for the Federal Student Loan RAG backend API.

## 📖 Table of Contents

- [✨ Features](#features)
- [📋 Prerequisites](#prerequisites)
- [🚀 Quick Start](#quick-start)
- [🔗 Backend Connection](#backend-connection)
- [📁 Files Structure](#files-structure)
- [💻 Usage](#usage)
- [🐳 Docker Deployment](#docker-deployment)
  - [Prerequisites for Docker](#prerequisites-for-docker)
  - [Container Discovery](#container-discovery)
  - [Option 1: Full-Stack with Enhanced Docker Script (Recommended)](#option-1-full-stack-with-enhanced-docker-script-recommended)
  - [Option 2: Multi-Container Setup (Frontend discovers Backend)](#option-2-multi-container-setup-frontend-discovers-backend)
  - [Option 3: Frontend Container Only](#option-3-frontend-container-only)
  - [Option 4: Development with Live Reload](#option-4-development-with-live-reload)
  - [Docker Management Commands](#docker-management-commands)
  - [Environment Variables & Configuration](#environment-variables--configuration)
  - [Network Architecture](#network-architecture)
- [🛠 Development](#development)
- [📄 Docker Files](#docker-files)
- [🔧 Troubleshooting](#troubleshooting)
  - [Common Issues](#common-issues)
  - [Health Checks](#health-checks)

## Features

- Clean chat interface for interacting with the RAG backend
- Real-time API health status indicator  
- Display of source count and processing time for responses
- Responsive design with Tailwind CSS
- No build process required - runs directly in browser

## Prerequisites

- Python 3 (for serving static files)
- Backend API running on http://localhost:8000

## Quick Start

1. **Start the backend first:**
   ```bash
   cd ../src/backend
   python simple_api.py
   ```

2. **Start the frontend:**
   ```bash
   ./start.sh
   ```
   
   Or manually:
   ```bash
   python3 serve.py 3000
   ```

3. **Open your browser:**
   Navigate to http://localhost:3000

## Backend Connection

The frontend connects to the backend API at `http://localhost:8000/ask`. The interface will show connection status in the header.

## Files Structure

- `index.html` - Main chat interface
- `serve.py` - Simple HTTP server with CORS support
- `start.sh` - Startup script with health checks
- `README.md` - This file

## Usage

1. Type your federal student loan question in the text area
2. Press Enter or click Send
3. View the AI response with source count and processing time
4. Continue the conversation as needed

## Docker Deployment

### Prerequisites for Docker

- Docker installed and running
- Backend API accessible (either running locally or in another container)

### Container Discovery

The frontend automatically detects the backend:
- **Local Development:** Uses `http://localhost:8000`
- **Docker Environment:** Looks for `rag-api` container on shared network
- **Fallback:** If container discovery fails, falls back to localhost

### Option 1: Full-Stack with Enhanced Docker Script (Recommended)

Deploy both frontend and backend with automatic cleanup:

```bash
# Quick start with enhanced script (includes cleanup & health checks)
./docker-start.sh
```

The enhanced script provides:
- **Pre-deployment cleanup** of stopped containers and dangling images
- **Environment validation** (.env file checks)
- **Health monitoring** and service status
- **Post-deployment cleanup** when stopping services

Or manually:
```bash
# Build and start both services
docker-compose up --build

### or for macOS

docker compose up --build

# Run in background (detached mode)
docker-compose up -d --build

### or for macOS

docker compose up -d --build

# Stop all services
docker-compose down

### or for macOS

docker compose down
```

This will start:
- **Frontend:** http://localhost:3000  
- **Backend API:** http://localhost:8000
- **API Docs:** http://localhost:8000/docs

### Option 2: Multi-Container Setup (Frontend discovers Backend)

**Step 1: Start Backend First**
```bash
cd ../src/backend
docker-compose up -d  # Creates the shared network

### or for macOS

docker compose up -d
```

**Step 2: Start Frontend (connects to backend container)**
```bash
cd frontend
docker-compose up --build

### or for macOS

docker compose up --build
```

The frontend will automatically discover and connect to the `rag-api` container via Docker networking.

### Option 3: Frontend Container Only

Build and run just the frontend container:

```bash
# Step 1: Build the Docker image
docker build -t student-loan-frontend .

# Step 2: Run the container
docker run -p 3000:3000 student-loan-frontend

# Alternative: Run in background with name
docker run -d --name frontend -p 3000:3000 student-loan-frontend

# Stop the container
docker stop frontend
docker rm frontend
```

### Option 4: Development with Live Reload

For development with file watching and automatic updates:

```bash
# Run with volume mounting for live changes
docker run -p 3000:3000 -v $(pwd):/app/frontend student-loan-frontend

# Or with docker-compose for development
docker-compose up --build  # Automatically includes volume mounts

### or for macOS

docker compose up --build
```

The docker-compose setup includes volume mounting by default, so any changes to frontend files will be reflected immediately without rebuilding the container.

### Docker Management Commands

```bash
# View running containers
docker ps

# View all containers
docker ps -a

# View logs (real-time)
docker logs -f frontend
docker-compose logs -f frontend

## OR
# for macOS
docker compose logs -f frontend

# View logs (last 100 lines)
docker logs --tail 100 frontend

# Check container health
docker inspect --format='{{.State.Health.Status}}' frontend

# Access container shell for debugging
docker exec -it frontend /bin/bash

# Remove image
docker rmi student-loan-frontend

# Clean up unused Docker resources
docker system prune

# Force cleanup (removes all unused containers, networks, images)
docker system prune -a --volumes
```

### Environment Variables & Configuration

The frontend container supports the following environment variables:

```bash
# Set custom backend URL (overrides auto-detection)
docker run -e BACKEND_URL=http://custom-backend:8000 -p 3000:3000 student-loan-frontend

# Enable debug mode for container discovery
docker run -e DEBUG=true -p 3000:3000 student-loan-frontend
```

### Network Architecture

```
┌─────────────────┐    ┌──────────────────┐
│   Frontend      │    │    Backend       │
│  (Port 3000)    │◄──►│  (Port 8000)     │
│                 │    │                  │
│ Auto-detects:   │    │ Service Name:    │
│ • localhost     │    │ • rag-api        │
│ • rag-api:8000  │    │                  │
└─────────────────┘    └──────────────────┘
          │                       │
          └───────────────────────┘
```

**Note:** When using frontend-only deployment, ensure the backend is running and accessible. The frontend will attempt container discovery first, then fall back to localhost:8000.

## Development

The frontend is vanilla HTML/CSS/JavaScript with Tailwind CSS via CDN. No build process or dependencies required.

For production deployment, you may need to update the API URL in `index.html` from `http://localhost:8000` to your production backend URL.

## Docker Files

- `Dockerfile` - Frontend container configuration (Python 3.11-slim base)
- `docker-compose.yml` - Frontend-only deployment with container discovery
- `docker-start.sh` - Enhanced startup script with cleanup, health checks, and environment validation
- `.dockerignore` - Files to exclude from Docker build context
- `.gitignore` - Version control exclusions for development

## Troubleshooting

### Common Issues

**1. Frontend can't connect to backend:**
```bash
# Check if backend container is running
docker ps | grep rag-api

# Check network connectivity
docker network ls
docker network inspect student-loan-network

# View frontend logs for connection errors
docker logs frontend
```

**2. Port conflicts:**
```bash
# Check what's using port 3000
lsof -i :3000

# Use different port
docker run -p 3001:3000 student-loan-frontend
```

**3. Container cleanup needed:**
```bash
# Run the enhanced cleanup script
./docker-start.sh  # Includes automatic cleanup

# Manual cleanup
docker stop $(docker ps -q)
docker rm $(docker ps -aq)
docker rmi $(docker images -q --filter "dangling=true")
```

**4. Environment file issues:**
```bash
# Ensure .env exists in project root
ls -la ../.env

# Check required variables
cat ../.env | grep -E "(OPENAI_API_KEY|TAVILY_API_KEY|COHERE_API_KEY)"
```

### Health Checks

The frontend container includes built-in health checks:
- **Endpoint:** `curl -f http://localhost:3000`
- **Interval:** 30 seconds
- **Timeout:** 10 seconds  
- **Retries:** 3
- **Start Period:** 10 seconds

Monitor health status:
```bash
docker inspect --format='{{.State.Health}}' frontend
```