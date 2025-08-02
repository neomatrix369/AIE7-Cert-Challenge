# Federal Student Loan Assistant - Frontend

A simple HTML/CSS/JavaScript chat interface for the Federal Student Loan RAG backend API.

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

### Option 1: Full-Stack with Docker Compose (Recommended)

Deploy both frontend and backend together:

```bash
# Quick start with script
./docker-start.sh
```

Or manually:
```bash
# Build and start both services
docker-compose up --build

# Run in background (detached mode)
docker-compose up -d --build

# Stop all services
docker-compose down
```

This will start:
- **Frontend:** http://localhost:3000  
- **Backend API:** http://localhost:8000
- **API Docs:** http://localhost:8000/docs

### Option 2: Frontend Container Only

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

### Option 3: Development with Live Reload

For development with file watching:

```bash
# Run with volume mounting for live changes
docker run -p 3000:3000 -v $(pwd):/app/frontend student-loan-frontend
```

### Docker Management Commands

```bash
# View running containers
docker ps

# View all containers
docker ps -a

# View logs
docker logs frontend
docker-compose logs frontend

# Remove image
docker rmi student-loan-frontend

# Clean up unused Docker resources
docker system prune
```

**Note:** When using frontend-only deployment, make sure the backend is running separately and accessible at http://localhost:8000.

## Development

The frontend is vanilla HTML/CSS/JavaScript with Tailwind CSS via CDN. No build process or dependencies required.

For production deployment, you may need to update the API URL in `index.html` from `http://localhost:8000` to your production backend URL.

## Docker Files

- `Dockerfile` - Frontend container configuration
- `docker-compose.yml` - Full-stack deployment with backend
- `docker-start.sh` - Startup script with health checks
- `.dockerignore` - Files to exclude from Docker build