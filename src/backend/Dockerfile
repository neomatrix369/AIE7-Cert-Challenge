# Federal Student Loan RAG API Dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install build dependencies
RUN pip install --upgrade pip setuptools wheel

# 
# File structure in container:
# 
# /app/
# ├── src/
# │   ├── backend/
# │   │   └── simple_api.py  ← 
# runs from here
# │   ├── core/
# │   ├── agents/
# │   └── ...
# ├── data/
# └── cache/
# 

# Copy requirements first (for better caching)
COPY src/backend/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire source code (this puts src/ contents into /app/)
COPY src/__init__.py ./src/__init__.py
COPY src/agents ./src/agents
COPY src/core ./src/core
COPY src/tools ./src/tools
COPY src/backend/simple_api.py ./src/backend/simple_api.py

# Copy data directory
COPY data ./data

# Create necessary directories
RUN mkdir -p cache

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application from the backend directory
CMD ["python", "/app/src/backend/simple_api.py"]