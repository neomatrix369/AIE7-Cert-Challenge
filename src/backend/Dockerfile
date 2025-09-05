# Federal Student Loan RAG System - Multi-stage Dockerfile
# Stage 1: Base Python environment with uv
FROM python:3.12-slim AS base

# Install system dependencies and uv
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir uv

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV UV_CACHE_DIR=/tmp/uv-cache

# 
# File structure in container:
# 
# /app/
# ├── src/
# │   ├── backend/
# │   │   └── simple_api.py  ← API runs from here
# │   ├── core/
# │   ├── agents/
# │   └── ...
# ├── data/
# ├── cache/
# └── notebooks/
# 

# Copy project files for dependency installation
COPY pyproject.toml uv.lock ./

# Install dependencies using uv (much faster than pip)
RUN uv sync --frozen --no-dev --no-install-project

# Copy source code
COPY src/ ./src/
COPY data/ ./data/

# Create necessary directories
RUN mkdir -p cache notebooks golden-masters metrics

# Stage 2: FastAPI Backend Service
FROM base AS backend

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the API server
CMD ["uv", "run", "python", "/app/src/backend/simple_api.py"]

# Stage 3: Jupyter Notebook Server
FROM base AS jupyter

# Jupyter dependencies are already installed in base stage via pyproject.toml
# No additional installation needed

# Expose Jupyter port
EXPOSE 8888

# Create Jupyter config directory
RUN mkdir -p /root/.jupyter

# Configure Jupyter
RUN echo "c.ServerApp.ip = '0.0.0.0'" >> /root/.jupyter/jupyter_lab_config.py \
    && echo "c.ServerApp.port = 8888" >> /root/.jupyter/jupyter_lab_config.py \
    && echo "c.ServerApp.open_browser = False" >> /root/.jupyter/jupyter_lab_config.py \
    && echo "c.ServerApp.allow_root = True" >> /root/.jupyter/jupyter_lab_config.py

# Default command (can be overridden in docker-compose)
CMD ["uv", "run", "jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]