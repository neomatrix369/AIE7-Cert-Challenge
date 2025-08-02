# Federal Student Loan Assistant Frontend Dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create frontend directory
RUN mkdir -p /app/frontend

# Copy frontend files
COPY index.html /app/frontend/
COPY serve.py /app/frontend/
COPY README.md /app/frontend/

# Make serve script executable
RUN chmod +x /app/frontend/serve.py

# Expose port
EXPOSE 3000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:3000 || exit 1

# Set working directory to frontend
WORKDIR /app/frontend

# Run the frontend server
CMD ["python3", "serve.py", "3000"]