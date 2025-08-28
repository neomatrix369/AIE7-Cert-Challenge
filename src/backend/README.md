# Simple Federal Student Loan RAG API

A streamlined FastAPI backend for the AIE7 Certification Challenge that provides a single `/ask` endpoint for federal student loan questions.

## 📖 Table of Contents

- [✅ Features](#features)
- [🚀 Quick Start](#quick-start)
  - [Option 1: Local Development](#option-1-local-development)
    - [1. Environment Setup](#1-environment-setup)
    - [2. Install Dependencies](#2-install-dependencies)
    - [3. Run the API](#3-run-the-api)
    - [4. Test the API](#4-test-the-api)
  - [Option 2: Docker Deployment](#option-2-docker-deployment)
    - [1. Environment Setup](#1-environment-setup-1)
    - [2. Build and Run with Docker Compose (Recommended)](#2-build-and-run-with-docker-compose-recommended)
    - [3. Manual Docker Build and Run](#3-manual-docker-build-and-run)
    - [4. Test the Dockerized API](#4-test-the-dockerized-api)
  - [Docker Management Commands](#docker-management-commands)
  - [Network and Volume Management](#network-and-volume-management)
- [🔗 API Usage](#api-usage)
  - [Ask Endpoint](#ask-endpoint)
  - [Enhanced Response Fields](#enhanced-response-fields)
  - [Other Endpoints](#other-endpoints)
- [❓ Example Questions](#example-questions)
- [📚 Knowledge Base](#knowledge-base)
  - [Federal Policy Documents (PDF)](#federal-policy-documents-pdf)
  - [Real Customer Data (CSV)](#real-customer-data-csv)
- [📊 Performance](#performance)
- [🏗️ Architecture](#architecture)
- [🛠️ Development](#development)
  - [Project Structure](#project-structure)
  - [Adding New Questions](#adding-new-questions)
  - [Monitoring](#monitoring)
- [🚀 Deployment](#deployment)
  - [Docker Production Deployment](#docker-production-deployment)
  - [Container Features](#container-features)
  - [Multi-Container Setup (Backend + Frontend)](#multi-container-setup-backend--frontend)
  - [Production Deployment Considerations](#production-deployment-considerations)
- [🔧 Troubleshooting](#troubleshooting)
  - [Common Issues](#common-issues)
  - [Performance Optimization](#performance-optimization)
  - [Log Analysis](#log-analysis)
- [💬 Support](#support)

## Features

✅ **Single Purpose API** - One `/ask` endpoint for student loan questions  
✅ **Best Performing RAG** - Uses Naive retrieval (highest RAGAS scores)  
✅ **Hybrid Knowledge Base** - Combines federal policies + real customer complaints  
✅ **Simple Request/Response** - JSON in, JSON out  
✅ **No Complex Features** - No file uploads, chat sessions, or user management  

## Quick Start

### Option 1: Local Development

#### 1. Environment Setup

```bash
# Copy environment variables
cp .env.example .env

# Add your API keys to .env:
OPENAI_API_KEY=your_openai_key_here
COHERE_API_KEY=your_cohere_key_here
TAVILY_API_KEY=your_tavily_key_here  # Optional
LANGCHAIN_API_KEY=your_langsmith_key_here  # Optional, for tracing
LANGCHAIN_TRACING_V2=true  # Optional, enables LangSmith tracing
LANGCHAIN_PROJECT=student-loan-rag  # Optional, LangSmith project name
```

#### 2. Install Dependencies

```bash
# Install requirements
pip install -r requirements.txt
```

#### 3. Run the API

```bash
# Start the server
python simple_api.py

# The API will be available at:
# http://localhost:8000
# API docs: http://localhost:8000/docs
```

#### 4. Test the API

```bash
# Quick test
python test_simple_api.py --quick

# Full test suite
python test_simple_api.py
```

### Option 2: Docker Deployment

#### 1. Environment Setup

```bash
# Ensure .env file exists in project root with API keys:
OPENAI_API_KEY=your_openai_key_here
COHERE_API_KEY=your_cohere_key_here
TAVILY_API_KEY=your_tavily_key_here  # Optional
LANGCHAIN_API_KEY=your_langsmith_key_here  # Optional, for tracing
LANGCHAIN_TRACING_V2=true  # Optional, enables LangSmith tracing  
LANGCHAIN_PROJECT=student-loan-rag  # Optional, LangSmith project name
```

#### 2. Build and Run with Docker Compose (Recommended)

**IMPORTANT:** Run these commands from the backend directory (`src/backend/`):

```bash
# Navigate to the backend directory
cd src/backend/

# Build and run with Docker Compose
docker-compose up --build

# Or run in background (detached mode)
docker-compose up --build -d

# View logs (real-time)
docker-compose logs -f

# View logs (last 100 lines)
docker-compose logs --tail 100

# Stop the container
docker-compose down

# Stop and remove volumes (complete cleanup)
docker-compose down -v
```

OR

```bash
# for macOS

# Navigate to the backend directory
cd src/backend/

# Build and run with Docker Compose
docker compose up --build

# Or run in background (detached mode)
docker compose up --build -d

# View logs (real-time)
docker compose logs -f

# View logs (last 100 lines)
docker compose logs --tail 100

# Stop the container
docker compose down

# Stop and remove volumes (complete cleanup)
docker compose down -v
```

> **Note:** The docker-compose.yml is configured to:
> - Build from the project root (`../../`) to access all `src/` dependencies
> - Mount volumes for persistent cache and live development
> - Include health checks and auto-restart policies

**_NOTE: Please give the app a good 'few' minutes or so to get started, as loading 2000+ docs inside the Docker container takes a bit of time. Till then we are not able to ping the backend server._**


#### 3. Manual Docker Build and Run

**IMPORTANT:** Build from project root, but .env must be in project root:

```bash
# Step 1: Navigate to PROJECT ROOT (AIE7-Cert-Challenge/)
cd ../../  # If you're in src/backend/, go back to project root

# Step 2: Ensure .env file exists in project root with API keys
ls .env  # Should show your .env file

# Step 3: Build the Docker image (FROM PROJECT ROOT)
docker build -f src/backend/Dockerfile -t rag-api .

# Step 4: Run the container (FROM PROJECT ROOT)
docker run -p 8000:8000 \
  --env-file .env \
  -v $(pwd)/data:/app/data:ro \
  --name rag-api-container \
  rag-api

# Stop the container
docker stop rag-api-container
docker rm rag-api-container
```

> **Why these folder positions matter:**
> - **Build context:** Must be project root to access `src/core/`, `src/agents/`, etc.
> - **Environment file:** `.env` must be in project root where API keys are stored
> - **Data volume:** `data/` folder is in project root
> - **Dependencies:** Backend imports modules from `src/core/`, `src/agents/`, etc.

#### 4. Test the Dockerized API

```bash
# Test health endpoint
curl http://localhost:8000/health

# Test ask endpoint
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "What are income-driven repayment plans?"}'

# Test with maximum response length
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "Explain federal student loan forgiveness options", "max_response_length": 1500}'

# Check API information
curl http://localhost:8000/api-info
```

### Docker Management Commands

```bash
# View running containers
docker ps

# Check container health status
docker inspect --format='{{.State.Health.Status}}' backend_rag-api_1

# Access container shell for debugging
docker exec -it backend_rag-api_1 /bin/bash

# View container resource usage
docker stats backend_rag-api_1

# Check container logs with timestamps
docker logs -t backend_rag-api_1

# Follow logs in real-time with timestamps
docker logs -ft backend_rag-api_1

# Export container logs to file
docker logs backend_rag-api_1 > rag-api.log 2>&1
```

### Network and Volume Management

```bash
# Inspect the shared network
docker network inspect backend_student-loan-network

# List all networks
docker network ls

# Check volume usage
docker volume ls
docker volume inspect backend_cache_volume

# Backup cache volume (optional)
docker run --rm -v backend_cache_volume:/data -v $(pwd):/backup alpine tar czf /backup/cache-backup.tar.gz -C /data .

# Restore cache volume (optional)
docker run --rm -v backend_cache_volume:/data -v $(pwd):/backup alpine tar xzf /backup/cache-backup.tar.gz -C /data
```

## API Usage

### Ask Endpoint

**POST** `/ask`

**Request:**
```json
{
  "question": "What should I do if I can't make my student loan payments?",
  "max_response_length": 2000  // Optional
}
```

**Response:**
```json
{
  "answer": "If you can't make your student loan payments, you have several options: 1. Contact your loan servicer immediately to discuss...",
  "sources_count": 5,
  "success": true,
  "message": "Question processed successfully",
  "source_details": [
    "Document context 1: Federal loan policies...",
    "Document context 2: Customer complaint about payment issues...",
    "Document context 3: Servicer guidelines for hardship..."
  ],
  "tools_used": [
    "ask_naive_llm_tool"
  ],
  "performance_metrics": {
    "response_time_ms": 3200,
    "retrieval_time_ms": 1920,
    "generation_time_ms": 1280,
    "tokens_used": 1850,
    "input_tokens": 420,
    "output_tokens": 1430,
    "retrieval_method": "naive",
    "total_contexts": 5
  }
}
```

### Enhanced Response Fields

The `/ask` endpoint returns comprehensive information about the RAG pipeline execution:

- **`answer`**: Generated response text
- **`sources_count`**: Number of knowledge sources used  
- **`success`**: Boolean indicating success
- **`message`**: Status message
- **`source_details`**: Raw contexts retrieved from documents (for transparency)
- **`tools_used`**: List of RAG tools/retrievers used in processing
- **`performance_metrics`**: Detailed metrics object containing:
  - `response_time_ms`: Total response time
  - `retrieval_time_ms`: Time spent on document retrieval
  - `generation_time_ms`: Time spent on text generation
  - `tokens_used`: Total tokens consumed
  - `input_tokens`: Input/prompt tokens
  - `output_tokens`: Generated tokens
  - `retrieval_method`: Retrieval method used
  - `total_contexts`: Number of contexts retrieved

### Other Endpoints

- **GET** `/` - API information
- **GET** `/health` - Health check
- **GET** `/api-info` - Detailed capabilities and knowledge base info
- **GET** `/docs` - Interactive API documentation

## Example Questions

- "What are the requirements for federal student loan forgiveness?"
- "How do I apply for income-driven repayment plans?"
- "What should I do if my loan servicer is not responding?"
- "What are the differences between federal and private student loans?"
- "Can I get help with Nelnet payment issues?"

## Knowledge Base

The API uses a **hybrid dataset** containing:

### Federal Policy Documents (PDF)
- Academic Calendars, Cost of Attendance, and Packaging
- Applications and Verification Guide  
- The Federal Pell Grant Program
- The Direct Loan Program

### Real Customer Data (CSV)
- 4,547 customer complaints (note: CSV complaints count further reduces down to 825 complaints and then to 480 after a cleaning/filtering process)
- Real servicer issues (Nelnet, Aidvantage, Mohela, etc.)
- Practical scenarios and solutions


### Data Requirements
The project includes a hybrid dataset:
- **4 PDF documents** (~4MB) - Federal student loan policies
- **CSV complaints file** (~12MB) - Real customer complaint data => 4,547 raw → 825 unfiltered -> 480 usable rows (10% retention after quality filtering)
- **Vector embeddings** (~39MB in memory) - Generated from documents

### Complaints Dataset Processing:
- **Raw CSV**: 4,547 total records
- **After loading Dataset**: 825 pre-quality checked complaints (18% retention)
- **Quality Filters Applied**:
  - ❌ Narratives < 100 characters
  - ❌ Excessive redaction (>5 XXXX tokens)
  - ❌ Empty/None/N/A content
- **Final Dataset**: 480 filtered complaints (11% retention)
- **Rationale**: Ensures meaningful content for
RAG retrieval

Note: the logs produced at runtime as well as the in the body of the [notebook](../../notebooks/Agentic%20RAG%20evaluation%20experiments.ipynb) make these statistics clear.

## Performance

Based on RAGAS evaluation results:
- **Faithfulness:** 0.90 (measures how factually accurate the generated answer is)
- **Answer Relevancy:** 0.79 (evaluates how relevant the answer is to the question)
- **Context Precision:** 0.69 (assesses the signal-to-noise ratio of the retrieved contexts)
- **Context Recall:** 0.64 (measures the ability of the retriever to retrieve all relevant information)
- **Answer Correctness:** 0.60 (evaluates the accuracy of the answer against the ground truth)

## Architecture

```
User Question → Naive RAG → Hybrid Knowledge Base → AI Response
                     ↓
               [Federal Policies + Customer Complaints]
```

**Retrieval Method:** Naive (standard chunking)  
**LLM:** GPT-4.1-nano for response generation  
**Vector Store:** Qdrant (in-memory for development)  
**Evaluation:** RAGAS framework validated

## Development

### Project Structure
```
simple_api.py              # Main FastAPI application
test_simple_api.py         # Test suite  
requirements.txt           # Dependencies
Dockerfile                 # Docker container configuration
docker-compose.yml         # Docker Compose setup
src/                       # RAG components
├── core/                  # Data loading and processing
├── agents/                # LangGraph agents and tools  
├── evaluation/            # RAGAS evaluation pipeline
└── tools/                 # External search tools
```

### Adding New Questions

The API automatically handles new questions through the RAG system. No manual updates needed.

### Monitoring

Check logs for:
- Question processing times
- RAG agent status
- Error handling
- Response quality

## Deployment

### Docker Production Deployment

The API is containerized and ready for production deployment:

**Option A: Docker Compose (from `src/backend/` directory):**
```bash
cd src/backend/
docker-compose up --build -d

## OR

docker compose up --build -d
```

**Option B: Manual Docker (from project root):**
```bash
# Ensure you're in project root (AIE7-Cert-Challenge/)
docker build -f src/backend/Dockerfile -t rag-api:latest .
docker run -d \
  --name rag-api-prod \
  -p 8000:8000 \
  --env-file .env \
  --restart unless-stopped \
  rag-api:latest
```

### Container Features
- **Health checks:** Built-in health monitoring at `/health` (30s intervals)
- **Persistent cache:** Volume-mounted cache directory for RAGAS evaluations
- **Read-only data:** Data directory mounted as read-only for security
- **Auto-restart:** Container restarts on failure (unless-stopped policy)
- **Resource monitoring:** Built-in container stats and logging
- **Development support:** Live code mounting for development

### Multi-Container Setup (Backend + Frontend)

The backend creates a shared Docker network that the frontend can connect to:

```bash
# Start backend (creates network)
cd src/backend/
docker-compose up -d

## OR

docker compose up -d

# Start frontend (connects to backend network)
cd ../../frontend/
docker-compose up --build

## OR

docker compose up --build
```

**Network Architecture:**
```
┌─────────────────┐    ┌──────────────────┐
│   Frontend      │    │    Backend       │
│  (Port 3000)    │◄──►│  (Port 8000)     │
│                 │    │                  │
│ Connects to:    │    │ Service Name:    │
│ • rag-api:8000  │    │ • rag-api        │
└─────────────────┘    └──────────────────┘
          │                       │
          └───────────────────────┘
```

### Production Deployment Considerations

1. **Environment Variables:** Use Docker secrets or secure env injection
2. **Vector Store:** Consider persistent Qdrant deployment for scale
3. **Rate Limiting:** Add rate limiting middleware for API protection
4. **Authentication:** Implement API key authentication if needed
5. **CORS:** Configure CORS origins for production domains
6. **Load Balancing:** Use reverse proxy (nginx/traefik) for multiple containers
7. **Monitoring:** Enable container monitoring, log aggregation, and metrics
8. **Resource Limits:** Set CPU and memory limits in production
9. **Security:** Run containers as non-root user, scan images for vulnerabilities
10. **Backup:** Regular backup of cache volumes and configuration

## Troubleshooting

### Common Issues

**1. "RAG agent is not initialized"**
```bash
# Check environment variables are set
docker exec -it backend_rag-api_1 env | grep -E "(OPENAI|COHERE|TAVILY)"

# Verify .env file is in project root
ls -la ../../.env

# Check OpenAI API key is valid
curl -H "Authorization: Bearer $OPENAI_API_KEY" https://api.openai.com/v1/models
```

**2. "Module not found" errors**
```bash
# Ensure Docker build context is correct (project root)
docker build -f src/backend/Dockerfile -t rag-api .  # From project root

# Check container filesystem structure
docker exec -it backend_rag-api_1 ls -la /app/src/

# Verify requirements are installed
docker exec -it backend_rag-api_1 pip list | grep -E "(langchain|openai|fastapi)"
```

**3. Slow responses or timeouts**
```bash
# Check container resource usage
docker stats backend_rag-api_1

# Monitor logs for performance issues
docker logs -f backend_rag-api_1 | grep -E "(⏱️|Processing|seconds)"

# Increase container memory limit
# Add to docker-compose.yml:
# deploy:
#   resources:
#     limits:
#       memory: 2G
```

**4. Empty or poor responses**
```bash
# Check hybrid dataset loading
docker logs backend_rag-api_1 | grep -E "(dataset|loading|Qdrant)"

# Verify data volume is mounted correctly
docker exec -it backend_rag-api_1 ls -la /app/data/

# Test with simple question
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is a federal student loan?"}'
```

**5. Docker network issues (Frontend can't reach Backend)**
```bash
# Check if backend network exists
docker network ls | grep student-loan

# Verify backend container is on the network
docker network inspect backend_student-loan-network

# Test network connectivity from frontend container
docker exec -it frontend curl http://rag-api:8000/health
```

**6. Port conflicts**
```bash
# Check what's using port 8000
lsof -i :8000

# Use different port mapping
docker run -p 8001:8000 rag-api
```

**7. Container health check failures**
```bash
# Check health status
docker inspect --format='{{.State.Health}}' backend_rag-api_1

# Manual health check
curl -f http://localhost:8000/health

# View health check logs
docker inspect backend_rag-api_1 | jq '.[0].State.Health.Log'
```

### Performance Optimization

**Resource Monitoring:**
```bash
# Monitor container resources in real-time
docker stats backend_rag-api_1

# Check memory usage patterns
docker exec -it backend_rag-api_1 free -h

# Monitor disk I/O for cache operations
docker exec -it backend_rag-api_1 df -h
```

**Cache Management:**
```bash
# Check cache volume size
docker volume inspect backend_cache_volume

# Clear cache if needed (will regenerate)
docker volume rm backend_cache_volume
docker-compose down && docker-compose up --build

## OR

docker compose down && docker compose up --build

# Monitor cache usage
docker exec -it backend_rag-api_1 du -sh /app/cache/
```

### Log Analysis

```bash
# Filter logs by component
docker logs backend_rag-api_1 | grep "🔍"  # RAG agent operations
docker logs backend_rag-api_1 | grep "📝"  # Question processing
docker logs backend_rag-api_1 | grep "⏱️"   # Performance metrics
docker logs backend_rag-api_1 | grep "❌"   # Errors

# Export logs with timestamps for analysis
docker logs -t backend_rag-api_1 > backend-analysis.log 2>&1

# Follow logs with filtering
docker logs -f backend_rag-api_1 | grep -E "(ERROR|WARNING|📝|✅)"
```

## Support

For issues related to:
- **API bugs:** Check the logs and test suite results
- **Knowledge gaps:** The hybrid dataset covers federal loans comprehensively
- **Performance:** Response times vary with question complexity

This API implements the requirements from the Project Completion Plan for a minimal viable product (MVP) focused specifically on federal student loan customer service.