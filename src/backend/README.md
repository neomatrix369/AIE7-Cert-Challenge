# Simple Federal Student Loan RAG API

A streamlined FastAPI backend for the AIE7 Certification Challenge that provides a single `/ask` endpoint for federal student loan questions.

## Features

✅ **Single Purpose API** - One `/ask` endpoint for student loan questions  
✅ **Best Performing RAG** - Uses Parent Document retrieval (highest RAGAS scores)  
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

# Or run in background
docker-compose up --build -d

# View logs
docker-compose logs -f

# Stop the container
docker-compose down
```

> **Note:** The docker-compose.yml is configured to build from the project root (`../../`) to access all `src/` dependencies while running from the backend directory.

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
    "ask_parent_document_llm_tool"
  ],
  "performance_metrics": {
    "response_time_ms": 3200,
    "retrieval_time_ms": 1920,
    "generation_time_ms": 1280,
    "tokens_used": 1850,
    "input_tokens": 420,
    "output_tokens": 1430,
    "retrieval_method": "parent_document",
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
- 4,547 filtered customer complaints
- Real servicer issues (Nelnet, Aidvantage, Mohela, etc.)
- Practical scenarios and solutions

## Performance

Based on RAGAS evaluation results:
- **Context Recall:** 0.89 (excellent retrieval quality)
- **Faithfulness:** 0.82 (high factual consistency)
- **Answer Relevancy:** 0.62 (good relevance)
- **Response Time:** ~3-8 seconds per question

## Architecture

```
User Question → Parent Document RAG → Hybrid Knowledge Base → AI Response
                     ↓
               [Federal Policies + Customer Complaints]
```

**Retrieval Method:** Parent Document (small-to-big chunking)  
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
- **Health checks:** Built-in health monitoring at `/health`
- **Persistent cache:** Volume-mounted cache directory
- **Read-only data:** Data directory mounted as read-only
- **Auto-restart:** Container restarts on failure
- **Resource limits:** Configure via Docker Compose

For production deployment considerations:

1. **Environment Variables:** Set in production environment or Docker secrets
2. **Vector Store:** Consider persistent Qdrant deployment  
3. **Rate Limiting:** Add rate limiting middleware
4. **Authentication:** Add API key authentication if needed
5. **CORS:** Configure CORS origins appropriately
6. **Load Balancing:** Use reverse proxy (nginx/traefik) for multiple containers
7. **Monitoring:** Enable container monitoring and log aggregation

## Troubleshooting

### Common Issues

**"RAG agent is not initialized"**
- Check environment variables are set
- Verify .env file is in project root
- Check OpenAI API key is valid

**"Module not found" errors**
- Ensure you're running from project root directory
- Check all requirements are installed
- Verify src/ folder structure is intact

**Slow responses**
- Normal for first request (model loading)
- Subsequent requests should be faster
- Check network connectivity

**Empty or poor responses**
- Verify hybrid dataset is loaded properly
- Check question is related to student loans
- Review logs for processing errors

## Support

For issues related to:
- **API bugs:** Check the logs and test suite results
- **Knowledge gaps:** The hybrid dataset covers federal loans comprehensively
- **Performance:** Response times vary with question complexity

This API implements the requirements from the Project Completion Plan for a minimal viable product (MVP) focused specifically on federal student loan customer service.