# Federal Student Loan AI Assistant

**AIE7 Certification Challenge** - Standard RAG system for federal student loan customer service

An intelligent assistant that combines official federal loan policies with real customer experiences to provide comprehensive guidance on student loan questions, repayment options, forgiveness programs, and servicer issues.

---

### Written report can be found [here](./docs/Report.md)
### Loom video can be found [here](https://www.loom.com/share/d89df95081c6407fbb705c03929e8f55?sid=c51a16f9-4cdb-4106-8cde-5aa344de4b63)

---

## 📖 Table of Contents

- [🚀 Quick Start](#-quick-start)
  - [1. Environment Setup](#1-environment-setup)
  - [2. Start Backend with Docker](#2-start-backend-with-docker-from-the-root-folder-of-the-project)
  - [3. Start the Frontend with Docker](#3-start-the-frontend-with-docker-new-terminal-from-the-root-folder-of-the-project)
  - [4. Open & Use the Assistant](#4-open--use-the-assistant)
- [✨ Core Features](#-core-features)
- [📁 Project Structure](#-project-structure)
- [🔗 API Usage](#-api-usage)
- [🛠 Development](#-development)
  - [Component Documentation](#component-documentation)
  - [Docker Deployment](#docker-deployment)
- [📋 Requirements](#-requirements)
  - [System Requirements](#system-requirements)
  - [Development Requirements](#development-requirements-if-not-using-docker)
    - [Backend Requirements](#backend-requirements)
    - [Frontend Requirements](#frontend-requirements)
  - [API Keys (Required)](#api-keys-required)
  - [General Environment Setup Dependencies](#general-environment-setup-dependencies)
    - [Core Tools](#core-tools)
  - [Native Environment Setup Dependencies](#native-environment-setup-dependencies)
    - [Python Dependencies (Backend)](#python-dependencies-backend)
    - [Node.js Dependencies (Frontend)](#nodejs-dependencies-frontend)
  - [Hardware Recommendations](#hardware-recommendations)
  - [Port Usage](#port-usage)
  - [Data Requirements](#data-requirements)

## 🚀 Quick Start

Get the entire RAG system running in 2 simple steps:

### 1. Environment Setup
```bash
# Copy environment template and add your API keys
cp .env-example .env

# Edit .env file with your required API keys:
# OPENAI_API_KEY=your_key_here (Required)
# COHERE_API_KEY=your_key_here (Required)  
# TAVILY_API_KEY=your_key_here (Required)
# LANGCHAIN_API_KEY=your_key_here (Optional - for tracing)
```

### 2. Start All Services with Docker
```bash
# Option 1: Automated setup (Recommended)
./start-services.sh

# Option 2: Manual Docker Compose
docker-compose up --build -d

# Option 3: Alternative setup for macOS
docker compose up --build -d
```

**🎉 That's it!** All services will start automatically:
- **📊 Qdrant** (Vector Database): http://localhost:6333/dashboard
- **🤖 Backend API**: http://localhost:8000 
- **📚 Jupyter Notebooks**: http://localhost:8888
- **📖 API Documentation**: http://localhost:8000/docs
- **🎨 Frontend Dashboard**: http://localhost:3000 (optional)

### ⏹️ Stop All Services
```bash
# Option 1: Using the stop script
./stop-services.sh

# Option 2: Direct Docker Compose
docker-compose down

# Option 3: Clean shutdown (removes containers)
./stop-services.sh --clean
```

**_NOTE: The system loads a hybrid dataset (PDF policies + customer complaints) which may take 1-2 minutes to initialize. Check the Qdrant dashboard to monitor vector ingestion progress._**

### 🔧 Service Management
```bash
# View logs for all services
docker-compose logs -f

# View logs for specific service
docker-compose logs -f backend
docker-compose logs -f jupyter

# Restart specific service  
docker-compose restart backend
```

### 3. Open & Use the System
Once all services are running, you can access:

- **🎨 Frontend Dashboard**: http://localhost:3000 - Interactive chat interface
- **📚 Jupyter Notebooks**: http://localhost:8888 - RAG experiments and analysis
- **📖 API Documentation**: http://localhost:8000/docs - REST API endpoints
- **📊 Qdrant Dashboard**: http://localhost:6333/dashboard - Vector database monitoring

### 4. Running RAG Experiments

#### Option 1: Using Docker (Recommended)
```bash
# Jupyter is already running at http://localhost:8888
# Open the notebooks directly in your browser
```

#### Option 2: Local Development
```bash
# Install dependencies locally
uv sync

# Start Jupyter from project root
uv run jupyter lab
```

When inside Jupyter Labs, you can access the main evaluation notebook:
- [Agentic RAG evaluation experiments.ipynb](./notebooks/Agentic%20RAG%20evaluation%20experiments.ipynb)

## ✨ Core Features

- **🎯 Federal Student Loan Expert** - Trained on official policies + real customer complaints
- **🔍 Standard RAG** - Multiple retrieval methods (Naive performs best)
- **💬 Chat Interface** - Clean, responsive web interface for questions
- **📊 Performance Metrics** - Response time, sources used, retrieval quality
- **🔌 API Ready** - Single `/ask` endpoint for integration
- **🐳 Docker Support** - Full containerization for both backend and frontend

## 📁 Project Structure

```
├── src/backend/          # FastAPI server with RAG endpoint
├── frontend/             # HTML/JS chat interface  
├── data/                 # Federal loan PDFs + complaints CSV
├── src/core/             # RAG retrieval implementations
├── src/agents/           # LangGraph agent orchestration
├── notebooks/            # Research & evaluation experiments
└── docs/                 # Project documentation
```

## 🔗 API Usage

**POST** `/ask` - Ask any federal student loan question
```json
{
  "question": "What are income-driven repayment plans?",
  "max_response_length": 2000
}
```

**Response includes:**
- Generated answer with sources
- Performance metrics (response time, tokens used)
- Source documents used for transparency

## 🛠 Development

This project implements cutting-edge RAG techniques:
- **Hybrid Dataset**: Official policies + real customer scenarios  
- **Multiple Retrievers**: Naive, Multi-Query, Parent-Document, Contextual Compression
- **Agent Framework**: LangGraph with tool orchestration
- **Evaluation**: RAGAS metrics for retrieval quality assessment

### Component Documentation
- **Backend Details**: [`src/backend/README.md`](src/backend/README.md)
- **Frontend Setup**: [`frontend/README.md`](frontend/README.md)

### Docker Deployment
```bash
# Full-stack deployment
cd src/backend && docker-compose up --build
cd frontend && docker-compose up --build
```

OR

```bash
# Full-stack deployment (for macOS)
cd src/backend && docker compose up --build
cd frontend && docker compose up --build
```


## 📋 Requirements

### System Requirements
- **Docker** (recommended) - For containerized deployment
- **Docker Compose** - For multi-container orchestration
- **Git** - For cloning the repository
- **Modern Browser** - Chrome, Firefox, Safari, or Edge

### Development Requirements (if not using Docker)

#### Backend Requirements
- **Python 3.11+** (tested with 3.11, minimum 3.8)
- **System Dependencies** (for native installation):
  - `gcc` and `g++` (build tools)
  - `curl` (for health checks)
  - `build-essential` (Linux/WSL)

#### Frontend Requirements  
- **Node.js 18+** with npm
- **Next.js 14.2+** (React framework)
- **TypeScript 5.1+** (for type safety)

### API Keys (Required)
Create a `.env` file in the project root with:
```bash
# Required API Keys
OPENAI_API_KEY=your_openai_key_here          # For LLM and embeddings
COHERE_API_KEY=your_cohere_key_here          # For reranking functionality

# Optional API Keys
TAVILY_API_KEY=your_tavily_key_here          # For external search (optional)
LANGCHAIN_API_KEY=your_langsmith_key_here    # For tracing/monitoring (optional)
```

### General environment setup dependencies

#### Core Tools
- **Git** (2.25+) - Version control and repository cloning
- **Docker** (20.10+) - Container runtime and orchestration
- **Docker Compose** (2.0+) - Multi-container application management
- **curl** or **wget** - For API testing and health checks

### Native environment setup dependencies
#### Python Dependencies (Backend)
The backend requires 50+ Python packages including:
- **AI/ML**: `langchain`, `langgraph`, `openai`, `cohere`, `ragas`
- **Vector DB**: `qdrant-client`, `langchain-qdrant`
- **Document Processing**: `pypdf2`, `pymupdf`, `unstructured`
- **Web Framework**: `fastapi`, `uvicorn`, `pydantic`
- **Data Science**: `numpy`, `pandas`, `matplotlib`, `seaborn`
- **Search**: `tavily-python`, `rank-bm25`
- **Utilities**: `python-dotenv`, `joblib`, `tqdm`

#### Node.js Dependencies (Frontend)
- **React 18.2+** with Next.js framework
- **UI Components**: `lucide-react` (icons)
- **Styling**: `tailwindcss`, `autoprefixer`, `postcss`
- **Development**: TypeScript, ESLint, development server

### Hardware Recommendations
- **Memory**: 4GB+ RAM (8GB+ recommended for better performance)
- **Storage**: 2GB+ free space for dependencies and data
- **CPU**: Multi-core processor recommended for faster RAG processing
- **Network**: Stable internet connection for API calls

### Port Usage
- **Backend API**: `localhost:8000`
- **Frontend**: `localhost:3000`
- **API Documentation**: `localhost:8000/docs`

### Data Requirements
The project includes a hybrid dataset:
- **4 PDF documents** (~4MB) - Federal student loan policies
- **CSV complaints file** (~12MB) - Real customer complaint data => 4,547 raw → 825 unfiltered -> 480 usable rows (11% retention after quality filtering)
- **Vector embeddings** (~39MB in memory) - Generated from documents

### Complaints Dataset Processing:
- **Raw CSV**: 4,547 total records
- **After loading Dataset**: 825 unfiltered complaints (18% retention)
- **Quality Filters Applied**:
  - ❌ Narratives < 100 characters
  - ❌ Excessive redaction (>5 XXXX tokens)
  - ❌ Empty/None/N/A content
- **Final Dataset**: 480 filtered complaints (11% retention)
- **Rationale**: Ensures meaningful content for
RAG retrieval

---

**Ready to help students navigate federal loan complexities with AI-powered guidance!** 🎓
