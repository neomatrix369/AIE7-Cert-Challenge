# Federal Student Loan AI Assistant

**AIE7 Certification Challenge** - Advanced RAG system for federal student loan customer service

An intelligent assistant that combines official federal loan policies with real customer experiences to provide comprehensive guidance on student loan questions, repayment options, forgiveness programs, and servicer issues.

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

Get the AI assistant running in 4 simple steps:

### 1. Environment Setup
```bash
# Copy environment template and add your API keys
cp .env-example .env
# Edit .env file with your API keys:
# OPENAI_API_KEY=your_key_here
# COHERE_API_KEY=your_key_here  
# TAVILY_API_KEY=your_key_here
# LANGCHAIN_API_KEY=your_key_here
```

### 2. Start Backend with Docker (from the root folder of the project)
```bash
cd src/backend
docker-compose up --build
# Backend running at: http://localhost:8000

# docker ps ### to find out the container running the backend
```

See [Docker deployment at src/backend/README.md](./src/backend/README.md#deployment) for detailed Docker setup options

To stop this docker container please do this:

```bash
cd src/backend
docker-compose down
```

**_NOTE: Please give the app a good 'few' minutes or so to get started, as loading 2000+ docs inside the Docker container takes a bit of time. Till then we are not able to ping the backend server._**

### 3. Start the Frontend with Docker (new terminal, from the root folder of the project)
```bash
cd frontend
./docker-start.sh
# Frontend running at: http://localhost:3000
```

See [Docker deployment at frontend/README.md](./frontend/README.md#docker-deployment) for detailed Docker setup options

To stop this docker container please do this:

```bash
cd frontend
docker-compose down
```

### 4. Open & Use the Assistant
Navigate to **http://localhost:3000** and start asking federal student loan questions!

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
- **Development Guide**: [`CLAUDE.md`](CLAUDE.md)

### Docker Deployment
```bash
# Full-stack deployment
cd src/backend && docker-compose up --build
cd frontend && docker-compose up --build
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
- **CSV complaints file** (~12MB) - Real customer complaint data
- **Vector embeddings** (~39MB in memory) - Generated from documents

---

**Ready to help students navigate federal loan complexities with AI-powered guidance!** 🎓
