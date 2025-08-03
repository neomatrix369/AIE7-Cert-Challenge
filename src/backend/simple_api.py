#!/usr/bin/env python3
"""
Simple FastAPI Backend for AIE7 Certification Challenge
Federal Student Loan Customer Service RAG API

Based on Project Completion Plan requirements:
- Single /ask endpoint for student loan questions
- Uses best performing retrieval method (Parent Document)
- Simple request/response structure
- No file upload, chat sessions, or complex features
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import logging
import time
from pathlib import Path
from typing import Optional, List

# Import our RAG components
from src.agents.build_graph_agent import get_graph_agent
from src.agents.llm_tools_for_toolbelt import ask_parent_document_llm_tool
from langchain_core.messages import HumanMessage
from src.evaluation.tool_calls_parser_for_eval import (
    process_agent_response,
    build_performance_metrics,
)
from src.utils.api_info_details import get_api_info_details

# Load environment variables
load_dotenv(dotenv_path=".env")

from src.utils.logging_config import setup_logging
from src.utils.api_validation import (
    validate_question_input,
    validate_agent_availability,
)
from src.utils.api_error_handling import handle_rag_agent_error, handle_unexpected_error

logger = setup_logging(__name__)

# Suppress verbose logging from third-party libraries
third_party_loggers = [
    "httpx",
    "httpcore",
    "openai",
    "urllib3",
    "requests",
    "uvicorn",
    "uvicorn.access",
    "langchain",
    "langchain_core",
    "langchain_openai",
    "qdrant_client",
    "cohere",
    "tavily",
]
for logger_name in third_party_loggers:
    logging.getLogger(logger_name).setLevel(logging.WARNING)

# Initialize FastAPI app
app = FastAPI(
    title="Federal Student Loan Assistant API",
    description="RAG-powered API for federal student loan customer service questions",
    version="1.0.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Data models
class StudentLoanQuestion(BaseModel):
    question: str
    max_response_length: Optional[int] = 2000


class StudentLoanResponse(BaseModel):
    answer: str
    sources_count: int
    success: bool
    message: Optional[str] = None

    # Raw data from LLM pipeline
    source_details: List[
        dict
    ]  # Contexts with relevance scores from extract_contexts_for_eval
    tools_used: List[str]  # Tools/retrievers used during processing
    performance_metrics: dict  # Basic timing measurements


# Global agent initialization
try:
    # Initialize the best performing agent (Parent Document from evaluation results)
    logger.info("🚀 Initializing Parent Document RAG agent...")
    rag_agent = get_graph_agent([ask_parent_document_llm_tool])
    logger.info("✅ RAG agent initialized successfully")
except Exception as e:
    logger.error(f"❌ Failed to initialize RAG agent: {str(e)}")
    rag_agent = None


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Federal Student Loan Assistant API",
        "description": "RAG-powered API for customer service agents",
        "endpoints": {
            "/ask": "POST - Ask questions about federal student loans",
            "/health": "GET - Health check",
            "/docs": "GET - API documentation",
        },
        "status": "ready" if rag_agent else "initializing",
    }


@app.get("/health")
async def health_check():
    """Enhanced health check endpoint with environment information"""
    logger.info(f"🔍 Health check requested:")
    logger.info(f"   - Environment: {'local'}")
    logger.info(f"   - RAG agent: {'ready' if rag_agent else 'not_initialized'}")

    return {
        "status": "healthy" if rag_agent else "unhealthy",
        "agent_status": "ready" if rag_agent else "not_initialized",
        "api_version": "1.0.0",
        "environment": "local",
        "vector_store": "qdrant_memory",  # We use in-memory Qdrant
        "features": {
            "rag_agent": bool(rag_agent),
            "hybrid_dataset": True,
            "performance_metrics": True,
            "source_transparency": True,
            "tools_tracking": True,
        },
    }


@app.post("/ask", response_model=StudentLoanResponse)
async def ask_student_loan_question(request: StudentLoanQuestion):
    """
    Ask a question about federal student loans

    This endpoint uses the best performing RAG retrieval method (Parent Document)
    based on RAGAS evaluation results. It searches through:
    - Federal student loan policy documents (PDF)
    - Real customer complaint scenarios (CSV)
    - Combined hybrid knowledge base for comprehensive answers
    """
    try:
        # Enhanced input validation using utility functions
        validate_question_input(request.question)
        validate_agent_availability(rag_agent)

        logger.info(f"📝 Processing question: {request.question[:100]}...")

        # Prepare input for the agent
        inputs = {"messages": [HumanMessage(content=request.question.strip())]}

        logger.info("🔍 Invoking RAG agent with Parent Document retrieval method")

        # Track performance timing
        start_time = time.time()

        # Invoke the RAG agent with detailed error handling
        try:
            response = rag_agent.invoke(inputs)
            end_time = time.time()
            logger.info("✅ RAG agent response received successfully")
        except Exception as e:
            handle_rag_agent_error(e, start_time, request.question)

        # Extract the final answer from agent response
        if not response or "messages" not in response:
            logger.error("❌ Invalid response structure from RAG agent")
            raise HTTPException(
                status_code=500, detail="Invalid response from RAG agent"
            )

        # Process agent response using extracted function
        processed = process_agent_response(response)
        contexts = processed["contexts"]
        tools_used = processed["tools_used"]
        answer = processed["final_answer"]
        messages = processed["messages"]

        logger.info(
            f"📋 Extracted {len(contexts)} contexts from {len(messages)} agent messages"
        )
        logger.info(f"🔧 Tools used in processing: {tools_used}")

        if not messages:
            raise HTTPException(
                status_code=500, detail="No response generated by RAG agent"
            )

        # # Truncate response if requested
        if request.max_response_length and len(answer) > request.max_response_length:
            answer = answer[: request.max_response_length] + "..."

        # Count sources (estimate based on typical RAG response)
        sources_count = 0
        if contexts:
            sources_count = len(
                contexts
            )  # Parent Document retriever typically returns 5 contexts
            logger.info(
                f"📚 Retrieved {sources_count} context sources from hybrid dataset"
            )

        # Calculate total processing time and log performance summary
        total_processing_time = time.time() - start_time

        logger.info(
            f"✅ Question processed successfully, response length: {len(answer)} chars"
        )
        logger.info(f"⏱️ Performance summary:")
        logger.info(f"   - Total processing: {total_processing_time:.3f}s")
        logger.info(f"   - Contexts retrieved: {len(contexts)}")
        logger.info(f"   - Tools used: {len(tools_used)}")

        # Build comprehensive performance metrics using refactored function
        performance_metrics = build_performance_metrics(
            start_time=start_time,
            end_time=end_time,
            messages=messages,
            contexts=contexts,
            question_text=request.question,
            answer_text=answer,
            retrieval_method="parent_document",
        )

        # Format source details with relevance scores
        formatted_source_details = []
        for context in contexts or []:
            if isinstance(context, dict) and "content" in context:
                # Context already has relevance score
                formatted_source_details.append(
                    {
                        "content": context["content"],
                        "relevance_score": context.get("relevance_score", 0.0),
                    }
                )
            elif isinstance(context, str):
                # Plain string context, no relevance score available
                formatted_source_details.append(
                    {"content": context, "relevance_score": 0.0}
                )

        return StudentLoanResponse(
            answer=answer,
            sources_count=sources_count,
            success=True,
            message="Question processed successfully",
            source_details=formatted_source_details,  # Formatted contexts with relevance scores
            tools_used=tools_used or [],  # Tools/retrievers used during processing
            performance_metrics=performance_metrics,
        )

    except HTTPException:
        # Re-raise HTTP exceptions as-is (they're already properly formatted)
        raise
    except Exception as e:
        handle_unexpected_error(e, getattr(request, "question", None))


@app.get("/api-info")
async def get_api_info():
    """Get detailed API information and capabilities"""
    return get_api_info_details()


if __name__ == "__main__":
    import uvicorn

    required_vars = [
        "OPENAI_API_KEY",
        "TAVILY_API_KEY",
        "LANGCHAIN_API_KEY",
        "LANGCHAIN_TRACING_V2",
        "LANGCHAIN_PROJECT",
    ]
    missing_vars = [var for var in required_vars if not os.getenv(var)]

    if missing_vars:
        logger.error(f"❌ Missing required environment variables: {missing_vars}")
        logger.error("Please set these in your .env file")
        exit(1)

    logger.info("🚀 Starting Federal Student Loan Assistant API...")
    logger.info("📋 Available at: http://localhost:8000")
    logger.info("📖 API docs at: http://localhost:8000/docs")

    # Using the below with reload renders the FastAPI to not load at all
    # uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
    # Hence resorting to the below
    uvicorn.run(app, host="0.0.0.0", port=8000)
