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
from src.evaluation.tool_calls_parser_for_eval import process_agent_response, build_performance_metrics

# Load environment variables
load_dotenv(dotenv_path=".env")

from src.utils.logging_config import setup_logging

logger = setup_logging(__name__)

# Suppress verbose logging from third-party libraries
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("uvicorn").setLevel(logging.WARNING)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logging.getLogger("langchain").setLevel(logging.WARNING)
logging.getLogger("langchain_core").setLevel(logging.WARNING)
logging.getLogger("langchain_openai").setLevel(logging.WARNING)
logging.getLogger("qdrant_client").setLevel(logging.WARNING)
logging.getLogger("cohere").setLevel(logging.WARNING)
logging.getLogger("tavily").setLevel(logging.WARNING)

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
    source_details: List[str]  # Raw contexts from extract_contexts_for_eval
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
    logger.info(f"   - Environment: {'vercel' if is_vercel_environment() else 'local'}")
    logger.info(f"   - Read-only: {IS_READONLY}")
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
        }
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
        # Enhanced input validation
        if not request.question or not request.question.strip():
            logger.warning("⚠️ Empty question submitted")
            raise HTTPException(status_code=400, detail="Question cannot be empty")

        # Validate question length (prevent extremely long inputs)
        if len(request.question) > 5000:
            logger.warning(f"⚠️ Question too long: {len(request.question)} characters")
            raise HTTPException(
                status_code=400, 
                detail="Question is too long. Please limit to 5000 characters or less."
            )
   
        # Check for potentially problematic content
        if len(request.question.strip()) < 3:
            logger.warning(f"⚠️ Question too short: '{request.question.strip()}'")
            raise HTTPException(
                status_code=400, 
                detail="Question is too short. Please provide a more detailed question."
            )

        # Check if agent is initialized
        if not rag_agent:
            raise HTTPException(
                status_code=503,
                detail="RAG agent is not initialized. Please try again later.",
            )

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
            end_time = time.time()
            error_time = end_time - start_time

            # Log detailed error information
            logger.error(f"❌ RAG agent failed after {error_time:.2f} seconds")
            logger.error(f"   - Error: {str(e)}")
            logger.error(f"   - Error type: {type(e).__name__}")
            logger.error(f"   - Question length: {len(request.question)} chars")

            # Provide specific error messages based on error type
            if "timeout" in str(e).lower():
                detail = "Request timed out. Please try with a shorter question or try again later."
            elif "api" in str(e).lower() or "openai" in str(e).lower():
                detail = "External API error. Please try again in a moment."
            elif "memory" in str(e).lower() or "resource" in str(e).lower():
                detail = "System resources temporarily unavailable. Please try again."
            else:
                detail = "Unable to process your question at this time. Please try again or rephrase your question."

            raise HTTPException(status_code=500, detail=detail)

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

        logger.info(f"📋 Extracted {len(contexts)} contexts from {len(messages)} agent messages")
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

        logger.info(f"✅ Question processed successfully, response length: {len(answer)} chars")
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
            retrieval_method="parent_document"
        )

        return StudentLoanResponse(
            answer=answer,
            sources_count=sources_count,
            success=True,
            message="Question processed successfully",
            source_details=contexts or [],  # Raw contexts from extract_contexts_for_eval
            tools_used=tools_used or [],  # Tools/retrievers used during processing
            performance_metrics=performance_metrics
        )

    except HTTPException:
        # Re-raise HTTP exceptions as-is (they're already properly formatted)
        raise
    except Exception as e:
        # Log the full exception details for debugging
        logger.error(f"❌ Unexpected error in /ask endpoint: {str(e)}")
        logger.error(f"   - Question: {request.question[:100] if hasattr(request, 'question') else 'Unknown'}...")
        logger.error(f"   - Error type: {type(e).__name__}")

        # Return user-friendly error message
        raise HTTPException(
            status_code=500, 
            detail="An internal error occurred while processing your question. Please try again or contact support if the issue persists."
        )


@app.get("/api-info")
async def get_api_info():
    """Get detailed API information and capabilities"""
    return {
        "api_name": "Federal Student Loan Assistant",
        "version": "1.0.0",
        "description": "RAG-powered API for federal student loan customer service",
        "capabilities": {
            "knowledge_base": {
                "pdf_documents": [
                    "Academic Calendars, Cost of Attendance, and Packaging",
                    "Applications and Verification Guide",
                    "The Federal Pell Grant Program",
                    "The Direct Loan Program",
                ],
                "customer_data": "4,547 real customer complaints and scenarios",
                "total_documents": "Hybrid dataset with policy + complaint knowledge",
            },
            "retrieval_method": "Parent Document (best performing from RAGAS evaluation)",
            "evaluation_metrics": {
                "context_recall": "0.89",
                "faithfulness": "0.82",
                "answer_relevancy": "0.62",
                "factual_correctness": "0.41",
            },
        },
        "usage": {
            "endpoint": "/ask",
            "method": "POST",
            "example_questions": [
                "What are the requirements for federal student loan forgiveness?",
                "How do I apply for income-driven repayment plans?",
                "What should I do if my loan servicer is not responding?",
                "What are the differences between federal and private student loans?",
                "How does loan consolidation work?",
            ],
        },
        "response_format": {
            "answer": "Generated response text",
            "sources_count": "Number of knowledge sources used",
            "success": "Boolean indicating success",
            "message": "Status message",
        },
    }


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
