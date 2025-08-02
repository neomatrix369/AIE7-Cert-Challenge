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
from typing import Optional

# Import our RAG components
from src.agents.build_graph_agent import get_graph_agent
from src.agents.llm_tools_for_toolbelt import ask_parent_document_llm_tool
from langchain_core.messages import HumanMessage
from src.evaluation.tool_calls_parser_for_eval import extract_contexts_for_eval

# Load environment variables
load_dotenv(dotenv_path=".env")

# Set up logging with third-party library noise suppression
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
# )
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
    """Health check endpoint"""
    return {
        "status": "healthy" if rag_agent else "unhealthy",
        "agent_status": "ready" if rag_agent else "not_initialized",
        "api_version": "1.0.0",
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
        # Validate input
        if not request.question or not request.question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")

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

        # Invoke the RAG agent
        try:
            response = rag_agent.invoke(inputs)
            logger.info("✅ RAG agent response received successfully")
        except Exception as e:
            logger.error(f"❌ RAG agent error: {str(e)}")
            raise HTTPException(
                status_code=500, detail=f"Failed to process question: {str(e)}"
            )

        # Extract the final answer from agent response
        if not response or "messages" not in response:
            logger.error("❌ Invalid response structure from RAG agent")
            raise HTTPException(
                status_code=500, detail="Invalid response from RAG agent"
            )

        messages = response["messages"]
        logger.info(f"📋 Extracting contexts from {len(messages)} agent messages")
        contexts = extract_contexts_for_eval(messages)
        if not messages:
            raise HTTPException(
                status_code=500, detail="No response generated by RAG agent"
            )

        # Get the final AI response
        final_message = messages[-1]
        answer = (
            final_message.content
            if hasattr(final_message, "content")
            else str(final_message)
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

        logger.info(
            f"✅ Question processed successfully, response length: {len(answer)} chars"
        )

        return StudentLoanResponse(
            answer=answer,
            sources_count=sources_count,
            success=True,
            message="Question processed successfully",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


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

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
