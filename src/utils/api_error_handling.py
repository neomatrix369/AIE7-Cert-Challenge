from fastapi import HTTPException
import logging
import time

logger = logging.getLogger(__name__)


def handle_rag_agent_error(error: Exception, start_time: float, question: str) -> None:
    error_time = time.time() - start_time
    logger.error(f"❌ RAG agent failed after {error_time:.2f} seconds")
    logger.error(f"   - Error: {str(error)}")
    logger.error(f"   - Error type: {type(error).__name__}")
    logger.error(f"   - Question length: {len(question)} chars")

    if "timeout" in str(error).lower():
        detail = (
            "Request timed out. Please try with a shorter question or try again later."
        )
    elif "api" in str(error).lower() or "openai" in str(error).lower():
        detail = "External API error. Please try again in a moment."
    elif "memory" in str(error).lower() or "resource" in str(error).lower():
        detail = "System resources temporarily unavailable. Please try again."
    else:
        detail = "Unable to process your question at this time. Please try again or rephrase your question."

    raise HTTPException(status_code=500, detail=detail)


def handle_unexpected_error(error: Exception, request_question: str = None) -> None:
    logger.error(f"❌ Unexpected error in /ask endpoint: {str(error)}")
    if request_question:
        logger.error(f"   - Question: {request_question[:100]}...")
    logger.error(f"   - Error type: {type(error).__name__}")

    raise HTTPException(
        status_code=500,
        detail="An internal error occurred while processing your question. Please try again or contact support if the issue persists.",
    )
