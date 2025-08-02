from fastapi import HTTPException
import logging

logger = logging.getLogger(__name__)


def validate_question_input(question: str) -> None:
    if not question or not question.strip():
        logger.warning("⚠️ Empty question submitted")
        raise HTTPException(status_code=400, detail="Question cannot be empty")
        
    if len(question) > 5000:
        logger.warning(f"⚠️ Question too long: {len(question)} characters")
        raise HTTPException(status_code=400, detail="Question is too long. Please limit to 5000 characters or less.")
        
    if len(question.strip()) < 3:
        logger.warning(f"⚠️ Question too short: '{question.strip()}'")
        raise HTTPException(status_code=400, detail="Question is too short. Please provide a more detailed question.")


def validate_agent_availability(rag_agent) -> None:
    if not rag_agent:
        raise HTTPException(status_code=503, detail="RAG agent is not initialized. Please try again later.")