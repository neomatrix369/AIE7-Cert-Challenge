from fastapi import HTTPException
import logging

logger = logging.getLogger(__name__)


def validate_question_input(question: str) -> None:
    """
    Validate student loan question input for API requests.

    Ensures questions meet quality standards for effective RAG processing by checking:
    - Non-empty content (prevents empty requests)
    - Reasonable length (3-5000 characters for optimal processing)
    - Meaningful content for vector retrieval and LLM processing

    Validation Rules:
    - Minimum length: 3 characters (prevents trivial inputs)
    - Maximum length: 5000 characters (prevents token limit issues)
    - Must contain non-whitespace content

    Args:
        question (str): User's federal student loan question

    Raises:
        HTTPException: 400 Bad Request with specific error details:
            - "Question cannot be empty" for empty/whitespace-only input
            - "Question is too long..." for questions exceeding 5000 chars
            - "Question is too short..." for questions under 3 chars

    Example:
        >>> validate_question_input("What are income-driven repayment plans?")
        # Passes validation (no exception)

        >>> validate_question_input("")
        # Raises HTTPException(400, "Question cannot be empty")

        >>> validate_question_input("Hi")
        # Raises HTTPException(400, "Question is too short...")

    Note:
        This function is designed for federal student loan queries but the validation
        logic is generic enough for other domain-specific RAG applications.
    """
    if not question or not question.strip():
        logger.warning("⚠️ Empty question submitted")
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    if len(question) > 5000:
        logger.warning(f"⚠️ Question too long: {len(question)} characters")
        raise HTTPException(
            status_code=400,
            detail="Question is too long. Please limit to 5000 characters or less.",
        )

    if len(question.strip()) < 3:
        logger.warning(f"⚠️ Question too short: '{question.strip()}'")
        raise HTTPException(
            status_code=400,
            detail="Question is too short. Please provide a more detailed question.",
        )


def validate_agent_availability(rag_agent) -> None:
    """
    Validate that the RAG agent is properly initialized and ready for processing.

    Ensures the LangGraph-based RAG agent is available before attempting to process
    user questions. This prevents 500 errors from uninitialized agent invocations.

    Args:
        rag_agent: LangGraph agent instance (typically from get_graph_agent())
                  Expected to be a compiled LangGraph with RAG tools bound

    Raises:
        HTTPException: 503 Service Unavailable if agent is None or not initialized

    Example:
        >>> from src.agents.build_graph_agent import get_graph_agent
        >>> from src.agents.llm_tools_for_toolbelt import ask_naive_llm_tool
        >>> agent = get_graph_agent([ask_naive_llm_tool])
        >>> validate_agent_availability(agent)
        # Passes validation (no exception)

        >>> validate_agent_availability(None)
        # Raises HTTPException(503, "RAG agent is not initialized...")

    Note:
        This validation is crucial during application startup when the vector store
        and RAG components may still be initializing. Returns 503 (temporary) rather
        than 500 (permanent) to indicate the service may be available later.
    """
    if not rag_agent:
        raise HTTPException(
            status_code=503,
            detail="RAG agent is not initialized. Please try again later.",
        )
