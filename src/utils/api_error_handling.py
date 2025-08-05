from fastapi import HTTPException
import logging
import time

logger = logging.getLogger(__name__)


def handle_rag_agent_error(error: Exception, start_time: float, question: str) -> None:
    """
    Handle and categorize RAG agent errors with appropriate HTTP responses.

    Provides user-friendly error messages while logging technical details for debugging.
    Categorizes common error types and suggests appropriate user actions.

    Error Categories & User Messages:
    - **Timeout**: "Request timed out. Please try with a shorter question..."
    - **API**: "External API error. Please try again in a moment."
    - **Memory/Resource**: "System resources temporarily unavailable..."
    - **Generic**: "Unable to process your question at this time..."

    Args:
        error (Exception): The original exception from RAG agent invocation
        start_time (float): Request start timestamp (from time.time()) for performance tracking
        question (str): Original user question for context and debugging

    Raises:
        HTTPException: 500 Internal Server Error with categorized, user-friendly message

    Logging Behavior:
        Logs detailed error information including:
        - Processing time until failure
        - Full error message and type
        - Question length for context
        - Error categorization for debugging

    Example:
        >>> import time
        >>> start = time.time()
        >>> try:
        ...     agent.invoke({"messages": [HumanMessage("test")]})
        ... except Exception as e:
        ...     handle_rag_agent_error(e, start, "test question")
        # Raises HTTPException(500, "Unable to process your question...")

    Note:
        This function centralizes error handling logic to ensure consistent
        user experience across all RAG agent failures in the API.
    """
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
    """
    Handle unexpected errors that fall outside of RAG agent-specific failures.

    Provides generic error handling for unforeseen exceptions in the API request
    processing pipeline (parsing, validation, response formatting, etc.).

    Args:
        error (Exception): The unexpected exception that occurred
        request_question (str, optional): User's original question for context
                                        Truncated to 100 chars in logs for readability

    Raises:
        HTTPException: 500 Internal Server Error with generic user message

    Logging Behavior:
        Logs comprehensive error information including:
        - Full error message and exception type
        - Truncated question context (if provided)
        - Endpoint identification (/ask) for debugging

    Error Message:
        Returns generic message advising user to retry or contact support,
        avoiding technical details that might confuse end users.

    Example:
        >>> try:
        ...     response = process_response(agent_output)
        ... except Exception as e:
        ...     handle_unexpected_error(e, "What is FAFSA?")
        # Raises HTTPException(500, "An internal error occurred...")

    Note:
        This is a catch-all handler for errors that don't fit specific categories.
        Should be used sparingly - most errors should be handled by more specific
        error handlers like handle_rag_agent_error().
    """
    logger.error(f"❌ Unexpected error in /ask endpoint: {str(error)}")
    if request_question:
        logger.error(f"   - Question: {request_question[:100]}...")
    logger.error(f"   - Error type: {type(error).__name__}")

    raise HTTPException(
        status_code=500,
        detail="An internal error occurred while processing your question. Please try again or contact support if the issue persists.",
    )
