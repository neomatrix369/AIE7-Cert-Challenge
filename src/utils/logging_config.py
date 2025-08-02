"""
Shared logging configuration to ensure consistent logging setup across all modules
and suppress verbose third-party library logging.
"""

import logging


def setup_logging(name: str = None) -> logging.Logger:
    """
    Set up logging with consistent configuration and third-party noise suppression.
    
    Args:
        name: Logger name (uses __name__ if not provided)
    
    Returns:
        Configured logger instance
    """
    # Configure basic logging (idempotent - won't duplicate if already configured)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        force=False  # Don't override existing configuration
    )
    
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
        "langchain_community",
        "qdrant_client",
        "cohere",
        "tavily",
        "asyncio",
        "aiohttp",
        "charset_normalizer",
        "multipart",
        "starlette",
        "fastapi"
    ]
    
    for logger_name in third_party_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)
    
    # Create and return logger
    return logging.getLogger(name or __name__)


def set_debug_mode():
    """Enable debug logging for troubleshooting"""
    logging.getLogger().setLevel(logging.DEBUG)
    

def set_quiet_mode():
    """Set to WARNING level to reduce noise"""
    logging.getLogger().setLevel(logging.WARNING)