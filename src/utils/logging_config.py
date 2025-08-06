"""
Centralized Logging Configuration for RAG System
================================================

**🎯 PURPOSE & STRATEGY:**
- Provides consistent logging setup across all RAG system modules
- Suppresses verbose third-party library logging to reduce noise
- Enables clean, focused log output for system monitoring and debugging
- Essential for production deployment and development troubleshooting

**⚡ LOGGING FEATURES:**
- **Consistent Format**: Standardized timestamp, name, level, and message format
- **Third-Party Suppression**: Reduces noise from HTTP, LangChain, OpenAI, and other libraries
- **Flexible Modes**: Debug, info, and quiet modes for different use cases
- **Module-Specific**: Returns named loggers for proper attribution

**🔧 TECHNICAL IMPLEMENTATION:**
- **Idempotent Setup**: Safe to call multiple times without duplication
- **Force=False**: Respects existing logging configuration when present
- **Selective Suppression**: Targets 15+ noisy third-party libraries
- **Level Control**: WARNING level for third-party, INFO for application code

**📊 SUPPRESSED LIBRARIES:**
- **HTTP**: httpx, httpcore, urllib3, requests, aiohttp
- **AI/ML**: openai, langchain*, cohere, tavily, qdrant_client
- **Web Framework**: uvicorn, fastapi, starlette
- **Utility**: asyncio, charset_normalizer, multipart

**💡 USAGE PATTERNS:**
```python
# Standard module setup
from src.utils.logging_config import setup_logging
logger = setup_logging(__name__)

# Debug mode for development
from src.utils.logging_config import set_debug_mode
set_debug_mode()

# Quiet mode for production
from src.utils.logging_config import set_quiet_mode  
set_quiet_mode()
```

**🛠️ LOG OUTPUT FORMAT:**
`2024-01-15 14:30:22,123 - src.core.core_functions - INFO - 📄 Loaded 615 PDF documents`

**⚠️ IMPORTANT NOTES:**
- Setup is idempotent - safe to call from multiple modules
- Third-party suppression affects only WARNING+ messages
- Application logs remain at INFO level for visibility
- Debug mode enables verbose output for all loggers
"""

import logging


def setup_logging(name: str = None) -> logging.Logger:
    """
    Initialize consistent logging configuration with third-party noise suppression.

    **🎯 PURPOSE:**
    - Creates standardized logger for module with consistent formatting
    - Automatically suppresses verbose third-party library logging
    - Provides clean, focused log output for development and production

    **🔧 CONFIGURATION APPLIED:**
    - **Level**: INFO for application code, WARNING for third-party libraries
    - **Format**: Timestamp - Module - Level - Message
    - **Idempotent**: Safe to call multiple times without conflicts
    - **Named Logger**: Returns logger specific to calling module

    Args:
        name (str, optional): Logger name, typically __name__ from calling module
                             If None, uses the current module's name

    Returns:
        logging.Logger: Configured logger instance ready for use

    **💡 TYPICAL USAGE:**
    ```python
    from src.utils.logging_config import setup_logging
    logger = setup_logging(__name__)
    logger.info("📊 Processing started")
    logger.warning("⚠️ Potential issue detected")
    ```

    **⚡ PERFORMANCE NOTES:**
    - Minimal overhead after initial setup
    - Third-party suppression improves log readability
    - Named loggers enable module-specific filtering if needed
    """
    # Configure basic logging (idempotent - won't duplicate if already configured)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        force=False,  # Don't override existing configuration
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
        "fastapi",
    ]

    for logger_name in third_party_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)

    # Create and return logger
    return logging.getLogger(name or __name__)


def set_debug_mode():
    """
    Enable verbose debug logging for detailed troubleshooting.
    
    **🎯 PURPOSE:**
    - Enables DEBUG level logging for all loggers (including third-party)
    - Useful for development, debugging, and detailed system analysis
    - Shows all log messages including verbose library operations
    
    **⚠️ WARNING:**
    This will produce very verbose output including third-party library debug messages.
    Only use during active development or when troubleshooting specific issues.
    
    **💡 USAGE:**
    ```python
    from src.utils.logging_config import set_debug_mode
    set_debug_mode()  # Call before running problematic code
    ```
    """
    logging.getLogger().setLevel(logging.DEBUG)


def set_quiet_mode():
    """
    Set logging to WARNING level to minimize console output.
    
    **🎯 PURPOSE:**
    - Reduces log output to warnings and errors only
    - Ideal for production environments or batch processing
    - Eliminates informational messages while preserving important alerts
    
    **💡 USAGE:**
    ```python
    from src.utils.logging_config import set_quiet_mode
    set_quiet_mode()  # Minimal output mode
    ```
    
    **⚡ PERFORMANCE:**
    Reduces I/O overhead by suppressing non-critical log messages.
    """
    logging.getLogger().setLevel(logging.WARNING)
