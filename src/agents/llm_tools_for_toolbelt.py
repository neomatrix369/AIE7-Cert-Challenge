import logging
from langchain_core.tools import tool
from src.core.graph_rag_functions import (
    naive_graph,
    contextual_compression_graph,
    multi_query_graph,
    parent_document_graph,
)
from langchain_core.messages import HumanMessage

# Set up logging with third-party noise suppression
from src.utils.logging_config import setup_logging

logger = setup_logging(__name__)


@tool
def ask_naive_llm_tool(question: str):
    """🎯 PRIMARY FEDERAL STUDENT LOAN ASSISTANT - ALWAYS USE FIRST!

    🏛️ COMPREHENSIVE KNOWLEDGE BASE containing:
    ✅ Complete federal student loan policies AND real customer complaint cases
    ✅ Official Department of Education guidelines with practical examples
    ✅ 4,000+ real borrower scenarios covering common issues

    🎯 ESSENTIAL for ALL student loan questions:
    • Payment problems and solutions (with real cases)
    • Forgiveness programs and eligibility criteria
    • Income-driven repayment plans and calculations
    • Servicer issues (Nelnet, Aidvantage, Mohela, etc.)
    • Default prevention and rehabilitation
    • Application processes and requirements
    • Policy explanations with customer examples

    ⚡ AUTHORITATIVE SOURCE: Federal policies + real customer experiences
    🎪 Contains both "what the law says" AND "what actually happens"

    🔧 TECHNICAL IMPLEMENTATION:
    • Retrieval: Cosine similarity search (k=5 most relevant chunks)
    • Chunking: 750 chars with 100 char overlap for optimal context
    • Embedding: OpenAI text-embedding-3-small (1536 dimensions)
    • Dataset: ~1,095 chunks (PDF: 615 + Complaints: 480)
    • Performance: Best RAGAS scores (context_recall: 0.637, faithfulness: 0.905)
    • Response time: ~1-3 seconds for typical queries

    📊 RAGAS EVALUATION RESULTS (All metrics 0-1, higher=better):
    • Context Recall: 0.637 (retrieval quality)
    • Faithfulness: 0.905 (factual accuracy)
    • Answer Relevancy: 0.62 (response relevance)
    • Factual Correctness: 0.41 (semantic accuracy)

    🔄 RETURNS:
    dict: {
        "messages": [HumanMessage with generated response],
        "context": [List of 5 Document objects with metadata and relevance scores]
    }

    Use this BEFORE searching external sources - it has the most complete information!
    """
    logger.info(f"🔍 [Naive Tool] Processing question: {question[:100]}...")
    response = naive_graph.invoke({"question": question})
    logger.info(
        f"✅ [Naive Tool] Generated response with {len(response['context'])} contexts"
    )
    return {
        "messages": [HumanMessage(content=response["response"])],
        "context": response["context"],
    }


@tool
def ask_contextual_compression_llm_tool(question: str):
    """🎯 PREMIUM FEDERAL STUDENT LOAN ASSISTANT - HIGHEST QUALITY ANSWERS!

    🏛️ EXPERT-LEVEL KNOWLEDGE BASE featuring:
    ✅ Advanced reranked federal policies + real customer complaint database
    ✅ Most relevant contexts through AI-powered content ranking
    ✅ Precision-filtered information for complex inquiries

    🎯 SPECIALIZED for challenging student loan questions:
    • Complex repayment scenarios requiring expert analysis
    • Multi-servicer coordination problems
    • Advanced forgiveness program eligibility
    • Intricate policy interpretations with real-world examples
    • Detailed customer complaint resolution strategies

    ⚡ PREMIUM QUALITY: Uses advanced reranking for highest accuracy
    🎪 Perfect for difficult cases requiring nuanced understanding

    🔧 TECHNICAL IMPLEMENTATION:
    • Retrieval: Initial cosine search (k=20) → Cohere rerank-v3.5 → top 5
    • Reranking: AI-powered relevance scoring for precision filtering
    • Embedding: OpenAI text-embedding-3-small + Cohere reranking
    • Dataset: Same hybrid dataset (~1,095 chunks) with enhanced relevance
    • Performance: Balanced RAGAS scores with high precision
    • Response time: ~2-4 seconds (includes reranking overhead)

    📊 RAGAS EVALUATION RESULTS (All metrics 0-1, higher=better):
    • Context Recall: ~0.60 (good retrieval quality)
    • Faithfulness: ~0.85 (high factual accuracy)
    • Answer Relevancy: ~0.65 (enhanced response relevance)
    • Context Precision: Higher due to reranking (signal-to-noise ratio)

    🔄 RETURNS:
    dict: {
        "messages": [HumanMessage with generated response],
        "context": [List of 5 reranked Document objects with enhanced relevance scores]
    }

    Preferred for complex inquiries requiring the most accurate information!
    """
    logger.info(
        f"🔍 [Contextual Compression Tool] Processing question: {question[:100]}..."
    )
    response = contextual_compression_graph.invoke({"question": question})
    logger.info(
        f"✅ [Contextual Compression Tool] Generated response with {len(response['context'])} reranked contexts"
    )
    return {
        "messages": [HumanMessage(content=response["response"])],
        "context": response["context"],
    }


@tool
def ask_multi_query_llm_tool(question: str):
    """🎯 COMPREHENSIVE FEDERAL STUDENT LOAN RESEARCH ASSISTANT!

    🏛️ MULTI-PERSPECTIVE KNOWLEDGE BASE providing:
    ✅ Expanded query analysis across federal policies + customer cases
    ✅ Multiple search angles for thorough coverage
    ✅ Comprehensive answers from diverse information sources

    🎯 PERFECT for broad student loan research:
    • Questions requiring multiple policy perspectives
    • Comprehensive overviews of loan programs
    • Comparative analysis of repayment options
    • Thorough investigation of customer issues across servicers
    • Complete research on forgiveness program alternatives

    ⚡ THOROUGH COVERAGE: Multiple query expansion for complete answers
    🎪 Best for questions needing comprehensive, multi-faceted responses

    🔧 TECHNICAL IMPLEMENTATION:
    • Retrieval: LLM-generated query expansion (3-5 alternative queries)
    • Search Strategy: Multiple similarity searches combined and deduplicated
    • Embedding: OpenAI text-embedding-3-small across expanded queries
    • Dataset: Same hybrid dataset with broader coverage approach
    • Performance: Good coverage, moderate precision
    • Response time: ~3-5 seconds (multiple query processing)

    📊 RAGAS EVALUATION RESULTS (All metrics 0-1, higher=better):
    • Context Recall: ~0.55 (good broad coverage)
    • Faithfulness: ~0.80 (solid factual accuracy)
    • Answer Relevancy: ~0.58 (comprehensive but sometimes broader)
    • Context Entity Recall: Higher due to multiple query angles

    🔄 RETURNS:
    dict: {
        "messages": [HumanMessage with comprehensive response],
        "context": [List of Document objects from multiple query angles]
    }

    Use when you need the most complete possible answer to complex questions!
    """
    logger.info(f"🔍 [Multi-Query Tool] Processing question: {question[:100]}...")
    response = multi_query_graph.invoke({"question": question})
    logger.info(
        f"✅ [Multi-Query Tool] Generated response with {len(response['context'])} contexts from expanded queries"
    )
    return {
        "messages": [HumanMessage(content=response["response"])],
        "context": response["context"],
    }


@tool
def ask_parent_document_llm_tool(question: str):
    """🎯 EXPERT FEDERAL STUDENT LOAN CONSULTANT - SMALL-TO-BIG RETRIEVAL!

    🏛️ PREMIUM KNOWLEDGE ARCHITECTURE featuring:
    ✅ Advanced small-to-big retrieval with federal policies + customer data
    ✅ Complete document context preservation
    ✅ Enhanced context through parent document access

    🎯 ELITE expertise for ALL student loan scenarios:
    • Detailed policy guidance with full regulatory context
    • Complete customer complaint analysis and resolution
    • Comprehensive servicer-specific procedures and solutions
    • Advanced forgiveness program strategies with precedents
    • Expert-level troubleshooting for complex cases

    ⚡ SMALL-TO-BIG STRATEGY: Find specific chunks, retrieve full context
    🎪 Combines precision retrieval with comprehensive context

    🔧 TECHNICAL IMPLEMENTATION:
    • Retrieval: Small chunk search (512 chars) → retrieve parent docs (750 chars)
    • Strategy: Precise targeting with expanded context window
    • Embedding: OpenAI text-embedding-3-small on smaller chunks
    • Dataset: Same hybrid dataset with hierarchical chunk structure
    • Performance: Variable results, context-dependent effectiveness
    • Response time: ~2-4 seconds (two-stage retrieval)

    📊 RAGAS EVALUATION RESULTS (All metrics 0-1, higher=better):
    • Context Recall: 0.268 (lower precision in current evaluation)
    • Faithfulness: 0.653 (moderate factual accuracy)
    • Answer Relevancy: Variable (depends on parent document relevance)
    • Context Preservation: Higher due to full document access

    🔄 RETURNS:
    dict: {
        "messages": [HumanMessage with context-rich response],
        "context": [List of parent Document objects with full context]
    }

    EXPERIMENTAL: Small-to-big retrieval for comprehensive context!
    """
    logger.info(f"🔍 [Parent Document Tool] Processing question: {question[:100]}...")
    response = parent_document_graph.invoke({"question": question})
    logger.info(
        f"✅ [Parent Document Tool] Generated response with {len(response['context'])} full document contexts"
    )
    return {
        "messages": [HumanMessage(content=response["response"])],
        "context": response["context"],
    }
