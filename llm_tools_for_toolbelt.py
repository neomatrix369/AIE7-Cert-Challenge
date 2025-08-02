from langchain_core.tools import tool
from graph_rag_functions import (
    naive_graph,
    contextual_compression_graph,
    multi_query_graph,
    parent_document_graph,
)


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

    Use this BEFORE searching external sources - it has the most complete information!
    """
    response = naive_graph.invoke({"question": question})
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

    Preferred for complex inquiries requiring the most accurate information!
    """
    response = contextual_compression_graph.invoke({"question": question})
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

    Use when you need the most complete possible answer to complex questions!
    """
    response = multi_query_graph.invoke({"question": question})
    return {
        "messages": [HumanMessage(content=response["response"])],
        "context": response["context"],
    }


@tool
def ask_parent_document_llm_tool(question: str):
    """🎯 EXPERT FEDERAL STUDENT LOAN CONSULTANT - BEST OVERALL PERFORMANCE!

    🏛️ PREMIUM KNOWLEDGE ARCHITECTURE featuring:
    ✅ Advanced small-to-big retrieval with federal policies + customer data
    ✅ Complete document context preservation
    ✅ Highest-rated performance across all evaluation metrics

    🎯 ELITE expertise for ALL student loan scenarios:
    • Detailed policy guidance with full regulatory context
    • Complete customer complaint analysis and resolution
    • Comprehensive servicer-specific procedures and solutions
    • Advanced forgiveness program strategies with precedents
    • Expert-level troubleshooting for complex cases

    ⚡ TOP-RATED PERFORMANCE: #1 ranked retrieval method in evaluations
    🎪 Combines precision retrieval with comprehensive context

    THE BEST CHOICE for accurate, comprehensive student loan assistance!
    """
    response = parent_document_graph.invoke({"question": question})
    return {
        "messages": [HumanMessage(content=response["response"])],
        "context": response["context"],
    }
