import logging
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools.arxiv.tool import ArxivQueryRun
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
import operator
from langchain_core.messages import BaseMessage
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, END
from src.tools.tavily_tools import (
    tavily_studentaid_search,
    tavily_mohela_search,
    tavily_student_loan_search,
)

# Set up logging with third-party noise suppression
from src.utils.logging_config import setup_logging

logger = setup_logging(__name__)

tavily_tool = TavilySearchResults(max_results=5)


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


def should_continue(state):
    last_message = state["messages"][-1]

    if last_message.tool_calls:
        logger.info(f"🔧 Agent requesting {len(last_message.tool_calls)} tool calls")
        return "action"

    logger.info("✅ Agent conversation complete, no more tool calls needed")
    return END


def get_graph_agent(additional_tools: list):
    """
    Create a LangGraph-based conversational agent with RAG tool orchestration.

    Builds a state-driven agent that can:
    - Select optimal RAG retrieval methods based on question complexity
    - Coordinate multiple tools (internal RAG + external search APIs)
    - Maintain conversation state and context across tool calls
    - Handle tool call failures and recursion gracefully

    Architecture Flow:
        User Question → Agent (GPT-4.1-nano) → Tool Selection → RAG/Search → Response
        │
        ├─ RAG Tools: ask_naive_llm_tool, ask_contextual_compression_llm_tool, etc.
        └─ External Tools: StudentAid.gov search, Mohela search, general web search

    Args:
        additional_tools (list): RAG tools to bind to the agent, typically:
            - ask_naive_llm_tool: Primary RAG tool (best RAGAS performance)
            - ask_contextual_compression_llm_tool: Premium reranked retrieval
            - ask_multi_query_llm_tool: Comprehensive multi-angle search
            - ask_parent_document_llm_tool: Small-to-big retrieval strategy

    Returns:
        Compiled LangGraph agent ready for student loan question processing.
        Agent supports recursive tool calls with 25-step limit for complex workflows.

    LangGraph Configuration:
        - Model: GPT-4.1-nano (temperature=0 for consistency)
        - Timeout: 120s for complex multi-tool operations
        - State: AgentState with message accumulation
        - Nodes: 'agent' (LLM) → 'action' (ToolNode) → 'agent' (response)

    Tool Belt Composition:
        - Internal RAG: 4 specialized federal loan retrieval methods
        - External APIs: 3 Tavily-powered search functions + general web search
        - Total: ~7-8 tools depending on additional_tools provided

    Example Usage:
        >>> from src.agents.llm_tools_for_toolbelt import ask_naive_llm_tool
        >>> agent = get_graph_agent([ask_naive_llm_tool])
        >>> response = agent.invoke({"messages": [HumanMessage("What is FAFSA?")]})
    """
    logger.info(
        f"🤖 Creating graph agent with {len(additional_tools)} additional tools"
    )
    model = ChatOpenAI(
        model="gpt-4.1-nano",
        temperature=0,  # Lower temperature for more consistent outputs
        request_timeout=120,  # Longer timeout for complex operations
    )

    def call_model(state):
        logger.info(f"🧠 LLM processing {len(state['messages'])} messages")
        messages = state["messages"]
        response = model.invoke(messages)
        logger.info(
            f"📝 LLM generated response with {len(response.content) if hasattr(response, 'content') else 0} characters"
        )
        return {"messages": [response]}

    tool_belt = additional_tools + [
        tavily_tool,
        Tool(
            name="StudentAid_Federal_Search",
            description="Search ONLY StudentAid.gov for official federal information: FAFSA applications, federal loan forgiveness programs, federal repayment plans, eligibility requirements",
            func=tavily_studentaid_search,
        ),
        Tool(
            name="Mohela_Servicer_Search",
            description="Search ONLY Mohela loan servicer for account-specific help: making payments, login issues, servicer-specific repayment options, customer service contacts",
            func=tavily_mohela_search,
        ),
        Tool(
            name="Student_Loan_Comparison_Search",
            description="Compare information across BOTH federal sources and Mohela when user needs comprehensive view or comparison of student loan options",
            func=tavily_student_loan_search,
        ),
    ]

    model = model.bind_tools(tool_belt)
    tool_node = ToolNode(tool_belt)

    logger.info(f"🔧 Agent toolbelt configured with {len(tool_belt)} total tools")

    uncompiled_graph = StateGraph(AgentState)

    uncompiled_graph.add_node("agent", call_model)
    uncompiled_graph.add_node("action", tool_node)

    uncompiled_graph.set_entry_point("agent")
    uncompiled_graph.add_conditional_edges(
        "agent", 
        should_continue,
        {
            "action": "action",
            END: END
        }
    )

    uncompiled_graph.add_edge("action", "agent")

    logger.info("✅ Agent graph compiled and ready")
    return uncompiled_graph.compile()
