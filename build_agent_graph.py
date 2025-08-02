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
from tavily_tools import (
    tavily_studentaid_search,
    tavily_mohela_search,
    tavily_student_loan_search,
)

tavily_tool = TavilySearchResults(max_results=5)

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

def should_continue(state):
    last_message = state["messages"][-1]

    if last_message.tool_calls:
        return "action"

    return END


def get_agent_graph(additional_tools: list):
    model = ChatOpenAI(
        model="gpt-4.1-nano",
        temperature=0,  # Lower temperature for more consistent outputs
        request_timeout=120,  # Longer timeout for complex operations
    )

    def call_model(state):
        messages = state["messages"]
        response = model.invoke(messages)
        return {"messages": [response]}

    tool_belt = [
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
    tool_belt.extend(additional_tools)

    model = model.bind_tools(tool_belt)
    tool_node = ToolNode(tool_belt)

    uncompiled_graph = StateGraph(AgentState)

    uncompiled_graph.add_node("agent", call_model)
    uncompiled_graph.add_node("action", tool_node)

    uncompiled_graph.set_entry_point("agent")
    uncompiled_graph.add_conditional_edges("agent", should_continue)

    uncompiled_graph.add_edge("action", "agent")

    return uncompiled_graph.compile()
