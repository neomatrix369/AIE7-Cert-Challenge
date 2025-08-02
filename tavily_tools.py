from tavily import TavilyClient
import os
from typing import Optional


def tavily_studentaid_search(query: str) -> str:
    """
    Search ONLY StudentAid.gov for official federal information: FAFSA applications,
    federal loan forgiveness programs, federal repayment plans, eligibility requirements.
    Use this when you need authoritative federal government information.

    Args:
        query: Search query for federal student aid topics

    Returns:
        Formatted search results from StudentAid.gov
    """
    try:
        client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

        response = client.search(
            query=f"site:studentaid.gov {query}",
            search_depth="advanced",
            max_results=3,
            include_answer=True,
            include_domains=["studentaid.gov"],
        )

        result = f"StudentAid.gov Search Results for: {query}\n\n"

        if response.get("answer"):
            result += f"Summary: {response['answer']}\n\n"

        result += "Official Federal Information:\n"
        for i, item in enumerate(response.get("results", []), 1):
            result += f"{i}. {item.get('title', 'No title')}\n"
            result += f"   {item.get('content', '')[:200]}...\n"
            result += f"   URL: {item.get('url', '')}\n\n"

        return result

    except Exception as e:
        return f"Error searching StudentAid.gov: {str(e)}"


def tavily_mohela_search(query: str) -> str:
    """
    Search ONLY Mohela loan servicer for account-specific help: making payments,
    login issues, servicer-specific repayment options, customer service contacts.
    Use this when users have Mohela-serviced loans and need servicer-specific help.

    Args:
        query: Search query for Mohela servicer-specific information

    Returns:
        Formatted search results from Mohela
    """
    try:
        client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

        response = client.search(
            query=f"site:mohela.com OR site:servicing.mohela.com {query}",
            search_depth="advanced",
            max_results=3,
            include_answer=True,
            include_domains=["mohela.com", "servicing.mohela.com"],
        )

        result = f"Mohela Search Results for: {query}\n\n"

        if response.get("answer"):
            result += f"Summary: {response['answer']}\n\n"

        result += "Mohela Servicer Information:\n"
        for i, item in enumerate(response.get("results", []), 1):
            result += f"{i}. {item.get('title', 'No title')}\n"
            result += f"   {item.get('content', '')[:200]}...\n"
            result += f"   URL: {item.get('url', '')}\n\n"

        return result

    except Exception as e:
        return f"Error searching Mohela: {str(e)}"


def tavily_student_loan_search(query: str, source: Optional[str] = None) -> str:
    """
    Compare information across BOTH federal sources and Mohela when user needs
    comprehensive view or comparison of student loan options. Use this when users
    want to see both federal policies and servicer-specific implementation, or when
    they're unsure which source has the information they need.

    Args:
        query: Search query for student loan information requiring comparison
        source: Optional - "studentaid" for StudentAid.gov only, "mohela" for Mohela only

    Returns:
        Formatted search results comparing both sources
    """
    try:
        client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

        if source == "studentaid":
            return tavily_studentaid_search(query)
        elif source == "mohela":
            return tavily_mohela_search(query)
        else:
            # Search both sources for comparison
            search_query = (
                f"student loan {query} site:studentaid.gov OR site:mohela.com"
            )

            response = client.search(
                query=search_query,
                search_depth="advanced",
                max_results=5,
                include_answer=True,
                include_domains=[
                    "studentaid.gov",
                    "mohela.com",
                    "servicing.mohela.com",
                ],
            )

            result = f"Comprehensive Student Loan Search Results for: {query}\n\n"

            if response.get("answer"):
                result += f"Overall Summary: {response['answer']}\n\n"

            # Group results by domain for comparison
            studentaid_results = []
            mohela_results = []

            for item in response.get("results", []):
                url = item.get("url", "")
                if "studentaid.gov" in url:
                    studentaid_results.append(item)
                elif "mohela.com" in url:
                    mohela_results.append(item)

            if studentaid_results:
                result += "📋 Federal Government Perspective (StudentAid.gov):\n"
                for i, item in enumerate(studentaid_results[:2], 1):
                    result += f"{i}. {item.get('title', 'No title')}\n"
                    result += f"   {item.get('content', '')[:150]}...\n"
                    result += f"   Source: {item.get('url', '')}\n\n"

            if mohela_results:
                result += "🏢 Loan Servicer Perspective (Mohela):\n"
                for i, item in enumerate(mohela_results[:2], 1):
                    result += f"{i}. {item.get('title', 'No title')}\n"
                    result += f"   {item.get('content', '')[:150]}...\n"
                    result += f"   Source: {item.get('url', '')}\n\n"

            if not studentaid_results and not mohela_results:
                result += "No specific results found from StudentAid.gov or Mohela. Consider using general web search.\n"

            return result

    except Exception as e:
        return f"Error searching student loan information: {str(e)}"
