import logging
from tavily import TavilyClient
import os
from typing import Optional

# Set up logging with third-party noise suppression
from src.utils.logging_config import setup_logging

logger = setup_logging(__name__)


def tavily_studentaid_search(query: str) -> str:
    """
    Search ONLY StudentAid.gov for official federal information: FAFSA applications,
    federal loan forgiveness programs, federal repayment plans, eligibility requirements.
    Use this when you need authoritative federal government information.

    Args:
        query (str): Search query for federal student aid topics

    Returns:
        str: Formatted search results containing:
            - Header with query information
            - AI-generated summary answer (if available)
            - Up to 3 official results with:
                • Title from StudentAid.gov page
                • Content snippet (200 characters max)
                • Full URL for verification
            - Error message if search fails

    Example Output:
        "StudentAid.gov Search Results for: income driven repayment

        Summary: Income-driven repayment plans calculate your monthly payment...

        Official Federal Information:
        1. Income-Driven Repayment Plans
           These plans calculate your monthly payment based on your income...
           URL: https://studentaid.gov/manage-loans/repayment/plans/income-driven

        2. Apply for Income-Driven Repayment
           Complete your IDR application online or by mail...
           URL: https://studentaid.gov/app/ibrInstructions.action"

    Technical Details:
        - Search depth: Advanced (comprehensive crawling)
        - Max results: 3 (focused, high-quality results)
        - Domain restriction: studentaid.gov only
        - Timeout: Handled with graceful error messages
    """
    try:
        logger.info(f"🔍 [StudentAid Search] Searching for: {query}")
        client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

        response = client.search(
            query=f"site:studentaid.gov {query}",
            search_depth="advanced",
            max_results=3,
            include_answer=True,
            include_domains=["studentaid.gov"],
        )

        logger.info(
            f"📚 [StudentAid Search] Found {len(response.get('results', []))} results"
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
        logger.error(f"❌ [StudentAid Search] Error: {str(e)}")
        return f"Error searching StudentAid.gov: {str(e)}"


def tavily_mohela_search(query: str) -> str:
    """
    Search ONLY Mohela loan servicer for account-specific help: making payments,
    login issues, servicer-specific repayment options, customer service contacts.
    Use this when users have Mohela-serviced loans and need servicer-specific help.

    Args:
        query (str): Search query for Mohela servicer-specific information

    Returns:
        str: Formatted search results containing:
            - Header with query information
            - AI-generated summary answer (if available)
            - Up to 3 Mohela-specific results with:
                • Title from Mohela website page
                • Content snippet (200 characters max)
                • Full URL for verification
            - Error message if search fails

    Example Output:
        "Mohela Search Results for: payment options

        Summary: Mohela offers several payment methods including...

        Mohela Servicer Information:
        1. Making Payments
           You can make payments online, by phone, or by mail...
           URL: https://servicing.mohela.com/payments

        2. Payment Plans
           Mohela offers standard and income-driven repayment plans...
           URL: https://mohela.com/repayment-options"

    Technical Details:
        - Search depth: Advanced (comprehensive crawling)
        - Max results: 3 (focused, servicer-specific results)
        - Domain restriction: mohela.com and servicing.mohela.com
        - Timeout: Handled with graceful error messages
    """
    try:
        logger.info(f"🔍 [Mohela Search] Searching for: {query}")
        client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

        response = client.search(
            query=f"site:mohela.com OR site:servicing.mohela.com {query}",
            search_depth="advanced",
            max_results=3,
            include_answer=True,
            include_domains=["mohela.com", "servicing.mohela.com"],
        )

        logger.info(
            f"📚 [Mohela Search] Found {len(response.get('results', []))} results"
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
        logger.error(f"❌ [Mohela Search] Error: {str(e)}")
        return f"Error searching Mohela: {str(e)}"


def tavily_student_loan_search(query: str, source: Optional[str] = None) -> str:
    """
    Compare information across BOTH federal sources and Mohela when user needs
    comprehensive view or comparison of student loan options. Use this when users
    want to see both federal policies and servicer-specific implementation, or when
    they're unsure which source has the information they need.

    Args:
        query (str): Search query for comprehensive student loan information
        source (Optional[str]): "studentaid" for StudentAid.gov only, "mohela" for Mohela only, None for both

    Returns:
        str: Formatted search results containing:
            - Header with query information
            - AI-generated summary answer (if available)
            - Up to 5 results from multiple sources:
                • Federal sources (StudentAid.gov)
                • Servicer sources (Mohela)
                • Title, content snippet (200 chars), and URL for each
            - Error message if search fails

    Example Output:
        "Comprehensive Student Loan Search Results for: loan forgiveness

        Summary: Student loan forgiveness programs include PSLF, IDR forgiveness...

        Federal and Servicer Information:
        1. Public Service Loan Forgiveness
           PSLF forgives remaining balance after 120 qualifying payments...
           URL: https://studentaid.gov/manage-loans/forgiveness-cancellation/public-service

        2. Mohela PSLF Processing
           As the PSLF servicer, Mohela handles PSLF applications...
           URL: https://mohela.com/pslf"

    Technical Details:
        - Search depth: Advanced (comprehensive crawling)
        - Max results: 5 (broader coverage across sources)
        - Domain scope: Multiple official sources (federal + servicers)
        - Comparison focus: Shows both policy and implementation perspectives
    """
    try:
        logger.info(
            f"🔍 [Comprehensive Search] Searching for: {query} (source: {source or 'both'})"
        )
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

            logger.info(
                f"📚 [Comprehensive Search] Found {len(response.get('results', []))} total results"
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
                logger.info(
                    f"📋 [Comprehensive Search] {len(studentaid_results)} StudentAid.gov results"
                )
                result += "📋 Federal Government Perspective (StudentAid.gov):\n"
                for i, item in enumerate(studentaid_results[:2], 1):
                    result += f"{i}. {item.get('title', 'No title')}\n"
                    result += f"   {item.get('content', '')[:150]}...\n"
                    result += f"   Source: {item.get('url', '')}\n\n"

            if mohela_results:
                logger.info(
                    f"🏢 [Comprehensive Search] {len(mohela_results)} Mohela results"
                )
                result += "🏢 Loan Servicer Perspective (Mohela):\n"
                for i, item in enumerate(mohela_results[:2], 1):
                    result += f"{i}. {item.get('title', 'No title')}\n"
                    result += f"   {item.get('content', '')[:150]}...\n"
                    result += f"   Source: {item.get('url', '')}\n\n"

            if not studentaid_results and not mohela_results:
                result += "No specific results found from StudentAid.gov or Mohela. Consider using general web search.\n"

            return result

    except Exception as e:
        logger.error(f"❌ [Comprehensive Search] Error: {str(e)}")
        return f"Error searching student loan information: {str(e)}"
