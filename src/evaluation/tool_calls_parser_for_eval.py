import re
import json
import logging

# Set up logging with third-party noise suppression
from src.utils.logging_config import setup_logging

logger = setup_logging(__name__)


class ToolCallParser:
    """Comprehensive parser for LLM tool calls from execution logs"""

    def parse(self, text: str) -> dict:
        """Parse text and return all tool calls and results"""
        tool_calls = []
        tool_results = []

        # Extract tool calls from AIMessage blocks (when available)
        self._extract_tool_calls(text, tool_calls)

        # Extract tool results from ToolMessage blocks (multiple patterns)
        self._extract_tool_results(text, tool_results)

        return {
            "calls": tool_calls,
            "results": tool_results,
            "summary": {
                "total_calls": len(tool_calls),
                "total_results": len(tool_results),
                "tools": list(set([t["tool"] for t in tool_calls + tool_results])),
                "nodes": list(
                    set([t.get("node") for t in tool_calls if t.get("node")])
                ),
            },
        }

    def _extract_tool_calls(self, text: str, tool_calls: list):
        """Extract tool calls from AIMessage blocks"""
        ai_pattern = r"Receiving update from node: '([^']+)'[\s\S]*?'tool_calls':\s*\[([\s\S]*?)\]"
        for node_match in re.finditer(ai_pattern, text):
            node = node_match.group(1)
            calls_str = node_match.group(2)

            # Parse individual calls
            call_pattern = r"'id':\s*'([^']+)',\s*'function':\s*\{[^}]*'arguments':\s*'([^']+)',\s*'name':\s*'([^']+)'"
            for call_match in re.finditer(call_pattern, calls_str):
                call_id, args_str, tool_name = call_match.groups()

                # Parse arguments JSON
                try:
                    args = json.loads(args_str.replace('\\"', '"'))
                except:
                    args = {"query": args_str.strip('"{}')}

                tool_calls.append(
                    {"id": call_id, "tool": tool_name, "args": args, "node": node}
                )

    def _extract_tool_results(self, text: str, tool_results: list):
        """Extract tool results from ToolMessage blocks - handles all content patterns"""

        # Pattern 1: Single quoted content - content='...'
        pattern1 = r"ToolMessage\(content='([^']*(?:\\.[^']*)*)'.*?name='([^']+)'.*?tool_call_id='([^']+)'"
        for match in re.finditer(pattern1, text, re.DOTALL):
            content, tool_name, call_id = match.groups()
            content = content.replace("\\n", "\n").replace('\\"', '"')
            tool_results.append(
                {
                    "id": call_id,
                    "tool": tool_name,
                    "content": content[:300] + "..." if len(content) > 300 else content,
                }
            )

        # Pattern 2: Double quoted content - content="..."
        pattern2 = r'ToolMessage\(content="([^"]*(?:\\.[^"]*)*)".*?name=\'([^\']+)\'.*?tool_call_id=\'([^\']+)\''
        for match in re.finditer(pattern2, text, re.DOTALL):
            content, tool_name, call_id = match.groups()
            content = content.replace("\\n", "\n").replace('\\"', '"')
            tool_results.append(
                {
                    "id": call_id,
                    "tool": tool_name,
                    "content": content[:300] + "..." if len(content) > 300 else content,
                }
            )

        # Pattern 3: JSON array content (no quotes) - content=[...]
        pattern3 = r"ToolMessage\(content=(\[[^\]]*\]).*?name='([^']+)'.*?tool_call_id='([^']+)'"
        for match in re.finditer(pattern3, text, re.DOTALL):
            content, tool_name, call_id = match.groups()
            tool_results.append(
                {
                    "id": call_id,
                    "tool": tool_name,
                    "content": content[:300] + "..." if len(content) > 300 else content,
                }
            )


# Simple usage function
def parse_logs(text: str) -> dict:
    """Main function to parse LLM tool execution logs"""
    return ToolCallParser().parse(text)


def parse_langchain_messages(messages) -> dict:
    """
    Parse LangChain message objects from LangGraph streaming response.

    Args:
        messages: List of LangChain message objects from values["messages"]

    Returns:
        dict: Parsed tool calls and results with extracted contexts
    """
    tool_calls = []
    tool_results = []
    contexts = []

    for message in messages:
        # Handle AI messages with tool calls
        if hasattr(message, "tool_calls") and message.tool_calls:
            for tool_call in message.tool_calls:
                tool_calls.append(
                    {
                        "id": tool_call.get("id", ""),
                        "tool": tool_call.get("name", ""),
                        "args": tool_call.get("args", {}),
                        "type": tool_call.get("type", "function"),
                    }
                )

        # Handle tool messages (results)
        elif hasattr(message, "name") and hasattr(message, "content"):
            tool_name = message.name
            content = message.content
            tool_call_id = getattr(message, "tool_call_id", "")

            tool_results.append(
                {"id": tool_call_id, "tool": tool_name, "content": content}
            )

            # Extract contexts for evaluation
            extracted_contexts = _extract_contexts_from_tool_result(tool_name, content)
            contexts.extend(extracted_contexts)

    return {
        "calls": tool_calls,
        "results": tool_results,
        "contexts": contexts,
        "summary": {
            "total_calls": len(tool_calls),
            "total_results": len(tool_results),
            "total_contexts": len(contexts),
            "tools": list(set([r["tool"] for r in tool_results])),
        },
    }


def _extract_contexts_from_tool_result(tool_name: str, content: str) -> list:
    """Extract contexts from a single tool result for evaluation"""
    contexts = []

    if not tool_name:
        return []

    # Handle RAG tool responses with custom format
    if any(
        rag_tool in tool_name.lower()
        for rag_tool in [
            "naive_llm",
            "contextual_compression",
            "multi_query",
            "parent_document",
        ]
    ):
        contexts.extend(_extract_rag_tool_contexts(content))
    elif tool_name == "tavily_search_results_json":
        contexts.extend(_extract_tavily_contexts(content))
    elif tool_name == "arxiv":
        contexts.extend(_extract_arxiv_contexts(content))
    elif "search" in tool_name.lower():
        contexts.extend(_extract_search_contexts(content))
    else:
        # Generic extraction - split into meaningful chunks
        if content and len(content.strip()) > 20:
            # Split by double newlines (paragraphs) or sentences
            chunks = content.split("\n\n")
            if len(chunks) == 1:
                # If no paragraphs, split by sentences
                import re

                chunks = re.split(r"(?<=[.!?])\s+", content)

            # Take meaningful chunks (not too short)
            meaningful_chunks = [
                chunk.strip() for chunk in chunks if len(chunk.strip()) > 30
            ]
            contexts.extend(meaningful_chunks[:5])  # Max 5 contexts per tool result

    return contexts


def parse_tool_call(message_obj) -> list:
    """
    Parse tool call results from a LangChain message object for context extraction.

    This function mimics the logic shown in the screenshot:
    - Extracts content from tool call results
    - Handles different result formats (Tavily, arXiv, etc.)
    - Returns a list of context strings suitable for RAG evaluation

    Args:
        message_obj: LangChain message object or dict representation

    Returns:
        list: List of context strings extracted from tool results
    """
    results = []

    # Handle case where message_obj is already parsed results
    if isinstance(message_obj, list):
        return _extract_contexts_from_results(message_obj)

    # Handle LangChain message object
    if hasattr(message_obj, "content"):
        # Try to parse as our tool call format first
        if isinstance(message_obj.content, str):
            parsed = ToolCallParser().parse(message_obj.content)
            if parsed["results"]:
                return _extract_contexts_from_results(parsed["results"])

    # Handle dict representation of message
    if isinstance(message_obj, dict):
        # Check if it has tool_calls or content
        if "tool_calls" in message_obj:
            # This is likely an AI message with tool calls
            return []  # Tool calls don't have contexts, only results do
        elif "content" in message_obj:
            # This might be a tool result message
            return _extract_context_from_content(message_obj["content"])

    return results


def _extract_contexts_from_results(results: list) -> list:
    """Extract context strings from parsed tool results"""
    contexts = []

    for result in results:
        if isinstance(result, dict):
            content = result.get("content", "")
            tool_name = result.get("tool", "")

            if not tool_name:
                continue

            # Handle different tool result formats
            if tool_name == "tavily_search_results_json":
                # Tavily returns JSON array of search results
                contexts.extend(_extract_tavily_contexts(content))
            elif tool_name == "arxiv":
                # arXiv returns paper summaries
                contexts.extend(_extract_arxiv_contexts(content))
            elif "search" in tool_name.lower():
                # Generic search results
                contexts.extend(_extract_search_contexts(content))
            else:
                # Fallback: use content as-is if it's meaningful
                if content and len(content.strip()) > 10:
                    contexts.append(content.strip())

    return contexts


def _extract_context_from_content(content: str) -> list:
    """Extract contexts from raw content string"""
    contexts = []

    # Try to parse as JSON first (for Tavily-like results)
    if content.strip().startswith("["):
        try:
            import json

            data = json.loads(content)
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        # Extract title and content from search results
                        text_parts = []
                        if "title" in item:
                            text_parts.append(item["title"])
                        if "content" in item:
                            text_parts.append(item["content"])
                        if text_parts:
                            contexts.append(" - ".join(text_parts))
            return contexts
        except:
            pass

    # For non-JSON content, split by common delimiters
    if "Title:" in content and "Summary:" in content:
        # Looks like arXiv format
        contexts.extend(_extract_arxiv_contexts(content))
    else:
        # Generic text content
        paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
        contexts.extend(paragraphs[:3])  # Take first 3 paragraphs

    return contexts


def _extract_tavily_contexts(content: str) -> list:
    """Extract contexts from Tavily search results"""
    contexts = []

    try:
        import json

        # Handle case where content might be truncated with "..."
        if content.endswith("..."):
            content = content[:-3] + "]"

        data = json.loads(content)
        for item in data:
            if isinstance(item, dict):
                text_parts = []
                if "title" in item:
                    text_parts.append(item["title"])
                if "content" in item:
                    text_parts.append(item["content"])
                if text_parts:
                    contexts.append(" - ".join(text_parts))
    except:
        # Fallback: treat as plain text
        contexts.append(content)

    return contexts


def _extract_arxiv_contexts(content: str) -> list:
    """Extract contexts from arXiv paper results"""
    contexts = []

    # Split by papers (usually separated by "Published:" or similar)
    papers = content.split("Published:")

    for paper in papers[1:]:  # Skip first empty split
        lines = paper.strip().split("\n")
        summary_parts = []

        for line in lines:
            if line.startswith("Title:"):
                summary_parts.append(line)
            elif line.startswith("Authors:"):
                summary_parts.append(line)
            elif line.startswith("Summary:"):
                # Take the summary but limit length
                summary_text = line + " " + " ".join(lines[lines.index(line) + 1 :])
                summary_parts.append(
                    summary_text[:500] + "..."
                    if len(summary_text) > 500
                    else summary_text
                )
                break

        if summary_parts:
            contexts.append("\n".join(summary_parts))

    return contexts


def _extract_search_contexts(content: str) -> list:
    """Extract contexts from generic search results"""
    contexts = []

    # Try JSON first
    if content.strip().startswith("[") or content.strip().startswith("{"):
        try:
            import json

            data = json.loads(content)
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and "content" in item:
                        contexts.append(item["content"])
            elif isinstance(data, dict) and "content" in data:
                contexts.append(data["content"])
        except:
            contexts.append(content)
    else:
        # Split by paragraphs
        paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
        contexts.extend(paragraphs[:5])  # Take first 5 paragraphs

    return contexts


def _extract_rag_tool_contexts(content: str) -> list:
    """Extract contexts from custom RAG tool responses that return {'messages': [...], 'context': [...]}"""
    contexts = []

    try:
        import ast
        import re

        # Handle the specific format: [{'messages': [...], 'context': [...]}]
        if content.strip().startswith("[{'messages':") and "'context':" in content:
            try:
                # Try to parse the entire list
                parsed_list = ast.literal_eval(content)
                if isinstance(parsed_list, list) and len(parsed_list) > 0:
                    # Get the first (and likely only) dict in the list
                    result_dict = parsed_list[0]
                    if isinstance(result_dict, dict) and "context" in result_dict:
                        context_data = result_dict["context"]

                        # Extract from context list with relevance scores
                        if isinstance(context_data, list):
                            for item in context_data:
                                if hasattr(item, "page_content"):
                                    # Check for relevance score in metadata
                                    relevance_score = None
                                    if hasattr(item, "metadata") and item.metadata:
                                        relevance_score = item.metadata.get(
                                            "relevance_score"
                                        )

                                    if relevance_score is not None:
                                        contexts.append(
                                            {
                                                "content": item.page_content,
                                                "relevance_score": relevance_score,
                                            }
                                        )
                                    else:
                                        contexts.append(item.page_content)
                                elif isinstance(item, dict) and "page_content" in item:
                                    # Check for relevance score in dict metadata
                                    relevance_score = None
                                    if "metadata" in item and item["metadata"]:
                                        relevance_score = item["metadata"].get(
                                            "relevance_score"
                                        )

                                    if relevance_score is not None:
                                        contexts.append(
                                            {
                                                "content": item["page_content"],
                                                "relevance_score": relevance_score,
                                            }
                                        )
                                    else:
                                        contexts.append(item["page_content"])
                                elif isinstance(item, str) and len(item.strip()) > 30:
                                    contexts.append(item.strip())
                        return contexts
            except Exception as e:
                # If parsing fails, try regex approach
                pass

        # Handle the case where content is a string representation of a dictionary
        if "{'messages':" in content and "'context':" in content:
            # Try to extract the context part using regex
            context_match = re.search(r"'context':\s*(\[.*?\])", content, re.DOTALL)
            if context_match:
                context_str = context_match.group(1)
                try:
                    # Try to safely evaluate the list
                    context_data = ast.literal_eval(context_str)
                    if isinstance(context_data, list):
                        for item in context_data:
                            if hasattr(item, "page_content"):
                                # LangChain Document object
                                contexts.append(item.page_content)
                            elif isinstance(item, dict) and "page_content" in item:
                                contexts.append(item["page_content"])
                            elif isinstance(item, str) and len(item.strip()) > 30:
                                contexts.append(item.strip())
                except:
                    pass

        # ❌ REMOVED: HumanMessage extraction that was pulling response content instead of contexts
        # The original code was extracting from HumanMessage(content=...) which contains the LLM response,
        # not the actual retrieved document contexts we need for evaluation

        # ✅ NEW: Enhanced fallback for complex serialized formats
        # If we still haven't extracted contexts, try to find Document-like patterns in the string
        if not contexts and "page_content" in content:
            # Look for page_content patterns in the serialized string
            page_content_pattern = r"page_content='([^']+)'"
            page_content_matches = re.findall(page_content_pattern, content)
            if page_content_matches:
                contexts.extend(
                    [match for match in page_content_matches if len(match.strip()) > 30]
                )

            # Alternative pattern with double quotes
            if not contexts:
                page_content_pattern_dq = r'page_content="([^"]+)"'
                page_content_matches_dq = re.findall(page_content_pattern_dq, content)
                if page_content_matches_dq:
                    contexts.extend(
                        [
                            match
                            for match in page_content_matches_dq
                            if len(match.strip()) > 30
                        ]
                    )

    except Exception as e:
        # Ultimate fallback: treat as regular text and extract meaningful chunks
        if content and len(content.strip()) > 30:
            # Split by sentences or paragraphs
            chunks = (
                content.split("\\n\\n")
                if "\\n\\n" in content
                else content.split("\n\n")
            )
            if len(chunks) == 1:
                import re

                chunks = re.split(r"(?<=[.!?])\s+", content)

            meaningful_chunks = [
                chunk.strip() for chunk in chunks if len(chunk.strip()) > 30
            ]
            contexts.extend(meaningful_chunks[:5])

    return contexts


def print_formatted_results(data):
    logger.info("=== PARSING RESULTS ===")
    logger.info(f"Tool calls: {data['summary']['total_calls']}")
    logger.info(f"Tool results: {data['summary']['total_results']}")
    logger.info(f"Tools used: {data['summary']['tools']}")
    if ("nodes" in data["summary"]) and data["summary"]["nodes"]:
        logger.info(f"Execution nodes: {data['summary']['nodes']}")

    logger.info("=== TOOL CALLS ===")
    for i, call in enumerate(data["calls"], 1):
        logger.info(f"{i}. {call['tool']} -> {call['args']}")

    logger.info("=== TOOL RESULTS ===")
    for i, result in enumerate(data["results"], 1):
        preview = result["content"][:100].replace("\n", " ")
        logger.info(f"{i}. {result['tool']} -> {preview}...")

    # Demo the new context extraction functionality
    logger.info("=== CONTEXT EXTRACTION DEMO ===")
    contexts = parse_tool_call(data["results"])
    logger.info(f"📚 Extracted {len(contexts)} contexts for evaluation")
    for i, context in enumerate(contexts[:3], 1):
        logger.debug(f"{i}. {context[:150]}...")


# Test and demo
if __name__ == "__main__":
    # Example usage with file
    try:
        with open("paste.txt", "r") as f:
            data = parse_logs(f.read())

        logger.info("=== PARSING RESULTS ===")
        logger.info(f"Tool calls: {data['summary']['total_calls']}")
        logger.info(f"Tool results: {data['summary']['total_results']}")
        logger.info(f"Tools used: {data['summary']['tools']}")
        if data["summary"]["nodes"]:
            logger.info(f"Execution nodes: {data['summary']['nodes']}")

        logger.info("=== TOOL CALLS ===")
        for i, call in enumerate(data["calls"], 1):
            logger.info(f"{i}. {call['tool']} -> {call['args']}")

        logger.info("=== TOOL RESULTS ===")
        for i, result in enumerate(data["results"], 1):
            preview = result["content"][:100].replace("\n", " ")
            logger.info(f"{i}. {result['tool']} -> {preview}...")

        # Demo the new context extraction functionality
        logger.info("=== CONTEXT EXTRACTION DEMO ===")
        contexts = parse_tool_call(data["results"])
        logger.info(f"📚 Extracted {len(contexts)} contexts for evaluation")
        for i, context in enumerate(contexts[:3], 1):
            logger.debug(f"{i}. {context[:150]}...")

    except FileNotFoundError:
        logger.warning(
            "No paste.txt file found. Use parse_logs(your_text) to parse any log text."
        )


def create_eval_sample(user_input: str, ai_response: str, tool_results: list) -> dict:
    """
    Create an evaluation sample in the format shown in the screenshot.

    This function mimics the evaluation pipeline logic:
    - Takes user input, AI response, and tool results
    - Extracts contexts from tool results
    - Returns structured eval sample

    Args:
        user_input: The original user query/input
        ai_response: The AI's final response
        tool_results: List of tool result objects from parse_logs()

    Returns:
        dict: Evaluation sample with retrieved_contexts populated
    """

    # Extract contexts from tool results
    retrieved_contexts = parse_tool_call(tool_results)

    # Create evaluation sample structure
    eval_sample = {
        "user_input": user_input,
        "response": ai_response,
        "retrieved_contexts": retrieved_contexts,
        # Additional metadata
        "tools_used": list(set([r.get("tool", "") for r in tool_results])),
        "num_contexts": len(retrieved_contexts),
        "num_tool_calls": len(tool_results),
    }

    return eval_sample


# Clean function for processing LangChain messages (no debug output)
def extract_contexts_for_eval(langchain_messages):
    """
    Extract contexts from LangChain messages for evaluation.

    Args:
        langchain_messages: List of LangChain message objects from values["messages"]
                           OR a list that contains context strings (mixed format handling)

    Returns:
        list: List of context strings for retrieved_contexts field
    """

    # Handle direct list of strings (edge case: ["{'messages': [HumanMessage(content='...', 'context1', 'context2', ...])
    if isinstance(langchain_messages, list):
        # Check if this is a list of strings with context data
        if all(isinstance(item, str) for item in langchain_messages):
            contexts = []
            for item in langchain_messages:
                # If item starts with serialized format, try to extract contexts from it
                if item.strip().startswith("[{'messages':") or item.strip().startswith(
                    "{'messages':"
                ):
                    # Try to extract contexts from this serialized format
                    extracted = _extract_rag_tool_contexts(item)
                    contexts.extend(extracted)
                # Otherwise, add as regular context string if meaningful
                elif len(item.strip()) > 30:
                    contexts.append(item.strip())
            if contexts:
                return contexts

    # Use our main parser function for LangChain message objects
    parsed_data = parse_langchain_messages(langchain_messages)

    # Return just the contexts (what goes into retrieved_contexts)
    return parsed_data.get("contexts", [])


# Debug version (if you want to see what's happening)
def process_message_for_eval_debug(message_obj):
    """
    Debug version that shows what's being processed.
    Use extract_contexts_for_eval() for clean output.
    """

    logger.debug("=== Starting message parsing ===")
    logger.debug(f"Message type: {type(message_obj)}")

    # Handle different input types
    if hasattr(message_obj, "__iter__") and not isinstance(message_obj, str):
        # It's a list of messages
        parsed_data = parse_langchain_messages(message_obj)
        tool_names = [r["tool"] for r in parsed_data.get("results", [])]
        logger.debug(f"Found tools: {tool_names}")
        logger.info(
            f"📚 Extracted {len(parsed_data.get('contexts', []))} contexts from message list"
        )
        return parsed_data.get("contexts", [])

    elif hasattr(message_obj, "name"):
        # Single LangChain message
        logger.debug(f"Found tool name: {message_obj.name}")
        parsed_data = parse_langchain_messages([message_obj])
        contexts = parsed_data.get("contexts", [])
        logger.info(f"📚 Extracted {len(contexts)} contexts from single message")
        return contexts

    else:
        logger.warning(f"⚠️ Unexpected message type: {type(message_obj)}")
        return []


def extract_token_usage_from_messages(messages, question_text="", answer_text=""):
    """
    Extract token usage information from LangChain message objects.

    Args:
        messages: List of LangChain message objects from agent response
        question_text: Original question text for fallback estimation
        answer_text: Generated answer text for fallback estimation

    Returns:
        dict: Token usage breakdown with input_tokens, output_tokens, total_tokens
    """
    input_tokens = 0
    output_tokens = 0
    total_tokens = 0

    for message in messages:
        # Check for usage_metadata (newer LangChain format)
        if hasattr(message, "usage_metadata"):
            usage = message.usage_metadata
            if hasattr(usage, "input_tokens"):
                input_tokens += usage.input_tokens
            if hasattr(usage, "output_tokens"):
                output_tokens += usage.output_tokens
            if hasattr(usage, "total_tokens"):
                total_tokens += usage.total_tokens
        # Check for response_metadata with token_usage (OpenAI format)
        elif hasattr(message, "response_metadata"):
            metadata = message.response_metadata
            if "token_usage" in metadata:
                token_usage = metadata["token_usage"]
                input_tokens += token_usage.get("prompt_tokens", 0)
                output_tokens += token_usage.get("completion_tokens", 0)
                total_tokens += token_usage.get("total_tokens", 0)

    # If no token data found in messages, estimate from text
    if total_tokens == 0:
        # Rough estimation: 1 token ≈ 0.75 words for GPT models
        estimated_input_tokens = (
            len(question_text.split()) / 0.75 if question_text else 0
        )
        estimated_output_tokens = len(answer_text.split()) / 0.75 if answer_text else 0
        input_tokens = int(estimated_input_tokens)
        output_tokens = int(estimated_output_tokens)
        total_tokens = input_tokens + output_tokens

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


def process_agent_response(response):
    """
    Extract common data from agent response messages.

    Args:
        response: Agent response dict with "messages" key

    Returns:
        dict: Contains contexts, tools_used, final_answer, messages
    """
    messages = response["messages"]
    contexts = extract_contexts_for_eval(messages)
    parsed_data = parse_langchain_messages(messages)
    tools_used_raw = parsed_data.get("summary", {}).get("tools", [])

    # Filter out None values and ensure all are strings
    tools_used = [str(tool) for tool in tools_used_raw if tool is not None]

    final_answer = messages[-1].content if messages else ""

    return {
        "contexts": contexts,
        "tools_used": tools_used,
        "final_answer": final_answer,
        "messages": messages,
    }


def build_performance_metrics(
    start_time,
    end_time,
    messages,
    contexts,
    question_text="",
    answer_text="",
    retrieval_method="parent_document",
):
    """
    Build comprehensive performance metrics from RAG agent execution.

    Args:
        start_time: Start timestamp (time.time())
        end_time: End timestamp (time.time())
        messages: List of LangChain message objects
        contexts: List of retrieved contexts
        question_text: Original question for token estimation
        answer_text: Generated answer for token estimation
        retrieval_method: The retrieval method used

    Returns:
        dict: Comprehensive performance metrics
    """
    # Calculate timing metrics
    total_time_ms = int((end_time - start_time) * 1000)
    # Estimate retrieval vs generation split (roughly 60/40 based on typical RAG patterns)
    retrieval_time_ms = int(total_time_ms * 0.6)
    generation_time_ms = int(total_time_ms * 0.4)

    # Extract token usage
    token_usage = extract_token_usage_from_messages(
        messages, question_text, answer_text
    )

    # Build comprehensive metrics
    performance_metrics = {
        "response_time_ms": total_time_ms,
        "retrieval_time_ms": retrieval_time_ms,
        "generation_time_ms": generation_time_ms,
        # -- we don't know for sure so best not to return to user
        # "confidence_score": 0.85,  # Typical confidence for Parent Document retrieval
        "tokens_used": token_usage["total_tokens"],
        "input_tokens": token_usage["input_tokens"],
        "output_tokens": token_usage["output_tokens"],
        "retrieval_method": retrieval_method,
        "total_contexts": len(contexts) if contexts else 0,
    }

    return performance_metrics
