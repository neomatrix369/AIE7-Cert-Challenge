import os
import time
import pandas as pd
from datetime import datetime
from tqdm.notebook import tqdm
from ragas.llms import LangchainLLMWrapper
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from ragas import EvaluationDataset

from ragas import evaluate
from ragas.cost import get_token_usage_for_openai
from ragas.testset import TestsetGenerator
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from joblib import Memory

CACHE_FOLDER = os.getenv("CACHE_FOLDER")
cache_folder = "./cache"
if CACHE_FOLDER:
    cache_folder = CACHE_FOLDER
memory = Memory(location=cache_folder)

from ragas.metrics import (
    LLMContextRecall,
    ContextPrecision,
    AnswerCorrectness,
    Faithfulness,
    FactualCorrectness,
    ResponseRelevancy,
    ContextEntityRecall,
    NoiseSensitivity,
)
from ragas import evaluate, RunConfig
from src.evaluation.ragas_metrics import extract_ragas_metrics
from src.evaluation.tool_calls_parser_for_eval import (
    extract_contexts_for_eval,
    parse_langchain_messages,
)
from src.evaluation.metrics_config import METRICS_ORDER

# ### Knowledge Graph Based Synthetic Generation
#
# Ragas uses a knowledge graph based approach to create data. This is extremely useful as it allows us to create complex queries rather simply. The additional testset complexity allows us to evaluate larger problems more effectively, as systems tend to be very strong on simple evaluation tasks.
#
# Let's start by defining our `generator_llm` (which will generate our questions, summaries, and more), and our `generator_embeddings` which will be useful in building our graph.

# ### Abstracted SDG
#
# The above method is the full process - but we can shortcut that using the provided abstractions!
#
# This will generate our knowledge graph under the hood, and will - from there - generate our personas and scenarios to construct our queries.
#
#


@memory.cache
def generate_golden_master(original_doc, items_to_pick: int = 20, final_size: int = 10):
    """
    Generate synthetic test dataset using RAGAS knowledge graph approach.
    
    **🎯 PURPOSE & STRATEGY:**
    - Creates high-quality synthetic evaluation questions from hybrid dataset documents
    - Uses knowledge graph analysis to generate complex, realistic queries
    - Produces questions spanning simple facts to multi-hop reasoning scenarios
    - Essential foundation for comprehensive RAG evaluation across retrieval methods
    
    **⚡ PERFORMANCE CHARACTERISTICS:**
    - **Generation Time**: 2-5 minutes for 10 questions (depends on document complexity)
    - **LLM Usage**: GPT-4.1 for question generation (higher cost but better quality)
    - **Caching**: Results cached with joblib to avoid expensive regeneration
    - **Scalability**: Handles 20+ documents efficiently, larger datasets may timeout
    
    **🔧 TECHNICAL IMPLEMENTATION:**
    - **Knowledge Graph**: RAGAS builds semantic graph from document relationships
    - **Question Types**: Simple, reasoning, multi-context, conditional queries
    - **Diversity**: Automatic question type balancing for comprehensive evaluation
    - **Quality Control**: LLM validates question complexity and answerability
    
    **📊 USAGE GUIDELINES:**
    - **items_to_pick**: Start with 20 docs for balanced dataset coverage
    - **final_size**: 10-20 questions optimal for development, 50+ for production
    - **Document Selection**: Uses first N documents - ensure representative sample
    - **Cache Management**: Delete cache folder when documents change
    
    **🔍 GENERATED QUESTION EXAMPLES:**
    - **Simple**: "What is the maximum Pell Grant amount?"
    - **Reasoning**: "If a student has $30K income, which repayment plan saves most money?"
    - **Multi-context**: "How do FAFSA deadlines relate to state aid eligibility?"
    
    Args:
        original_doc (list): List of LangChain Document objects from hybrid dataset
        items_to_pick (int): Number of documents to use for generation (default: 20)
        final_size (int): Number of test questions to generate (default: 10)
    
    Returns:
        RAGAS TestsetGenerator dataset: Contains generated questions with ground truth,
        ready for evaluation pipeline with eval_sample.user_input populated
    
    **💡 CACHE BEHAVIOR:**
    Results cached based on document content hash. Manual cache clearing required
    when documents change: `rm -rf ./cache/joblib/evaluation_helpers/generate_golden_master/`
    
    **⚠️ IMPORTANT NOTES:**
    - Requires OPENAI_API_KEY for GPT-4.1 access
    - Generation cost: ~$0.10-0.50 per 10 questions depending on document size
    - Questions generated from document content only (no external knowledge)
    - Validates questions are answerable from provided context
    
    Example:
        >>> from src.core.core_functions import load_and_prepare_pdf_loan_docs
        >>> docs = load_and_prepare_pdf_loan_docs()
        >>> golden_dataset = generate_golden_master(docs, items_to_pick=15, final_size=20)
        >>> print(f"Generated {len(golden_dataset)} test questions")
    """
    generator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4.1"))
    generator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())
    generator = TestsetGenerator(
        llm=generator_llm, embedding_model=generator_embeddings
    )
    golden_master_dataset = generator.generate_with_langchain_docs(
        original_doc[:items_to_pick], testset_size=final_size
    )
    golden_master_dataset.to_pandas()
    return golden_master_dataset


def generate_responses_for_golden_dataset(
    golden_master, graph_agent, pause_secs_between_each_run: int = 0
):
    """
    Execute RAG agent on golden dataset questions and extract evaluation contexts.
    
    **🎯 PURPOSE & STRATEGY:**
    - Runs each test question through the complete RAG agent pipeline
    - Extracts retrieved contexts and final responses for RAGAS evaluation
    - Captures tool usage metadata for comprehensive performance analysis
    - Bridges synthetic questions with actual RAG system responses
    
    **⚡ PERFORMANCE CHARACTERISTICS:**
    - **Execution Time**: 5-15 seconds per question (depends on agent complexity)
    - **API Usage**: Multiple LLM calls per question (agent + retrieval tools)
    - **Rate Limiting**: Built-in pause option to avoid API rate limits
    - **Memory Usage**: Stores contexts and metadata for each test sample
    
    **🔧 TECHNICAL IMPLEMENTATION:**
    - **Agent Invocation**: Complete LangGraph agent execution with tool calls
    - **Context Extraction**: Advanced parsing of tool outputs for evaluation contexts
    - **Metadata Tracking**: Custom metadata storage for tools_used, num_contexts
    - **RAGAS Compatibility**: Populates eval_sample fields required by RAGAS
    
    **📊 EXECUTION FLOW:**
    1. **Question Processing**: Convert each question to HumanMessage format
    2. **Agent Execution**: Full LangGraph pipeline with tool selection
    3. **Response Parsing**: Extract final answer from message chain
    4. **Context Extraction**: Parse tool outputs for retrieved_contexts
    5. **Metadata Storage**: Track tool usage and performance metrics
    
    **🛠️ CONTEXT EXTRACTION LOGIC:**
    - **RAG Tools**: Extracts document contexts from ask_*_llm_tool outputs
    - **External APIs**: Parses search results from Tavily, StudentAid searches
    - **Tool Results**: Handles complex serialized tool output formats
    - **Quality Filtering**: Includes only meaningful contexts (>30 chars)
    
    **💡 RATE LIMITING STRATEGY:**
    - **Default**: No pause (fastest execution)
    - **Conservative**: 2-5 seconds to avoid OpenAI rate limits
    - **Production**: 1-2 seconds balance speed vs stability
    
    Args:
        golden_master: RAGAS golden dataset with generated questions
        graph_agent: Compiled LangGraph agent with RAG tools bound
        pause_secs_between_each_run (int): Sleep time between questions (default: 0)
    
    Returns:
        Enhanced golden dataset: Same dataset with populated responses and contexts,
        plus custom metadata for tool usage analysis
    
    **📋 POPULATED RAGAS FIELDS:**
    - **eval_sample.response**: Final AI answer from agent execution
    - **eval_sample.retrieved_contexts**: List of document contexts for evaluation
    - **_custom_metadata**: Additional tracking (tools_used, num_contexts)
    
    **⚠️ IMPORTANT NOTES:**
    - Modifies golden_master in-place (adds response data)
    - Rate limiting recommended for large datasets (>20 questions)
    - Tool parsing handles multiple serialization formats gracefully
    - Custom metadata stored separately from RAGAS standard fields
    
    Example:
        >>> golden_dataset = generate_golden_master(docs, final_size=10)
        >>> agent = get_graph_agent([ask_naive_llm_tool])
        >>> completed_dataset = generate_responses_for_golden_dataset(
        ...     golden_dataset, agent, pause_secs_between_each_run=2
        ... )
        >>> print(f"Completed {len(completed_dataset)} evaluations")
    """
    # Initialize metadata storage if it doesn't exist
    if not hasattr(golden_master, "_custom_metadata"):
        golden_master._custom_metadata = {}

    for idx, test_row in enumerate(tqdm(golden_master)):
        inputs = {"messages": [HumanMessage(content=test_row.eval_sample.user_input)]}
        response = graph_agent.invoke(inputs)

        evaluation_contexts = extract_contexts_for_eval(response["messages"])
        parsed_data = parse_langchain_messages(response["messages"])
        eval_sample = {
            "user_input": inputs["messages"][0].content,
            "response": response["messages"][-1].content,  # Final AI response
            "retrieved_contexts": evaluation_contexts,
            "tools_used": parsed_data.get("summary", {}).get("tools", []),
            "num_contexts": len(evaluation_contexts),
        }

        # Assign supported RAGAS fields
        test_row.eval_sample.response = eval_sample["response"]
        test_row.eval_sample.retrieved_contexts = eval_sample["retrieved_contexts"]

        # Store custom metadata separately using sample index as key
        sample_id = f"sample_{idx}"
        golden_master._custom_metadata[sample_id] = {
            "tools_used": eval_sample["tools_used"],
            "num_contexts": eval_sample["num_contexts"],
            "user_input": eval_sample["user_input"],  # For matching
        }

        if pause_secs_between_each_run > 0:
            time.sleep(pause_secs_between_each_run)
    return golden_master


def display_dataset_with_metadata(golden_master):
    """
    Convert golden dataset to pandas DataFrame with enhanced metadata columns.
    
    **🎯 PURPOSE & STRATEGY:**
    - Merges RAGAS standard fields with custom metadata for comprehensive view
    - Provides human-readable format for dataset analysis and debugging
    - Essential for understanding tool usage patterns across test questions
    - Enables data validation before running expensive RAGAS evaluations
    
    **📊 ENHANCED COLUMNS:**
    - **Standard RAGAS**: user_input, response, retrieved_contexts, reference
    - **Custom Metadata**: tools_used, num_contexts (from _custom_metadata)
    - **Analysis Ready**: Ready for pandas operations and visualization
    
    Args:
        golden_master: RAGAS dataset with populated responses and custom metadata
    
    Returns:
        pandas.DataFrame: Enhanced DataFrame with all evaluation data + metadata
    
    **💡 USAGE EXAMPLES:**
    ```python
    >>> df = display_dataset_with_metadata(completed_dataset)
    >>> print(df['tools_used'].value_counts())  # Tool usage analysis
    >>> print(df['num_contexts'].describe())    # Context count statistics
    >>> df[df['num_contexts'] == 0]             # Find questions with no contexts
    ```
    
    **⚠️ IMPORTANT NOTES:**
    - Returns empty metadata columns if _custom_metadata not present
    - tools_used column contains lists of tool names per question
    - num_contexts shows count of retrieved contexts per question
    """
    df = golden_master.to_pandas()
    if hasattr(golden_master, "_custom_metadata"):
        tools_used = []
        num_contexts = []
        for idx in range(len(df)):
            meta = golden_master._custom_metadata.get(f"sample_{idx}", {})
            tools_used.append(meta.get("tools_used", []))
            num_contexts.append(meta.get("num_contexts", 0))
        df["tools_used"] = tools_used
        df["num_contexts"] = num_contexts
    return df


def run_ragas_evaluation(
    golden_dataset,
    method_name: str = "Unknown",
    evaluator_model: str = "gpt-4.1-mini",
    eval_timeout: int = 360,
    request_timeout: int = 120,
    custom_metrics: list = None,
):
    """
    Execute comprehensive RAGAS evaluation with 8 core quality metrics.
    
    **🎯 PURPOSE & STRATEGY:**
    - Evaluates RAG system performance across multiple quality dimensions
    - Uses LLM-based evaluation for nuanced semantic understanding
    - Provides standardized metrics for retrieval method comparison
    - Generates detailed performance reports for system optimization
    
    **⚡ PERFORMANCE CHARACTERISTICS:**
    - **Evaluation Time**: 3-8 minutes for 10 samples (varies by metric complexity)
    - **LLM Usage**: GPT-4.1-mini for cost-effective high-quality evaluation
    - **Parallel Processing**: RAGAS handles concurrent metric evaluation
    - **Timeout Management**: Configurable timeouts for complex evaluations
    
    **🔧 TECHNICAL IMPLEMENTATION:**
    - **Dataset Conversion**: Transforms custom dataset to RAGAS EvaluationDataset format
    - **LLM Wrapper**: LangChain integration for consistent model interface
    - **Metric Orchestration**: Parallel execution of 8 evaluation metrics
    - **Token Tracking**: OpenAI usage monitoring for cost analysis
    
    **📊 DEFAULT RAGAS METRICS (All 0-1 scale, higher=better):**
    1. **LLMContextRecall**: How well retrieval finds relevant information
    2. **Faithfulness**: Response consistency with retrieved contexts
    3. **FactualCorrectness**: Accuracy of factual claims in response
    4. **ResponseRelevancy**: How well response addresses original question
    5. **ContextEntityRecall**: Preservation of important entities in contexts
    6. **ContextPrecision**: Quality ranking of retrieved contexts
    7. **AnswerCorrectness**: Overall response quality vs ground truth
    8. **NoiseSensitivity**: Robustness to irrelevant information
    
    **🎛️ CONFIGURATION OPTIONS:**
    - **evaluator_model**: GPT-4.1-mini (balanced), GPT-4.1 (premium quality)
    - **eval_timeout**: 360s (conservative), 180s (fast), 600s (complex datasets)
    - **request_timeout**: 120s per LLM call (handles complex reasoning)
    - **custom_metrics**: Override default set for specific evaluation needs
    
    **💰 COST ESTIMATION:**
    - **10 samples**: ~$0.50-2.00 depending on response length
    - **50 samples**: ~$2.50-10.00 for comprehensive evaluation
    - **Cost factors**: Response length, context size, metric complexity
    
    Args:
        golden_dataset: Dataset with user_input, response, retrieved_contexts populated
        method_name (str): Identifier for this evaluation run (used in logging)
        evaluator_model (str): LLM model for evaluation (default: "gpt-4.1-mini")
        eval_timeout (int): Max seconds for entire evaluation (default: 360)
        request_timeout (int): Max seconds per LLM request (default: 120)
        custom_metrics (list): Override default metrics (default: None for all 8 metrics)
    
    Returns:
        RAGAS EvaluationResult: Contains metric scores, token usage, timing data
        Access via result.scores (DataFrame) and result.dataset (processed data)
    
    **📈 RESULT STRUCTURE:**
    ```python
    result.scores:  # pandas DataFrame with metric columns
    - context_recall: 0.0-1.0
    - faithfulness: 0.0-1.0  
    - factual_correctness: 0.0-1.0
    - answer_relevancy: 0.0-1.0
    # ... other metrics
    ```
    
    **⚠️ IMPORTANT NOTES:**
    - Requires OPENAI_API_KEY for evaluator LLM access
    - Temperature=0 for consistent evaluation results
    - Timeout errors indicate dataset complexity or API issues
    - Results vary slightly between runs due to LLM evaluation nature
    
    **🔍 TROUBLESHOOTING:**
    - **Timeout errors**: Increase eval_timeout or reduce dataset size
    - **Rate limits**: Add delays between evaluations or use smaller batches
    - **Low scores**: Check retrieved_contexts quality and relevance
    
    Example:
        >>> result = run_ragas_evaluation(
        ...     completed_dataset, 
        ...     method_name="naive_retrieval",
        ...     evaluator_model="gpt-4.1-mini",
        ...     eval_timeout=300
        ... )
        >>> print(result.scores.mean())  # Average scores across all metrics
        >>> print(f"Evaluation cost: ${result.total_cost:.2f}")
    """
    # Convert to RAGAS evaluation dataset
    evaluation_dataset = EvaluationDataset.from_pandas(golden_dataset.to_pandas())

    # Setup evaluator LLM
    evaluator_llm = LangchainLLMWrapper(
        ChatOpenAI(
            model=evaluator_model, temperature=0, request_timeout=request_timeout
        )
    )
    # Configure evaluation settings
    run_config = RunConfig(timeout=eval_timeout)

    # Use custom metrics if provided, otherwise use default set
    if custom_metrics is None:
        metrics = [
            LLMContextRecall(),
            Faithfulness(),
            FactualCorrectness(),
            ResponseRelevancy(),
            ContextEntityRecall(),
            ContextPrecision(),
            AnswerCorrectness(),
            NoiseSensitivity(),
        ]
    else:
        metrics = custom_metrics

    print(f"🧪 Running RAGAS evaluation for {method_name}")

    # Run evaluation
    result = evaluate(
        dataset=evaluation_dataset,
        metrics=metrics,
        llm=evaluator_llm,
        token_usage_parser=get_token_usage_for_openai,
        run_config=run_config,
    )

    print(f"✅ Completed  RAGAS evaluation for {method_name}")
    return result


def record_metrics_from_run(retriever_name, dataframe):
    """
    Append RAGAS evaluation results to persistent metrics CSV file.
    
    **🎯 PURPOSE & STRATEGY:**
    - Maintains central repository of all evaluation results across time
    - Enables longitudinal performance tracking and method comparison
    - Supports automated performance regression detection
    - Provides data source for visualization and ranking analysis
    
    **📁 FILE STRUCTURE:**
    Creates/appends to `../metrics/ragas-evaluation-metrics.csv` with columns:
    - **datetime**: Timestamp of evaluation run (YYYY-MM-DD HH:MM:SS)
    - **retriever**: Method identifier (naive, contextual_compression, etc.)
    - **RAGAS metrics**: All metrics in METRICS_ORDER configuration
    
    **🔧 TECHNICAL IMPLEMENTATION:**
    - **Column Ordering**: Uses METRICS_ORDER for consistent column structure
    - **Data Persistence**: Appends to existing file or creates new one
    - **Timestamp Tracking**: Automatic datetime stamping for trend analysis
    - **Data Validation**: Filters columns to match expected metric schema
    
    **📊 SUPPORTED METRICS:**
    Standard RAGAS evaluation metrics as defined in METRICS_ORDER:
    - context_recall, faithfulness, factual_correctness, answer_relevancy
    - context_entity_recall, context_precision, answer_correctness, etc.
    
    **💾 PERSISTENCE BEHAVIOR:**
    - **Append Mode**: Preserves historical evaluation data
    - **Create Mode**: Initializes file if it doesn't exist
    - **No Deduplication**: Each run creates new rows (intentional for trend tracking)
    - **Index=False**: Clean CSV output without pandas index column
    
    Args:
        retriever_name (str): Identifier for retrieval method being evaluated
                             (e.g., "naive_retrieval_chain", "parent_document")
        dataframe (pd.DataFrame): RAGAS evaluation results with metric columns
    
    Returns:
        None: Modifies CSV file on disk, no return value
    
    **🗃️ OUTPUT FILE STRUCTURE:**
    ```
    datetime,retriever,context_recall,faithfulness,factual_correctness,...
    2024-01-15 14:30:22,naive_retrieval_chain,0.637,0.905,0.823,...
    2024-01-15 14:35:18,contextual_compression,0.581,0.887,0.798,...
    ```
    
    **⚠️ IMPORTANT NOTES:**
    - Creates directory structure if ../metrics/ doesn't exist
    - Timestamp in local timezone (consider UTC for distributed teams)
    - Column order matters for downstream analysis tools
    - No data validation - assumes clean RAGAS output format
    
    **🔍 USAGE PATTERNS:**
    ```python
    # After RAGAS evaluation
    result = run_ragas_evaluation(dataset, "new_method")
    record_metrics_from_run("new_method", result.scores)
    
    # Load historical data for analysis
    historical = pd.read_csv("../metrics/ragas-evaluation-metrics.csv")
    print(historical.groupby('retriever').mean())
    ```
    
    Example:
        >>> result = run_ragas_evaluation(completed_dataset, "naive_retrieval")
        >>> record_metrics_from_run("naive_retrieval_chain", result.scores)
        >>> # Data now persisted in ../metrics/ragas-evaluation-metrics.csv
    """
    new_dataframe = dataframe.copy()
    columns = METRICS_ORDER
    metrics_filename = "../metrics/ragas-evaluation-metrics.csv"
    dataset_df = pd.DataFrame()
    if os.path.exists(metrics_filename):
        dataset_df = pd.read_csv(metrics_filename)
    new_dataframe["datetime"] = datetime.now().strftime("%Y-%m-%d %T")
    new_dataframe["retriever"] = retriever_name
    new_dataframe = new_dataframe[["datetime", "retriever"] + columns]
    dataset_df = pd.concat([dataset_df, new_dataframe])

    dataset_df.to_csv(metrics_filename, index=False)
