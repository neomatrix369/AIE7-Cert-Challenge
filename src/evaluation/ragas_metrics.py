def extract_ragas_metrics(ragas_result, model_name: str = ""):
    """
    Extract comprehensive metrics from RAGAS evaluation results with cost analysis.
    
    **🎯 PURPOSE & STRATEGY:**
    - Processes RAGAS evaluation results into standardized metrics format
    - Calculates token usage and cost analysis across all evaluation runs
    - Provides comprehensive performance and efficiency metrics
    - Essential for cost-performance trade-off analysis across retrieval methods
    
    **⚡ PERFORMANCE METRICS EXTRACTED:**
    - **RAGAS Scores**: All 8 standard RAGAS quality metrics (0-1 scale)
    - **Token Usage**: Input/output tokens with per-run averages
    - **Cost Analysis**: Total and per-run costs using current API pricing
    - **Execution Stats**: Run counts and aggregated statistics
    
    **🔧 TECHNICAL IMPLEMENTATION:**
    - **Safe Extraction**: Handles various RAGAS result formats gracefully
    - **NaN Filtering**: Removes invalid scores for accurate averages
    - **Cost Modeling**: Uses current API pricing for 15+ model types
    - **Fallback Defaults**: Graceful handling of missing data fields
    
    **💰 SUPPORTED MODELS & PRICING:**
    - **GPT Family**: gpt-4.1, gpt-4.1-nano, gpt-4.1-mini, gpt-4o-mini, gpt-4o
    - **Claude Family**: claude-3-haiku, claude-3-sonnet, claude-3-opus
    - **Embeddings**: text-embedding-3-small, text-embedding-3-large
    - **Reranking**: rerank-v3.5 (Cohere)
    - **Fallback**: gpt-4o-mini pricing for unknown models
    
    **📊 OUTPUT METRICS:**
    ```python
    {
        # Execution Statistics
        "Total_Runs": 10,
        "Total_Cost": 0.0234,
        "Avg_Cost_Per_Run": 0.00234,
        
        # Token Usage
        "Total_Input_Tokens": 15420,
        "Total_Output_Tokens": 3280,
        "Avg_Input_Tokens_Per_Run": 1542.0,
        "Avg_Output_Tokens_Per_Run": 328.0,
        
        # RAGAS Quality Metrics (all 0-1 scale, higher=better)
        "context_recall": 0.637,
        "faithfulness": 0.905,
        "factual_correctness": 0.823,
        # ... all other RAGAS metrics
    }
    ```
    
    Args:
        ragas_result: RAGAS evaluation result object with scores, costs, and usage data
        model_name (str): Model identifier for cost calculation (e.g., "gpt-4.1-mini")
    
    Returns:
        dict: Comprehensive metrics including RAGAS scores, tokens, costs, and statistics
    
    **💡 COST CALCULATION METHODOLOGY:**
    - Uses per-million-token pricing from official API documentation
    - Separates input and output token costs (different rates)
    - Provides both total and per-run cost breakdowns
    - Essential for ROI analysis and method selection
    
    **⚠️ IMPORTANT NOTES:**
    - Costs based on current API pricing (update get_model_costs for changes)
    - Token usage extracted from RAGAS cost callback when available
    - NaN scores filtered out to prevent skewed averages
    - Latency metrics set to 0 (not available in standard RAGAS results)
    
    Example:
        >>> result = run_ragas_evaluation(dataset, "naive_retrieval", "gpt-4.1-mini")
        >>> metrics = extract_ragas_metrics(result, "gpt-4.1-mini")
        >>> print(f"Cost per run: ${metrics['Avg_Cost_Per_Run']:.4f}")
        >>> print(f"Context recall: {metrics['context_recall']:.3f}")
    """
    import numpy as np

    def get_value(obj, key):
        """Get value from dict key or object attribute"""
        return obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)

    def safe_mean(values):
        """Calculate mean, filtering out NaN values"""
        if not values:
            return 0
        arr = np.array(values, dtype=float)
        valid = arr[~np.isnan(arr)]
        return float(np.mean(valid)) if len(valid) > 0 else 0

    def get_model_costs(model_name):
        PER_MILLION = 1_000_000
        """Get per-token costs for common models"""
        costs = {
            "gpt-4.1": (2.50 / PER_MILLION, 10.00 / PER_MILLION),
            "gpt-4.1-nano": (0.15 / PER_MILLION, 0.60 / PER_MILLION),
            "gpt-4.1-mini": (0.15 / PER_MILLION, 0.60 / PER_MILLION),
            "gpt-4o-mini": (0.000000150, 0.000000600),
            "gpt-4o": (0.000002500, 0.000010000),
            "gpt-4-turbo": (0.000010000, 0.000030000),
            "gpt-3.5-turbo": (0.000000500, 0.000001500),
            "claude-3-haiku": (0.000000250, 0.000001250),
            "claude-3-sonnet": (0.000003000, 0.000015000),
            "claude-3-opus": (0.000015000, 0.000075000),
            "text-embedding-3-small": (0.02 / PER_MILLION, 0.0),
            "text-embedding-3-large": (0.13 / PER_MILLION, 0.0),
            "rerank-v3.5": (2.00 / PER_MILLION, 0.0),
        }

        # Try exact match, then partial match
        if model_name in costs:
            return costs[model_name]

        for model_key in costs:
            if model_key in model_name.lower():
                return costs[model_key]

        return costs["gpt-4o-mini"]  # Default

    # Extract data
    scores = get_value(ragas_result, "scores") or []
    scores_dict = get_value(ragas_result, "_scores_dict") or {}
    cost_cb = get_value(ragas_result, "cost_cb") or {}
    usage_data = get_value(cost_cb, "usage_data") or []

    # Calculate runs
    total_runs = len(scores) if scores else 1

    # Calculate RAGAS scores (averages from score lists)
    ragas_scores = {}
    for metric, values in scores_dict.items():
        if isinstance(values, list):
            ragas_scores[metric] = safe_mean(values)

    # Calculate tokens and cost
    total_input = sum(get_value(usage, "input_tokens") or 0 for usage in usage_data)
    total_output = sum(get_value(usage, "output_tokens") or 0 for usage in usage_data)

    input_cost, output_cost = get_model_costs(model_name)
    total_cost = (total_input * input_cost) + (total_output * output_cost)

    # Build metrics
    metrics = {
        "Total_Runs": total_runs,
        "Total_Cost": total_cost,
        "Total_Input_Tokens": total_input,
        "Total_Output_Tokens": total_output,
        "Total_Latency_Sec": 0,  # Not available in this data
        "Avg_Cost_Per_Run": total_cost / total_runs,
        "Avg_Input_Tokens_Per_Run": total_input / total_runs,
        "Avg_Output_Tokens_Per_Run": total_output / total_runs,
        "Avg_Latency_Sec": 0,
        **ragas_scores,
    }

    return metrics
