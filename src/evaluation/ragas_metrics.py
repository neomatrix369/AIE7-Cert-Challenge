def extract_ragas_metrics(ragas_result, model_name: str = ""):
    """Extract cost, latency, and token metrics from RAGAS evaluation result"""
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
