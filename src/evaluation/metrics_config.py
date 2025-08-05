#!/usr/bin/env python3
"""
RAGAS Metrics Configuration
===========================
Centralized configuration for RAGAS evaluation metrics order and weights.
This ensures consistency across all evaluation and visualization modules.
"""

# Official metric order - DO NOT CHANGE without updating all dependent code
METRICS_ORDER = [
    "context_recall",
    "faithfulness", 
    "factual_correctness",
    "answer_relevancy",
    "context_entity_recall",
    "context_precision",
    "answer_correctness",
    "noise_sensitivity_relevant",
]

# Weights for overall score calculation
# Weight Distribution Summary:
#   - Primary RAG metrics: 0.85 (85% of total)
#     * context_recall, faithfulness, factual_correctness, answer_relevancy
#   - Secondary metrics: 0.10 (10% of total) 
#     * context_entity_recall, noise_sensitivity_relevant
#   - Supporting precision metrics: 0.05 (5% of total)
#     * context_precision, answer_correctness
#   Total: 1.00 ✅
METRICS_WEIGHTS = {
    # Primary metrics (85%)
    "context_recall": 0.25,
    "faithfulness": 0.25,
    "factual_correctness": 0.18,
    "answer_relevancy": 0.17,
    # Secondary metrics (10%)
    "context_entity_recall": 0.05,         
    "noise_sensitivity_relevant": 0.05,    
    # Supporting precision metrics (5%)
    "context_precision": 0.02,             
    "answer_correctness": 0.03,            
}

# Validation
assert len(METRICS_ORDER) == len(METRICS_WEIGHTS), "Metrics order and weights must have same length"
assert set(METRICS_ORDER) == set(METRICS_WEIGHTS.keys()), "Metrics order and weights must have same keys"
assert abs(sum(METRICS_WEIGHTS.values()) - 1.0) < 0.001, f"Weights must sum to 1.0, got {sum(METRICS_WEIGHTS.values())}"