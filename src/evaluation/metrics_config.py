#!/usr/bin/env python3
"""
RAGAS Metrics Configuration Management
=====================================

**🎯 PURPOSE & STRATEGY:**
- Centralized configuration for RAGAS evaluation metrics order and weights
- Ensures consistency across all evaluation, ranking, and visualization modules
- Provides single source of truth for metric definitions and importance
- Enables easy adjustment of evaluation priorities without code changes

**🔧 TECHNICAL IMPLEMENTATION:**
- **METRICS_ORDER**: Defines consistent column ordering across all outputs
- **METRICS_WEIGHTS**: Specifies relative importance for overall scoring
- **Validation**: Built-in checks ensure configuration integrity
- **Immutable Design**: Changes require deliberate updates across dependent modules

**📊 METRIC CATEGORIES & WEIGHTINGS:**

**Primary RAG Metrics (85% total weight):**
- **context_recall (25%)**: How well retrieval finds relevant information
- **faithfulness (25%)**: Response consistency with retrieved contexts
- **factual_correctness (18%)**: Accuracy of factual claims in responses
- **answer_relevancy (17%)**: How well response addresses original question

**Secondary Quality Metrics (10% total weight):**
- **context_entity_recall (5%)**: Preservation of important entities in contexts
- **noise_sensitivity_relevant (5%)**: Robustness to irrelevant information

**Supporting Precision Metrics (5% total weight):**
- **context_precision (2%)**: Quality ranking of retrieved contexts
- **answer_correctness (3%)**: Overall response quality vs ground truth

**💡 WEIGHTING RATIONALE:**
- **Primary Focus**: Core RAG functionality (retrieval + generation quality)
- **Balanced Importance**: context_recall and faithfulness equally critical
- **Practical Emphasis**: factual_correctness and answer_relevancy for user value
- **Quality Assurance**: Secondary metrics ensure robustness and completeness
- **Precision Support**: Supporting metrics provide additional quality validation

**⚠️ CONFIGURATION CONSTRAINTS:**
- **Immutable Order**: METRICS_ORDER changes require updates across all modules
- **Weight Sum**: METRICS_WEIGHTS must sum to exactly 1.0 (validated)
- **Key Consistency**: Same metrics in both METRICS_ORDER and METRICS_WEIGHTS
- **Dependency Impact**: Changes affect visualization, ranking, and export modules

**🔍 USAGE ACROSS MODULES:**
- **evaluation_helpers.py**: RAGAS evaluation metric selection
- **ragas_rank_retrievers.py**: Performance ranking and comparison
- **heatmap_visualization.py**: Visual metric ordering and importance display
- **visualize_retriever_performance.py**: Metric display configuration

**🛠️ MODIFICATION GUIDELINES:**
When updating weights or order:
1. Update METRICS_WEIGHTS to sum to 1.0
2. Ensure METRICS_ORDER matches METRICS_WEIGHTS keys
3. Test all visualization modules
4. Update documentation in dependent modules
5. Regenerate any cached evaluation results

Example Usage:
    >>> from src.evaluation.metrics_config import METRICS_ORDER, METRICS_WEIGHTS
    >>> print(f"Primary metric: {METRICS_ORDER[0]} (weight: {METRICS_WEIGHTS[METRICS_ORDER[0]]})")
    >>> total_weight = sum(METRICS_WEIGHTS.values())
    >>> print(f"Total weight validation: {total_weight}")  # Should be 1.0
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
assert len(METRICS_ORDER) == len(
    METRICS_WEIGHTS
), "Metrics order and weights must have same length"
assert set(METRICS_ORDER) == set(
    METRICS_WEIGHTS.keys()
), "Metrics order and weights must have same keys"
assert (
    abs(sum(METRICS_WEIGHTS.values()) - 1.0) < 0.001
), f"Weights must sum to 1.0, got {sum(METRICS_WEIGHTS.values())}"
