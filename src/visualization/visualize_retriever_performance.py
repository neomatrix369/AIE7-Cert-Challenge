#!/usr/bin/env python3
"""
RAGAS Retriever Performance Heatmap Visualization
=================================================

**🎯 PURPOSE & STRATEGY:**
- Creates publication-ready heatmap visualization of RAGAS metrics across retrieval methods
- Provides immediate visual comparison of retriever performance strengths/weaknesses
- Supports decision-making for optimal retrieval method selection
- Essential tool for communicating evaluation results to stakeholders

**📊 VISUALIZATION FEATURES:**
- **Color-coded Performance**: Green=better, Red=worse, with numeric overlays
- **Method Ranking**: Retrievers sorted by overall performance average
- **Metric Explanations**: Comprehensive legend explaining each RAGAS metric
- **Best Method Highlighting**: Purple border around top-performing retriever

**🔧 TECHNICAL IMPLEMENTATION:**
- **Data Source**: ragas-evaluation-metrics.csv with historical evaluation results
- **Metrics**: Uses METRICS_ORDER configuration for consistent column ordering
- **Visualization**: matplotlib with RdYlGn colormap for intuitive interpretation
- **Layout**: Optimized for readability with explanatory text below chart

**📈 PERFORMANCE ANALYSIS:**
- **Overall Ranking**: Simple average across all metrics for method ordering
- **Individual Metrics**: Cell values show precise scores (0-1 scale)
- **Visual Patterns**: Easy identification of method strengths and weaknesses
- **Comparative Analysis**: Side-by-side method comparison across all dimensions

**🎨 DESIGN ELEMENTS:**
- **Heatmap Colors**: Intuitive green (good) to red (poor) gradient
- **Text Overlays**: Precise numeric values in each cell for exact scores
- **Method Labels**: Human-readable names (title case, underscore removal)
- **Metric Labels**: Formatted for readability with rotation for space
- **Legend**: Detailed explanations of each RAGAS metric meaning

Usage:
    python visualize_retriever_performance.py
    
Outputs:
    - Interactive matplotlib heatmap window
    - Visual comparison of all retrieval methods
    - Performance insights for method selection

**⚠️ REQUIREMENTS:**
- ragas-evaluation-metrics.csv must exist with evaluation data
- matplotlib, pandas, numpy for visualization
- METRICS_ORDER configuration from metrics_config module
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from src.evaluation.metrics_config import METRICS_ORDER

# Configure matplotlib for better visualization
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 11

# Load evaluation results CSV file
df = pd.read_csv("../metrics/ragas-evaluation-metrics.csv")

# Get metrics columns from configuration
metrics = METRICS_ORDER

# Calculate performance averages by retrieval method
data = df.groupby("retriever")[metrics].mean()

# Rank retrievers by overall performance (simple average)
data["overall"] = data.mean(axis=1)
data = data.sort_values("overall", ascending=False)
data = data.drop("overall", axis=1)

# Create performance heatmap visualization
plt.figure(figsize=(12, 8))
heatmap = plt.imshow(data.values, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

# Configure axis labels with readable formatting
plt.yticks(range(len(data)), [name.replace("_", " ").title() for name in data.index])
plt.xticks(
    range(len(metrics)), [m.replace("_", " ").title() for m in metrics], rotation=45, ha='right'
)

# Add precise performance values to each cell
for i in range(len(data)):
    for j in range(len(metrics)):
        score = data.values[i, j]
        # Use white text for dark cells, black for light cells
        text_color = "white" if score < 0.4 else "black"
        plt.text(
            j, i, f"{score:.3f}",
            ha="center", va="center", 
            color=text_color, fontweight="bold", fontsize=10
        )

# Highlight best performing retriever with purple border
plt.gca().add_patch(
    plt.Rectangle(
        (-0.5, -0.5), len(metrics), 1, 
        fill=False, edgecolor="purple", linewidth=4
    )
)

# Enhanced title and colorbar
plt.title(
    "RAGAS Retrieval Method Performance Comparison\n" +
    "(Green=Better Performance, Purple Border=Overall Winner)",
    fontsize=14, fontweight="bold", pad=20
)
colorbar = plt.colorbar(heatmap, label="Performance Score (0-1 scale)", shrink=0.8)
colorbar.ax.tick_params(labelsize=10)

# Add comprehensive metric explanations below the heatmap
explanations = """
📊 RAGAS METRIC DEFINITIONS (All metrics: 0-1 scale, Higher = Better):

🎯 CORE RETRIEVAL METRICS:
• Context Recall: How well the retrieval system finds relevant information from the knowledge base
  └─ Measures retrieval completeness - did we get the right documents?
• Context Precision: How well the most relevant chunks are ranked higher in retrieval results  
  └─ Measures retrieval ranking quality - are the best results at the top?
• Context Entity Recall: How well important entities (names, places, numbers) are preserved in contexts
  └─ Measures information preservation - did we keep critical details?

🤖 RESPONSE QUALITY METRICS:
• Faithfulness: How well the generated answer stays consistent with the retrieved document content
  └─ Measures hallucination prevention - no making things up!
• Answer Relevancy: How well the response directly addresses the original user question
  └─ Measures response focus - does the answer actually help the user?
• Factual Correctness: How accurate and factually correct the claims in the answer are
  └─ Measures accuracy - are the facts stated correctly?

🛡️ ROBUSTNESS METRICS:
• Answer Correctness: Overall answer quality combining factual accuracy and semantic similarity
  └─ Measures comprehensive response quality against ground truth
• Noise Sensitivity: How well the system handles irrelevant or confusing information
  └─ Measures robustness - can it ignore distracting content?

💡 INTERPRETATION GUIDE:
🟢 Green (0.7-1.0): Excellent performance  🟡 Yellow (0.4-0.7): Good performance  🔴 Red (0.0-0.4): Needs improvement
"""

# Position explanation text below the heatmap
plt.figtext(
    0.1, -0.15, explanations,
    fontsize=9, verticalalignment="top",
    bbox=dict(boxstyle="round,pad=1", facecolor="lightblue", alpha=0.9, edgecolor="navy"),
    wrap=True
)

# Adjust layout to accommodate explanation text
plt.subplots_adjust(bottom=0.6)
plt.tight_layout()
plt.show()
