#!/usr/bin/env python3
"""
Simple Heatmap Visualization
============================
Bare minimum code to load CSV and display heatmap.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from src.evaluation.metrics_config import METRICS_ORDER

# Load CSV file
df = pd.read_csv("../metrics/ragas-evaluation-metrics.csv")

# Get metrics columns
metrics = METRICS_ORDER

# Calculate averages by retriever
data = df.groupby("retriever")[metrics].mean()

# Sort by overall performance (simple average)
data["overall"] = data.mean(axis=1)
data = data.sort_values("overall", ascending=False)
data = data.drop("overall", axis=1)

# Create heatmap
plt.figure(figsize=(10, 6))
plt.imshow(data.values, cmap="RdYlGn", aspect="auto")

# Labels
plt.yticks(range(len(data)), [name.replace("_", " ").title() for name in data.index])
plt.xticks(
    range(len(metrics)), [m.replace("_", " ").title() for m in metrics], rotation=45
)

# Add values to cells
for i in range(len(data)):
    for j in range(len(metrics)):
        plt.text(
            j,
            i,
            f"{data.values[i,j]:.3f}",
            ha="center",
            va="center",
            color="black" if data.values[i, j] < 0.5 else "black",
        )

# Highlight best retriever (first row)
plt.gca().add_patch(
    plt.Rectangle(
        (-0.5, -0.5), len(metrics), 1, fill=False, edgecolor="purple", linewidth=3
    )
)

plt.title("RAGAS Metrics Heatmap\n(Green=Better, Purple Border=Best Retriever)")
plt.colorbar(label="Performance Score")

# Add metric explanations at the bottom
explanations = """
📊 WHAT EACH METRIC MEANS:

• Context Recall (context_recall): How well the system finds relevant information from the documents
  • (measures how well the retrieval system finds relevant context)
• Faithfulness (faithfulness): How well the answer sticks to what's actually in the retrieved documents
  • (measures factual consistency between context and response)
• Factual Correctness (factual_correctness): How accurate and correct the facts in the answer are
  • (measures factual accuracy of the response)
• Answer Relevancy (answer_relevancy): How well the answer actually addresses the original question
  • (measures how relevant the answer is to the question)
• Context Entity Recall (context_entity_recall): How well important names/places/things are preserved
  • (measures how well entities are retrieved in context)
• Context Precision (context_precision): How well the most relevant chunks are ranked higher in retrieval results
  • (measures the signal-to-noise ratio of retrieved context and ranking quality)
• Answer Correctness (answer_correctness): How well the answer matches the ground truth in both factual content and semantic meaning
  • (combines factual similarity and semantic similarity using weighted evaluation)
• Noise Sensitivity (noise_sensitivity_relevant): How well the system ignores irrelevant or confusing information
  • (measures robustness to irrelevant information)

Green/Higher scores = Better performance | Red/Lower scores = Needs improvement
"""

plt.figtext(
    0.125,
    -0.45,
    explanations,
    fontsize=9,
    verticalalignment="bottom",
    bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8),
)

plt.tight_layout()
plt.show()
