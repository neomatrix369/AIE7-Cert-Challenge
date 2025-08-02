#!/usr/bin/env python3
"""
Simple Heatmap Visualization
============================
Bare minimum code to load CSV and display heatmap.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

    # Load CSV file
df = pd.read_csv("ragas-evaluation-metrics.csv")

# Get metrics columns
metrics = ['context_recall', 'faithfulness', 'factual_correctness', 
            'answer_relevancy', 'context_entity_recall', 'noise_sensitivity_relevant']

# Calculate averages by retriever
data = df.groupby('retriever')[metrics].mean()

# Sort by overall performance (simple average)
data['overall'] = data.mean(axis=1)
data = data.sort_values('overall', ascending=False)
data = data.drop('overall', axis=1)

# Create heatmap
plt.figure(figsize=(10, 6))
plt.imshow(data.values, cmap='RdYlGn', aspect='auto')

# Labels
plt.yticks(range(len(data)), [name.replace('_', ' ').title() for name in data.index])
plt.xticks(range(len(metrics)), [m.replace('_', ' ').title() for m in metrics], rotation=45)

# Add values to cells
for i in range(len(data)):
    for j in range(len(metrics)):
        plt.text(j, i, f'{data.values[i,j]:.3f}', ha='center', va='center', 
                color='black' if data.values[i,j] < 0.5 else 'black')

# Highlight best retriever (first row)
plt.gca().add_patch(plt.Rectangle((-0.5, -0.5), len(metrics), 1, 
                                    fill=False, edgecolor='purple', linewidth=3))

plt.title('RAGAS Metrics Heatmap\n(Green=Better, Purple Border=Best Retriever)')
plt.colorbar(label='Performance Score')

# Add metric explanations at the bottom
explanations = """
📊 WHAT EACH METRIC MEANS:

• Context Recall: How well the system finds relevant information from the documents
• Faithfulness: How well the answer sticks to what's actually in the retrieved documents  
• Factual Correctness: How accurate and correct the facts in the answer are
• Answer Relevancy: How well the answer actually addresses the original question
• Context Entity Recall: How well important names/places/things are preserved
• Noise Sensitivity: How well the system ignores irrelevant or confusing information

Higher scores = Better performance | Lower scores = Needs improvement
"""

plt.figtext(0.25, -0.3, explanations, fontsize=9, verticalalignment='bottom', 
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

plt.tight_layout()
plt.show()