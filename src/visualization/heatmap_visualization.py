#!/usr/bin/env python3
"""
Advanced RAGAS Metrics Heatmap Visualization Suite
==================================================

**🎯 PURPOSE & STRATEGY:**
- Comprehensive visual analysis of RAGAS evaluation results across retrieval methods
- Publication-ready heatmaps with professional styling and clear interpretation
- Multi-chart analysis including performance comparison and metric importance
- Automated export capabilities for documentation and presentations

**📊 VISUALIZATION CAPABILITIES:**
- **Main Heatmap**: Color-coded performance matrix with champion highlighting
- **Metric Analysis**: Importance weights and per-metric champion identification
- **Statistical Summary**: Performance gaps, ranking, and key insights
- **Export Ready**: High-resolution PNG output for professional use

**🔧 TECHNICAL FEATURES:**
- **Weighted Scoring**: Uses METRICS_WEIGHTS configuration for overall ranking
- **Multi-Run Aggregation**: Averages multiple evaluation runs per retriever
- **Adaptive Styling**: Color-coded text, borders, and visual hierarchy
- **Error Handling**: Graceful failure with helpful error messages

**🎨 DESIGN PRINCIPLES:**
- **Intuitive Colors**: Green=good, Red=poor for immediate understanding
- **Clear Hierarchy**: Gold borders and crowns for top performers
- **Information Dense**: Maximum insight per visual element
- **Professional Output**: Publication-ready quality and formatting

**💡 KEY INSIGHTS PROVIDED:**
- Overall ranking of retrieval methodss
- Individual metric performance patterns
- Specialized vs generalist retriever identification
- Performance gaps and competitive analysis
- Metric importance and evaluation methodology transparency

Usage:
    python heatmap_visualization.py
    
Outputs:
    - Interactive matplotlib windows
    - ragas_metrics_heatmap.png (300 DPI)
    - metric_importance_analysis.png (300 DPI)
    - Console analysis summary

**⚠️ REQUIREMENTS:**
- matplotlib, numpy for visualization
- RAGAS evaluation results CSV file
- METRICS_ORDER and METRICS_WEIGHTS configuration
"""

import csv
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from collections import defaultdict
from src.evaluation.metrics_config import METRICS_ORDER, METRICS_WEIGHTS


def load_and_process_data(csv_path="../metrics/ragas-evaluation-metrics.csv"):
    """
    Load RAGAS evaluation results and calculate performance averages.
    
    **🎯 PURPOSE & STRATEGY:**
    - Processes historical evaluation results across multiple runs per retriever
    - Calculates weighted overall scores for comprehensive ranking
    - Groups multiple evaluation runs for statistical reliability
    - Prepares data for heatmap visualization and analysis
    
    **🔧 TECHNICAL PROCESSING:**
    - **CSV Loading**: Reads evaluation results with multiple runs per method
    - **Grouping**: Aggregates by retriever name for average calculation
    - **Weighted Scoring**: Uses METRICS_WEIGHTS for overall performance
    - **Ranking**: Sorts by overall weighted score (best to worst)
    
    Args:
        csv_path (str): Path to RAGAS evaluation metrics CSV file
    
    Returns:
        tuple: (sorted_data, metrics) where sorted_data is list of (retriever, scores)
    
    **📊 METRICS PROCESSED:**
    All RAGAS metrics from METRICS_ORDER configuration, typically including:
    - context_recall, faithfulness, factual_correctness, answer_relevancy
    - context_entity_recall, context_precision, answer_correctness
    - noise_sensitivity_relevant
    """
    data = []
    with open(csv_path, "r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append(row)

    # Group by retriever and calculate averages
    retriever_data = defaultdict(list)
    for row in data:
        retriever = row["retriever"]
        retriever_data[retriever].append(
            {
                "context_recall": float(row["context_recall"]),
                "faithfulness": float(row["faithfulness"]),
                "factual_correctness": float(row["factual_correctness"]),
                "answer_relevancy": float(row["answer_relevancy"]),
                "context_entity_recall": float(row["context_entity_recall"]),
                "context_precision": float(row["context_precision"]),
                "answer_correctness": float(row["answer_correctness"]),
                "noise_sensitivity_relevant": float(row["noise_sensitivity_relevant"]),
            }
        )

    # Calculate averages and overall scores
    processed = {}
    metrics = METRICS_ORDER
    weights = METRICS_WEIGHTS

    for retriever, runs in retriever_data.items():
        avg_metrics = {}
        for metric in metrics:
            values = [run[metric] for run in runs]
            avg_metrics[metric] = sum(values) / len(values)

        # Calculate overall score
        overall_score = sum(avg_metrics[metric] * weights[metric] for metric in metrics)
        avg_metrics["overall_score"] = overall_score
        avg_metrics["run_count"] = len(runs)

        processed[retriever] = avg_metrics

    # Sort by overall score (best to worst)
    sorted_retrievers = sorted(
        processed.items(), key=lambda x: x[1]["overall_score"], reverse=True
    )

    return sorted_retrievers, metrics


def create_ragas_heatmap(sorted_data, metrics):
    """
    Create publication-ready RAGAS metrics heatmap with champion highlighting.
    
    **🎯 PURPOSE & STRATEGY:**
    - Creates visually compelling comparison of all retrieval methods
    - Highlights best performer with gold border and crown emoji
    - Uses color coding for immediate performance pattern recognition
    - Includes comprehensive performance summary and metric guide
    
    **🎨 VISUALIZATION FEATURES:**
    - **Color Mapping**: RdYlGn (Red-Yellow-Green) for intuitive good/bad indication
    - **Best Highlight**: Gold border + crown emoji for top performer
    - **Ranking**: Numbered ranks (#1, #2, etc.) with color coding
    - **Value Overlay**: Precise scores in each cell with adaptive text color
    - **Grid Lines**: White grid for better cell separation
    
    **📈 LAYOUT ELEMENTS:**
    - **Main Heatmap**: 14x8 figure for readability
    - **Performance Summary**: Blue box with key statistics
    - **Metric Guide**: Explanations of each RAGAS metric
    - **Colorbar**: Scale explanation (0.0 = Poor, 1.0 = Excellent)
    
    Args:
        sorted_data: List of (retriever_name, metrics_dict) sorted by performance
        metrics: List of metric names in display order
    
    Returns:
        matplotlib.figure.Figure: Configured heatmap ready for display/export
    
    **💡 DESIGN DECISIONS:**
    - All RAGAS metrics are higher=better, so consistent green=good interpretation
    - Gold highlighting for #1 performer creates clear visual hierarchy
    - Adaptive text color (white/black) ensures readability on all backgrounds
    - Performance gap calculation shows competitive landscape
    """

    # Prepare data
    retrievers = [data[0].replace("_", " ").title() for data in sorted_data]
    metric_labels = [metric.replace("_", " ").title() for metric in metrics]

    # Create data matrix
    data_matrix = []
    for retriever, data in sorted_data:
        row = [data[metric] for metric in metrics]
        data_matrix.append(row)

    data_matrix = np.array(data_matrix)

    # Best retriever is first in sorted list
    best_retriever_idx = 0
    best_retriever_name = retrievers[0]

    # Create figure with larger size for readability
    fig, ax = plt.subplots(figsize=(14, 8))

    # Create heatmap
    # All RAGAS metrics: Higher values = Better performance
    im = ax.imshow(data_matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

    # Set ticks and labels
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_yticks(np.arange(len(retrievers)))
    ax.set_xticklabels(metric_labels, rotation=45, ha="right", fontsize=11)
    ax.set_yticklabels(retrievers, fontsize=11)

    # Add value annotations
    for i in range(len(retrievers)):
        for j in range(len(metrics)):
            value = data_matrix[i, j]

            # Text color based on background intensity
            text_color = "white" if value < 0.5 else "black"

            # Bold for best retriever
            weight = "bold" if i == best_retriever_idx else "normal"
            size = 11 if i == best_retriever_idx else 10

            ax.text(
                j,
                i,
                f"{value:.3f}",
                ha="center",
                va="center",
                color=text_color,
                fontweight=weight,
                fontsize=size,
            )

    # Highlight best retriever with gold border
    rect = patches.Rectangle(
        (-0.5, best_retriever_idx - 0.5),
        len(metrics),
        1,
        linewidth=5,
        edgecolor="gold",
        facecolor="none",
        alpha=0.8,
    )
    ax.add_patch(rect)

    # Add crown emoji for best retriever
    ax.text(-0.8, best_retriever_idx, "👑", fontsize=25, va="center", ha="center")

    # Add rank numbers on the right
    for i, (retriever, data) in enumerate(sorted_data):
        rank_color = (
            "gold" if i == 0 else "silver" if i == 1 else "orange" if i == 2 else "gray"
        )
        ax.text(
            len(metrics) + 0.3,
            i,
            f"#{i+1}",
            ha="center",
            va="center",
            fontsize=14,
            fontweight="bold",
            color=rank_color,
        )

    # Title and labels
    ax.set_title(
        f"🎯 RAGAS Metrics Heatmap Analysis\n👑 Best Performer: {best_retriever_name}",
        fontsize=16,
        fontweight="bold",
        pad=25,
    )
    ax.set_xlabel("RAGAS Evaluation Metrics", fontsize=13, fontweight="bold")
    ax.set_ylabel("Retrievers (Ranked Best → Worst)", fontsize=13, fontweight="bold")

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.1)
    cbar.set_label(
        "Performance Score\n(0.0 = Poor, 1.0 = Excellent)",
        rotation=270,
        labelpad=20,
        fontsize=11,
    )

    # Add grid for better readability
    ax.set_xticks(np.arange(len(metrics) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(retrievers) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=2)

    # Performance summary with statistical insights
    best_score = sorted_data[0][1]["overall_score"]
    worst_score = sorted_data[-1][1]["overall_score"]
    performance_gap = best_score - worst_score
    
    # Calculate competitive landscape insight
    competitiveness = "Highly Competitive" if performance_gap < 0.1 else "Clear Leader" if performance_gap > 0.2 else "Moderately Competitive"

    summary_text = f"""
📊 PERFORMANCE SUMMARY:
🏆 Best: {best_retriever_name} ({best_score:.3f})
📉 Worst: {retrievers[-1]} ({worst_score:.3f})
📈 Gap: {performance_gap:.3f} ({competitiveness})
📋 Methods: {len(retrievers)} retrieval approaches analyzed

🔍 METRIC GUIDE (All Higher = Better):
• Context Recall: Retrieval completeness (find relevant docs)
• Faithfulness: Answer-context alignment (no hallucination)
• Factual Correctness: Accuracy of stated facts
• Answer Relevancy: Question-answer relevance match
• Context Entity Recall: Important entity preservation
• Noise Sensitivity: Robustness to irrelevant information

💡 INTERPRETATION:
🟢 Green: Strong performance (0.7-1.0)
🟡 Yellow: Moderate performance (0.4-0.7)
🔴 Red: Needs improvement (0.0-0.4)
"""

    # Add text box
    props = dict(boxstyle="round", facecolor="lightblue", alpha=0.8)
    fig.text(
        0.02,
        0.98,
        summary_text,
        fontsize=9,
        verticalalignment="top",
        bbox=props,
        transform=fig.transFigure,
    )

    plt.tight_layout()
    return fig


def create_metric_importance_chart(sorted_data, metrics):
    """
    Create dual chart showing metric weights and per-metric champions.
    
    **🎯 PURPOSE & STRATEGY:**
    - Shows relative importance of each metric in overall scoring
    - Identifies which retriever excels at each specific metric
    - Provides insight into evaluation methodology and retriever strengths
    - Complements main heatmap with detailed metric-level analysis
    
    **📈 CHART COMPONENTS:**
    
    **Left Chart - Metric Importance:**
    - Bar heights show METRICS_WEIGHTS values
    - Percentage labels indicate contribution to overall score
    - Color-coded bars for visual distinction
    - Reveals evaluation priorities and weightings
    
    **Right Chart - Metric Champions:**
    - Shows best performer for each individual metric
    - Bar heights indicate champion's score in that metric
    - Labels show champion name and exact score
    - Identifies specialized strengths across retrievers
    
    **💡 ANALYTICAL VALUE:**
    - **Weight Analysis**: Understanding evaluation criteria emphasis
    - **Specialist Identification**: Finding retrievers that excel in specific areas
    - **Balanced vs Specialized**: Comparing generalists vs specialists
    - **Improvement Targets**: Identifying metrics where retrievers could improve
    
    Args:
        sorted_data: List of (retriever_name, metrics_dict) sorted by performance
        metrics: List of metric names in display order
    
    Returns:
        matplotlib.figure.Figure: Dual-chart analysis ready for display/export
    
    **🎨 VISUAL DESIGN:**
    - 16x6 figure split into two equal subplots
    - Set3 colormap for distinct, pleasant colors
    - Rotated labels for readability
    - Consistent color mapping between charts
    """

    # Calculate who performs best in each metric
    metric_champions = {}
    for metric in metrics:
        best_performer = max(sorted_data, key=lambda x: x[1][metric])
        metric_champions[metric] = (best_performer[0], best_performer[1][metric])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Chart 1: Metric importance (weights) - dynamically ordered
    weights = METRICS_WEIGHTS  # Import from config
    weight_values = [weights[metric] for metric in metrics]
    metric_names = [m.replace("_", " ").title() for m in metrics]

    colors = plt.cm.Set3(np.arange(len(metrics)))
    bars = ax1.bar(metric_names, weight_values, color=colors)
    ax1.set_title(
        "📊 Metric Importance Weights\n(Used in Overall Scoring)", fontweight="bold"
    )
    ax1.set_ylabel("Weight in Overall Score")
    ax1.set_ylim(0, 0.3)

    # Add percentage labels
    for bar, weight in zip(bars, weight_values):
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.01,
            f"{weight*100:.0f}%",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha="right")

    # Chart 2: Best performer per metric
    champions = [
        metric_champions[metric][0].replace("_", " ").title() for metric in metrics
    ]
    champion_scores = [metric_champions[metric][1] for metric in metrics]

    bars2 = ax2.bar(metric_names, champion_scores, color=colors)
    ax2.set_title("🏆 Metric Champions\n(Best Performer per Metric)", fontweight="bold")
    ax2.set_ylabel("Best Score Achieved")
    ax2.set_ylim(0, 1)

    # Add champion names and scores
    for i, (bar, champion, score) in enumerate(zip(bars2, champions, champion_scores)):
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.02,
            f"{champion}\n{score:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    return fig


def main():
    """
    Execute complete heatmap analysis pipeline with file export.
    
    **🎯 EXECUTION PIPELINE:**
    1. **Data Loading**: Load and process RAGAS evaluation results
    2. **Heatmap Creation**: Generate main performance comparison heatmap
    3. **Analysis Charts**: Create metric importance and champion analysis
    4. **File Export**: Save high-resolution PNG files for documentation
    5. **Interactive Display**: Show plots for immediate analysis
    
    **💾 OUTPUT FILES:**
    - **ragas_metrics_heatmap.png**: Main performance comparison (300 DPI)
    - **metric_importance_analysis.png**: Metric weights and champions (300 DPI)
    
    **📈 CONSOLE OUTPUT:**
    - Performance summary with best performer identification
    - File save confirmations
    - Key insights and interpretation guide
    - Error handling for missing files or dependencies
    
    **⚠️ ERROR HANDLING:**
    - FileNotFoundError: Missing evaluation results CSV
    - ImportError: Missing matplotlib dependency
    - Generic exceptions: Unexpected processing errors
    
    **💡 USAGE:**
    Run from command line: `python heatmap_visualization.py`
    Or import and call: `from heatmap_visualization import main; main()`
    """
    try:
        print("🎨 RAGAS METRICS HEATMAP ANALYZER")
        print("=" * 40)

        # Load data
        sorted_data, metrics = load_and_process_data()

        print(f"📊 Loaded data for {len(sorted_data)} retrievers")
        print(f"🏆 Best Performer: {sorted_data[0][0].replace('_', ' ').title()}")
        print(f"📈 Best Score: {sorted_data[0][1]['overall_score']:.3f}")

        # Create heatmap
        print("\n🎨 Creating RAGAS metrics heatmap...")
        fig1 = create_ragas_heatmap(sorted_data, metrics)
        fig1.savefig(
            "ragas_metrics_heatmap.png", dpi=300, bbox_inches="tight", facecolor="white"
        )
        print("✅ Saved: ragas_metrics_heatmap.png")

        # Create metric importance chart
        print("🎨 Creating metric importance analysis...")
        fig2 = create_metric_importance_chart(sorted_data, metrics)
        fig2.savefig(
            "metric_importance_analysis.png",
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )
        print("✅ Saved: metric_importance_analysis.png")

        # Show plots
        plt.show()

        print("\n🎉 Heatmap analysis complete!")
        print("📋 Key Insights:")
        print(f"   • Green areas = Strong performance")
        print(f"   • Red areas = Weak performance")
        print(f"   • Gold border = Best overall retriever")
        print(f"   • All metrics: Higher values = Better performance")

    except FileNotFoundError:
        print("❌ Error: ragas-evaluation-metrics.csv not found!")
    except ImportError:
        print("❌ Error: matplotlib not available!")
        print("Install with: pip install matplotlib")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
