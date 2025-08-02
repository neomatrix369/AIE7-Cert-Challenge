#!/usr/bin/env python3
"""
RAGAS Metrics Heatmap Visualization
===================================
Creates a heatmap showing all RAGAS metrics with proper direction and magnitude consideration.
Highlights the best performing retriever.
"""

import csv
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from collections import defaultdict


def load_and_process_data(csv_path="../metrics/ragas-evaluation-metrics.csv"):
    """Load CSV data and calculate averages."""
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
                "noise_sensitivity_relevant": float(row["noise_sensitivity_relevant"]),
            }
        )

    # Calculate averages and overall scores
    processed = {}
    metrics = [
        "context_recall",
        "faithfulness",
        "factual_correctness",
        "answer_relevancy",
        "context_entity_recall",
        "noise_sensitivity_relevant",
    ]

    # Weights for overall score calculation
    weights = {
        "context_recall": 0.25,
        "faithfulness": 0.25,
        "factual_correctness": 0.20,
        "answer_relevancy": 0.20,
        "context_entity_recall": 0.05,
        "noise_sensitivity_relevant": 0.05,
    }

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
    """Create comprehensive RAGAS metrics heatmap with best retriever highlighted."""

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

    # Add crown for best retriever
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

    # Performance summary box
    best_score = sorted_data[0][1]["overall_score"]
    worst_score = sorted_data[-1][1]["overall_score"]
    performance_gap = best_score - worst_score

    summary_text = f"""
📊 PERFORMANCE SUMMARY:
🏆 Best: {best_retriever_name} ({best_score:.3f})
📉 Worst: {retrievers[-1]} ({worst_score:.3f})
📈 Gap: {performance_gap:.3f}

🔍 METRIC GUIDE (All Higher = Better):
• Context Recall: Retrieval completeness
• Faithfulness: Answer-context alignment  
• Factual Correctness: Accuracy of facts
• Answer Relevancy: Question-answer match
• Context Entity Recall: Entity preservation
• Noise Sensitivity: Robustness to noise
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
    """Create chart showing metric importance and best performers."""

    # Calculate who performs best in each metric
    metric_champions = {}
    for metric in metrics:
        best_performer = max(sorted_data, key=lambda x: x[1][metric])
        metric_champions[metric] = (best_performer[0], best_performer[1][metric])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Chart 1: Metric importance (weights)
    weights = [0.25, 0.25, 0.20, 0.20, 0.05, 0.05]  # Corresponding to metrics order
    metric_names = [m.replace("_", " ").title() for m in metrics]

    colors = plt.cm.Set3(np.arange(len(metrics)))
    bars = ax1.bar(metric_names, weights, color=colors)
    ax1.set_title(
        "📊 Metric Importance Weights\n(Used in Overall Scoring)", fontweight="bold"
    )
    ax1.set_ylabel("Weight in Overall Score")
    ax1.set_ylim(0, 0.3)

    # Add percentage labels
    for bar, weight in zip(bars, weights):
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
    """Main function to create heatmap visualizations."""
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
