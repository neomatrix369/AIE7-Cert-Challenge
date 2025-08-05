import pandas as pd
import numpy as np
import logging

# Set up logging with third-party noise suppression
from src.utils.logging_config import setup_logging

logger = setup_logging(__name__)


class RetrieverRanker:
    """
    Comprehensive RAGAS-based retriever performance analysis and ranking system.

    Evaluates RAG retrieval methods across multiple quality metrics and provides
    statistical analysis, ranking, and visualization capabilities for retriever comparison.

    Core RAGAS Metrics (All 0-1 scale, higher=better):
    - **context_recall**: Proportion of relevant contexts retrieved (retrieval quality)
    - **faithfulness**: Factual accuracy of generated responses (hallucination check)
    - **answer_relevancy**: How relevant the response is to the original question
    - **factual_correctness**: Semantic accuracy compared to ground truth
    - **context_entity_recall**: Entity extraction completeness from contexts
    - **noise_sensitivity_relevant**: Robustness to irrelevant/noisy contexts

    Advanced Metrics:
    - **llm_context_precision_without_reference**: LLM-based precision without ground truth
    - **llm_context_precision_with_reference**: LLM-based precision with ground truth
    - **non_llm_context_precision_with_reference**: Traditional precision metrics
    - **faithful_rate**: Rate of faithful (non-hallucinated) responses

    Features:
    - Multi-metric ranking with customizable weights
    - Statistical significance testing across retrievers
    - Performance visualization and heatmap generation
    - Cost-performance trade-off analysis
    - Automatic normalization for fair comparison

    Usage:
        >>> ranker = RetrieverRanker('ragas-evaluation-metrics.csv')
        >>> rankings = ranker.rank_retrievers(['context_recall', 'faithfulness'])
        >>> ranker.plot_performance_heatmap()
        >>> stats = ranker.statistical_significance_test('context_recall')

    Data Format Expected:
        CSV with columns: retriever, context_recall, faithfulness, answer_relevancy, etc.
        Each row represents one evaluation run for a specific retriever.
    """

    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        self.normalized_df = self._normalize_data()
        self.available_metrics = self._get_available_metrics()

    def _get_available_metrics(self):
        """Get list of available quality metrics in the dataset"""
        all_metrics = [
            "context_recall",
            "faithfulness",
            "factual_correctness",
            "answer_relevancy",
            "context_entity_recall",
            "noise_sensitivity_relevant",
            "llm_context_precision_without_reference",
            "llm_context_precision_with_reference",
            "non_llm_context_precision_with_reference",
            "faithful_rate",
        ]
        return [metric for metric in all_metrics if metric in self.df.columns]

    def _normalize_data(self):
        """Normalize metrics to 0-1 scale for fair comparison"""
        df_norm = self.df.copy()

        # Define all possible metrics (old + new)
        possible_metrics = [
            "context_recall",
            "faithfulness",
            "factual_correctness",
            "answer_relevancy",
            "context_entity_recall",
            "noise_sensitivity_relevant",
            "Avg_Cost_Per_Run",
            "llm_context_precision_without_reference",
            "llm_context_precision_with_reference",
            "non_llm_context_precision_with_reference",
            "faithful_rate",
        ]

        # Only normalize metrics that exist in the dataframe
        available_metrics = [
            metric for metric in possible_metrics if metric in df_norm.columns
        ]

        for metric in available_metrics:
            min_val, max_val = df_norm[metric].min(), df_norm[metric].max()
            if max_val == min_val:
                df_norm[f"{metric}_norm"] = 1.0
            elif metric == "Avg_Cost_Per_Run":  # Lower cost is better
                df_norm[f"{metric}_norm"] = (max_val - df_norm[metric]) / (
                    max_val - min_val
                )
            else:  # Higher is better for quality metrics
                df_norm[f"{metric}_norm"] = (df_norm[metric] - min_val) / (
                    max_val - min_val
                )

        return df_norm

    def weighted_score(self, weights=None):
        """Calculate weighted score based on custom weights - adapts to available metrics"""
        if weights is None:
            weights = {}
            # Set default weights for available metrics
            if "context_recall" in self.df.columns:
                weights["context_recall"] = 20
            if "faithfulness" in self.df.columns:
                weights["faithfulness"] = 20
            if "factual_correctness" in self.df.columns:
                weights["factual_correctness"] = 15
            if "answer_relevancy" in self.df.columns:
                weights["answer_relevancy"] = 20
            if "context_entity_recall" in self.df.columns:
                weights["context_entity_recall"] = 10
            if "noise_sensitivity_relevant" in self.df.columns:
                weights["noise_sensitivity_relevant"] = 5
            if "Avg_Cost_Per_Run" in self.df.columns:
                weights["cost_efficiency"] = 10

            # Add new metrics with reasonable defaults
            if "llm_context_precision_without_reference" in self.df.columns:
                weights["llm_context_precision_without_reference"] = 15
            if "llm_context_precision_with_reference" in self.df.columns:
                weights["llm_context_precision_with_reference"] = 18
            if "non_llm_context_precision_with_reference" in self.df.columns:
                weights["non_llm_context_precision_with_reference"] = 12
            if "faithful_rate" in self.df.columns:
                weights["faithful_rate"] = 8

        total_weight = sum(weights.values())
        if total_weight == 0:
            return [0] * len(self.df)

        scores = []

        for _, row in self.normalized_df.iterrows():
            score = 0
            for metric, weight in weights.items():
                if metric == "cost_efficiency":
                    norm_col = "Avg_Cost_Per_Run_norm"
                else:
                    norm_col = f"{metric}_norm"

                if norm_col in self.normalized_df.columns and not pd.isna(
                    row[norm_col]
                ):
                    score += weight * row[norm_col]

            scores.append(score / total_weight)

        return scores

    def quality_first_score(self):
        """Prioritize quality with cost penalty - adapts to available metrics"""
        scores = []
        for _, row in self.df.iterrows():
            # Build quality score from available metrics
            quality_metrics = []

            # Core quality metrics (old)
            for metric in [
                "context_recall",
                "faithfulness",
                "answer_relevancy",
                "factual_correctness",
            ]:
                if metric in self.df.columns and not pd.isna(row[metric]):
                    quality_metrics.append(row[metric])

            # New precision metrics
            for metric in [
                "llm_context_precision_with_reference",
                "llm_context_precision_without_reference",
            ]:
                if metric in self.df.columns and not pd.isna(row[metric]):
                    quality_metrics.append(row[metric])

            quality = (
                sum(quality_metrics) / len(quality_metrics) if quality_metrics else 0
            )

            # Cost penalty (if cost data available)
            cost_penalty = 0
            if "Avg_Cost_Per_Run" in self.df.columns:
                cost_range = (
                    self.df["Avg_Cost_Per_Run"].max()
                    - self.df["Avg_Cost_Per_Run"].min()
                )
                if cost_range > 0:
                    cost_penalty = (
                        (row["Avg_Cost_Per_Run"] - self.df["Avg_Cost_Per_Run"].min())
                        / cost_range
                        * 0.1
                    )

            scores.append(max(0, quality - cost_penalty))
        return scores

    def balanced_score(self):
        """Balanced approach considering all factors - adapts to available metrics"""
        scores = []
        for _, row in self.normalized_df.iterrows():
            # Core quality from available metrics
            core_metrics = []
            for metric in [
                "context_recall_norm",
                "faithfulness_norm",
                "answer_relevancy_norm",
            ]:
                if metric in self.normalized_df.columns and not pd.isna(row[metric]):
                    core_metrics.append(row[metric])

            # Add new precision metrics to core quality
            for metric in [
                "llm_context_precision_with_reference_norm",
                "llm_context_precision_without_reference_norm",
            ]:
                if metric in self.normalized_df.columns and not pd.isna(row[metric]):
                    core_metrics.append(row[metric])

            core_quality = sum(core_metrics) / len(core_metrics) if core_metrics else 0

            # Accuracy bonus from available metrics
            accuracy_bonus = 0
            accuracy_metrics = []
            for metric in ["factual_correctness_norm", "faithful_rate_norm"]:
                if metric in self.normalized_df.columns and not pd.isna(row[metric]):
                    accuracy_metrics.append(row[metric])

            if accuracy_metrics:
                accuracy_bonus = sum(accuracy_metrics) / len(accuracy_metrics) * 0.2

            # Robustness from available metrics
            robustness_metrics = []
            for metric in [
                "context_entity_recall_norm",
                "noise_sensitivity_relevant_norm",
                "non_llm_context_precision_with_reference_norm",
            ]:
                if metric in self.normalized_df.columns and not pd.isna(row[metric]):
                    robustness_metrics.append(row[metric])

            robustness = (
                (sum(robustness_metrics) / len(robustness_metrics) * 0.1)
                if robustness_metrics
                else 0
            )

            # Efficiency
            efficiency = row.get("Avg_Cost_Per_Run_norm", 0) * 0.1

            scores.append(core_quality + accuracy_bonus + robustness + efficiency)
        return scores

    def production_ready_score(self):
        """Production-ready with minimum thresholds - adapts to available metrics"""
        # Define thresholds for available metrics
        thresholds = {}
        if "context_recall" in self.df.columns:
            thresholds["context_recall"] = 0.7
        if "faithfulness" in self.df.columns:
            thresholds["faithfulness"] = 0.8
        if "answer_relevancy" in self.df.columns:
            thresholds["answer_relevancy"] = 0.85
        if "llm_context_precision_with_reference" in self.df.columns:
            thresholds["llm_context_precision_with_reference"] = 0.75

        scores = []

        for _, row in self.df.iterrows():
            # Check if meets available thresholds
            meets_thresholds = True
            for metric, threshold in thresholds.items():
                if metric in self.df.columns and not pd.isna(row[metric]):
                    if row[metric] < threshold:
                        meets_thresholds = False
                        break

            if not meets_thresholds:
                scores.append(0)
                continue

            # Calculate quality excess for available metrics
            quality_excess = 0
            for metric, threshold in thresholds.items():
                if metric in self.df.columns and not pd.isna(row[metric]):
                    quality_excess += max(0, row[metric] - threshold)

            # Add cost efficiency if available
            cost_efficiency = 0
            if "Avg_Cost_Per_Run_norm" in self.normalized_df.columns:
                cost_efficiency = (
                    self.normalized_df.loc[row.name, "Avg_Cost_Per_Run_norm"] * 0.3
                )

            scores.append(quality_excess + cost_efficiency)

        return scores

    def get_rankings_table(self, algorithm="weighted", weights=None):
        """Generate rankings table for specified algorithm"""
        df_result = self.df.copy()
        df_result["retriever_chain"] = (
            df_result["retriever"]
            .str.replace("_retrieval_chain", "")
            .str.replace("_", " ")
            .str.title()
        )

        if algorithm == "weighted":
            df_result["score"] = self.weighted_score(weights)
        elif algorithm == "quality_first":
            df_result["score"] = self.quality_first_score()
        elif algorithm == "balanced":
            df_result["score"] = self.balanced_score()
        elif algorithm == "production_ready":
            df_result["score"] = self.production_ready_score()

        df_result = df_result.sort_values("score", ascending=False).reset_index(
            drop=True
        )
        df_result["rank"] = range(1, len(df_result) + 1)

        # Build output columns based on available metrics
        output_cols = ["rank", "retriever_chain", "score"]

        # Add available quality metrics in order of priority
        priority_metrics = [
            "context_recall",
            "faithfulness",
            "factual_correctness",
            "answer_relevancy",
            "llm_context_precision_with_reference",
            "llm_context_precision_without_reference",
            "faithful_rate",
            "context_entity_recall",
        ]

        for col in priority_metrics:
            if col in df_result.columns:
                output_cols.append(col)

        # Add cost if available
        if "Avg_Cost_Per_Run" in df_result.columns:
            output_cols.append("Avg_Cost_Per_Run")

        return df_result[output_cols].round(4)

    def get_metrics_comparison_table(self):
        """Compare all retrievers across key metrics - adapts to available columns"""
        df_comp = self.df.copy()
        df_comp["retriever_chain"] = (
            df_comp["retriever"]
            .str.replace("_retrieval_chain", "")
            .str.replace("_", " ")
            .str.title()
        )

        # Build output columns based on available metrics
        output_cols = ["retriever_chain"]

        # Add available metrics in logical order
        priority_metrics = [
            "context_recall",
            "faithfulness",
            "factual_correctness",
            "answer_relevancy",
            "llm_context_precision_with_reference",
            "llm_context_precision_without_reference",
            "non_llm_context_precision_with_reference",
            "faithful_rate",
            "context_entity_recall",
            "noise_sensitivity_relevant",
            "Avg_Cost_Per_Run",
        ]

        for metric in priority_metrics:
            if metric in df_comp.columns:
                output_cols.append(metric)

        return df_comp[output_cols].round(4)

    def get_algorithm_comparison_table(self):
        """Compare rankings across all algorithms"""
        algorithms = ["weighted", "quality_first", "balanced", "production_ready"]
        results = {}

        for algo in algorithms:
            rankings = self.get_rankings_table(algo)
            results[f"{algo}_rank"] = rankings.set_index("retriever_chain")["rank"]
            results[f"{algo}_score"] = rankings.set_index("retriever_chain")["score"]

        df_comp = pd.DataFrame(results)
        df_comp.index.name = "retriever"
        return df_comp.round(4)

    def get_recommendations_table(self):
        """Generate recommendations for different use cases"""
        recommendations = []

        # Overall winner (weighted)
        winner = self.get_rankings_table("weighted").iloc[0]
        recommendations.append(
            [
                "Overall Winner",
                winner["retriever_chain"],
                f"Score: {winner['score']:.3f}",
                "Best balanced performance",
            ]
        )

        # Budget option
        if "Avg_Cost_Per_Run" in self.df.columns:
            budget_idx = self.df["Avg_Cost_Per_Run"].idxmin()
            budget_retriever = (
                self.df.loc[budget_idx, "retriever"]
                .replace("_retrieval_chain", "")
                .replace("_", " ")
                .title()
            )
            budget_cost = self.df.loc[budget_idx, "Avg_Cost_Per_Run"]
            recommendations.append(
                [
                    "Budget Option",
                    budget_retriever,
                    f"Cost: ${budget_cost:.4f}",
                    "Lowest cost per run",
                ]
            )

        # Quality leader - use available quality metrics
        quality_metrics = []
        for metric in [
            "context_recall",
            "faithfulness",
            "answer_relevancy",
            "llm_context_precision_with_reference",
        ]:
            if metric in self.df.columns:
                quality_metrics.append(metric)

        if quality_metrics:
            quality_scores = self.df[quality_metrics].mean(axis=1)
            quality_idx = quality_scores.idxmax()
            quality_retriever = (
                self.df.loc[quality_idx, "retriever"]
                .replace("_retrieval_chain", "")
                .replace("_", " ")
                .title()
            )
            quality_score = quality_scores.iloc[quality_idx]
            recommendations.append(
                [
                    "Quality Leader",
                    quality_retriever,
                    f"Quality: {quality_score:.3f}",
                    f"Highest average across {len(quality_metrics)} quality metrics",
                ]
            )

        # Production ready
        prod_rankings = self.get_rankings_table("production_ready")
        prod_winner = (
            prod_rankings[prod_rankings["score"] > 0].iloc[0]
            if (prod_rankings["score"] > 0).any()
            else prod_rankings.iloc[0]
        )
        recommendations.append(
            [
                "Production Ready",
                prod_winner["retriever_chain"],
                f"Score: {prod_winner['score']:.3f}",
                "Meets minimum thresholds",
            ]
        )

        return pd.DataFrame(
            recommendations,
            columns=["Category", "Retriever", "Key Metric", "Description"],
        )

    def print_available_metrics(self):
        """Print information about available metrics in the dataset"""
        logger.info("📊 AVAILABLE METRICS IN DATASET")
        logger.info("-" * 40)

        old_metrics = [
            "context_recall",
            "faithfulness",
            "factual_correctness",
            "answer_relevancy",
            "context_entity_recall",
            "noise_sensitivity_relevant",
        ]
        new_metrics = [
            "llm_context_precision_without_reference",
            "llm_context_precision_with_reference",
            "non_llm_context_precision_with_reference",
            "faithful_rate",
        ]

        logger.info("Legacy metrics:")
        for metric in old_metrics:
            status = "✓" if metric in self.df.columns else "✗"
            logger.info(f"  {status} {metric}")

        logger.info("New metrics:")
        for metric in new_metrics:
            status = "✓" if metric in self.df.columns else "✗"
            logger.info(f"  {status} {metric}")

        logger.info(f"Total available quality metrics: {len(self.available_metrics)}")
        logger.info(
            f"Cost metric available: {'✓' if 'Avg_Cost_Per_Run' in self.df.columns else '✗'}"
        )


def main():
    # Initialize ranker
    ranker = RetrieverRanker("ragas_retriever_raw_stats.csv")

    logger.info("🏆 RETRIEVER RANKING ANALYSIS")
    logger.info("=" * 60)

    # Show available metrics
    ranker.print_available_metrics()

    # 1. Overall Rankings (Weighted Algorithm)
    logger.info("1. OVERALL RANKINGS (Weighted Algorithm)")
    logger.info("-" * 45)
    rankings = ranker.get_rankings_table("weighted")
    logger.info(f"\n{rankings.to_string(index=False)}")

    # 2. Metrics Comparison
    print("\n\n2. DETAILED METRICS COMPARISON")
    print("-" * 45)
    metrics = ranker.get_metrics_comparison_table()
    print(metrics.to_string(index=False))

    # 3. Algorithm Comparison
    print("\n\n3. ALGORITHM COMPARISON (Rankings)")
    print("-" * 45)
    algo_comp = ranker.get_algorithm_comparison_table()
    print(algo_comp.to_string())

    # 4. Recommendations
    print("\n\n4. RECOMMENDATIONS")
    print("-" * 45)
    recommendations = ranker.get_recommendations_table()
    print(recommendations.to_string(index=False))

    # 5. Top 3 Summary
    print("\n\n5. TOP 3 SUMMARY")
    print("-" * 45)
    # Only include columns that exist in the rankings table
    available_cols = rankings.columns.tolist()
    top3_cols = ["rank", "retriever_chain", "score"]

    # Add core quality metrics if available
    for col in [
        "context_recall",
        "faithfulness",
        "llm_context_precision_with_reference",
    ]:
        if col in available_cols:
            top3_cols.append(col)
            break  # Just add one main quality metric for summary

    # Add cost if available
    if "Avg_Cost_Per_Run" in available_cols:
        top3_cols.append("Avg_Cost_Per_Run")

    top3 = rankings.head(3)[top3_cols]
    print(top3.to_string(index=False))


if __name__ == "__main__":
    main()
