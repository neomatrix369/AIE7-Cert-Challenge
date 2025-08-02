"""
Evaluation Result Caching Utility

This module provides utilities to cache and load raw evaluate() result
to avoid re-running expensive evaluations.
"""

import os
import pickle
import json
from typing import Dict, Any, Optional, List
from datetime import datetime
import uuid


def _extract_serializable_data(obj) -> Any:
    """Extract serializable data from complex objects"""
    if obj is None:
        return None

    # Handle basic types
    if isinstance(obj, (str, int, float, bool)):
        return obj

    # Handle datetime objects
    if hasattr(obj, "isoformat"):
        try:
            return obj.isoformat()
        except:
            return str(obj)

    # Handle UUID objects
    if hasattr(obj, "hex"):
        return str(obj)

    # Handle lists
    if isinstance(obj, (list, tuple)):
        return [_extract_serializable_data(item) for item in obj]

    # Handle dictionaries
    if isinstance(obj, dict):
        return {key: _extract_serializable_data(value) for key, value in obj.items()}

    # Handle objects with __dict__
    if hasattr(obj, "__dict__"):
        try:
            return _extract_serializable_data(obj.__dict__)
        except:
            pass

    # Handle objects with specific attributes we care about
    serializable_attrs = {}

    # Common attributes to extract from evaluation result
    attrs_to_check = [
        "experiment_id",
        "experiment_name",
        "results",
        "examples",
        "_results",
        "total_cost",
        "start_time",
        "end_time",
        "run_id",
        "evaluation_result",
        "feedback",
        "scores",
        "run",
        "data",
        "value",
        "score",
        "avg",
    ]

    for attr in attrs_to_check:
        if hasattr(obj, attr):
            try:
                val = getattr(obj, attr)
                serializable_attrs[attr] = _extract_serializable_data(val)
            except:
                # If we can't serialize this attribute, convert to string
                try:
                    serializable_attrs[attr] = str(getattr(obj, attr))
                except:
                    serializable_attrs[attr] = f"<{type(getattr(obj, attr)).__name__}>"

    # If we found some attributes, return them
    if serializable_attrs:
        serializable_attrs["_object_type"] = type(obj).__name__
        return serializable_attrs

    # Last resort: convert to string
    return str(obj)


def save_evaluation_result(
    evaluation_result: Any, cache_file: str = "evaluation_result_cache.pkl"
) -> bool:
    """Save raw evaluate() result to cache file

    Args:
        evaluation_result: value is evaluate() result
        cache_file: Path to cache file

    Returns:
        True if saved successfully, False otherwise
    """
    try:
        print(f"🔄 Extracting serializable data from evaluation results...")

        # Extract serializable data from evaluation results
        print("   Processing retriever...")
        serializable_result = _extract_serializable_data(evaluation_result)

        cache_data = {
            "evaluation_result": serializable_result,
            "metadata": {
                "save_time": datetime.now().isoformat(),
                "is_serialized": True,
                "cache_version": "1.0",
            },
        }

        # Save as pickle (preserves object structure)
        with open(cache_file, "wb") as f:
            pickle.dump(cache_data, f)

        # Also save as JSON for cross-platform compatibility and inspection
        json_file = cache_file.replace(".pkl", ".json")
        with open(json_file, "w") as f:
            json.dump(cache_data, f, indent=2, default=str)

        # Save metadata separately for inspection
        json_meta_file = cache_file.replace(".pkl", "_metadata.json")
        with open(json_meta_file, "w") as f:
            json.dump(cache_data["metadata"], f, indent=2)

        print(f"✅ Evaluation results cached to: {cache_file}")
        print(f"📋 JSON version saved to: {json_file}")
        print(f"📋 Metadata saved to: {json_meta_file}")

        return True

    except Exception as e:
        print(f"❌ Failed to save evaluation results: {e}")
        import traceback

        print(f"   Error details: {traceback.format_exc()}")
        return False


def load_evaluation_result(
    cache_file: str = "evaluation_result_cache.pkl",
) -> Optional[Any]:
    """Load raw evaluate() result from cache file

    Args:
        cache_file: Path to cache file

    Returns:
        A single evaluation result if successful, None otherwise
    """
    if not os.path.exists(cache_file):
        print(f"❌ Cache file not found: {cache_file}")
        return None

    try:
        with open(cache_file, "rb") as f:
            cache_data = pickle.load(f)

        evaluation_result = cache_data["evaluation_result"]
        metadata = cache_data.get("metadata", {})

        print(f"✅ Loaded evaluation results from: {cache_file}")
        print(f"📅 Cached on: {metadata.get('save_time', 'Unknown')}")
        return evaluation_result

    except Exception as e:
        print(f"❌ Failed to load evaluation results: {e}")
        return None


def check_cache_exists(cache_file: str = "evaluation_result_cache.pkl") -> bool:
    """Check if evaluation cache file exists

    Args:
        cache_file: Path to cache file

    Returns:
        True if cache exists, False otherwise
    """
    return os.path.exists(cache_file)


def get_cached_evaluation_or_run(
    evaluation_function,
    cache_file: str = "evaluation_result_cache.pkl",
    force_rerun: bool = False,
) -> Dict[str, Any]:
    """Get cached evaluation results or run evaluation function

    Args:
        evaluation_function: Function that returns evaluation_result object when called
        cache_file: Path to cache file
        force_rerun: If True, ignore cache and run evaluation function

    Returns:
        Dictionary of evaluation result
    """
    # Try to load from cache first (unless forced to rerun)
    if not force_rerun:
        cached_result = load_evaluation_result(cache_file)
        if cached_result is not None:
            return cached_result

    # Run evaluation function
    print("🚀 Running fresh evaluation...")
    evaluation_result = evaluation_function()

    # Cache the result
    save_evaluation_result(evaluation_result, cache_file)

    return evaluation_result


# Example usage functions
def create_usage_example():
    """Create example code for using this caching utility"""

    example_code = """
# Example usage in notebook:

from evaluation_cache import save_evaluation_result, load_evaluation_result, get_cached_evaluation_or_run

# Save to cache
save_evaluation_result(evaluation_result, "my_evaluation.pkl")

# Later, load from cache
cached_result = load_evaluation_result("my_evaluation.pkl")
if cached_result:
    # Use cached result with existing analysis code
    from performance_analysis_from_evaluate import analyze_from_evaluate_result
    analyzer = analyze_from_evaluate_result(cached_result)

run_evaluation = ...

# This will use cache if available, otherwise run evaluation
evaluation_result = get_cached_evaluation_or_run(run_evaluation, "evaluation.pkl")

# Use with existing analysis code
analyzer = analyze_from_evaluate_result(evaluation_result)
"""

    return example_code


if __name__ == "__main__":
    print("Evaluation Result Caching Utility")
    print("=" * 40)
    print("\nThis utility helps cache expensive evaluate() result.")
    print("\nUsage example:")
    print(create_usage_example())
