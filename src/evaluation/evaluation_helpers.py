import os
import time
import pandas as pd
from datetime import datetime
from tqdm.notebook import tqdm
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from ragas import EvaluationDataset

from ragas import evaluate
from ragas.cost import get_token_usage_for_openai
from ragas.testset import TestsetGenerator
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper


from ragas.metrics import (
    LLMContextRecall,
    Faithfulness,
    FactualCorrectness,
    ResponseRelevancy,
    ContextEntityRecall,
    NoiseSensitivity,
)
from ragas import evaluate, RunConfig
from src.evaluation.ragas_metrics import extract_ragas_metrics
from src.evaluation.tool_calls_parser_for_eval import (
    extract_contexts_for_eval,
    parse_langchain_messages,
)

# ### Knowledge Graph Based Synthetic Generation
#
# Ragas uses a knowledge graph based approach to create data. This is extremely useful as it allows us to create complex queries rather simply. The additional testset complexity allows us to evaluate larger problems more effectively, as systems tend to be very strong on simple evaluation tasks.
#
# Let's start by defining our `generator_llm` (which will generate our questions, summaries, and more), and our `generator_embeddings` which will be useful in building our graph.

# ### Abstracted SDG
#
# The above method is the full process - but we can shortcut that using the provided abstractions!
#
# This will generate our knowledge graph under the hood, and will - from there - generate our personas and scenarios to construct our queries.
#
#

@memory.cache
def generate_golden_master(original_doc, items_to_pick: int = 20, final_size: int = 10):
    generator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4.1"))
    generator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())
    generator = TestsetGenerator(
        llm=generator_llm, embedding_model=generator_embeddings
    )
    golden_master_dataset = generator.generate_with_langchain_docs(
        original_doc[:items_to_pick], testset_size=final_size
    )
    golden_master_dataset.to_pandas()
    return golden_master_dataset


def generate_responses_for_golden_dataset(
    golden_master, graph_agent, pause_secs_between_each_run: int = 0
):
    for test_row in tqdm(golden_master):
        inputs = {"messages": [HumanMessage(content=test_row.eval_sample.user_input)]}
        response = graph_agent.invoke(inputs)

        evaluation_contexts = extract_contexts_for_eval(response["messages"])
        parsed_data = parse_langchain_messages(response["messages"])
        eval_sample = {
            "user_input": inputs["messages"][0].content,
            "response": response["messages"][-1].content,  # Final AI response
            "retrieved_contexts": evaluation_contexts,
            "tools_used": parsed_data.get("summary", {}).get("tools", []),
            "num_contexts": len(evaluation_contexts),
        }
        test_row.eval_sample.response = eval_sample["response"]
        test_row.eval_sample.retrieved_contexts = eval_sample["retrieved_contexts"]
        test_row.eval_sample.tools_used = eval_sample["tools_used"]

        if pause_secs_between_each_run > 0:
            time.sleep(pause_secs_between_each_run)
    return golden_master


def run_ragas_evaluation(
    golden_dataset,
    method_name: str = "Unknown",
    evaluator_model: str = "gpt-4.1-mini",
    eval_timeout: int = 360,
    request_timeout: int = 120,
    custom_metrics: list = None,
):
    # Convert to RAGAS evaluation dataset
    evaluation_dataset = EvaluationDataset.from_pandas(golden_dataset.to_pandas())

    # Setup evaluator LLM
    evaluator_llm = LangchainLLMWrapper(
        ChatOpenAI(
            model=evaluator_model, temperature=0, request_timeout=request_timeout
        )
    )
    # Configure evaluation settings
    run_config = RunConfig(timeout=eval_timeout)

    # Use custom metrics if provided, otherwise use default set
    if custom_metrics is None:
        metrics = [
            LLMContextRecall(),
            Faithfulness(),
            FactualCorrectness(),
            ResponseRelevancy(),
            ContextEntityRecall(),
            NoiseSensitivity(),
        ]
    else:
        metrics = custom_metrics

    print(f"🧪 Running RAGAS evaluation for {method_name}")

    # Run evaluation
    result = evaluate(
        dataset=evaluation_dataset,
        metrics=metrics,
        llm=evaluator_llm,
        token_usage_parser=get_token_usage_for_openai,
        run_config=run_config,
    )

    print(f"✅ Completed  RAGAS evaluation for {method_name}")
    return result


def record_metrics_from_run(retriever_name, dataframe):
    new_dataframe = dataframe.copy()
    columns = [
        "context_recall",
        "faithfulness",
        "factual_correctness",
        "answer_relevancy",
        "context_entity_recall",
        "noise_sensitivity_relevant",
    ]
    metrics_filename = "../metrics/ragas-evaluation-metrics.csv"
    dataset_df = pd.DataFrame()
    if os.path.exists(metrics_filename):
        dataset_df = pd.read_csv(metrics_filename)
    new_dataframe["datetime"] = datetime.now().strftime("%Y-%m-%d %T")
    new_dataframe["retriever"] = retriever_name
    new_dataframe = new_dataframe[["datetime", "retriever"] + columns]
    dataset_df = pd.concat([dataset_df, new_dataframe])

    dataset_df.to_csv(metrics_filename, index=False)
