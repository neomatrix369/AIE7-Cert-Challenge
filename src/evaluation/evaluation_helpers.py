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

from ragas.metrics import (
    LLMContextRecall,
    Faithfulness,
    FactualCorrectness,
    ResponseRelevancy,
    ContextEntityRecall,
    NoiseSensitivity,
)
from ragas import evaluate, RunConfig
from ragas_metrics import extract_ragas_metrics
from tool_calls_parser_for_eval import (
    extract_contexts_for_eval,
    parse_langchain_messages,
)


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
    metrics_filename = "ragas-evaluation-metrics.csv"
    dataset_df = pd.DataFrame()
    if os.path.exists(metrics_filename):
        dataset_df = pd.read_csv(metrics_filename)
    new_dataframe["datetime"] = datetime.now().strftime("%Y-%m-%d %T")
    new_dataframe["retriever"] = retriever_name
    new_dataframe = new_dataframe[["datetime", "retriever"] + columns]
    dataset_df = pd.concat([dataset_df, new_dataframe])

    dataset_df.to_csv(metrics_filename, index=False)
