# List of retrievers covered:
#
# - Naive Retrieval
# - Best-Matching 25 (BM25)
# - Multi-Query Retrieval
# - Parent-Document Retrieval
# - Contextual Compression (a.k.a. Rerank)
# - Ensemble Retrieval
# - Semantic chunking

# Standard library imports
import gc
import os
import getpass
import nltk

# Third-party imports
from datetime import datetime, timedelta
from dotenv import load_dotenv
from operator import itemgetter

# LangChain imports
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import Qdrant
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers import ParentDocumentRetriever, EnsembleRetriever
from langchain.storage import InMemoryStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_cohere import CohereRerank
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models

# RAGAS imports
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas import evaluate, EvaluationDataset, RunConfig
from ragas.testset import TestsetGenerator
from ragas.cost import get_token_usage_for_openai

# Local imports
from evaluation_cache import save_evaluation_result, load_evaluation_result
from langsmith import Client

load_dotenv(dotenv_path=".env")


def check_if_env_var_is_set(env_var_name: str, human_readable_string: str = "API Key"):
    api_key = os.getenv(env_var_name)

    if api_key:
        print(f"{env_var_name} is present")
    else:
        print(f"{env_var_name} is NOT present, paste key at the prompt:")
        os.environ[env_var_name] = getpass.getpass(
            f"Please enter your {human_readable_string}: "
        )


print(check_if_env_var_is_set("OPENAI_API_KEY", "OpenAI API key"))
print(check_if_env_var_is_set("COHERE_API_KEY", "Cohere API key"))

# print(check_if_env_var_is_set("LANGSMITH_API_KEY", "LangSmith API key"))
# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_API_KEY"] = getpass.getpass("LangChain API Key:")
# os.environ["LANGCHAIN_PROJECT"] = f"Trace retrievers - {uuid4().hex[0:8]}"

# ## Task 2: Data Collection and Preparation


# ### Data Preparation


def load_and_prepare_data():
    loader = CSVLoader(
        file_path=f"./data/complaints.csv",
        metadata_columns=[
            "Date received",
            "Product",
            "Sub-product",
            "Issue",
            "Sub-issue",
            "Consumer complaint narrative",
            "Company public response",
            "Company",
            "State",
            "ZIP code",
            "Tags",
            "Consumer consent provided?",
            "Submitted via",
            "Date sent to company",
            "Company response to consumer",
            "Timely response?",
            "Consumer disputed?",
            "Complaint ID",
        ],
    )

    print("Loading loan complaint data...")
    loan_complaint_data = loader.load()

    for doc in loan_complaint_data:
        doc.page_content = doc.metadata["Consumer complaint narrative"]

    gc.collect()

    ### Cleaning the original documents

    print(f"Original documents count: {len(loan_complaint_data)}")

    filtered_docs = []
    for doc in loan_complaint_data:
        narrative = doc.metadata.get("Consumer complaint narrative", "")
        if (
            len(narrative.strip()) < 100
            or narrative.count("XXXX") > 5
            or narrative.strip() in ["", "None", "N/A"]
        ):
            continue

        doc.page_content = f"Customer Issue: {doc.metadata.get('Issue', 'Unknown')}\n"
        doc.page_content += f"Product: {doc.metadata.get('Product', 'Unknown')}\n"
        doc.page_content += f"Complaint Details: {narrative}"

        filtered_docs.append(doc)

        print(f"Documents count after filtering: {len(filtered_docs)}")

        gc.collect()
    return filtered_docs.copy()


# ## Task 3: Setting up QDrant!

small_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")


def get_vector_store():
    return Qdrant.from_documents(
        loan_complaint_data,
        small_embeddings,
        location=":memory:",
        collection_name="LoanComplaints",
    )


# ## Task 4: Naive RAG Chain


def get_naive_retriever(vectorstore):
    return vectorstore.as_retriever(search_kwargs={"k": 10})


# ### A - Augmented


def get_rag_prompt():
    RAG_TEMPLATE = """\
    You are a helpful and kind assistant. Use the context provided below to answer the question.

    If you do not know the answer, or are unsure, say you don't know.

    Query:
    {question}

    Context:
    {context}
    """

    return ChatPromptTemplate.from_template(RAG_TEMPLATE)


# ### G - Generation
#


def get_chat_model(model_name: str):
    return ChatOpenAI(
        model=model_name,
        temperature=0.1,  # Lower temperature for more consistent outputs
        request_timeout=120,  # Longer timeout for complex operations
    )


# ### LCEL RAG Chain
#
# We're going to use LCEL to construct our chain.
#
# > NOTE: This chain will be exactly the same across the various examples with the exception of our Retriever!


def get_naive_retrieval_chain(naive_retriever, rag_prompt, chat_model):
    return (
        # INVOKE CHAIN WITH: {"question" : "<<SOME USER QUESTION>>"}
        # "question" : populated by getting the value of the "question" key
        # "context"  : populated by getting the value of the "question" key and chaining it into the base_retriever
        {
            "context": itemgetter("question") | naive_retriever,
            "question": itemgetter("question"),
        }
        # "context"  : is assigned to a RunnablePassthrough object (will not be called or considered in the next step)
        #              by getting the value of the "context" key from the previous step
        | RunnablePassthrough.assign(context=itemgetter("context"))
        # "response" : the "context" and "question" values are used to format our prompt object and then piped
        #              into the LLM and stored in a key called "response"
        # "context"  : populated by getting the value of the "context" key from the previous step
        | {"response": rag_prompt | chat_model, "context": itemgetter("context")}
    )


# ## Task 5: Best-Matching 25 (BM25) Retriever
#
# Taking a step back in time - [BM25](https://www.nowpublishers.com/article/Details/INR-019) is based on [Bag-Of-Words](https://en.wikipedia.org/wiki/Bag-of-words_model) which is a sparse representation of text.


def get_bm25_retriever(dataset):
    return BM25Retriever.from_documents(
        dataset,
    )


def get_bm25_retriever_chain(bm25_retriever, rag_prompt, chat_model):
    bm25_retrieval_chain = (
        {
            "context": itemgetter("question") | bm25_retriever,
            "question": itemgetter("question"),
        }
        | RunnablePassthrough.assign(context=itemgetter("context"))
        | {"response": rag_prompt | chat_model, "context": itemgetter("context")}
    )


# ## Task 6: Contextual Compression (Using Reranking)
#
# Contextual Compression is a fairly straightforward idea: We want to "compress" our retrieved context into just the most useful bits.
#
# There are a few ways we can achieve this - but we're going to look at a specific example called reranking.
#
# The basic idea here is this:
#
# - We retrieve lots of documents that are very likely related to our query vector
# - We "compress" those documents into a smaller set of *more* related documents using a reranking algorithm.
#
# We'll be leveraging Cohere's Rerank model for our reranker today!
#
# All we need to do is the following:
#
# - Create a basic retriever
# - Create a compressor (reranker, in this case)
#


def get_contextual_compression_retriever(naive_retriever):
    compressor = CohereRerank(model="rerank-v3.5")
    return ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=naive_retriever
    )


def get_compression_retriever_chain(
    contextual_compression_retriever, rag_prompt, chat_model
):
    return (
        {
            "context": itemgetter("question") | contextual_compression_retriever,
            "question": itemgetter("question"),
        }
        | RunnablePassthrough.assign(context=itemgetter("context"))
        | {"response": rag_prompt | chat_model, "context": itemgetter("context")}
    )


# ## Task 7: Multi-Query Retriever
#
# Typically in RAG we have a single query - the one provided by the user.
#
# What if we had....more than one query!
#
# In essence, a Multi-Query Retriever works by:
#
# 1. Taking the original user query and creating `n` number of new user queries using an LLM.
# 2. Retrieving documents for each query.
# 3. Using all unique retrieved documents as context


def get_multi_query_retriever(naive_retriever):
    return MultiQueryRetriever.from_llm(retriever=naive_retriever, llm=chat_model)


def get_multi_query_retrieval_chain(multi_query_retriever):
    return (
        {
            "context": itemgetter("question") | multi_query_retriever,
            "question": itemgetter("question"),
        }
        | RunnablePassthrough.assign(context=itemgetter("context"))
        | {"response": rag_prompt | chat_model, "context": itemgetter("context")}
    )


# ## Task 8: Parent Document Retriever
#
# A "small-to-big" strategy - the Parent Document Retriever works based on a simple strategy:
#
# 1. Each un-split "document" will be designated as a "parent document" (You could use larger chunks of document as well, but our data format allows us to consider the overall document as the parent chunk)
# 2. Store those "parent documents" in a memory store (not a VectorStore)
# 3. We will chunk each of those documents into smaller documents, and associate them with their respective parents, and store those in a VectorStore. We'll call those "child chunks".
# 4. When we query our Retriever, we will do a similarity search comparing our query vector to the "child chunks".
# 5. Instead of returning the "child chunks", we'll return their associated "parent chunks".
#
# Okay, maybe that was a few steps - but the basic idea is this:
#
# - Search for small documents
# - Return big documents
#
# The intuition is that we're likely to find the most relevant information by limiting the amount of semantic information that is encoded in each embedding vector - but we're likely to miss relevant surrounding context if we only use that information.


def get_parent_document_retriever(
    loan_complaint_data,
    vectorstore,
):
    parent_docs = loan_complaint_data
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=750)

    vectorstore.client.create_collection(
        collection_name="full_documents",
        vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE),
    )

    parent_document_vectorstore = Qdrant(
        client=vectorstore.client,  # ✅ Reuse existing client
        embeddings=small_embeddings,  # ✅ Reuse embeddings
        collection_name="full_documents",
    )

    store = InMemoryStore()

    parent_document_retriever = ParentDocumentRetriever(
        vectorstore=parent_document_vectorstore,
        docstore=store,
        child_splitter=child_splitter,
    )

    parent_document_retriever.add_documents(parent_docs, ids=None)

    return parent_document_retriever


def get_parent_document_retrieval_chain(
    parent_document_retriever, rag_prompt, chat_model
):
    return (
        {
            "context": itemgetter("question") | parent_document_retriever,
            "question": itemgetter("question"),
        }
        | RunnablePassthrough.assign(context=itemgetter("context"))
        | {"response": rag_prompt | chat_model, "context": itemgetter("context")}
    )


# ## Task 9: Ensemble Retriever
#
# In brief, an Ensemble Retriever simply takes 2, or more, retrievers and combines their retrieved documents based on a rank-fusion algorithm.
#
# In this case - we're using the [Reciprocal Rank Fusion](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) algorithm.
#
# Setting it up is as easy as providing a list of our desired retrievers - and the weights for each retriever.


def get_ensemble_retriever(
    retriever_list: list = [
        bm25_retriever,
        naive_retriever,
        parent_document_retriever,
        compression_retriever,
        multi_query_retriever,
    ],
    equal_weighting: list = [],
):
    if not equal_weighting:
        equal_weighting = [1 / len(retriever_list)] * len(retriever_list)

    return EnsembleRetriever(retrievers=retriever_list, weights=equal_weighting)


def get_ensemble_retrieval_chain(ensemble_retriever, rag_prompt, chat_model):
    return (
        {
            "context": itemgetter("question") | ensemble_retriever,
            "question": itemgetter("question"),
        }
        | RunnablePassthrough.assign(context=itemgetter("context"))
        | {"response": rag_prompt | chat_model, "context": itemgetter("context")}
    )


# ## Task 10: Semantic Chunking
#
# While this is not a retrieval method - it *is* an effective way of increasing retrieval performance on corpora that have clean semantic breaks in them.
#
# Essentially, Semantic Chunking is implemented by:
#
# 1. Embedding all sentences in the corpus.
# 2. Combining or splitting sequences of sentences based on their semantic similarity based on a number of [possible thresholding methods](https://python.langchain.com/docs/how_to/semantic-chunker/):
#   - `percentile`
#   - `standard_deviation`
#   - `interquartile`
#   - `gradient`
# 3. Each sequence of related sentences is kept as a document!
#

# The `breakpoint_threshold_type` parameter controls when the semantic chunker creates chunk boundaries based on embedding similarity between sentences:
#
# **Four Threshold Types:**
#
# 1. _"percentile" (default)_
# - Splits when sentence embedding distance exceeds the 95th percentile of all distances
# - Effect: Creates chunks at the most semantically distinct boundaries
# - Behavior: More conservative splitting, larger chunks
#
# 2. _"standard_deviation"_
# - Splits when distance exceeds 3 standard deviations from mean
# - Effect: Better predictable performance, especially for normally distributed content
# - Behavior: More consistent chunk sizes
#
# 3. _"interquartile"_
# - Uses IQR * 1.5 scaling factor to determine breakpoints
# - Effect: Middle-ground approach, robust to outliers
# - Behavior: Balanced chunk distribution
#
# 4. _"gradient"_
# - Detects anomalies in embedding distance gradients
# - Effect: Best for domain-specific/highly correlated content
# - Behavior: Finds subtle semantic transitions
#
# **Impact:** _The threshold type determines sensitivity to semantic changes - more sensitive types create smaller, more focused chunks while less sensitive types create larger, more comprehensive chunks._

# We'll use the `percentile` thresholding method for this example which will:
#
# Calculate all distances between sentences, and then break apart sequences of setences that exceed a given percentile among all distances.


def get_semantic_chunker(loan_complaint_data, vectorstore):
    semantic_chunker = SemanticChunker(
        small_embeddings, breakpoint_threshold_type="percentile"
    )

    semantic_documents = semantic_chunker.split_documents(loan_complaint_data[:20])

    vectorstore.client.create_collection(
        collection_name="Loan_Complaint_Data_Semantic_Chunks",
        vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE),
    )

    semantic_vectorstore = Qdrant(
        client=vectorstore.client,  # ✅ Reuse existing client
        embeddings=small_embeddings,  # ✅ Reuse embeddings
        collection_name="Loan_Complaint_Data_Semantic_Chunks",
    )

    _ = semantic_vectorstore.add_documents(semantic_documents)

    return semantic_vectorstore.as_retriever(search_kwargs={"k": 10})


def get_semantic_retrieval_chain(semantic_retriever, rag_prompt, chat_model):
    return (
        {
            "context": itemgetter("question") | semantic_retriever,
            "question": itemgetter("question"),
        }
        | RunnablePassthrough.assign(context=itemgetter("context"))
        | {"response": rag_prompt | chat_model, "context": itemgetter("context")}
    )


nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")


generator_llm = LangchainLLMWrapper(
    get_chat_model(
        "gpt-4.1-nano",  # Less capable than mini for reasoning tasks, but okay for the task
    )
)
generator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())


evaluator_llm = LangchainLLMWrapper(get_chat_model(model="gpt-4.1-mini"))


transformer_llm = generator_llm
embedding_model = generator_embeddings


def generate_golden_master():
    generator = TestsetGenerator(llm=generator_llm, embedding_model=embedding_model)

    golden_master = None

    generator = TestsetGenerator(
        llm=generator_llm, embedding_model=generator_embeddings
    )
    golden_master = generator.generate_with_langchain_docs(
        loan_complaint_data[:20], testset_size=10
    )
    print(golden_master.to_pandas())
    return golden_master


def create_examples_on_langsmith():
    langsmith_client = Client(
        timeout_ms=60000, retry_config={"max_retries": 5}  # 60 seconds
    )

    dataset_name = "Loan Synthetic Data (s09)"

    existing_datasets = langsmith_client.list_datasets()
    dataset_exists = any(dataset.name == dataset_name for dataset in existing_datasets)

    if dataset_exists:
        langsmith_dataset = langsmith_client.read_dataset(dataset_name=dataset_name)
        print(f"Using existing dataset: {dataset_name}")
    else:
        langsmith_dataset = langsmith_client.create_dataset(
            dataset_name=dataset_name,
            description="Loan Synthetic Data (for s09 exercise)",
        )
        print(f"Created new dataset: {dataset_name}")
    return langsmith_dataset


# ## Ragas Evaluation

retriever_chains_list = {
    "naive_retrieval_chain": {"rag_chain": naive_retrieval_chain},
    "bm25_retrieval_chain": {"rag_chain": bm25_retrieval_chain},
    "contextual_compression_retrieval_chain": {
        "rag_chain": contextual_compression_retrieval_chain
    },
    "multi_query_retrieval_chain": {"rag_chain": multi_query_retrieval_chain},
    "parent_document_retrieval_chain": {"rag_chain": parent_document_retrieval_chain},
    "ensemble_retrieval_chain": {"rag_chain": ensemble_retrieval_chain},
}


def simplest_copy_method(original_dataset):
    """
    Simplest method: Use copy.deepcopy()
    This creates a completely independent copy
    """
    dataset_copy = copy.deepcopy(original_dataset)
    return dataset_copy


def create_evaluation_dataset_after_applying_retrieval_chains(
    retriever_chains_list=retriever_chains_list,
):
    for retriever_chain in tqdm(retriever_chains_list.keys()):
        copy_of_golden_master = simplest_copy_method(golden_master)
        retriever_chains_list[retriever_chain]["dataset"] = copy_of_golden_master
        rag_chain = retriever_chains_list[retriever_chain]["rag_chain"]
        for test_row in copy_of_golden_master:
            response = rag_chain.invoke({"question": test_row.eval_sample.user_input})
            test_row.eval_sample.response = response["response"].content
            test_row.eval_sample.retrieved_contexts = [
                context.page_content for context in response["context"]
            ]

    for retriever_chain in tqdm(retriever_chains_list.keys()):
        copy_of_golden_master = retriever_chains_list[retriever_chain]["dataset"]
        retriever_chains_list[retriever_chain]["evaluation_dataset"] = (
            EvaluationDataset.from_pandas(copy_of_golden_master.to_pandas())
        )


pipeline_stages_folder_name = ".pipeline-stages"


def create_pipeline_folder():
    os.makedirs(pipeline_stages_folder_name, exist_ok=True)


# from ragas.metrics import ContextRelevance -- not available for current version of RAGAS


def run_ragas_evaluations(retriever_chains_list=retriever_chains_list):
    evaluation_results = {}
    custom_run_config = RunConfig(timeout=360)

    for retriever_chain in tqdm(retriever_chains_list.keys()):
        evaluation_results_filename = f"{pipeline_stages_folder_name}/ragas_evaluation_results_{retriever_chain}.pkl"
        if os.path.exists(evaluation_results_filename):
            print(f"{retriever_chain} already processed, skipping to the next one...")
            retriever_chains_list[retriever_chain]["evaluation_result"] = (
                load_evaluation_result(evaluation_results_filename)
            )
            continue

        result = evaluate(
            dataset=retriever_chains_list[retriever_chain]["evaluation_dataset"],
            metrics=[
                # STRONGLY related to retrievers
                LLMContextRecall(),  # Retrieval completeness
                LLMContextPrecisionWithoutReference(),  # Retrieval relevance
                LLMContextPrecisionWithReference(),
                NonLLMContextPrecisionWithReference(),
                ContextEntityRecall(),  # Entity-based retrieval quality
                NoiseSensitivity(),  # Noise handling in retrieval
                # ContextRelevance(),          # Overall context relevance to query -- not available for current version of RAGAS
                # MILDLY related to retrievers
                Faithfulness(),  # Keep - generation quality depends on retrieval
                MultiModalFaithfulness(),  # Add if using multimodal - context consistency
            ],
            llm=evaluator_llm,
            token_usage_parser=get_token_usage_for_openai,
            run_config=custom_run_config,
        )
        print(f"Saving {retriever_chain}...")
        retriever_chains_list[retriever_chain]["evaluation_result"] = result
        save_evaluation_result(result, evaluation_results_filename)

        print(
            f"Finished evaluating and saving {retriever_chain} moving to the next one..."
        )
    return save_evaluation_result


# ## Evaluation and Performance Analysis
#
# Now that we have evaluation data from LangSmith, let's analyze the performance of different retrievers across multiple dimensions: **Performance**, **Cost**, and **Latency**.


def extract_ragas_metrics(ragas_result, model_name: str = ""):
    """Extract cost, latency, and token metrics from RAGAS evaluation result"""
    import numpy as np

    def get_value(obj, key):
        """Get value from dict key or object attribute"""
        return obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)

    def safe_mean(values):
        """Calculate mean, filtering out NaN values"""
        if not values:
            return 0
        arr = np.array(values, dtype=float)
        valid = arr[~np.isnan(arr)]
        return float(np.mean(valid)) if len(valid) > 0 else 0

    def get_model_costs(model_name):
        PER_MILLION = 1_000_000
        """Get per-token costs for common models"""
        costs = {
            "gpt-4.1": (2.50 / PER_MILLION, 10.00 / PER_MILLION),
            "gpt-4.1-nano": (0.15 / PER_MILLION, 0.60 / PER_MILLION),
            "gpt-4.1-mini": (0.15 / PER_MILLION, 0.60 / PER_MILLION),
            "gpt-4o-mini": (0.000000150, 0.000000600),
            "gpt-4o": (0.000002500, 0.000010000),
            "gpt-4-turbo": (0.000010000, 0.000030000),
            "gpt-3.5-turbo": (0.000000500, 0.000001500),
            "claude-3-haiku": (0.000000250, 0.000001250),
            "claude-3-sonnet": (0.000003000, 0.000015000),
            "claude-3-opus": (0.000015000, 0.000075000),
            "text-embedding-3-small": (0.02 / PER_MILLION, 0.0),
            "text-embedding-3-large": (0.13 / PER_MILLION, 0.0),
            "rerank-v3.5": (2.00 / PER_MILLION, 0.0),
        }

        # Try exact match, then partial match
        if model_name in costs:
            return costs[model_name]

        for model_key in costs:
            if model_key in model_name.lower():
                return costs[model_key]

        return costs["gpt-4o-mini"]  # Default

    # Extract data
    scores = get_value(ragas_result, "scores") or []
    scores_dict = get_value(ragas_result, "_scores_dict") or {}
    cost_cb = get_value(ragas_result, "cost_cb") or {}
    usage_data = get_value(cost_cb, "usage_data") or []

    # Calculate runs
    total_runs = len(scores) if scores else 1

    # Calculate RAGAS scores (averages from score lists)
    ragas_scores = {}
    for metric, values in scores_dict.items():
        if isinstance(values, list):
            ragas_scores[metric] = safe_mean(values)

    # Calculate tokens and cost
    total_input = sum(get_value(usage, "input_tokens") or 0 for usage in usage_data)
    total_output = sum(get_value(usage, "output_tokens") or 0 for usage in usage_data)

    input_cost, output_cost = get_model_costs(model_name)
    total_cost = (total_input * input_cost) + (total_output * output_cost)

    # Build metrics
    metrics = {
        "Total_Runs": total_runs,
        "Total_Cost": total_cost,
        "Total_Input_Tokens": total_input,
        "Total_Output_Tokens": total_output,
        "Total_Latency_Sec": 0,  # Not available in this data
        "Avg_Cost_Per_Run": total_cost / total_runs,
        "Avg_Input_Tokens_Per_Run": total_input / total_runs,
        "Avg_Output_Tokens_Per_Run": total_output / total_runs,
        "Avg_Latency_Sec": 0,
        **ragas_scores,
    }

    return metrics


def gather_and_save_raw_stats(retriever_chains_list=retriever_chains_list):
    raw_stats_df = pd.DataFrame()
    for retriever_chain in tqdm(retriever_chains_list.keys()):
        result = retriever_chains_list[retriever_chain]["evaluation_result"]
        retriever_chains_list[retriever_chain]["evaluation_metrics"] = (
            extract_ragas_metrics(result, "gpt-4.1-mini").copy()
        )
        each_retriever_df = pd.concat(
            [
                pd.DataFrame([{"retriever": retriever_chain}]),
                pd.DataFrame(
                    [retriever_chains_list[retriever_chain]["evaluation_metrics"]]
                ),
            ],
            axis=1,
        )
        raw_stats_df = pd.concat([raw_stats_df, each_retriever_df])

    raw_stats_df.to_csv("ragas_retriever_raw_stats.csv", index=False)
    return raw_stats_df
