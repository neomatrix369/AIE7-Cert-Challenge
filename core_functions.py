import os
from getpass import getpass
from dotenv import load_dotenv
from langchain_community.document_loaders import CSVLoader
from tqdm.notebook import tqdm
import gc
from joblib import Memory

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

from ragas.testset import TestsetGenerator

from langchain.text_splitter import RecursiveCharacterTextSplitter

memory = Memory(location="./cache")

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


check_if_env_var_is_set("OPENAI_API_KEY", "OpenAI API key")
check_if_env_var_is_set("COHERE_API_KEY", "Cohere API key")

# ## Generating Synthetic Test Data
#
# ### Data Preparation
#
from langchain_community.document_loaders import DirectoryLoader


from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import PyMuPDFLoader


def load_and_prepare_pdf_loan_docs():
    print("Loading student loan pdfs (knowledge) data...")
    path = "data/"
    loader = DirectoryLoader(path, glob="*.pdf", loader_cls=PyMuPDFLoader)
    docs = loader.load()
    gc.collect()
    print(f"Documents count: {len(docs)}")
    return docs


def load_and_prepare_csv_loan_docs():
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

    print("Loading student loan complaints data...")
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


# ## LangChain RAG
#
# Now we'll construct our LangChain RAG, which we will be evaluating using the above created test data!

# ### R - Retrieval
#
# Let's start with building our retrieval pipeline, which will involve loading the same data we used to create our synthetic test set above.
#
# > NOTE: We need to use the same data - as our test set is specifically designed for this data.
# Now that we have our data loaded, let's split it into chunks!


def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=750, chunk_overlap=100)
    split_documents = text_splitter.split_documents(docs)
    len(split_documents)
    return split_documents


# #### ❓ Question:
#
# What is the purpose of the `chunk_overlap` parameter in the `RecursiveCharacterTextSplitter`?

# Next up, we'll need to provide an embedding model that we can use to construct our vector store.

from langchain_openai import OpenAIEmbeddings

# Now we can build our in memory QDrant vector store.


from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams


def get_vector_store(split_documents):
    client = QdrantClient(":memory:")

    client.create_collection(
        collection_name="loan_data",
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
    )

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = QdrantVectorStore(
        client=client,
        collection_name="loan_data",
        embedding=embeddings,
    )

    # We can now add our documents to our vector store.
    vector_store.add_documents(documents=split_documents)
    return vector_store


# Let's define our retriever.


def get_retriever(vector_store):
    return vector_store.as_retriever(search_kwargs={"k": 5})


# Now we can produce a node for retrieval!


def retrieve(state, retriever):
    retrieved_docs = retriever.invoke(state["question"])
    return {"context": retrieved_docs}


# ### Augmented
#
# Let's create a simple RAG prompt!

from langchain.prompts import ChatPromptTemplate


def get_rag_prompt():
    RAG_PROMPT = """\
  You are a helpful assistant who answers questions based on provided context. You must only use the provided context, and cannot use your own knowledge.

  ### Question
  {question}

  ### Context
  {context}
  """

    return ChatPromptTemplate.from_template(RAG_PROMPT)


# ### Generation
#
# We'll also need an LLM to generate responses - we'll use `gpt-4o-nano` to avoid using the same model as our judge model.

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4.1-nano")


# Then we can create a `generate` node!


def generate(state):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = rag_prompt.format_messages(
        question=state["question"], context=docs_content
    )
    response = llm.invoke(messages)
    return {"response": response.content}


# ### Building RAG Graph with LangGraph
#
# Let's create some state for our LangGraph RAG graph!

from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
from langchain_core.documents import Document


class State(TypedDict):
    question: str
    context: List[Document]
    response: str


# Now we can build our simple graph!
#
# > NOTE: We're using `add_sequence` since we will always move from retrieval to generation. This is essentially building a chain in LangGraph.


graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()


# Let's do a test to make sure it's doing what we'd expect.

# response = graph.invoke({"question" : "What are the different kinds of loans?"})


# ## Evaluating the App with Ragas
#
# Now we can finally do our evaluation!
#
# We'll start by running the queries we generated usign SDG above through our application to get context and responses.


def run_queries_thru_rag_graph(dataset):
    for test_row in dataset:
        response = graph.invoke({"question": test_row.eval_sample.user_input})
        test_row.eval_sample.response = response["response"]
        test_row.eval_sample.retrieved_contexts = [
            context.page_content for context in response["context"]
        ]


# dataset.samples[0].eval_sample.response


# Then we can convert that table into a `EvaluationDataset` which will make the process of evaluation smoother.


# from ragas import EvaluationDataset

# evaluation_dataset = EvaluationDataset.from_pandas(golden_master_dataset.to_pandas())

# # We'll need to select a judge model - in this case we're using the same model that was used to generate our Synthetic Data.


# from ragas import evaluate
# from ragas.llms import LangchainLLMWrapper

# evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4.1-mini"))

# # Next up - we simply evaluate on our desired metrics!

# from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness, ResponseRelevancy, ContextEntityRecall, NoiseSensitivity
# from ragas import evaluate, RunConfig

# custom_run_config = RunConfig(timeout=360)

# result = evaluate(
#     dataset=evaluation_dataset,
#     metrics=[LLMContextRecall(), Faithfulness(), FactualCorrectness(), ResponseRelevancy(), ContextEntityRecall(), NoiseSensitivity()],
#     llm=evaluator_llm,
#     run_config=custom_run_config
# )
# # result


# # ## Making Adjustments and Re-Evaluating
# #
# # Now that we've got our baseline - let's make a change and see how the model improves or doesn't improve!
# #
# # > NOTE: This will be using Cohere's Rerank model - please be sure to [sign-up for an API key!](https://docs.cohere.com/reference/about)

# #
# # We'll first set our retriever to return more documents, which will allow us to take advantage of the reranking.


# adjusted_example_retriever = vector_store.as_retriever(search_kwargs={"k": 20})

# # Reranking, or contextual compression, is a technique that uses a reranker to compress the retrieved documents into a smaller set of documents.
# #
# # This is essentially a slower, more accurate form of semantic similarity that we use on a smaller subset of our documents.

# from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
# from langchain_cohere import CohereRerank

# def retrieve_adjusted(state):
#   compressor = CohereRerank(model="rerank-v3.5")
#   compression_retriever = ContextualCompressionRetriever(
#     base_compressor=compressor, base_retriever=adjusted_example_retriever, search_kwargs={"k": 5}
#   )
#   retrieved_docs = compression_retriever.invoke(state["question"])
#   return {"context" : retrieved_docs}


# # We can simply rebuild our graph with the new retriever!

# class AdjustedState(TypedDict):
#   question: str
#   context: List[Document]
#   response: str

# adjusted_graph_builder = StateGraph(AdjustedState).add_sequence([retrieve_adjusted, generate])
# adjusted_graph_builder.add_edge(START, "retrieve_adjusted")
# adjusted_graph = adjusted_graph_builder.compile()


# # response = adjusted_graph.invoke({"question" : "What are the different kinds of loans?"})
# # response["response"]

# import time
# import copy

# rerank_dataset = copy.deepcopy(golden_master_dataset)

# for test_row in rerank_dataset:
#   response = adjusted_graph.invoke({"question" : test_row.eval_sample.user_input})
#   test_row.eval_sample.response = response["response"]
#   test_row.eval_sample.retrieved_contexts = [context.page_content for context in response["context"]]
#   time.sleep(2) # To try to avoid rate limiting.

# # rerank_dataset.samples[0].eval_sample.response

# rerank_evaluation_dataset = EvaluationDataset.from_pandas(rerank_dataset.to_pandas())

# result = evaluate(
#     dataset=rerank_evaluation_dataset,
#     metrics=[LLMContextRecall(), Faithfulness(), FactualCorrectness(), ResponseRelevancy(), ContextEntityRecall(), NoiseSensitivity()],
#     llm=evaluator_llm,
#     run_config=custom_run_config
# )
