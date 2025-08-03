import os
import logging
from qdrant_client.http.models import Distance, VectorParams
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient, models
from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from langchain.prompts import ChatPromptTemplate
from langgraph.graph import START, END, StateGraph
from langchain_core.documents import Document
from typing_extensions import List, TypedDict

from dotenv import load_dotenv

load_dotenv(dotenv_path="../../.env")

from joblib import Memory

# Set up logging with third-party noise suppression
from src.utils.logging_config import setup_logging

logger = setup_logging(__name__)

CACHE_FOLDER = os.getenv("CACHE_FOLDER")
cache_folder = "./cache"
if CACHE_FOLDER:
    cache_folder = CACHE_FOLDER
memory = Memory(location=cache_folder)
logger.info(f"🗄️ Joblib cache location: {cache_folder}")

from datetime import datetime


from src.core.core_functions import (
    load_and_prepare_pdf_loan_docs,
    load_and_prepare_csv_loan_docs,
)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")


@memory.cache
def get_vectorstore_after_loading_students_loan_data_into_qdrant():
    logger.info(f"📚 Starting to load student loan hybrid dataset")
    student_loan_pdf_docs_dataset = load_and_prepare_pdf_loan_docs()
    student_loan_complaint_docs_dataset = load_and_prepare_csv_loan_docs()
    student_loan_docs_dataset = (
        student_loan_pdf_docs_dataset + student_loan_complaint_docs_dataset
    )
    logger.info(
        f"📊 Total hybrid dataset documents: {len(student_loan_docs_dataset)} (PDFs: {len(student_loan_pdf_docs_dataset)}, Complaints: {len(student_loan_complaint_docs_dataset)})"
    )
    logger.info(f"✅ Finished loading student loan hybrid dataset")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=750, chunk_overlap=100)
    split_documents = text_splitter.split_documents(student_loan_docs_dataset)
    logger.info(
        f"📄 Split hybrid dataset into {len(split_documents)} chunks (size=750, overlap=100)"
    )

    logger.info(f"🗃️ Starting Qdrant in-memory database")
    client = QdrantClient(":memory:")

    logger.info(f"📦 Creating Qdrant collection 'loan_data'")
    client.create_collection(
        collection_name="loan_data",
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
    )

    internal_vector_store = QdrantVectorStore(
        client=client,
        collection_name="loan_data",
        embedding=embeddings,
    )

    logger.info(f"⬆️ Adding {len(split_documents)} documents to Qdrant collection")
    _ = internal_vector_store.add_documents(documents=split_documents)

    logger.info(f"✅ Qdrant vector store ready with hybrid dataset")
    return student_loan_docs_dataset, internal_vector_store


student_loan_docs_dataset, vector_store = (
    get_vectorstore_after_loading_students_loan_data_into_qdrant()
)

### Naive Retriever

naive_retriever = vector_store.as_retriever(search_kwargs={"k": 5})


def naive_retrieve(state):
    logger.info(f"🔍 [Naive] Retrieving docs for: {state['question'][:100]}...")
    # Use similarity_search_with_score to get relevance scores
    docs_with_scores = vector_store.similarity_search_with_score(state["question"], k=5)
    retrieved_docs = [doc for doc, score in docs_with_scores]

    # Add relevance scores as metadata for each document
    for i, (doc, score) in enumerate(docs_with_scores):
        if not hasattr(retrieved_docs[i], "metadata"):
            retrieved_docs[i].metadata = {}
        retrieved_docs[i].metadata["relevance_score"] = float(score)

    logger.info(
        f"📚 [Naive] Retrieved {len(retrieved_docs)} documents with relevance scores"
    )
    return {"context": retrieved_docs}


RAG_PROMPT = """
You are a helpful assistant who answers questions based on provided context. You must only use the provided context, and cannot use your own knowledge.

### Question
{question}

### Context
{context}

Return a confidence score and a reason as json for the score once finished based on the outcome of the query.
"""

rag_prompt = ChatPromptTemplate.from_template(RAG_PROMPT)

llm = ChatOpenAI(
    model="gpt-4.1-nano",
    temperature=0,  # Lower temperature for more consistent outputs
    request_timeout=120,  # Longer timeout for complex operations
)


@memory.cache
def invoke_llm(messages):
    """Extracted method for LLM invocation"""
    return llm.invoke(messages)


def generate(state):
    logger.info(
        f"🤖 [Generate Function] Executing LLM call for question: {state['question'][:50]}..."
    )
    logger.info(
        f"🤖 Generating response using {len(state['context'])} context documents"
    )
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = rag_prompt.format_messages(
        question=state["question"], context=docs_content
    )
    response = invoke_llm(messages)
    logger.info(f"✅ Generated response with {len(response.content)} characters")
    return {"response": response.content}


class NaiveState(TypedDict):
    question: str
    context: List[Document]
    response: str


naive_graph_builder = StateGraph(NaiveState)
naive_graph_builder.add_node("naive_retrieve", naive_retrieve)
naive_graph_builder.add_node("generate", generate)
naive_graph_builder.add_edge(START, "naive_retrieve")
naive_graph_builder.add_edge("naive_retrieve", "generate")
naive_graph_builder.add_edge("generate", END)
naive_graph = naive_graph_builder.compile()

### Contextual Compression Retriever
contextual_compression_retriever = vector_store.as_retriever(search_kwargs={"k": 20})


def contextual_compression_retrieve(state):
    logger.info(
        f"🔍 [Contextual Compression] Retrieving docs for: {state['question'][:100]}..."
    )
    compressor = CohereRerank(model="rerank-v3.5")
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=contextual_compression_retriever,
        search_kwargs={"k": 5},
    )
    retrieved_docs = compression_retriever.invoke(state["question"])
    logger.info(
        f"📚 [Contextual Compression] Retrieved {len(retrieved_docs)} documents after reranking"
    )
    return {"context": retrieved_docs}


class ContextualCompressionState(TypedDict):
    question: str
    context: List[Document]
    response: str


contextual_compression_graph_builder = StateGraph(ContextualCompressionState)
contextual_compression_graph_builder.add_node(
    "contextual_compression_retrieve", contextual_compression_retrieve
)
contextual_compression_graph_builder.add_node("generate", generate)
contextual_compression_graph_builder.add_edge(START, "contextual_compression_retrieve")
contextual_compression_graph_builder.add_edge(
    "contextual_compression_retrieve", "generate"
)
contextual_compression_graph_builder.add_edge("generate", END)
contextual_compression_graph = contextual_compression_graph_builder.compile()

### Multi Query Retriever
from langchain.retrievers.multi_query import MultiQueryRetriever

multi_query_retriever = MultiQueryRetriever.from_llm(retriever=naive_retriever, llm=llm)


def multi_query_retrieve(state):
    logger.info(f"🔍 [Multi-Query] Retrieving docs for: {state['question'][:100]}...")
    retrieved_docs = multi_query_retriever.invoke(state["question"])
    logger.info(
        f"📚 [Multi-Query] Retrieved {len(retrieved_docs)} documents from expanded queries"
    )
    return {"context": retrieved_docs}


class MultiQueryState(TypedDict):
    question: str
    context: List[Document]
    response: str


multi_query_graph_builder = StateGraph(MultiQueryState)
multi_query_graph_builder.add_node("multi_query_retrieve", multi_query_retrieve)
multi_query_graph_builder.add_node("generate", generate)
multi_query_graph_builder.add_edge(START, "multi_query_retrieve")
multi_query_graph_builder.add_edge("multi_query_retrieve", "generate")
multi_query_graph_builder.add_edge("generate", END)
multi_query_graph = multi_query_graph_builder.compile()

### Parent-Document Retriever

from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore

parent_docs = student_loan_docs_dataset.copy()
child_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)

vector_store.client.create_collection(
    collection_name="full_documents",
    vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE),
)

parent_document_vectorstore = Qdrant(
    client=vector_store.client,  # ✅ Reuse existing client
    embeddings=embeddings,  # ✅ Reuse embeddings
    collection_name="full_documents",
)

store = InMemoryStore()

parent_document_retriever = ParentDocumentRetriever(
    vectorstore=parent_document_vectorstore,
    docstore=store,
    child_splitter=child_splitter,
)

parent_document_retriever.add_documents(parent_docs, ids=None)


def parent_document_retrieve(state):
    logger.info(
        f"🔍 [Parent Document] Retrieving docs for: {state['question'][:100]}..."
    )

    # Get child chunks with scores from the vectorstore first
    child_docs_with_scores = parent_document_vectorstore.similarity_search_with_score(
        state["question"], k=5
    )
    logger.info(f"child_docs_with_scores: {child_docs_with_scores}")

    # Get the parent documents using the retriever
    retrieved_docs = parent_document_retriever.similarity_search_with_score(
        state["question"]
    )
    logger.info(f"parent_docs_with_scores: {retrieved_docs}")

    # Map child chunk scores to parent documents
    # For simplicity, we'll use the highest relevance score from child chunks for each parent
    child_score_map = {}
    for child_doc, score in child_docs_with_scores:
        # Use page_content as a key to match with parent docs
        content_key = child_doc.page_content[:100]  # First 100 chars as identifier
        if content_key not in child_score_map:
            child_score_map[content_key] = float(score)

    # Add relevance scores to parent documents
    for doc in retrieved_docs:
        if not hasattr(doc, "metadata"):
            doc.metadata = {}
        # Find the best matching child score or use a default
        best_score = 0.0
        for content_key, score in child_score_map.items():
            if content_key in doc.page_content:
                best_score = max(best_score, score)
        doc.metadata["relevance_score"] = best_score

    logger.info(
        f"📚 [Parent Document] Retrieved {len(retrieved_docs)} full documents with relevance scores"
    )
    return {"context": retrieved_docs}


class ParentDocumentState(TypedDict):
    question: str
    context: List[Document]
    response: str


parent_document_graph_builder = StateGraph(ParentDocumentState)
parent_document_graph_builder.add_node(
    "parent_document_retrieve", parent_document_retrieve
)
parent_document_graph_builder.add_node("generate", generate)
parent_document_graph_builder.add_edge(START, "parent_document_retrieve")
parent_document_graph_builder.add_edge("parent_document_retrieve", "generate")
parent_document_graph_builder.add_edge("generate", END)
parent_document_graph = parent_document_graph_builder.compile()
