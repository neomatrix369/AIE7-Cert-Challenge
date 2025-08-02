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

from joblib import Memory
CACHE_FOLDER = os.getenv('CACHE_FOLDER')
cache_folder = "./cache"
if CACHE_FOLDER:
   cache_folder = CACHE_FOLDER
memory = Memory(location=cache_folder)

from datetime import datetime

from dotenv import load_dotenv

from src.core.core_functions import (
    load_and_prepare_pdf_loan_docs,
    load_and_prepare_csv_loan_docs,
)

# Set up logging with third-party noise suppression  
from src.utils.logging_config import setup_logging
logger = setup_logging(__name__)

load_dotenv(dotenv_path="../../.env")

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

@memory.cache
def get_vectorstore_after_loading_students_loan_data_into_qdrant():
    logger.info(f"📚 Starting to load student loan hybrid dataset")
    student_loan_pdf_docs_dataset = load_and_prepare_pdf_loan_docs()
    student_loan_complaint_docs_dataset = load_and_prepare_csv_loan_docs()
    student_loan_docs_dataset = (
        student_loan_pdf_docs_dataset + student_loan_complaint_docs_dataset
    )
    logger.info(f"📊 Total hybrid dataset documents: {len(student_loan_docs_dataset)} (PDFs: {len(student_loan_pdf_docs_dataset)}, Complaints: {len(student_loan_complaint_docs_dataset)})")
    logger.info(f"✅ Finished loading student loan hybrid dataset")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=750, chunk_overlap=100)
    split_documents = text_splitter.split_documents(student_loan_docs_dataset)
    logger.info(f"📄 Split hybrid dataset into {len(split_documents)} chunks (size=750, overlap=100)")

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


student_loan_docs_dataset, vector_store = get_vectorstore_after_loading_students_loan_data_into_qdrant()

### Naive Retriever

naive_retriever = vector_store.as_retriever(search_kwargs={"k": 5})


def naive_retrieve(state):
    logger.info(f"🔍 [Naive] Retrieving docs for: {state['question'][:100]}...")
    retrieved_docs = naive_retriever.invoke(state["question"])
    logger.info(f"📚 [Naive] Retrieved {len(retrieved_docs)} documents")
    return {"context": retrieved_docs}


RAG_PROMPT = """\
You are a helpful assistant who answers questions based on provided context. You must only use the provided context, and cannot use your own knowledge.

### Question
{question}

### Context
{context}
"""

rag_prompt = ChatPromptTemplate.from_template(RAG_PROMPT)

llm = ChatOpenAI(
    model="gpt-4.1-nano",
    temperature=0,  # Lower temperature for more consistent outputs
    request_timeout=120,  # Longer timeout for complex operations
)


def generate(state):
    logger.info(f"🤖 Generating response using {len(state['context'])} context documents")
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = rag_prompt.format_messages(
        question=state["question"], context=docs_content
    )
    response = llm.invoke(messages)
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
    logger.info(f"🔍 [Contextual Compression] Retrieving docs for: {state['question'][:100]}...")
    compressor = CohereRerank(model="rerank-v3.5")
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=contextual_compression_retriever,
        search_kwargs={"k": 5},
    )
    retrieved_docs = compression_retriever.invoke(state["question"])
    logger.info(f"📚 [Contextual Compression] Retrieved {len(retrieved_docs)} documents after reranking")
    return {"context": retrieved_docs}


class ContextualCompressionState(TypedDict):
    question: str
    context: List[Document]
    response: str


contextual_compression_graph_builder = StateGraph(ContextualCompressionState)
contextual_compression_graph_builder.add_node("contextual_compression_retrieve", contextual_compression_retrieve)
contextual_compression_graph_builder.add_node("generate", generate)
contextual_compression_graph_builder.add_edge(START, "contextual_compression_retrieve")
contextual_compression_graph_builder.add_edge("contextual_compression_retrieve", "generate")
contextual_compression_graph_builder.add_edge("generate", END)
contextual_compression_graph = contextual_compression_graph_builder.compile()

### Multi Query Retriever
from langchain.retrievers.multi_query import MultiQueryRetriever

multi_query_retriever = MultiQueryRetriever.from_llm(retriever=naive_retriever, llm=llm)


def multi_query_retrieve(state):
    logger.info(f"🔍 [Multi-Query] Retrieving docs for: {state['question'][:100]}...")
    retrieved_docs = multi_query_retriever.invoke(state["question"])
    logger.info(f"📚 [Multi-Query] Retrieved {len(retrieved_docs)} documents from expanded queries")
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
    logger.info(f"🔍 [Parent Document] Retrieving docs for: {state['question'][:100]}...")
    retrieved_docs = parent_document_retriever.invoke(state["question"])
    logger.info(f"📚 [Parent Document] Retrieved {len(retrieved_docs)} full documents from child chunks")
    return {"context": retrieved_docs}


class ParentDocumentState(TypedDict):
    question: str
    context: List[Document]
    response: str


parent_document_graph_builder = StateGraph(ParentDocumentState)
parent_document_graph_builder.add_node("parent_document_retrieve", parent_document_retrieve)
parent_document_graph_builder.add_node("generate", generate)
parent_document_graph_builder.add_edge(START, "parent_document_retrieve")
parent_document_graph_builder.add_edge("parent_document_retrieve", "generate")
parent_document_graph_builder.add_edge("generate", END)
parent_document_graph = parent_document_graph_builder.compile()
