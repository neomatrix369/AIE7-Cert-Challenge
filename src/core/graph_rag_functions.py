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

from src.core.core_functions import (
    load_and_prepare_pdf_loan_docs,
    load_and_prepare_csv_loan_docs,
)

load_dotenv(dotenv_path="../../.env")

student_loan_pdf_docs_dataset = load_and_prepare_pdf_loan_docs()
student_loan_complaint_docs_dataset = load_and_prepare_csv_loan_docs()
student_loan_docs_dataset = (
    student_loan_pdf_docs_dataset + student_loan_complaint_docs_dataset
)
print(f"Total documents count: {len(student_loan_docs_dataset)}")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=750, chunk_overlap=100)
split_documents = text_splitter.split_documents(student_loan_docs_dataset)
len(split_documents)

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
_ = vector_store.add_documents(documents=split_documents)

### Naive Retriever

naive_retriever = vector_store.as_retriever(search_kwargs={"k": 5})


def naive_retrieve(state):
    retrieved_docs = naive_retriever.invoke(state["question"])
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
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = rag_prompt.format_messages(
        question=state["question"], context=docs_content
    )
    response = llm.invoke(messages)
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
    compressor = CohereRerank(model="rerank-v3.5")
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=contextual_compression_retriever,
        search_kwargs={"k": 5},
    )
    retrieved_docs = compression_retriever.invoke(state["question"])
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
    retrieved_docs = multi_query_retriever.invoke(state["question"])
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
    retrieved_docs = parent_document_retriever.invoke(state["question"])
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
