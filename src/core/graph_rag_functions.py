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
    """
    Load and process hybrid federal student loan dataset into Qdrant vector store.

    Data Processing Pipeline:
    1. Load 4 federal student loan PDF documents (~615 chunks)
       - Academic Calendars & Cost of Attendance
       - Applications & Verification Guide
       - Federal Pell Grant Program
       - Direct Loan Program

    2. Load customer complaints CSV (4,547 → 825 → 480 after quality filtering)
       - Filters: narratives < 100 chars, excessive redaction (>5 XXXX), empty content
       - Structured format: "Customer Issue: [Issue]\\nProduct: [Product]\\nComplaint Details: [narrative]"

    3. Text chunking optimized for hybrid content:
       - Chunk size: 750 characters (optimal for both PDFs and complaints)
       - Overlap: 100 characters for context preservation
       - Strategy: RecursiveCharacterTextSplitter for intelligent boundaries

    4. Vector embedding generation:
       - Model: OpenAI text-embedding-3-small (1536 dimensions)
       - Distance metric: Cosine similarity for semantic search
       - Total vectors: ~1,095 (PDF: 615 + Complaints: 480)

    5. Qdrant vector store configuration:
       - Deployment: In-memory (:memory:) for development
       - Collection: 'loan_data' with cosine distance
       - Memory footprint: ~39.2MB including embeddings and metadata

    Returns:
        tuple: (student_loan_docs_dataset, internal_vector_store)
            - student_loan_docs_dataset (list): Raw combined documents before chunking
            - internal_vector_store (QdrantVectorStore): Ready-to-use vector store with hybrid knowledge base

    Performance Characteristics:
        - Loading time: ~30-60 seconds (cached after first run)
        - Memory usage: 39.2MB (efficient for hybrid dataset size)
        - Search performance: Sub-second retrieval for k=5 queries
        - Retention rate: 10.7% from raw CSV (4,547 → 480 quality complaints)

    Cache Behavior:
        - Cached with joblib Memory to avoid expensive reprocessing
        - Cache invalidation: Manual deletion of cache folder required for data updates
        - Cache location: ./cache/ (configurable via CACHE_FOLDER env var)

    Usage Note:
        This function is called once at module import to initialize global vector_store.
        Subsequent calls use cached results for fast startup.

    Example:
        >>> docs, vs = get_vectorstore_after_loading_students_loan_data_into_qdrant()
        >>> retriever = vs.as_retriever(search_kwargs={"k": 5})
        >>> results = retriever.get_relevant_documents("What is FAFSA?")
    """
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
    logger.info(
        f"text_splitter: Chunk Size: {text_splitter._chunk_size} | Chunk Overlap: {text_splitter._chunk_overlap}"
    )
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
    """
    Simple cosine similarity retrieval - the baseline RAG method.

    **🎯 RETRIEVAL STRATEGY: Direct Vector Similarity**
    - Uses raw cosine similarity between question embedding and document embeddings
    - No query enhancement, reranking, or document expansion
    - Single-step retrieval: Question → Vector Search → Top-K Documents

    **⚡ PERFORMANCE CHARACTERISTICS:**
    - **Speed**: Fastest (single vector search operation)
    - **RAGAS Metrics**: Best overall performer (context_recall: 0.637, faithfulness: 0.905)
    - **Token Usage**: Lowest (no additional LLM calls)
    - **Reliability**: Most predictable results

    **🔧 TECHNICAL IMPLEMENTATION:**
    - Embedding Model: OpenAI text-embedding-3-small (1536 dimensions)
    - Distance Metric: Cosine similarity for semantic matching
    - Retrieval Count: Fixed k=5 documents
    - Relevance Scoring: Includes cosine distance scores as metadata

    **📊 WHEN TO USE:**
    - ✅ General-purpose queries with clear intent
    - ✅ Performance-critical applications (fastest response)
    - ✅ When you need consistent, reliable results
    - ✅ Budget-conscious deployments (minimal API costs)

    **🆚 KEY DIFFERENCES vs Other Methods:**
    - **vs Contextual Compression**: No reranking - accepts vector similarity as final ranking
    - **vs Multi-Query**: Single query approach - no query expansion or diversification
    - **vs Parent Document**: Returns exact chunks - no document expansion or context restoration

    Args:
        state (dict): LangGraph state containing 'question' key

    Returns:
        dict: Updated state with 'context' key containing 5 retrieved documents,
              each with relevance_score metadata for evaluation
    """
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
    """
    AI-powered reranking retrieval for premium quality results.

    **🎯 RETRIEVAL STRATEGY: Retrieve-Then-Rerank Pipeline**
    - Step 1: Retrieve 20 candidates using cosine similarity (broad recall)
    - Step 2: AI reranking with Cohere Rerank-v3.5 for semantic relevance
    - Step 3: Return top 5 most relevant after intelligent compression

    **⚡ PERFORMANCE CHARACTERISTICS:**
    - **Speed**: Slower (2-step process: retrieval + reranking)
    - **Quality**: Premium semantic matching beyond vector similarity
    - **RAGAS Metrics**: Balanced performance with improved semantic relevance
    - **Cost**: Higher (Cohere API calls for reranking)

    **🔧 TECHNICAL IMPLEMENTATION:**
    - Base Retrieval: k=20 candidates from vector store
    - AI Reranker: Cohere Rerank-v3.5 model for cross-encoder scoring
    - Final Selection: Top 5 after AI-powered reranking
    - Pipeline: LangChain ContextualCompressionRetriever orchestration

    **📊 WHEN TO USE:**
    - ✅ Complex questions requiring nuanced semantic understanding
    - ✅ When precision is more important than speed
    - ✅ Questions with multiple possible interpretations
    - ✅ Premium applications where quality justifies cost

    **🆚 KEY DIFFERENCES vs Other Methods:**
    - **vs Naive**: Adds AI reranking step - smarter relevance scoring beyond cosine similarity
    - **vs Multi-Query**: Single query with better ranking vs multiple queries with standard ranking
    - **vs Parent Document**: Works with fixed chunks - no document size expansion

    **💡 UNIQUE ADVANTAGE:**
    Cross-encoder reranking understands query-document relationships better than
    bi-encoder similarity, leading to more contextually appropriate results.

    Args:
        state (dict): LangGraph state containing 'question' key

    Returns:
        dict: Updated state with 'context' key containing 5 AI-reranked documents
              optimized for semantic relevance
    """
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
    """
    LLM-powered query expansion for comprehensive coverage.

    **🎯 RETRIEVAL STRATEGY: Query Diversification Approach**
    - Step 1: LLM generates multiple query variations from original question
    - Step 2: Execute parallel vector searches for each generated query
    - Step 3: Combine and deduplicate results from all query variations
    - Step 4: Return diverse document set covering multiple query angles

    **⚡ PERFORMANCE CHARACTERISTICS:**
    - **Speed**: Moderate (LLM query generation + multiple vector searches)
    - **Coverage**: Excellent (captures different aspects/interpretations)
    - **RAGAS Metrics**: Good query expansion with solid performance
    - **Robustness**: Handles ambiguous or multi-faceted questions well

    **🔧 TECHNICAL IMPLEMENTATION:**
    - Query Generator: GPT-4.1-nano creates 3-5 alternative phrasings
    - Base Retriever: Naive retriever (k=5) for each generated query
    - Deduplication: LangChain handles duplicate document removal
    - Result Aggregation: Combines unique documents from all query variations

    **📊 WHEN TO USE:**
    - ✅ Ambiguous questions with multiple valid interpretations
    - ✅ Complex topics requiring different perspectives
    - ✅ When user intent is unclear or broad
    - ✅ Research-style queries needing comprehensive coverage

    **🆚 KEY DIFFERENCES vs Other Methods:**
    - **vs Naive**: Multiple query angles vs single query - better coverage but slower
    - **vs Contextual Compression**: Query expansion vs result reranking - different optimization focus
    - **vs Parent Document**: Multiple queries on same chunks vs single query on larger context

    **💡 UNIQUE ADVANTAGE:**
    Captures documents that might be missed by single query due to vocabulary
    mismatch or different conceptual framing of the same information.

    **🔍 EXAMPLE QUERY EXPANSION:**
    Original: "How do I apply for income-driven repayment?"
    Generated:
    - "IDR application process steps"
    - "Income-based repayment plan enrollment"
    - "How to submit income driven repayment forms"

    Args:
        state (dict): LangGraph state containing 'question' key

    Returns:
        dict: Updated state with 'context' key containing deduplicated documents
              from multiple query perspectives (typically 5-15 documents)
    """
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
# child_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
### Same as parent splitting
child_splitter = RecursiveCharacterTextSplitter(chunk_size=750, chunk_overlap=100)
logger.info(
    f"child_splitter: Chunk Size: {child_splitter._chunk_size} | Chunk Overlap: {child_splitter._chunk_overlap}"
)

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
    """
    Small-to-big hierarchical retrieval for maximum context preservation.

    **🎯 RETRIEVAL STRATEGY: Hierarchical Document Expansion**
    - Step 1: Index small child chunks (750 chars) for precise semantic matching
    - Step 2: Vector search finds most relevant child chunks
    - Step 3: Retrieve full parent documents containing matching child chunks
    - Step 4: Return expanded context with complete document boundaries

    **⚡ PERFORMANCE CHARACTERISTICS:**
    - **Speed**: Slower (dual vector store operations + document mapping)
    - **Context**: Maximum - returns full documents vs small chunks
    - **RAGAS Metrics**: Currently lowest performer (context_recall: 0.268, faithfulness: 0.653)
    - **Memory**: Higher (stores both child chunks and full parent documents)

    **🔧 TECHNICAL IMPLEMENTATION:**
    - Child Vectorstore: 750-char chunks in 'full_documents' collection
    - Parent Docstore: InMemoryStore with complete original documents
    - Retrieval Process: ParentDocumentRetriever orchestrates child→parent mapping
    - Score Mapping: Child chunk relevance scores mapped to parent documents

    **📊 WHEN TO USE:**
    - ✅ Questions requiring full document context (policies, procedures)
    - ✅ Multi-section documents where context spans chunk boundaries
    - ✅ Legal/regulatory content where complete context is critical
    - ⚠️ Currently underperforming - consider optimization before production use

    **🆚 KEY DIFFERENCES vs Other Methods:**
    - **vs Naive**: Returns full documents vs 750-char chunks - much more context
    - **vs Contextual Compression**: Context expansion vs result reranking - different goals
    - **vs Multi-Query**: Document size expansion vs query diversification

    **💡 THEORETICAL ADVANTAGE:**
    Provides complete document context to prevent information fragmentation,
    especially valuable for complex documents where relevant information spans
    multiple chunks.

    **⚠️ CURRENT PERFORMANCE NOTE:**
    This method currently shows lower RAGAS scores, possibly due to:
    - Information dilution in large parent documents
    - Misalignment between child chunk relevance and parent document utility
    - Need for better chunk→parent scoring methodology

    **🔧 IMPLEMENTATION DETAILS:**
    - Child Splitter: Same as parent (750 chars, 100 overlap) - consider smaller chunks
    - Dual Storage: Qdrant for child vectors + InMemoryStore for parent docs
    - Score Transfer: Maps highest child relevance score to parent document

    Args:
        state (dict): LangGraph state containing 'question' key

    Returns:
        dict: Updated state with 'context' key containing full parent documents
              with relevance scores transferred from matching child chunks
    """
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
