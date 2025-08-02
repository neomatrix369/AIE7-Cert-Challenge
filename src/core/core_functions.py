import os
from getpass import getpass
from dotenv import load_dotenv
from langchain_community.document_loaders import CSVLoader
from tqdm.notebook import tqdm
import gc
from joblib import Memory

from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader

from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

from langgraph.graph import START, END, StateGraph
from typing_extensions import List, TypedDict
from langchain_core.documents import Document

memory = Memory(location="./cache")

load_dotenv(dotenv_path="../../.env")

DATA_FOLDER = os.environ['DATA_FOLDER']

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


#
# ### Data Preparation
#


def load_and_prepare_pdf_loan_docs(folder: str = "../data/"):
    print("Current working directory:", os.getcwd())
    if DATA_FOLDER:
        folder = DATA_FOLDER
    if not os.path.exists(folder):
        folder = "../" + folder
    print("Loading student loan pdfs (knowledge) data...")
    loader = DirectoryLoader(folder, glob="*.pdf", loader_cls=PyMuPDFLoader)
    docs = loader.load()
    gc.collect()
    print(f"Documents count: {len(docs)}")
    return docs


def load_and_prepare_csv_loan_docs(folder: str = "../data/"):
    print("Current working directory:", os.getcwd())
    if DATA_FOLDER:
        folder = DATA_FOLDER
    if not os.path.exists(folder):
        folder = "../" + folder
    loader = CSVLoader(
        file_path=f"{folder}/complaints.csv",
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



def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=750, chunk_overlap=100)
    split_documents = text_splitter.split_documents(docs)
    len(split_documents)
    return split_documents


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


class State(TypedDict):
    question: str
    context: List[Document]
    response: str


graph_builder = StateGraph(State)
graph_builder.add_node("retrieve", retrieve)
graph_builder.add_node("generate", generate)
graph_builder.add_edge(START, "retrieve")
graph_builder.add_edge("retrieve", "generate")
graph_builder.add_edge("generate", END)
graph = graph_builder.compile()
