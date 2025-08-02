# Import required FastAPI components for building the API
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
# Import Pydantic for data validation and settings management
from pydantic import BaseModel
# Import OpenAI client for interacting with OpenAI's API
from openai import OpenAI
import os
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv
import uuid
from pathlib import Path
import json
import asyncio
from datetime import datetime
import logging
import sys
import tempfile

# Import aimakerspace components for PDF processing and indexing
from aimakerspace.text_utils import PDFLoader, CharacterTextSplitter
from aimakerspace.vectordatabase import VectorDatabase, QdrantVectorDatabase

load_dotenv()

# Set up logging configuration
def setup_logger(name: str = "ai_makerspace", log_level: str = "INFO") -> logging.Logger:
    """Set up a logger with both file and console handlers."""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear any existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler - create logs directory if it doesn't exist
    try:
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)
        
        # Create log file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = logs_dir / f"{name}_{timestamp}.log"
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)
        
        # Log the setup
        logger.info(f"Logger '{name}' initialized with level {log_level}")
        logger.info(f"Log file: {log_file}")
        logger.info(f"Environment: {'Vercel' if os.getenv('VERCEL') else 'Local'}")
    except Exception as e:
        # If file logging fails (e.g., on Vercel), just use console logging
        logger.warning(f"File logging not available: {e}")
        logger.info(f"Logger '{name}' initialized with console-only logging")
    
    return logger

def get_logger(name: str = "ai_makerspace") -> logging.Logger:
    """Get a logger instance. If not already configured, it will be set up."""
    logger = logging.getLogger(name)
    
    # If logger doesn't have handlers, set it up
    if not logger.handlers:
        # Get log level from environment or default to INFO
        log_level = os.getenv("LOG_LEVEL", "INFO")
        setup_logger(name, log_level)
    
    return logger

# Initialize logger
logger = get_logger("api")

# Initialize FastAPI application with a title
app = FastAPI(title="OpenAI Chat API")

# Get the frontend URL from environment or use a default
FRONTEND_URL = os.getenv("NEXT_PUBLIC_API_URL", "http://localhost:3000")

# Feature flags
USE_QDRANT = os.getenv("USE_QDRANT", "false").lower() == "true"
USE_BROWSER_STORAGE = os.getenv("USE_BROWSER_STORAGE", "true").lower() == "true"

# Debug environment variables
logger.info("🔍 Environment Variables:")
logger.info(f"   - VERCEL: {os.getenv('VERCEL')}")
logger.info(f"   - USE_QDRANT: {os.getenv('USE_QDRANT')} -> {USE_QDRANT}")
logger.info(f"   - USE_BROWSER_STORAGE: {os.getenv('USE_BROWSER_STORAGE')} -> {USE_BROWSER_STORAGE}")
logger.info(f"   - QDRANT_URL: {os.getenv('QDRANT_URL', 'Not set')}")
logger.info(f"   - QDRANT_API_KEY: {'Set' if os.getenv('QDRANT_API_KEY') else 'Not set'}")

# Environment detection
def is_vercel_environment():
    """Check if we're running on Vercel"""
    return os.getenv("VERCEL") == "1"

def is_local_environment():
    """Check if we're running locally"""
    return not is_vercel_environment()

# Auto-detect Qdrant for Vercel environments
if is_vercel_environment() and os.getenv("USE_QDRANT") is None:
    if os.getenv("QDRANT_URL") and os.getenv("QDRANT_API_KEY"):
        USE_QDRANT = True
        logger.info(f"   - Auto-enabled USE_QDRANT: {USE_QDRANT}")
    else:
        logger.warning("⚠️ Vercel environment detected but Qdrant credentials not found")
        logger.info("   - Using browser storage fallback")

# Create uploads directory if it doesn't exist
UPLOADS_DIR = Path("uploads")
INDEXES_DIR = Path("indexes")
CHAT_HISTORY_DIR = Path("chat_history")
FILE_METADATA_PATH = Path(tempfile.gettempdir()) / "file_metadata.json"

# Check if we're in a read-only environment (like Vercel)
def is_readonly_environment():
    """Check if we're in a read-only environment like Vercel"""
    try:
        # Try to create a test file
        test_file = UPLOADS_DIR / "test_write.txt"
        UPLOADS_DIR.mkdir(exist_ok=True)
        with open(test_file, 'w') as f:
            f.write("test")
        os.remove(test_file)
        return False
    except (OSError, PermissionError):
        return True

# Global readonly flag
IS_READONLY = is_readonly_environment()

# Create directories if not readonly
if not IS_READONLY:
    UPLOADS_DIR.mkdir(exist_ok=True)
    INDEXES_DIR.mkdir(exist_ok=True)
    CHAT_HISTORY_DIR.mkdir(exist_ok=True)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_URL, "http://localhost:3000", "https://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Supported file extensions
SUPPORTED_EXTENSIONS = ['.pdf', '.md', '.txt', '.csv', '.json']

# File type detection
def get_file_type(filename: str) -> str:
    """Determine file type based on extension"""
    ext = Path(filename).suffix.lower()
    if ext == '.pdf':
        return 'pdf'
    elif ext in ['.md', '.txt']:
        return 'text'
    elif ext == '.csv':
        return 'csv'
    elif ext == '.json':
        return 'json'
    else:
        return 'unknown'

def is_supported_file(filename: str) -> bool:
    """Check if file type is supported"""
    return get_file_type(filename) != 'unknown'

# Data models
class ChatMessage(BaseModel):
    role: str
    content: str
    timestamp: Optional[str] = None

class ChatSession(BaseModel):
    session_id: str
    created_at: str
    file_ids: List[str]
    messages: List[ChatMessage]

class FileChatRequest(BaseModel):
    user_message: str
    file_ids: List[str]
    session_id: Optional[str] = None
    persona: Optional[str] = None
    domain: Optional[str] = None

class GeneralChatRequest(BaseModel):
    user_message: str
    session_id: Optional[str] = None
    persona: Optional[str] = None
    domain: Optional[str] = None

class FileUploadResponse(BaseModel):
    filename: str
    file_id: str
    message: str
    indexing_status: str
    use_browser_storage: bool = False
    file_content: Optional[str] = None  # Base64 encoded file content for browser storage
    vector_store_type: str = "memory"  # "memory", "qdrant", or "browser"

class PreIndexedFileRequest(BaseModel):
    file_id: str
    filename: str
    chunks: List[str]
    embeddings: Optional[List[List[float]]] = None

# Vector database factory
def create_vector_database(file_id: str = None):
    """Create appropriate vector database based on configuration"""
    logger.info(f"🔍 Creating vector database for file_id: {file_id}")
    logger.info(f"   - USE_QDRANT: {USE_QDRANT}")
    logger.info(f"   - QDRANT_URL: {os.getenv('QDRANT_URL', 'Not set')}")
    logger.info(f"   - QDRANT_API_KEY: {'Set' if os.getenv('QDRANT_API_KEY') else 'Not set'}")
    
    if USE_QDRANT:
        try:
            logger.info(f"🔗 Attempting to create QdrantVectorDatabase...")
            vector_db = QdrantVectorDatabase(collection_name=f"documents_{file_id}")
            logger.info(f"✅ Successfully created QdrantVectorDatabase")
            return vector_db
        except Exception as e:
            logger.error(f"❌ Failed to create QdrantVectorDatabase: {str(e)}")
            logger.warning(f"⚠️ Falling back to in-memory VectorDatabase")
            return VectorDatabase()
    else:
        logger.info(f"💾 Creating in-memory vector database")
        return VectorDatabase()

# Global variables for in-memory storage
vector_databases = {}
indexing_status = {}
chat_sessions = {}
memory_stored_files = {}

def load_vector_database_metadata():
    """Load vector database metadata from persistent storage"""
    logger.info("🔄 Loading vector database metadata...")
    logger.info(f"   - IS_READONLY: {IS_READONLY}")
    logger.info(f"   - INDEXES_DIR: {INDEXES_DIR}")
    logger.info(f"   - INDEXES_DIR.exists(): {INDEXES_DIR.exists()}")
    logger.info(f"   - is_vercel_environment(): {is_vercel_environment()}")
    
    try:
        # For Vercel, we can't rely on file system persistence
        # Instead, we'll rely on the fact that Qdrant stores the data
        # and we'll check Qdrant directly when needed
        if is_vercel_environment():
            logger.info("   - Vercel environment detected, skipping file-based metadata loading")
            logger.info("   - Will rely on Qdrant for data persistence")
            return
        
        if not IS_READONLY and INDEXES_DIR.exists():
            index_files = list(INDEXES_DIR.glob("*.json"))
            logger.info(f"   - Found {len(index_files)} index files")
            
            for index_file in index_files:
                logger.info(f"   - Processing index file: {index_file}")
                try:
                    with open(index_file, 'r') as f:
                        metadata = json.load(f)
                    
                    file_id = metadata.get("file_id")
                    if file_id:
                        # Create a placeholder entry for the vector database
                        # The actual vector database will be loaded from Qdrant when needed
                        vector_databases[file_id] = {
                            "vector_db": None,  # Will be loaded on demand
                            "chunks": [],  # Not stored in metadata
                            "filename": metadata.get("filename", f"File_{file_id[:8]}"),
                            "metadata": metadata
                        }
                        
                        # Update indexing status
                        indexing_status[file_id] = {
                            "status": metadata.get("status", "completed"),
                            "message": f"Loaded from persistent storage"
                        }
                        
                        logger.info(f"📊 Loaded metadata for file {file_id}")
                        logger.info(f"   - Filename: {metadata.get('filename', 'Unknown')}")
                        logger.info(f"   - Status: {metadata.get('status', 'Unknown')}")
                    else:
                        logger.warning(f"⚠️ No file_id found in metadata: {metadata}")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to load metadata from {index_file}: {str(e)}")
        else:
            logger.info(f"   - Skipping metadata load: IS_READONLY={IS_READONLY}, INDEXES_DIR.exists()={INDEXES_DIR.exists()}")
    except Exception as e:
        logger.warning(f"⚠️ Failed to load vector database metadata: {str(e)}")
    
    logger.info(f"📊 Final vector_databases keys: {list(vector_databases.keys())}")
    logger.info(f"📊 Final indexing_status keys: {list(indexing_status.keys())}")

# Load metadata on startup
load_vector_database_metadata()

# Store file metadata immediately for list_files endpoint
file_metadata = {}

def save_file_metadata():
    """Save file_metadata to disk (only for local environments)"""
    if is_vercel_environment():
        logger.info("🔄 Vercel environment detected, skipping file metadata persistence to disk")
        return
    
    try:
        with open(FILE_METADATA_PATH, 'w') as f:
            json.dump(file_metadata, f)
        logger.info(f"💾 Saved file_metadata to disk: {len(file_metadata)} files")
    except Exception as e:
        logger.warning(f"⚠️ Failed to save file_metadata: {e}")

def load_file_metadata_from_qdrant():
    """Load file metadata from Qdrant for Vercel environments"""
    if not is_vercel_environment() or not USE_QDRANT:
        return
    
    logger.info("🔄 Loading file metadata from Qdrant for Vercel...")
    try:
        # Get Qdrant client directly
        import os
        
        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")
        
        if not qdrant_url or not qdrant_api_key:
            logger.warning("⚠️ Qdrant credentials not found, skipping metadata loading")
            return
        
        logger.info(f"🔗 Connecting to Qdrant at {qdrant_url}")
        from qdrant_client import QdrantClient
        client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        
        # Test connection
        try:
            collections = client.get_collections()
            logger.info(f"📊 Found {len(collections.collections)} collections in Qdrant")
        except Exception as e:
            logger.error(f"❌ Failed to connect to Qdrant: {str(e)}")
            return
        
        for collection in collections.collections:
            collection_name = collection.name
            if collection_name.startswith("documents_"):
                file_id = collection_name.replace("documents_", "")
                logger.info(f"🔍 Checking collection for file_id: {file_id}")
                
                try:
                    # Get collection info to check if it has data
                    collection_info = client.get_collection(collection_name=collection_name)
                    if collection_info.points_count == 0:
                        logger.warning(f"⚠️ Collection {collection_name} is empty")
                        continue
                    
                    # Try to get metadata from collection info or search
                    filename = None
                    
                    # Method 1: Try to get from collection info
                    try:
                        # Get all points from the collection to find metadata
                        points = client.scroll(
                            collection_name=collection_name,
                            limit=1,
                            with_payload=True
                        )
                        
                        if points[0] and len(points[0]) > 0:
                            point = points[0][0]
                            metadata = point.payload
                            if metadata and "filename" in metadata:
                                filename = metadata["filename"]
                                logger.info(f"📊 Found filename in collection scroll: {filename}")
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to get metadata from scroll: {str(e)}")
                    
                    # Method 2: Try search if scroll failed
                    if not filename:
                        try:
                            # Create a dummy embedding for search
                            dummy_embedding = [0.0] * 1536  # Default embedding size
                            
                            search_results = client.search(
                                collection_name=collection_name,
                                query_vector=dummy_embedding,
                                limit=1,
                                with_payload=True
                            )
                            
                            if search_results and len(search_results) > 0:
                                result = search_results[0]
                                metadata = result.payload
                                if metadata and "filename" in metadata:
                                    filename = metadata["filename"]
                                    logger.info(f"📊 Found filename in search: {filename}")
                        except Exception as e:
                            logger.warning(f"⚠️ Failed to get metadata from search: {str(e)}")
                    
                    # Use filename if found, otherwise use default
                    if filename:
                        file_metadata[file_id] = {
                            "filename": filename,
                            "vector_store_type": "qdrant",
                            "uploaded_at": datetime.now().isoformat()
                        }
                        logger.info(f"📊 Loaded metadata for {file_id}: {filename}")
                    else:
                        # Use a default filename based on file_id
                        file_metadata[file_id] = {
                            "filename": f"File_{file_id[:8]}",
                            "vector_store_type": "qdrant",
                            "uploaded_at": datetime.now().isoformat()
                        }
                        logger.warning(f"⚠️ No filename found for {file_id}, using default: File_{file_id[:8]}")
                        
                except Exception as e:
                    logger.warning(f"⚠️ Error loading metadata for {file_id}: {str(e)}")
                    # Still add to file_metadata with default name
                    file_metadata[file_id] = {
                        "filename": f"File_{file_id[:8]}",
                        "vector_store_type": "qdrant",
                        "uploaded_at": datetime.now().isoformat()
                    }
    except Exception as e:
        logger.warning(f"⚠️ Failed to load file metadata from Qdrant: {str(e)}")
    
    logger.info(f"📊 Loaded {len(file_metadata)} files from Qdrant metadata")

# Load file_metadata from disk on startup (for local environments)
if not is_vercel_environment():
    try:
        if FILE_METADATA_PATH.exists():
            with open(FILE_METADATA_PATH, 'r') as f:
                file_metadata = json.load(f)
            logger.info(f"📊 Loaded file_metadata from disk: {len(file_metadata)} files")
        else:
            file_metadata = {}
            logger.info("📊 No existing file_metadata found, starting fresh")
    except Exception as e:
        logger.warning(f"⚠️ Failed to load file_metadata: {e}")
        file_metadata = {}
else:
    # For Vercel, load metadata from Qdrant
    file_metadata = {}
    load_file_metadata_from_qdrant()

def save_chat_session(session: ChatSession):
    """Save a chat session to disk"""
    try:
        CHAT_HISTORY_DIR.mkdir(exist_ok=True)
        session_file = CHAT_HISTORY_DIR / f"{session.session_id}.json"
        with open(session_file, 'w') as f:
            json.dump(session.model_dump(), f, indent=2)
        logger.info(f"💾 Saved chat session: {session.session_id}")
    except Exception as e:
        logger.error(f"❌ Error saving chat session: {str(e)}")

def load_chat_session(session_id: str) -> Optional[ChatSession]:
    """Load chat session from file or memory"""
    if IS_READONLY:
        return chat_sessions.get(session_id)
    
    try:
        session_file = CHAT_HISTORY_DIR / f"{session_id}.json"
        if session_file.exists():
            with open(session_file, 'r') as f:
                data = json.load(f)
                return ChatSession(**data)
    except Exception as e:
        logger.warning(f"Warning: Failed to load chat session: {e}")
    return None

# Text extraction functions
def extract_text_content(content: bytes) -> List[str]:
    """Extract text content from various file types"""
    try:
        text = content.decode('utf-8')
        return [text]
    except UnicodeDecodeError:
        return []

def extract_csv_content(content: bytes) -> List[str]:
    """Extract and format CSV content"""
    try:
        import csv
        from io import StringIO
        
        text = content.decode('utf-8')
        csv_file = StringIO(text)
        reader = csv.DictReader(csv_file)
        
        # Convert CSV to formatted text
        rows = list(reader)
        if not rows:
            return []
        
        # Get headers
        headers = list(rows[0].keys())
        
        # Format as text
        formatted_rows = []
        for i, row in enumerate(rows):
            row_text = f"Row {i+1}: " + " | ".join([f"{header}: {value}" for header, value in row.items()])
            formatted_rows.append(row_text)
        
        return formatted_rows
    except Exception as e:
        logger.error(f"Error processing CSV: {e}")
        return []

def extract_json_content(content: bytes) -> List[str]:
    """Extract and format JSON content"""
    try:
        import json
        
        data = json.loads(content.decode('utf-8'))
        
        def flatten_json(obj, prefix=""):
            """Flatten JSON object into key-value pairs"""
            items = []
            if isinstance(obj, dict):
                for key, value in obj.items():
                    new_prefix = f"{prefix}.{key}" if prefix else key
                    items.extend(flatten_json(value, new_prefix))
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    new_prefix = f"{prefix}[{i}]" if prefix else f"[{i}]"
                    items.extend(flatten_json(item, new_prefix))
            else:
                items.append(f"{prefix}: {obj}")
            return items
        
        flattened = flatten_json(data)
        return flattened
    except Exception as e:
        logger.error(f"Error processing JSON: {e}")
        return []

# Health check endpoint
@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    logger.info(f"🔍 Health check requested:")
    logger.info(f"   - IS_READONLY: {IS_READONLY}")
    logger.info(f"   - USE_QDRANT: {USE_QDRANT}")
    logger.info(f"   - USE_BROWSER_STORAGE: {USE_BROWSER_STORAGE}")
    logger.info(f"   - Vercel Environment: {is_vercel_environment()}")
    
    return {
        "status": "ok",
        "readonly": IS_READONLY,
        "environment": "vercel" if is_vercel_environment() else "local",
        "vector_store": "qdrant" if USE_QDRANT else "memory",
        "browser_storage": USE_BROWSER_STORAGE,
        "features": {
            "qdrant": USE_QDRANT,
            "browser_storage": USE_BROWSER_STORAGE,
            "readonly": IS_READONLY
        }
    }

# File listing endpoint
@app.get("/api/files")
async def list_files():
    """List all uploaded files"""
    # For Vercel environments, refresh metadata from Qdrant on every request
    if is_vercel_environment() and USE_QDRANT and not file_metadata:
        logger.info("🔄 Vercel environment detected, refreshing metadata from Qdrant")
        load_file_metadata_from_qdrant()
    
    files = []
    
    if IS_READONLY:
        # In read-only mode, return files from memory and vector databases
        for file_id in set(list(memory_stored_files.keys()) + list(vector_databases.keys())):
            status_info = indexing_status.get(file_id, {"status": "unknown", "message": "File not found"})
            
            # Get the actual filename from file_metadata or vector_databases
            actual_filename = f"File_{file_id[:8]}"
            actual_vector_store_type = "browser"
            
            if file_id in file_metadata:
                actual_filename = file_metadata[file_id]["filename"]
                actual_vector_store_type = file_metadata[file_id]["vector_store_type"]
                logger.info(f"✅ Found metadata for {file_id}: {actual_filename} ({actual_vector_store_type})")
            elif file_id in vector_databases and "filename" in vector_databases[file_id]:
                actual_filename = vector_databases[file_id]["filename"]
                actual_vector_store_type = "memory" if file_id in vector_databases else "browser"
                logger.info(f"✅ Found vector database info for {file_id}: {actual_filename}")
            else:
                # Try to refresh metadata from Qdrant for Vercel environments
                if is_vercel_environment() and USE_QDRANT:
                    refreshed_filename = refresh_file_metadata_from_qdrant(file_id)
                    if refreshed_filename:
                        actual_filename = refreshed_filename
                        actual_vector_store_type = "qdrant"
                        logger.info(f"🔄 Refreshed metadata for {file_id}: {actual_filename}")
                    else:
                        logger.warning(f"⚠️ No metadata found for {file_id}, using generic: {actual_filename}")
                else:
                    logger.warning(f"⚠️ No metadata found for {file_id}, using generic: {actual_filename}")
            
            # Determine if file is ready for chat
            ready_for_chat = bool(status_info["status"] in ["completed", "ready"] or file_id in vector_databases)
            
            files.append({
                "file_id": file_id,
                "filename": actual_filename,
                "indexing_status": status_info["status"],
                "message": status_info["message"],
                "vector_store_type": actual_vector_store_type,
                "ready_for_chat": ready_for_chat
            })
    else:
        # In non-read-only mode, scan the uploads directory
        for extension in SUPPORTED_EXTENSIONS:
            for file_path in UPLOADS_DIR.glob(f"*{extension}"):
                # Extract file ID from filename
                filename_parts = file_path.name.split("_", 1)
                if len(filename_parts) == 2:
                    file_id = filename_parts[0]
                    filename = filename_parts[1]
                    
                    status_info = indexing_status.get(file_id, {"status": "unknown", "message": "File not found"})
                    
                    # Determine if file is ready for chat
                    ready_for_chat = bool(status_info["status"] in ["completed", "ready"] or file_id in vector_databases)
                    
                    files.append({
                        "file_id": file_id,
                        "filename": filename,
                        "indexing_status": status_info["status"],
                        "message": status_info["message"],
                        "vector_store_type": "memory" if file_id in vector_databases else "disk",
                        "ready_for_chat": ready_for_chat
                    })
    
    return {"files": files}

# File status endpoint
@app.get("/api/files/{file_id}/status")
async def get_file_status(file_id: str):
    """Get indexing status for a specific file"""
    status_info = indexing_status.get(file_id, {"status": "unknown", "message": "File not found"})
    
    # Determine if file is ready for chat
    in_vector_databases = file_id in vector_databases
    ready_for_chat = bool(status_info["status"] in ["completed", "ready"] or in_vector_databases)
    
    logger.info(f"🔍 File status check for {file_id}:")
    logger.info(f"   - Status info: {status_info}")
    logger.info(f"   - In vector_databases: {in_vector_databases}")
    logger.info(f"   - Ready for chat: {ready_for_chat}")
    logger.info(f"   - Current vector_databases keys: {list(vector_databases.keys())}")
    
    return {
        **status_info,
        "ready_for_chat": ready_for_chat
    }

# Chat history endpoint
@app.get("/api/chat-history")
async def get_chat_history():
    """Get all chat sessions"""
    if IS_READONLY:
        sessions = list(chat_sessions.values())
    else:
        sessions = []
        for session_file in CHAT_HISTORY_DIR.glob("*.json"):
            try:
                with open(session_file, 'r') as f:
                    data = json.load(f)
                    sessions.append(ChatSession(**data))
            except Exception as e:
                logger.warning(f"Warning: Failed to load session {session_file}: {e}")
    
    return {"sessions": [session.dict() for session in sessions]}

# Get specific chat session endpoint
@app.get("/api/chat-history/{session_id}")
async def get_chat_session(session_id: str):
    """Get a specific chat session by session_id"""
    try:
        # First check in memory
        if session_id in chat_sessions:
            session = chat_sessions[session_id]
            logger.info(f"📊 Found chat session {session_id} in memory")
            return session.dict()
        
        # If not in memory, try to load from disk (for local environments)
        if not IS_READONLY:
            session = load_chat_session(session_id)
            if session:
                logger.info(f"📊 Loaded chat session {session_id} from disk")
                return session.dict()
        
        # If not found anywhere, return 404
        logger.warning(f"⚠️ Chat session {session_id} not found")
        raise HTTPException(status_code=404, detail="Chat session not found")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error getting chat session {session_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get chat session: {str(e)}")

# File indexing function
async def index_file(file_content: bytes, file_id: str, filename: str):
    """Index a file using the aimakerspace library"""
    chunks = []  # Initialize chunks variable
    
    try:
        # Update status to indexing
        indexing_status[file_id] = {
            "status": "indexing",
            "message": "Processing file and creating embeddings..."
        }
        
        file_type = get_file_type(filename)
        
        # Calculate vector store type
        vector_store_type = "qdrant" if USE_QDRANT else "memory"
        
        logger.info(f"🔍 Starting vector database indexing for {file_id}")
        logger.info(f"   - Chunks count: {len(chunks)}")
        logger.info(f"   - File type: {file_type}")
        logger.info(f"   - Vector store type: {vector_store_type}")
        
        if file_type == 'pdf':
            # Handle PDF files with improved text extraction
            if IS_READONLY:
                # Store in memory
                memory_stored_files[file_id] = file_content
                temp_file_path = f"/tmp/{file_id}_{filename}"
                with open(temp_file_path, 'wb') as f:
                    f.write(file_content)
            else:
                # Store on disk
                file_path = UPLOADS_DIR / f"{file_id}_{filename}"
                with open(file_path, "wb") as f:
                    f.write(file_content)
                temp_file_path = str(file_path)
            
            logger.info(f"🔍 Processing PDF: {filename} ({len(file_content)} bytes)")
            
            # Try multiple PDF text extraction methods
            documents = []
            
            # Method 1: Try PDFLoader from aimakerspace
            try:
                pdf_loader = PDFLoader(temp_file_path)
                documents = pdf_loader.load_documents()
                logger.info(f"✅ PDFLoader extracted {len(documents)} documents")
            except Exception as e:
                logger.warning(f"⚠️ PDFLoader failed: {str(e)}")
                documents = []
            
            # Method 2: Try PyPDF2 if PDFLoader failed
            if not documents:
                try:
                    import PyPDF2
                    with open(temp_file_path, 'rb') as file:
                        pdf_reader = PyPDF2.PdfReader(file)
                        documents = []
                        for page_num in range(len(pdf_reader.pages)):
                            page = pdf_reader.pages[page_num]
                            text = page.extract_text()
                            if text.strip():
                                documents.append(text)
                    logger.info(f"✅ PyPDF2 extracted {len(documents)} pages")
                except Exception as e:
                    logger.warning(f"⚠️ PyPDF2 failed: {str(e)}")
                    documents = []
            
            # Method 3: Basic text extraction as last resort
            if not documents:
                try:
                    import PyPDF2
                    with open(temp_file_path, 'rb') as file:
                        pdf_reader = PyPDF2.PdfReader(file)
                        documents = []
                        for page in pdf_reader.pages:
                            text = page.extract_text()
                            if text.strip():
                                documents.append(text)
                    logger.info(f"✅ Basic extraction got {len(documents)} pages")
                except Exception as e:
                    logger.warning(f"⚠️ Basic extraction failed: {str(e)}")
                    documents = []
            
            if not documents:
                raise ValueError("Could not extract text from PDF using any method")
            
            # Split text into chunks
            # OPTIMIZATION: Use smaller chunks for better performance
            splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
            chunks = splitter.split_texts(documents)
            
            # OPTIMIZATION: For small files, use even smaller chunks and lower limits
            total_text_length = sum(len(doc) for doc in documents)
            max_chunks = 50  # Default limit
            
            if total_text_length < 10000:  # Small files (< 10KB)
                max_chunks = 20  # Even fewer chunks for small files
                logger.info(f"📝 Small file detected ({total_text_length} chars), limiting to {max_chunks} chunks")
            elif total_text_length < 50000:  # Medium files (< 50KB)
                max_chunks = 35
                logger.info(f"📄 Medium file detected ({total_text_length} chars), limiting to {max_chunks} chunks")
            
            if len(chunks) > max_chunks:
                logger.warning(f"⚠️ Limiting chunks from {len(chunks)} to {max_chunks} for performance")
                chunks = chunks[:max_chunks]
            
        elif file_type == 'text':
            # Handle markdown and text files
            documents = extract_text_content(file_content)
            
            if not documents:
                raise ValueError("No text could be extracted from the file")
            
            # Split text into chunks
            splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
            chunks = splitter.split_texts(documents)
            
            # OPTIMIZATION: For small files, use even smaller chunks and lower limits
            total_text_length = sum(len(doc) for doc in documents)
            max_chunks = 50  # Default limit
            
            if total_text_length < 10000:  # Small files (< 10KB)
                max_chunks = 20  # Even fewer chunks for small files
                logger.info(f"📝 Small file detected ({total_text_length} chars), limiting to {max_chunks} chunks")
            elif total_text_length < 50000:  # Medium files (< 50KB)
                max_chunks = 35
                logger.info(f"📄 Medium file detected ({total_text_length} chars), limiting to {max_chunks} chunks")
            
            if len(chunks) > max_chunks:
                logger.warning(f"⚠️ Limiting chunks from {len(chunks)} to {max_chunks} for performance")
                chunks = chunks[:max_chunks]
            
        elif file_type == 'csv':
            # Handle CSV files
            documents = extract_csv_content(file_content)
            
            if not documents:
                raise ValueError("No text could be extracted from the file")
            
            # Split text into chunks
            splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
            chunks = splitter.split_texts(documents)
            
            # OPTIMIZATION: For small files, use even smaller chunks and lower limits
            total_text_length = sum(len(doc) for doc in documents)
            max_chunks = 50  # Default limit
            
            if total_text_length < 10000:  # Small files (< 10KB)
                max_chunks = 20  # Even fewer chunks for small files
                logger.info(f"📝 Small file detected ({total_text_length} chars), limiting to {max_chunks} chunks")
            elif total_text_length < 50000:  # Medium files (< 50KB)
                max_chunks = 35
                logger.info(f"📄 Medium file detected ({total_text_length} chars), limiting to {max_chunks} chunks")
            
            if len(chunks) > max_chunks:
                logger.warning(f"⚠️ Limiting chunks from {len(chunks)} to {max_chunks} for performance")
                chunks = chunks[:max_chunks]
            
        elif file_type == 'json':
            # Handle JSON files
            documents = extract_json_content(file_content)
            
            if not documents:
                raise ValueError("No text could be extracted from the file")
            
            # Split text into chunks
            splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
            chunks = splitter.split_texts(documents)
            
            # OPTIMIZATION: For small files, use even smaller chunks and lower limits
            total_text_length = sum(len(doc) for doc in documents)
            max_chunks = 50  # Default limit
            
            if total_text_length < 10000:  # Small files (< 10KB)
                max_chunks = 20  # Even fewer chunks for small files
                logger.info(f"📝 Small file detected ({total_text_length} chars), limiting to {max_chunks} chunks")
            elif total_text_length < 50000:  # Medium files (< 50KB)
                max_chunks = 35
                logger.info(f"📄 Medium file detected ({total_text_length} chars), limiting to {max_chunks} chunks")
            
            if len(chunks) > max_chunks:
                logger.warning(f"⚠️ Limiting chunks from {len(chunks)} to {max_chunks} for performance")
                chunks = chunks[:max_chunks]
            
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
        
        # Update status
        indexing_status[file_id] = {
            "status": "indexing", 
            "message": f"Creating embeddings for {len(chunks)} text chunks..."
        }
        
        # Create vector database based on configuration
        vector_db = create_vector_database(file_id)
        logger.info(f"✅ Created vector database: {type(vector_db).__name__}")
        
        try:
            await vector_db.abuild_from_list(chunks, metadata={"file_id": file_id, "filename": filename})
            logger.info(f"✅ Successfully indexed {len(chunks)} chunks into vector database")
            
            # Store the vector database in memory for quick access
            vector_databases[file_id] = {
                "vector_db": vector_db,
                "chunks": chunks,
                "filename": filename
            }
            
            logger.info(f"📊 Stored file {file_id} in vector_databases")
            logger.info(f"📊 Current vector_databases keys: {list(vector_databases.keys())}")
            
            # Save the vector database metadata
            index_data = {
                "file_id": file_id,
                "chunks_count": len(chunks),
                "indexed_at": asyncio.get_event_loop().time(),
                "status": "completed",
                "vector_store_type": "qdrant" if USE_QDRANT else "memory",
                "filename": filename  # Add filename to metadata
            }
            
            logger.info(f"💾 Saving index metadata for {file_id}:")
            logger.info(f"   - Filename: {filename}")
            logger.info(f"   - Chunks count: {len(chunks)}")
            logger.info(f"   - Vector store type: {index_data['vector_store_type']}")
            
            # Store metadata in memory for Vercel (since file system is not persistent)
            if is_vercel_environment():
                logger.info(f"   - Vercel environment: storing metadata in memory")
                # Store in the global vector_databases dictionary
                vector_databases[file_id] = {
                    "vector_db": vector_db,
                    "chunks": chunks,
                    "filename": filename,
                    "metadata": index_data
                }
                # Also update file_metadata for consistency
                file_metadata[file_id] = {
                    "filename": filename,
                    "vector_store_type": "qdrant",
                    "uploaded_at": datetime.now().isoformat()
                }
                logger.info(f"   - ✅ Updated file_metadata for {file_id}: {filename}")
                
                # Also store metadata in Qdrant for persistent access
                store_metadata_in_qdrant(file_id, filename)
            elif not IS_READONLY:
                index_file_path = INDEXES_DIR / f"{file_id}.json"
                logger.info(f"   - Saving to: {index_file_path}")
                with open(index_file_path, 'w') as f:
                    json.dump(index_data, f)
                logger.info(f"   - ✅ Metadata saved successfully")
            else:
                logger.info(f"   - Skipping metadata save (read-only environment)")
            
            # Update status to completed
            indexing_status[file_id] = {
                "status": "completed",
                "message": f"Successfully indexed {len(chunks)} text chunks"
            }
            
            # Clean up temp file if in read-only mode
            if IS_READONLY and file_type == 'pdf':
                try:
                    os.remove(temp_file_path)
                except:
                    pass
            
        except Exception as e:
            logger.error(f"❌ Error during vector database indexing: {str(e)}")
            # Update status to failed
            indexing_status[file_id] = {
                "status": "failed",
                "message": f"Vector database indexing failed: {str(e)}"
            }
            raise
        
    except Exception as e:
        # Update status to failed
        indexing_status[file_id] = {
            "status": "failed",
            "message": f"Indexing failed: {str(e)}"
        }
        raise

# Define PDF chat endpoint with enhanced RAG functionality
@app.post("/api/chat-file")
async def chat_with_file(request: FileChatRequest):
    try:
        # Validate file IDs
        if not request.file_ids:
            raise HTTPException(status_code=400, detail="At least one file ID is required")
        
        logger.info(f"🔍 Chat request for files: {request.file_ids}")
        logger.info(f"📊 Available vector databases: {list(vector_databases.keys())}")
        logger.info(f"📊 Available indexing status: {list(indexing_status.keys())}")
        
        # Check if all files are indexed
        missing_files = []
        failed_files = []
        for file_id in request.file_ids:
            logger.info(f"🔍 Checking file {file_id}:")
            logger.info(f"   - In vector_databases: {file_id in vector_databases}")
            logger.info(f"   - In indexing_status: {file_id in indexing_status}")
            
            if file_id not in vector_databases:
                # For Vercel environment, try to check if file exists in Qdrant
                if is_vercel_environment() and USE_QDRANT:
                    logger.info(f"   - Checking Qdrant for file {file_id}")
                    try:
                        # Try to create a vector database and check if it has data
                        temp_vector_db = create_vector_database(file_id)
                        
                        # Try to get the actual filename from Qdrant metadata
                        actual_filename = f"File_{file_id[:8]}"  # Default fallback
                        try:
                            # Search for any document to get metadata
                            search_results = temp_vector_db.search_by_text("", k=1, return_as_text=False)
                            if search_results and len(search_results) > 0:
                                # Get metadata from the first result
                                metadata = search_results[0][1] if len(search_results[0]) > 1 else {}
                                if isinstance(metadata, dict) and "filename" in metadata:
                                    actual_filename = metadata["filename"]
                                    logger.info(f"   - Found filename in Qdrant metadata: {actual_filename}")
                                else:
                                    logger.info(f"   - No filename in Qdrant metadata, using default")
                            else:
                                logger.info(f"   - No documents found in Qdrant, using default filename")
                        except Exception as e:
                            logger.warning(f"   - Could not retrieve filename from Qdrant: {str(e)}")
                        
                        # If no metadata found, try to get from file_metadata
                        if actual_filename == f"File_{file_id[:8]}" and file_id in file_metadata:
                            actual_filename = file_metadata[file_id].get("filename", actual_filename)
                            logger.info(f"   - Found filename in file_metadata: {actual_filename}")
                        
                        # Update file_metadata with the found information
                        file_metadata[file_id] = {
                            "filename": actual_filename,
                            "vector_store_type": "qdrant",
                            "uploaded_at": datetime.now().isoformat()
                        }
                        
                        # Save file_metadata to disk
                        save_file_metadata()
                        
                        vector_databases[file_id] = {
                            "vector_db": temp_vector_db,
                            "chunks": [],
                            "filename": actual_filename,
                            "metadata": {"file_id": file_id, "status": "completed", "filename": actual_filename}
                        }
                        indexing_status[file_id] = {
                            "status": "completed",
                            "message": "Found in Qdrant"
                        }
                        logger.info(f"   - ✅ File found in Qdrant with filename: {actual_filename}")
                        continue
                    except Exception as e:
                        logger.warning(f"   - ⚠️ File not found in Qdrant: {str(e)}")
                
                # Check if file has failed indexing
                status_info = indexing_status.get(file_id, {"status": "unknown", "message": "File not found"})
                logger.info(f"   - Status: {status_info}")
                if status_info["status"] == "failed":
                    failed_files.append(f"{file_id} (failed: {status_info['message']})")
                else:
                    missing_files.append(f"{file_id} (status: {status_info['status']})")
            else:
                logger.info(f"   - ✅ File found in vector_databases")
        
        if missing_files or failed_files:
            error_details = []
            if missing_files:
                error_details.append(f"Missing/not indexed: {', '.join(missing_files)}")
            if failed_files:
                error_details.append(f"Failed indexing: {', '.join(failed_files)}")
            
            raise HTTPException(
                status_code=400, 
                detail=f"Files not found or not indexed: {'; '.join(error_details)}"
            )
        
        # Get or create chat session
        session_id = request.session_id or str(uuid.uuid4())
        if session_id not in chat_sessions:
            chat_sessions[session_id] = ChatSession(
                session_id=session_id,
                created_at=datetime.now().isoformat(),
                file_ids=request.file_ids,
                messages=[]
            )
        
        session = chat_sessions[session_id]
        
        # Collect relevant chunks from all files
        all_relevant_chunks = []
        file_names = []
        
        for file_id in request.file_ids:
            file_data = vector_databases[file_id]
            
            # Check if vector_db is a placeholder and needs to be loaded from Qdrant
            if file_data["vector_db"] is None and USE_QDRANT:
                logger.info(f"🔄 Loading vector database from Qdrant for {file_id}")
                try:
                    # Create a new vector database instance and load from Qdrant
                    vector_db = create_vector_database(file_id)
                    # The vector database should already contain the data from Qdrant
                    file_data["vector_db"] = vector_db
                    logger.info(f"✅ Successfully loaded vector database from Qdrant for {file_id}")
                except Exception as e:
                    logger.error(f"❌ Failed to load vector database from Qdrant for {file_id}: {str(e)}")
                    raise HTTPException(
                        status_code=500,
                        detail=f"Failed to load vector database for file {file_id}: {str(e)}"
                    )
            
            vector_db = file_data["vector_db"]
            
            # Search for relevant chunks
            relevant_chunks = vector_db.search_by_text(request.user_message, k=2, return_as_text=True)
            all_relevant_chunks.extend(relevant_chunks)
            
            # Get file name for context
            filename = file_data.get("filename", f"File_{file_id[:8]}")
            file_names.append(filename)
        
        if not all_relevant_chunks:
            raise HTTPException(
                status_code=400, 
                detail="No relevant content found in the selected files"
            )
        
        # Create context from relevant chunks
        context = "\n\n".join(all_relevant_chunks)
        
        # Build system prompt with persona and domain guidance
        system_prompt = "You are a helpful AI assistant that answers questions based on the provided file content. "
        
        if request.persona:
            system_prompt += f"\n\nPersona: {request.persona}"
        
        if request.domain:
            system_prompt += f"\n\nDomain Context: {request.domain}"
        
        system_prompt += f"\n\nRelevant file content:\n{context}\n\nAnswer the user's question based on this content. If the answer cannot be found in the provided content, say so."
        
        # Get OpenAI API key
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="OpenAI API key not configured")
        
        # Create OpenAI client
        client = OpenAI(api_key=api_key)
        
        # Add user message to session
        session.messages.append(ChatMessage(
            role="user",
            content=request.user_message,
            timestamp=datetime.now().isoformat()
        ))
        
        # Save session
        save_chat_session(session)
        
        # Create chat completion
        def generate_response():
            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": request.user_message}
                    ],
                    stream=True,
                    temperature=0.7,
                    max_tokens=1000
                )
                
                for chunk in response:
                    if chunk.choices[0].delta.content:
                        yield f"data: {json.dumps({'content': chunk.choices[0].delta.content})}\n\n"
                
                # Add assistant response to session
                session.messages.append(ChatMessage(
                    role="assistant",
                    content="[Streamed response]",
                    timestamp=datetime.now().isoformat()
                ))
                save_chat_session(session)
                
            except Exception as e:
                error_message = f"Error generating response: {str(e)}"
                yield f"data: {json.dumps({'error': error_message})}\n\n"
        
        return StreamingResponse(generate_response(), media_type="text/plain")
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to chat with file: {str(e)}")

# Define general chat endpoint
@app.post("/api/chat")
async def general_chat(request: GeneralChatRequest):
    try:
        # Get OpenAI API key
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="OpenAI API key not configured")
        
        # Create OpenAI client
        client = OpenAI(api_key=api_key)
        
        # Build system prompt with persona and domain guidance
        system_prompt = "You are a helpful AI assistant."
        
        if request.persona:
            system_prompt += f"\n\nPersona: {request.persona}"
        
        if request.domain:
            system_prompt += f"\n\nDomain Context: {request.domain}"
        
        # Get or create chat session
        session_id = request.session_id or str(uuid.uuid4())
        if session_id not in chat_sessions:
            chat_sessions[session_id] = ChatSession(
                session_id=session_id,
                created_at=datetime.now().isoformat(),
                file_ids=[],
                messages=[]
            )
        
        session = chat_sessions[session_id]
        
        # Add user message to session
        session.messages.append(ChatMessage(
            role="user",
            content=request.user_message,
            timestamp=datetime.now().isoformat()
        ))
        
        # Save session
        save_chat_session(session)
        
        # Create chat completion
        def generate_response():
            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": request.user_message}
                    ],
                    stream=True,
                    temperature=0.7,
                    max_tokens=1000
                )
                
                for chunk in response:
                    if chunk.choices[0].delta.content:
                        yield f"data: {json.dumps({'content': chunk.choices[0].delta.content})}\n\n"
                
                # Add assistant response to session
                session.messages.append(ChatMessage(
                    role="assistant",
                    content="[Streamed response]",
                    timestamp=datetime.now().isoformat()
                ))
                save_chat_session(session)
                
            except Exception as e:
                error_message = f"Error generating response: {str(e)}"
                yield f"data: {json.dumps({'error': error_message})}\n\n"
        
        return StreamingResponse(generate_response(), media_type="text/plain")
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate chat response: {str(e)}")

# Define file upload endpoint
@app.post("/api/upload-file", response_model=FileUploadResponse)
async def upload_file(file: UploadFile = File(...)):
    try:
        # Validate file type
        if not is_supported_file(file.filename):
            supported_extensions = ', '.join(SUPPORTED_EXTENSIONS)
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type. Supported types: {supported_extensions}"
            )
        
        # Validate file size (10MB limit)
        content = await file.read()
        if len(content) > 10 * 1024 * 1024:  # 10MB
            raise HTTPException(status_code=400, detail="File size too large. Maximum size is 10MB")
        
        # Generate unique file ID
        file_id = str(uuid.uuid4())
        filename = file.filename
        
        # Initialize indexing status for both modes (consistent behavior)
        indexing_status[file_id] = {
            "status": "pending",
            "message": "File uploaded, indexing will start shortly..."
        }
        
        # Determine vector store type
        vector_store_type = "qdrant" if USE_QDRANT else "memory"
        logger.info(f"🔧 Vector store type for {file_id}: {vector_store_type}")
        
        # Store file metadata immediately for list_files endpoint
        file_metadata[file_id] = {
            "filename": filename,
            "vector_store_type": vector_store_type,
            "uploaded_at": datetime.now().isoformat()
        }
        logger.info(f"💾 Stored metadata for {file_id}: filename={filename}, vector_store_type={vector_store_type}")
        
        # For Vercel environments, metadata is stored in Qdrant during indexing
        # For local environments, save to disk
        if not is_vercel_environment():
            save_file_metadata()
        
        if (IS_READONLY and USE_BROWSER_STORAGE and not USE_QDRANT) or (is_vercel_environment() and not USE_QDRANT):
            # Browser storage mode: when read-only + browser storage enabled + no Qdrant, OR Vercel + no Qdrant
            # If Qdrant is available, use server storage mode even on Vercel
            import base64
            file_content_b64 = base64.b64encode(content).decode('utf-8')
            
            logger.info(f"📤 Sending file {file_id} to browser storage mode")
            logger.info(f"   - File size: {len(content)} bytes")
            logger.info(f"   - Base64 content length: {len(file_content_b64)} chars")
            logger.info(f"   - Browser storage enabled: {USE_BROWSER_STORAGE}")
            logger.info(f"   - Read-only mode: {IS_READONLY}")
            logger.info(f"   - Vercel environment: {is_vercel_environment()}")
            logger.info(f"   - Qdrant available: {USE_QDRANT}")
            
            return FileUploadResponse(
                filename=filename,
                file_id=file_id,
                message=f"{get_file_type(filename).upper()} uploaded successfully (stored in browser)",
                indexing_status="pending",
                use_browser_storage=True,
                file_content=file_content_b64,
                vector_store_type=vector_store_type
            )
        else:
            # Save the file to disk or start indexing directly
            if not IS_READONLY:
                file_path = UPLOADS_DIR / f"{file_id}_{filename}"
                with open(file_path, "wb") as buffer:
                    buffer.write(content)
            
            # Handle indexing differently for Vercel vs local
            logger.info(f"🔍 Environment check for {file_id}:")
            logger.info(f"   - IS_READONLY: {IS_READONLY}")
            logger.info(f"   - is_vercel_environment(): {is_vercel_environment()}")
            logger.info(f"   - USE_QDRANT: {USE_QDRANT}")
            logger.info(f"   - File content length: {len(content)} bytes")
            
            if is_vercel_environment():
                # For Vercel, run indexing synchronously to avoid task killing
                logger.info(f"🚀 Starting synchronous indexing for Vercel environment: {file_id}")
                try:
                    await index_file(content, file_id, filename)
                    logger.info(f"✅ Synchronous indexing completed for {file_id}")
                except Exception as e:
                    logger.error(f"❌ Synchronous indexing failed for {file_id}: {str(e)}")
                    indexing_status[file_id] = {
                        "status": "failed",
                        "message": f"Indexing failed: {str(e)}"
                    }
            else:
                # For local development, use background task
                logger.info(f"🚀 Starting background indexing for local environment: {file_id}")
                asyncio.create_task(index_file(content, file_id, filename))
            
            # After updating file_metadata[file_id] on upload, persist to disk
            file_metadata[file_id] = {
                "filename": filename,
                "vector_store_type": vector_store_type,
                "uploaded_at": datetime.now().isoformat()
            }
            try:
                save_file_metadata()
            except Exception as e:
                logger.warning(f"⚠️ Failed to save file_metadata: {e}")
            
            return FileUploadResponse(
                filename=filename,
                file_id=file_id,
                message=f"{get_file_type(filename).upper()} uploaded successfully",
                indexing_status="pending",
                use_browser_storage=False,
                vector_store_type=vector_store_type
            )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload file: {str(e)}")

# Define pre-indexed file endpoint (for browser storage mode)
@app.post("/api/pre-indexed-file")
async def accept_pre_indexed_file(request: PreIndexedFileRequest):
    """Accept pre-indexed file data from the frontend for browser-stored files"""
    import time
    start_time = time.time()
    
    try:
        logger.info(f"🔍 Processing pre-indexed file: {request.file_id} ({request.filename})")
        logger.info(f"📊 Received {len(request.chunks)} chunks")
        
        # Validate input
        if not request.chunks:
            raise ValueError("No chunks provided")
        
        # Get OpenAI API key from environment
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not configured on backend")
        
        # Create OpenAI client with timeout
        import httpx
        client = OpenAI(
            api_key=api_key,
            http_client=httpx.Client(timeout=30.0)  # 30 second timeout
        )
        
        logger.info(f"🔗 Creating embeddings for {len(request.chunks)} chunks...")
        embedding_start_time = time.time()
        
        # OPTIMIZATION: Create embeddings in smaller batches for better performance
        embeddings = []
        batch_size = 50  # Reduced from 100 to avoid timeouts
        
        for i in range(0, len(request.chunks), batch_size):
            batch_chunks = request.chunks[i:i + batch_size]
            try:
                response = client.embeddings.create(
                    input=batch_chunks,
                    model="text-embedding-3-small"
                )
                batch_embeddings = [data.embedding for data in response.data]
                embeddings.extend(batch_embeddings)
                logger.info(f"✅ Created embeddings batch {i//batch_size + 1}/{(len(request.chunks) + batch_size - 1)//batch_size} ({len(batch_chunks)} chunks)")
            except Exception as e:
                logger.error(f"❌ Error creating embeddings batch {i//batch_size + 1}: {str(e)}")
                # Update status to failed
                indexing_status[request.file_id] = {
                    "status": "failed",
                    "message": f"Embedding creation failed: {str(e)}"
                }
                raise
        
        embedding_time = time.time() - embedding_start_time
        logger.info(f"✅ Created {len(embeddings)} embeddings in {embedding_time:.2f} seconds")
        
        # Create vector database based on configuration
        vector_start_time = time.time()
        vector_db = create_vector_database(request.file_id)
        
        # OPTIMIZATION: Insert chunks and embeddings in smaller batches
        import numpy as np
        insert_batch_size = 25  # Reduced batch size for better performance
        
        for i in range(0, len(request.chunks), insert_batch_size):
            batch_chunks = request.chunks[i:i + insert_batch_size]
            batch_embeddings = embeddings[i:i + insert_batch_size]
            
            try:
                for j, (chunk, embedding) in enumerate(zip(batch_chunks, batch_embeddings)):
                    # Check if the vector database supports metadata by checking its type
                    if isinstance(vector_db, QdrantVectorDatabase):
                        # QdrantVectorDatabase supports metadata
                        vector_db.insert(chunk, np.array(embedding), metadata={"file_id": request.file_id, "filename": request.filename})
                    else:
                        # In-memory VectorDatabase doesn't support metadata
                        vector_db.insert(chunk, np.array(embedding))
                
                logger.info(f"✅ Inserted batch {i//insert_batch_size + 1}/{(len(request.chunks) + insert_batch_size - 1)//insert_batch_size} into vector database")
            except Exception as e:
                logger.error(f"❌ Error inserting batch {i//insert_batch_size + 1}: {str(e)}")
                # Update status to failed
                indexing_status[request.file_id] = {
                    "status": "failed",
                    "message": f"Vector database insertion failed: {str(e)}"
                }
                raise
        
        vector_time = time.time() - vector_start_time
        logger.info(f"✅ Vector database operations completed in {vector_time:.2f} seconds")
        
        # Store the vector database in memory for quick access
        vector_databases[request.file_id] = {
            "vector_db": vector_db,
            "chunks": request.chunks,
            "filename": request.filename
        }
        
        logger.info(f"📊 Stored file {request.file_id} in vector_databases")
        logger.info(f"📊 Current vector_databases keys: {list(vector_databases.keys())}")
        
        # Update indexing status
        indexing_status[request.file_id] = {
            "status": "completed",
            "message": f"Successfully indexed {len(request.chunks)} text chunks from browser storage"
        }
        
        logger.info(f"📊 Updated indexing status for {request.file_id}")
        logger.info(f"📊 Current indexing_status keys: {list(indexing_status.keys())}")
        
        total_time = time.time() - start_time
        logger.info(f"✅ Successfully indexed file {request.file_id} with {len(request.chunks)} chunks in {total_time:.2f} seconds")
        logger.info(f"📊 Performance breakdown:")
        logger.info(f"   - Embedding creation: {embedding_time:.2f}s ({embedding_time/total_time*100:.1f}%)")
        logger.info(f"   - Vector database: {vector_time:.2f}s ({vector_time/total_time*100:.1f}%)")
        logger.info(f"   - Other operations: {total_time-embedding_time-vector_time:.2f}s ({(total_time-embedding_time-vector_time)/total_time*100:.1f}%)")
        
        return {"message": "File indexed successfully", "chunks_count": len(request.chunks)}
        
    except Exception as e:
        total_time = time.time() - start_time
        logger.error(f"❌ Error indexing file {request.file_id} after {total_time:.2f} seconds: {str(e)}")
        # Update status to failed
        indexing_status[request.file_id] = {
            "status": "failed",
            "message": f"Indexing failed: {str(e)}"
        }
        raise HTTPException(status_code=500, detail=f"Failed to index file: {str(e)}")

# Define file deletion endpoint
@app.delete("/api/files/{file_id}")
async def delete_file(file_id: str):
    """Delete a file from memory and vector database"""
    try:
        deleted = False
        
        # Remove from vector database
        if file_id in vector_databases:
            vector_data = vector_databases[file_id]
            vector_db = vector_data["vector_db"]
            
            # If using Qdrant, we need to delete the collection
            if USE_QDRANT and hasattr(vector_db, 'client'):
                try:
                    collection_name = f"documents_{file_id}"
                    vector_db.client.delete_collection(collection_name=collection_name)
                    logger.info(f"✅ Deleted Qdrant collection for file {file_id}")
                except Exception as e:
                    logger.warning(f"⚠️ Warning: Could not delete Qdrant collection: {str(e)}")
                    # Try direct client deletion as fallback
                    try:
                        import os
                        from qdrant_client import QdrantClient
                        qdrant_url = os.getenv("QDRANT_URL")
                        qdrant_api_key = os.getenv("QDRANT_API_KEY")
                        if qdrant_url and qdrant_api_key:
                            direct_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
                            direct_client.delete_collection(collection_name=collection_name)
                            logger.info(f"✅ Deleted Qdrant collection using direct client for file {file_id}")
                    except Exception as e2:
                        logger.error(f"❌ Failed to delete Qdrant collection with direct client: {str(e2)}")
            
            del vector_databases[file_id]
            deleted = True
        
        # Remove from indexing status
        if file_id in indexing_status:
            del indexing_status[file_id]
            deleted = True
        
        # Remove from memory stored files (read-only mode)
        if file_id in memory_stored_files:
            del memory_stored_files[file_id]
            deleted = True
        
        # Remove from file metadata
        if file_id in file_metadata:
            del file_metadata[file_id]
            deleted = True
            # Save updated metadata to disk (for local environments)
            if not is_vercel_environment():
                save_file_metadata()
        
        # Remove from disk (non-read-only mode)
        if not IS_READONLY:
            # Find and delete the file from disk
            for extension in SUPPORTED_EXTENSIONS:
                file_path = UPLOADS_DIR / f"{file_id}_*{extension}"
                for matching_file in UPLOADS_DIR.glob(f"{file_id}_*{extension}"):
                    try:
                        matching_file.unlink()
                        deleted = True
                        break
                    except FileNotFoundError:
                        pass
        
        # Remove from chat sessions that reference this file
        for session_id in list(chat_sessions.keys()):
            session = chat_sessions[session_id]
            if file_id in session.file_ids:
                session.file_ids.remove(file_id)
                # If no files left in session, remove the session
                if not session.file_ids:
                    del chat_sessions[session_id]
        
        if deleted:
            return {"message": f"File {file_id} deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail="File not found")
            
    except Exception as e:
        logger.error(f"Error deleting file {file_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete file: {str(e)}")

# Define delete all files endpoint
@app.delete("/api/files")
async def delete_all_files():
    """Delete all files from memory and vector database"""
    try:
        deleted_count = 0
        
        # Get all file IDs
        file_ids = list(vector_databases.keys())
        
        for file_id in file_ids:
            try:
                # Remove from vector database
                if file_id in vector_databases:
                    vector_data = vector_databases[file_id]
                    vector_db = vector_data["vector_db"]
                    
                    # If using Qdrant, we need to delete the collection
                    if USE_QDRANT and hasattr(vector_db, 'client'):
                        try:
                            collection_name = f"documents_{file_id}"
                            vector_db.client.delete_collection(collection_name=collection_name)
                            logger.info(f"✅ Deleted Qdrant collection for file {file_id}")
                        except Exception as e:
                            logger.warning(f"⚠️ Warning: Could not delete Qdrant collection: {str(e)}")
                            # Try direct client deletion as fallback
                            try:
                                import os
                                from qdrant_client import QdrantClient
                                qdrant_url = os.getenv("QDRANT_URL")
                                qdrant_api_key = os.getenv("QDRANT_API_KEY")
                                if qdrant_url and qdrant_api_key:
                                    direct_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
                                    direct_client.delete_collection(collection_name=collection_name)
                                    logger.info(f"✅ Deleted Qdrant collection using direct client for file {file_id}")
                            except Exception as e2:
                                logger.error(f"❌ Failed to delete Qdrant collection with direct client: {str(e2)}")
                    
                    del vector_databases[file_id]
                
                # Remove from indexing status
                if file_id in indexing_status:
                    del indexing_status[file_id]
                
                # Remove from memory stored files (read-only mode)
                if file_id in memory_stored_files:
                    del memory_stored_files[file_id]
                
                # Remove from file metadata
                if file_id in file_metadata:
                    del file_metadata[file_id]
                
                deleted_count += 1
                
            except Exception as e:
                logger.warning(f"Warning: Failed to delete file {file_id}: {e}")
        
        # Clear all chat sessions
        chat_sessions.clear()
        
        # For Vercel environments, clear all Qdrant collections
        if is_vercel_environment() and USE_QDRANT:
            qdrant_deleted = clear_all_qdrant_collections()
            logger.info(f"🗑️ Cleared {qdrant_deleted} Qdrant collections for Vercel")
        
        # Save updated metadata to disk (for local environments)
        if not is_vercel_environment():
            save_file_metadata()
        
        # Remove from disk (non-read-only mode)
        if not IS_READONLY:
            try:
                for file_path in UPLOADS_DIR.glob("*"):
                    file_path.unlink()
            except Exception as e:
                logger.warning(f"Warning: Failed to delete files from disk: {e}")
        
        return {"message": f"Deleted {deleted_count} files successfully"}
        
    except Exception as e:
        logger.error(f"Error deleting all files: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete all files: {str(e)}")

# Test endpoint for PDF processing
@app.post("/api/test-pdf")
async def test_pdf_processing(file: UploadFile = File(...)):
    """Test endpoint for PDF processing"""
    try:
        content = await file.read()
        file_id = str(uuid.uuid4())
        filename = file.filename
        
        # Create temp file
        temp_file_path = f"/tmp/{file_id}_{filename}"
        with open(temp_file_path, 'wb') as f:
            f.write(content)
        
        # Test PDF processing
        documents = []
        method = "Failed"
        
        # Method 1: Try PDFLoader
        try:
            pdf_loader = PDFLoader(temp_file_path)
            documents = pdf_loader.load_documents()
            method = "PDFLoader"
        except Exception as e:
            logger.warning(f"PDFLoader failed: {str(e)}")
            
            # Method 2: Try PyPDF2
            try:
                import PyPDF2
                with open(temp_file_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    documents = []
                    for page_num in range(len(pdf_reader.pages)):
                        page = pdf_reader.pages[page_num]
                        text = page.extract_text()
                        if text.strip():
                            documents.append(text)
                    method = "PyPDF2"
            except Exception as e2:
                logger.warning(f"PyPDF2 failed: {str(e2)}")
                
                # Method 3: Basic extraction
                try:
                    import PyPDF2
                    with open(temp_file_path, 'rb') as file:
                        pdf_reader = PyPDF2.PdfReader(file)
                        documents = []
                        for page in pdf_reader.pages:
                            text = page.extract_text()
                            if text.strip():
                                documents.append(text)
                    method = "Basic extraction"
                except Exception as e3:
                    logger.warning(f"Basic extraction failed: {str(e3)}")
                    documents = []
                    method = "Failed"
        
        # Clean up temp file
        try:
            os.remove(temp_file_path)
        except:
            pass
        
        return {
            "filename": filename,
            "file_size": len(content),
            "extraction_method": method,
            "pages_extracted": len(documents),
            "total_text_length": sum(len(doc) for doc in documents),
            "sample_text": documents[0][:200] + "..." if documents else "No text extracted"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF processing test failed: {str(e)}")

def clear_all_qdrant_collections():
    """Clear all Qdrant collections for Vercel environments"""
    if not is_vercel_environment() or not USE_QDRANT:
        return
    
    try:
        import os
        from qdrant_client import QdrantClient
        
        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")
        
        if not qdrant_url or not qdrant_api_key:
            logger.warning("⚠️ Qdrant credentials not found, skipping collection clearing")
            return
        
        client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        collections = client.get_collections()
        
        deleted_count = 0
        for collection in collections.collections:
            collection_name = collection.name
            if collection_name.startswith("documents_"):
                try:
                    client.delete_collection(collection_name=collection_name)
                    deleted_count += 1
                    logger.info(f"✅ Deleted Qdrant collection: {collection_name}")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to delete collection {collection_name}: {str(e)}")
        
        logger.info(f"🗑️ Cleared {deleted_count} Qdrant collections")
        return deleted_count
    except Exception as e:
        logger.error(f"❌ Failed to clear Qdrant collections: {str(e)}")
        return 0

def refresh_file_metadata_from_qdrant(file_id: str):
    """Refresh file metadata from Qdrant for a specific file"""
    if not is_vercel_environment() or not USE_QDRANT:
        return None
    
    try:
        import os
        from qdrant_client import QdrantClient
        
        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")
        
        if not qdrant_url or not qdrant_api_key:
            return None
        
        client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        collection_name = f"documents_{file_id}"
        
        # Check if collection exists
        try:
            collection_info = client.get_collection(collection_name=collection_name)
            if collection_info.points_count == 0:
                return None
        except Exception:
            return None
        
        # Try to get metadata from collection
        try:
            points = client.scroll(
                collection_name=collection_name,
                limit=1,
                with_payload=True
            )
            
            if points[0] and len(points[0]) > 0:
                point = points[0][0]
                metadata = point.payload
                if metadata and "filename" in metadata:
                    filename = metadata["filename"]
                    file_metadata[file_id] = {
                        "filename": filename,
                        "vector_store_type": "qdrant",
                        "uploaded_at": datetime.now().isoformat()
                    }
                    logger.info(f"🔄 Refreshed metadata for {file_id}: {filename}")
                    return filename
        except Exception as e:
            logger.warning(f"⚠️ Failed to refresh metadata for {file_id}: {str(e)}")
        
        return None
    except Exception as e:
        logger.warning(f"⚠️ Failed to refresh metadata for {file_id}: {str(e)}")
        return None

def store_metadata_in_qdrant(file_id: str, filename: str):
    """Store metadata in Qdrant collection for persistent access"""
    if not is_vercel_environment() or not USE_QDRANT:
        return
    
    try:
        import os
        from qdrant_client import QdrantClient
        
        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")
        
        if not qdrant_url or not qdrant_api_key:
            return
        
        client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        collection_name = f"documents_{file_id}"
        
        # Create a metadata point with the filename
        metadata_point = {
            "id": f"{file_id}_metadata",
            "vector": [0.0] * 1536,  # Dummy vector
            "payload": {
                "filename": filename,
                "file_id": file_id,
                "metadata_type": "file_info",
                "uploaded_at": datetime.now().isoformat()
            }
        }
        
        # Upsert the metadata point
        client.upsert(
            collection_name=collection_name,
            points=[metadata_point]
        )
        
        logger.info(f"💾 Stored metadata in Qdrant for {file_id}: {filename}")
    except Exception as e:
        logger.warning(f"⚠️ Failed to store metadata in Qdrant for {file_id}: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
