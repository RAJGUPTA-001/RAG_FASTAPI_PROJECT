# chroma_utils.py - Docker-compatible version

# Disable ChromaDB telemetry first
import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"

# Imports
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredHTMLLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from typing import List
from langchain_core.documents import Document
from pathlib import Path
from google import genai
from google.genai import types
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import time
import tempfile
import logging
from dataclasses import dataclass
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables with Docker compatibility
load_dotenv()

# Environment variables with defaults
embed_model = os.getenv("EMBEDDING_MODEL")
api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

if not api_key:
    logger.warning("⚠️ No API key found. Set GEMINI_API_KEY or GOOGLE_API_KEY environment variable")
else:
    os.environ["GOOGLE_API_KEY"] = api_key

@dataclass
class UploadLimits:
    """Configuration for upload limits"""
    max_documents: int = 20           # Maximum documents per batch
    max_pages_per_doc: int = 1000     # Maximum pages per document
    pdf_max_size: int = 50 * 1024 * 1024    # 50 MB for PDFs
    docx_max_size: int = 25 * 1024 * 1024   # 25 MB for DOCX
    txt_max_size: int = 10 * 1024 * 1024    # 10 MB for text files
    html_max_size: int = 15 * 1024 * 1024   # 15 MB for HTML
    
    def get_max_size_for_file(self, file_path: str) -> int:
        """Get maximum allowed size for a specific file type"""
        if file_path.endswith('.pdf'):
            return self.pdf_max_size
        elif file_path.endswith('.docx'):
            return self.docx_max_size
        elif file_path.endswith('.txt'):
            return self.txt_max_size
        elif file_path.endswith('.html'):
            return self.html_max_size
        else:
            return self.pdf_max_size  # Default to PDF limit

upload_limits = UploadLimits()

def format_file_size(size_bytes: int) -> str:
    """Convert bytes to human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"

def get_document_page_count(file_path: str) -> int:
    """Get page count for any supported document type"""
    ext = Path(file_path).suffix.lower()
    
    if ext == '.pdf':
        return get_pdf_page_count(file_path)
    elif ext == '.docx':
        return get_docx_page_count(file_path)
    else:
        # Text and HTML files are considered single page
        return 1

def get_pdf_page_count(file_path: str) -> int:
    """Get page count for PDF files using langchain loader"""
    try:
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        return len(documents)
    except Exception as e:
        logger.warning(f"Could not get accurate page count for {file_path}, estimating from file size")
        # Fallback: estimate pages from file size (rough estimate: 100KB per page)
        file_size = os.path.getsize(file_path)
        estimated_pages = max(1, file_size // (100 * 1024))
        return estimated_pages

def get_docx_page_count(file_path: str) -> int:
    """Get approximate page count for DOCX files"""
    try:
        loader = Docx2txtLoader(file_path)
        documents = loader.load()
        
        if documents:
            content = documents[0].page_content
            # Estimate pages: ~500 words per page, ~5 chars per word
            char_count = len(content)
            estimated_pages = max(1, char_count // (500 * 5))
            return estimated_pages
        return 1
    except Exception as e:
        logger.warning(f"Could not get page count for {file_path}: {e}")
        return 1

def validate_single_document(file_path: str, limits: UploadLimits) -> dict:
    """Validate a single document against all criteria"""
    result = {
        "valid": True,
        "errors": [],
        "file_path": file_path,
        "file_size": 0,
        "page_count": 0,
        "file_type": ""
    }
    
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            result["valid"] = False
            result["errors"].append(f"File not found: {file_path}")
            return result
        
        # Get file info
        file_size = os.path.getsize(file_path)
        file_type = Path(file_path).suffix.lower()
        result["file_size"] = file_size
        result["file_type"] = file_type
        
        # Check supported file types
        if file_type not in ['.pdf', '.docx', '.txt', '.html']:
            result["valid"] = False
            result["errors"].append(f"Unsupported file type: {file_type}")
            return result
        
        # Check file size
        max_size = limits.get_max_size_for_file(file_path)
        if file_size > max_size:
            result["valid"] = False
            result["errors"].append(
                f"File size {format_file_size(file_size)} exceeds maximum {format_file_size(max_size)} for {file_type} files"
            )
        
        # Check page count
        try:
            page_count = get_document_page_count(file_path)
            result["page_count"] = page_count
            
            if page_count > limits.max_pages_per_doc:
                result["valid"] = False
                result["errors"].append(
                    f"Document has {page_count} pages, exceeding maximum of {limits.max_pages_per_doc} pages"
                )
        except Exception as e:
            result["valid"] = False
            result["errors"].append(f"Could not determine page count: {str(e)}")
        
        return result
        
    except Exception as e:
        result["valid"] = False
        result["errors"].append(f"Validation error: {str(e)}")
        return result

def validate_document_batch(file_paths: List[str], limits: UploadLimits = None) -> dict:
    """Validate a batch of documents against upload limits"""
    if limits is None:
        limits = upload_limits
    
    validation_result = {
        "valid": True,
        "total_documents": len(file_paths),
        "total_pages": 0,
        "total_size": 0,
        "errors": [],
        "documents": [],
        "summary": {}
    }
    
    # Check document count limit
    if len(file_paths) > limits.max_documents:
        validation_result["valid"] = False
        validation_result["errors"].append(
            f"Too many documents: {len(file_paths)} exceeds maximum of {limits.max_documents}"
        )
        return validation_result
    
    # Validate each document
    for file_path in file_paths:
        doc_result = validate_single_document(file_path, limits)
        validation_result["documents"].append(doc_result)
        
        if not doc_result["valid"]:
            validation_result["valid"] = False
            validation_result["errors"].extend(doc_result["errors"])
        else:
            validation_result["total_pages"] += doc_result["page_count"]
            validation_result["total_size"] += doc_result["file_size"]
    
    # Create summary
    validation_result["summary"] = {
        "total_documents": validation_result["total_documents"],
        "total_pages": validation_result["total_pages"],
        "total_size_formatted": format_file_size(validation_result["total_size"]),
        "valid_documents": sum(1 for doc in validation_result["documents"] if doc["valid"]),
        "invalid_documents": sum(1 for doc in validation_result["documents"] if not doc["valid"])
    }
    
    return validation_result



# Initialize text splitter and embedding function
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, 
    chunk_overlap=200, 
    length_function=len
)

embedding_function = GoogleGenerativeAIEmbeddings(
    model=embed_model,
    task_type="retrieval_document"
)

# Docker-compatible ChromaDB initialization
def initialize_vectorstore():
    """Initialize vectorstore with Docker compatibility"""
    chroma_host = os.getenv("CHROMA_HOST", "localhost")
    chroma_port = int(os.getenv("CHROMA_PORT", "8001"))
    
    try:
        if chroma_host != "localhost":
            import chromadb
            from chromadb.config import Settings
            
            logger.info(f"🐳 Connecting to external ChromaDB at {chroma_host}:{chroma_port}")
            
            client = chromadb.HttpClient(
                host=chroma_host,
                port=chroma_port,
                settings=Settings(allow_reset=True)
            )
            
            vectorstore_instance = Chroma(
                client=client,
                collection_name="document_collection",
                embedding_function=embedding_function
            )
            
            # Test connection
            vectorstore_instance.get(limit=1)
            logger.info(f"✅ Connected to external ChromaDB successfully")
            return vectorstore_instance
            
        else:
            raise Exception("Using local ChromaDB")
            
    except Exception as e:
        logger.info(f"🔄 Using local ChromaDB: {e}")
        
        # Use Docker-friendly path
        persist_dir = "/app/chroma_db" if os.path.exists("/app") else "./chroma_db"
        os.makedirs(persist_dir, exist_ok=True)
        
        vectorstore_instance = Chroma(
            persist_directory=persist_dir,
            embedding_function=embedding_function,
            collection_name="document_collection"
        )
        
        logger.info(f"✅ Using local ChromaDB: {persist_dir}")
        return vectorstore_instance

# Initialize vectorstore
vectorstore = initialize_vectorstore()

def validate_file_path(file_path: str) -> str:
    """Validate file exists and is readable, return actual path - Docker compatible"""
    if not file_path:
        raise ValueError("File path is empty")
    
    if not os.path.exists(file_path):
        # Try to find in temp directory
        temp_dir = tempfile.gettempdir()
        filename = os.path.basename(file_path)
        temp_path = os.path.join(temp_dir, filename)
        
        if os.path.exists(temp_path):
            logger.info(f"📍 Found file in temp directory: {temp_path}")
            return temp_path
        
        # Try Docker-specific paths
        docker_paths = [
            os.path.join("/app/uploads", filename),
            os.path.join("/app/temp", filename),
            os.path.join("/app", filename),
            # Fallback to current directory paths
            os.path.join(os.getcwd(), filename),
            os.path.join(os.getcwd(), "uploads", filename),
            os.path.join(os.getcwd(), "temp", filename)
        ]
        
        for path in docker_paths:
            if os.path.exists(path):
                logger.info(f"📍 Found file at: {path}")
                return path
        
        raise FileNotFoundError(f"File not found: {file_path}")
    
    if not os.access(file_path, os.R_OK):
        raise PermissionError(f"Cannot read file: {file_path}")
    
    file_size = os.path.getsize(file_path)
    if file_size == 0:
        raise ValueError(f"File is empty: {file_path}")
    
    logger.info(f"✓ File validated: {file_path} ({file_size} bytes)")
    return file_path

def safe_load_text_file(file_path: str) -> List[Document]:
    """Safely load text file with multiple encoding attempts"""
    encodings = ['utf-8', 'utf-8-sig', 'latin1', 'cp1252', 'iso-8859-1']
    
    # Try TextLoader with different encodings
    for encoding in encodings:
        try:
            loader = TextLoader(file_path, encoding=encoding)
            documents = loader.load()
            
            if documents and documents[0].page_content.strip():
                logger.info(f"✓ Successfully loaded {file_path} with {encoding}")
                return documents
            else:
                logger.warning(f"⚠️ File loaded but empty with {encoding}")
                
        except UnicodeDecodeError:
            logger.debug(f"❌ Encoding {encoding} failed for {file_path}")
            continue
        except Exception as e:
            logger.debug(f"💥 Error with {encoding} for {file_path}: {e}")
            continue
    
    # Manual fallback - read file directly and create Document
    logger.info(f"🔄 Attempting manual text loading for {file_path}")
    
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                content = f.read()
            
            if content.strip():
                doc = Document(
                    page_content=content,
                    metadata={"source": file_path, "encoding": encoding}
                )
                logger.info(f"✓ Manual loading successful with {encoding}")
                return [doc]
                
        except UnicodeDecodeError:
            continue
        except Exception as e:
            logger.debug(f"Manual loading error with {encoding}: {e}")
            continue
    
    raise Exception(f"Failed to load {file_path} with any encoding method")

def load_and_split_document(file_path: str) -> List[Document]:
    """Load and split document with comprehensive error handling"""
    try:
        # Validate file first
        validated_path = validate_file_path(file_path)
        if validated_path != file_path:
            file_path = validated_path
        
        logger.info(f"🔄 Loading document: {file_path}")
        
        # Load based on file type
        if file_path.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            
        elif file_path.endswith('.docx'):
            loader = Docx2txtLoader(file_path)
            documents = loader.load()
            
        elif file_path.endswith('.html'):
            loader = UnstructuredHTMLLoader(file_path)
            documents = loader.load()
            
        elif file_path.endswith('.txt'):
            documents = safe_load_text_file(file_path)
            
        else:
            raise ValueError(f"Unsupported file type: {file_path}")
        
        # Validate documents were loaded
        if not documents:
            raise ValueError(f"No content loaded from {file_path}")
        
        if not any(doc.page_content.strip() for doc in documents):
            raise ValueError(f"All documents from {file_path} are empty")
        
        logger.info(f"✓ Loaded {len(documents)} document(s) from {file_path}")
        
        # Split documents
        splits = text_splitter.split_documents(documents)
        logger.info(f"✓ Created {len(splits)} text chunks")
        
        return splits
        
    except Exception as e:
        error_msg = f"Failed to load and split document {file_path}: {str(e)}"
        logger.error(f"❌ {error_msg}")
        raise Exception(error_msg)

def index_document_to_chroma_batch(splits: List[Document], file_id: int, batch_size: int = 50) -> bool:
    """Index document chunks in batches with quota handling"""
    try:
        total_batches = (len(splits) + batch_size - 1) // batch_size
        successful_batches = 0
        
        logger.info(f"🔄 Indexing {len(splits)} chunks in {total_batches} batches of {batch_size}...")
        
        for i in range(0, len(splits), batch_size):
            batch = splits[i:i + batch_size]
            batch_num = i // batch_size + 1
            max_batch_retries = 3
            
            for attempt in range(max_batch_retries):
                try:
                    vectorstore.add_documents(batch)
                    successful_batches += 1
                    logger.info(f"✓ Batch {batch_num}/{total_batches}: Indexed {len(batch)} chunks")
                    break
                    
                except Exception as batch_error:
                    error_str = str(batch_error)
                    
                    if "429" in error_str or "quota" in error_str.lower():
                        if attempt < max_batch_retries - 1:
                            wait_time = 30 * (2 ** attempt)  # Exponential backoff
                            logger.warning(f"⚠️ Quota limit hit at batch {batch_num}. Waiting {wait_time}s...")
                            time.sleep(wait_time)
                            continue
                        else:
                            logger.error(f"❌ Quota limit exceeded for batch {batch_num} after {max_batch_retries} attempts")
                            raise batch_error
                    else:
                        logger.error(f"❌ Batch {batch_num} failed: {batch_error}")
                        raise batch_error
            
            # Small delay between batches to be gentle on API
            time.sleep(1)
        
        logger.info(f"✅ Successfully indexed {successful_batches}/{total_batches} batches for file_id {file_id}")
        return successful_batches == total_batches
        
    except Exception as e:
        logger.error(f"❌ Batch indexing failed: {e}")
        return False

def index_document_to_chroma(file_path: str, file_id: int, max_retries: int = 3) -> bool:
    """Index document with retry logic and comprehensive error handling"""
    
    for attempt in range(max_retries):
        try:
            logger.info(f"📊 Indexing attempt {attempt + 1}/{max_retries} for file_id {file_id}")
            
            # Load and split document
            splits = load_and_split_document(file_path)
            
            if not splits:
                raise ValueError("No text chunks created from document")
            
            # Add metadata to each split
            for idx, split in enumerate(splits):
                split.metadata['file_id'] = file_id
                split.metadata['chunk_index'] = idx
                split.metadata['total_chunks'] = len(splits)
                split.metadata['source_file'] = os.path.basename(file_path)
            
            # Index in batches
            success = index_document_to_chroma_batch(splits, file_id)
            
            if success:
                logger.info(f"✅ Successfully indexed file_id {file_id}: {len(splits)} chunks")
                return True
            else:
                raise Exception("Batch indexing failed")
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"❌ Attempt {attempt + 1} failed: {error_msg}")
            
            # Don't retry for file-related errors
            if any(term in error_msg.lower() for term in [
                "not found", "no such file", "permission", "empty", 
                "unsupported file type", "failed to load"
            ]):
                logger.error(f"💥 File error - not retrying: {error_msg}")
                break
            
            # Retry with exponential backoff for other errors
            if attempt < max_retries - 1:
                wait_time = 10 * (2 ** attempt)
                logger.info(f"⏳ Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
                continue
    
    logger.error(f"💥 Failed to index file_id {file_id} after {max_retries} attempts")
    return False

def delete_doc_from_chroma(file_id: int) -> bool:
    """Delete document from Chroma with error handling"""
    try:
        logger.info(f"🗑️ Deleting documents for file_id {file_id}")
        
        # Check if documents exist
        docs = vectorstore.get(where={"file_id": file_id})
        
        if not docs or not docs.get('ids'):
            logger.warning(f"⚠️ No documents found for file_id {file_id}")
            return True
        
        doc_count = len(docs['ids'])
        logger.info(f"📋 Found {doc_count} document chunks for file_id {file_id}")
        
        # Delete documents
        vectorstore._collection.delete(where={"file_id": file_id})
        
        # Verify deletion
        remaining_docs = vectorstore.get(where={"file_id": file_id})
        if remaining_docs and remaining_docs.get('ids'):
            logger.warning(f"⚠️ Some documents may not have been deleted for file_id {file_id}")
        
        logger.info(f"✅ Deleted documents for file_id {file_id}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error deleting document with file_id {file_id}: {str(e)}")
        return False

def get_document_info(file_id: int) -> dict:
    """Get information about a document in the vector store"""
    try:
        docs = vectorstore.get(where={"file_id": file_id})
        
        if not docs or not docs.get('ids'):
            return {"exists": False, "chunk_count": 0}
        
        chunk_count = len(docs['ids'])
        
        # Get metadata from first document
        metadata = docs.get('metadatas', [{}])[0] if docs.get('metadatas') else {}
        
        return {
            "exists": True,
            "chunk_count": chunk_count,
            "source_file": metadata.get('source_file', 'unknown'),
            "total_chunks": metadata.get('total_chunks', chunk_count)
        }
        
    except Exception as e:
        logger.error(f"❌ Error getting document info for file_id {file_id}: {str(e)}")
        return {"exists": False, "error": str(e)}

def list_all_documents() -> List[dict]:
    """List all documents in the vector store"""
    try:
        # Get all documents
        all_docs = vectorstore.get()
        
        if not all_docs or not all_docs.get('metadatas'):
            return []
        
        # Group by file_id
        file_info = {}
        for metadata in all_docs['metadatas']:
            if metadata:
                file_id = metadata.get('file_id')
                if file_id is not None:
                    if file_id not in file_info:
                        file_info[file_id] = {
                            'file_id': file_id,
                            'source_file': metadata.get('source_file', 'unknown'),
                            'chunk_count': 0
                        }
                    file_info[file_id]['chunk_count'] += 1
        
        return list(file_info.values())
        
    except Exception as e:
        logger.error(f"❌ Error listing documents: {str(e)}")
        return []

def test_file_loading(file_path: str) -> bool:
    """Test file loading with detailed output"""
    try:
        logger.info(f"🧪 Testing file loading for: {file_path}")
        splits = load_and_split_document(file_path)
        logger.info(f"✅ Test successful: {len(splits)} chunks created")
        
        # Show first chunk preview
        if splits:
            first_chunk = splits[0].page_content[:200]
            logger.info(f"📄 First chunk preview: {first_chunk}...")
        
        return True
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False

def get_vectorstore_stats() -> dict:
    """Get statistics about the vector store"""
    try:
        all_docs = vectorstore.get()
        
        if not all_docs:
            return {"total_chunks": 0, "total_documents": 0}
        
        total_chunks = len(all_docs.get('ids', []))
        
        # Count unique file_ids
        file_ids = set()
        if all_docs.get('metadatas'):
            for metadata in all_docs['metadatas']:
                if metadata and metadata.get('file_id') is not None:
                    file_ids.add(metadata['file_id'])
        
        return {
            "total_chunks": total_chunks,
            "total_documents": len(file_ids),
            "file_ids": sorted(list(file_ids))
        }
        
    except Exception as e:
        logger.error(f"❌ Error getting vectorstore stats: {str(e)}")
        return {"error": str(e)}

def health_check() -> dict:
    """Health check for Docker containers"""
    try:
        # Test vectorstore connection
        stats = get_vectorstore_stats()
        
        # Test embedding function
        test_embedding = embedding_function.embed_query("test")
        
        return {
            "status": "healthy",
            "vectorstore": "connected",
            "embeddings": "working",
            "stats": stats
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }

# Example usage and testing
if __name__ == "__main__":
    # Test the functions
    print("🚀 Vector store module loaded successfully")
    
    # Show current stats
    stats = get_vectorstore_stats()
    print(f"📊 Current vectorstore stats: {stats}")
    
    # List existing documents
    docs = list_all_documents()
    print(f"📋 Existing documents: {len(docs)}")
    for doc in docs[:5]:  # Show first 5
        print(f"  File ID {doc['file_id']}: {doc['source_file']} ({doc['chunk_count']} chunks)")
    
    # Health check
    health = health_check()
    print(f"🏥 Health status: {health['status']}")
