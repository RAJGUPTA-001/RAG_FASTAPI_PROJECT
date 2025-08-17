
# RAG Document Retrieval and Query API

A comprehensive **Retrieval-Augmented Generation (RAG)** system built with FastAPI that enables document upload, processing, and intelligent querying using vector embeddings and language models.

## 🚀 Features

- **Multi-format Document Support**: PDF, DOCX, TXT, and HTML files
- **Vector-based Document Retrieval**: ChromaDB integration for semantic search
- **Intelligent Query Processing**: RAG pipeline with LangChain
- **Session Management**: Persistent chat history across conversations
- **Docker Support**: Container-ready with external ChromaDB support
- **Comprehensive API**: RESTful endpoints for all operations
- **File Validation**: Size, page count, and format validation
- **Batch Processing**: Multiple document upload with error handling

## 📋 Table of Contents

- [Setup and Installation](#setup-and-installation)
- [API Usage and Testing Guidelines](#api-usage-and-testing-guidelines)
- [Configuration for Different LLM Providers](#configuration-for-different-llm-providers)
- [Testing](#testing)
- [Docker Deployment](#docker-deployment)
- [Architecture](#architecture)
- [Contributing](#contributing)

## 🛠️ Setup and Installation Instructions

### Prerequisites

- Python 3.8+  (developed on ==3.11.9) 
- Git
- Virtual environment tool (venv, conda, etc.) (preferred)

### 1. Clone the Repository

```bash
git clone <https://github.com/RAJGUPTA-001/RAG_FASTAPI_PROJECT.git>
cd rag-document-api
```




### 2. Create and Activate Virtual Environment

```bash
# Using venv
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate

# Using conda
conda create -n rag-env python=3.11.9
conda activate rag-env
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Environment Configuration

Create a `.env` file in the project root:

```env
# Required API Keys
GEMINI_API_KEY=your_gemini_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
GROQ_API_KEY=your_groq_api_key_here

# Model Configuration
LLM_for_chat=openai/gpt-oss-120b
EMBEDDING_MODEL=models/embedding-001

# ChromaDB Configuration
CHROMA_HOST=localhost
CHROMA_PORT=8001

# Optional: Disable telemetry
ANONYMIZED_TELEMETRY=False
```

### 5. Initialize the Application

```bash
# Run the FastAPI server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 6. Access the Application

- **Web Interface**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Alternative Documentation**: http://localhost:8000/redoc

## 📚 API Usage and Testing Guidelines

### Core Endpoints

#### 🏥 Health Check
```bash
GET /health
```
**Response**: System health status and timestamp

#### 📄 Document Management

**Upload Documents**
```bash
POST /upload-docs
Content-Type: multipart/form-data

# Upload multiple files
curl -X POST "http://localhost:8000/upload-docs" \
  -F "files=@document1.pdf" \
  -F "files=@document2.docx"
```

**List All Documents**
```bash
GET /list-docs
```

**Delete Document**
```bash
POST /delete-doc
Content-Type: application/json

{
  "file_id": 1
}
```

#### 💬 Chat Interface

**Send Query**
```bash
POST /chat
Content-Type: application/json

{
  "question": "What is machine learning?",
  "session_id": "user-session-123",
  "model": "openai/gpt-oss-120b"
}
```

#### 📊 Metadata and Limits

**Get Upload Limits**
```bash
GET /upload-limits
```

**Get All Documents Metadata**
```bash
GET /api/documents/metadata
```

**Get Specific Document Metadata**
```bash
GET /api/documents/{file_id}/metadata
```

#### 🔍 File Validation

**Validate Files (Dry Run)**
```bash
POST /validate-files
Content-Type: multipart/form-data

# Test files without uploading
curl -X POST "http://localhost:8000/validate-files" \
  -F "files=@test.pdf"
```

### 📋 Upload Limits and Constraints

| File Type | Max Size | Max Pages | Max Files per Batch |
|-----------|----------|-----------|-------------------|
| PDF       | 50 MB    | 1000      | 20                |
| DOCX      | 25 MB    | 1000      | 20                |
| TXT       | 10 MB    | N/A       | 20                |
| HTML      | 15 MB    | N/A       | 20                |

### 🧪 Testing Examples

**Python Testing Script**
```python
import requests

base_url = "http://localhost:8000"

# Test health
health = requests.get(f"{base_url}/health")
print(f"Health: {health.json()}")

# Test chat
chat_response = requests.post(f"{base_url}/chat", json={
    "question": "Explain artificial intelligence",
    "session_id": "test-session"
})
print(f"Chat Response: {chat_response.json()}")

# Test document upload
with open("sample.pdf", "rb") as f:
    files = {"files": ("sample.pdf", f, "application/pdf")}
    upload_response = requests.post(f"{base_url}/upload-docs", files=files)
    print(f"Upload Response: {upload_response.json()}")
```

## ⚙️ Configuration for Different LLM Providers

### 🤖 Groq Configuration (Default)

```env
# Groq API (Default)
GROQ_API_KEY=your_groq_api_key
LLM_for_chat=openai/gpt-oss-120b
```

**Supported Groq Models:**
- `openai/gpt-oss-120b` (default)
- `llama2-70b-4096`
- `mixtral-8x7b-32768`
- `gemma-7b-it`

### 🔍 Google Gemini Configuration

```env
# Google Gemini
GOOGLE_API_KEY=your_google_api_key
GEMINI_API_KEY=your_gemini_api_key
EMBEDDING_MODEL=models/embedding-001
```

### 🔧 Custom LLM Provider Setup

To add a new LLM provider, modify `langchain_utils.py`:

```python
# Example: Adding OpenAI GPT
from langchain_openai import ChatOpenAI

def get_rag_chain(model_name: str):
    if model_name.startswith("gpt-"):
        llm = ChatOpenAI(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            model_name=model_name,
            temperature=0
        )
    elif model_name.startswith("claude-"):
        # Add Anthropic Claude support
        from langchain_anthropic import ChatAnthropic
        llm = ChatAnthropic(
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            model=model_name
        )
    else:
        # Default to Groq
        llm = ChatGroq(
            groq_api_key=GROQ_API_KEY,
            model_name=model_name,
            temperature=0
        )
    # ... rest of the chain setup
```

### 📊 Embedding Model Configuration

**Google Embeddings (Default)**
```env
EMBEDDING_MODEL=models/embedding-001
```

**OpenAI Embeddings**
```python
# In chroma_utils.py
from langchain_openai import OpenAIEmbeddings

embedding_function = OpenAIEmbeddings(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    model="text-embedding-ada-002"
)
```


## 🧪 Testing

### Running Tests

```bash
# Run all tests
pytest -v


# Run with coverage
pip install pytest-cov
pytest --cov=. --cov-report=html

# Run specific test files
pytest tests/test_api.py
pytest tests/test_chroma_utils.py
pytest tests/test_db_utils.py
pytest tests/test_integration.py
```

### Test Structure

```
tests/
├── __init__.py
├── test_api.py              # API endpoint tests
├── test_chroma_utils.py     # Document processing tests
├── test_db_utils.py         # Database operation tests
├── test_integration.py      # End-to-end workflow tests
├── test_config.py           # Test configuration
└── fixtures/                # Test data files
```

### Test Configuration

Tests use environment variables from `tests/test_config.py`. For CI/CD, set:

```bash
export GEMINI_API_KEY="test-key"
export GROQ_API_KEY="test-key"
export LLM_for_chat="openai/gpt-oss-120b"
```

## 🐳 Docker Deployment

### Local Development with Docker

```bash
# Build the image
docker build -t rag-api .

# Run with environment file
docker run --env-file .env -p 8000:8000 rag-api
```

### Docker Compose with External ChromaDB

```yaml
version: '3.8'

services:
  # ChromaDB service
  chromadb:
    image: chromadb/chroma:latest
    container_name: rag-chromadb
    ports:
      - "8001:8000"
    volumes:
      - chroma_data:/chroma/chroma
    environment:
      - CHROMA_SERVER_HOST=0.0.0.0
      - CHROMA_SERVER_HTTP_PORT=8000
    networks:
      - rag-network
    restart: unless-stopped

  # Main RAG application
  rag-app:
    build: .
    container_name: rag-app
    ports:
      - "8000:8000"
    volumes:
      - app_data:/app/data
      - chroma_local:/app/chroma_db
      - ./uploads:/app/uploads
    environment:
      # ChromaDB connection
      - CHROMA_HOST=chromadb
      - CHROMA_PORT=8000
      
      # API Keys (set these in .env file)
      - GEMINI_API_KEY=${GEMINI_API_KEY}
      - GROQ_API_KEY=${GROQ_API_KEY}
      - GOOGLE_API_KEY=${GOOGLE_API_KEY}
      
      # Model configuration
      - EMBEDDING_MODEL=${EMBEDDING_MODEL:-models/embedding-001}
      - LLM_for_chat=${LLM_for_chat:-openai/gpt-oss-120b}
      
      # Application settings
      - ANONYMIZED_TELEMETRY=False
    depends_on:
      - chromadb
    networks:
      - rag-network
    restart: unless-stopped

volumes:
  chroma_data:
  chroma_local:
  app_data:

networks:
  rag-network:
    driver: bridge

```

## 🏗️ Architecture

### System Components

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   FastAPI App   │    │   LangChain RAG  │    │   ChromaDB      │
│                 │────│                  │────│   Vector Store  │
│   • Endpoints   │    │   • Retrieval    │    │                 │
│   • Validation  │    │   • Generation   │    │   • Embeddings  │
│   • File Upload │    │   • Chat History │    │   • Similarity  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                        │                        │
         ▼                        ▼                        ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   SQLite DB     │    │   Document       │    │   Google        │
│                 │    │   Processors     │    │   Gemini API    │
│   • Metadata    │    │                  │    │                 │
│   • Chat Logs   │    │   • PDF/DOCX     │    │   • Embeddings  │
│   • Sessions    │    │   • TXT/HTML     │    │   • Chat Model  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Data Flow

1. **Document Upload** → Validation → Processing → Chunking → Embedding → Vector Storage
2. **User Query** → Session Retrieval → Vector Search → Context Assembly → LLM Generation → Response

## 📝 API Response Formats

### Successful Upload Response
```json
{
  "uploaded": [
    {
      "file": "document.pdf",
      "file_id": 1,
      "status": "success",
      "page_count": 10,
      "file_size": "2.5 MB"
    }
  ],
  "failed": [],
  "summary": {
    "total_files": 1,
    "successful": 1,
    "failed": 0,
    "success_rate": "100.0%"
  }
}
```

### Chat Response
```json
{
  "answer": "Artificial Intelligence (AI) refers to...",
  "session_id": "user-session-123",
  "model": "openai/gpt-oss-120b"
}
```

### Error Response
```json
{
  "error": "Document validation failed",
  "validation_errors": ["File size exceeds limit"],
  "invalid_documents": [
    {
      "file": "large_document.pdf",
      "errors": ["File size 60.0 MB exceeds maximum 50.0 MB"],
      "file_size": "60.0 MB",
      "page_count": 500
    }
  ]
}
```



## 🆘 Troubleshooting

### Common Issues

**ChromaDB Connection Error**
```bash
# Check if ChromaDB is running
curl http://localhost:8001/api/v1/heartbeat

# Reset ChromaDB data
rm -rf ./chroma_db
```

**API Key Issues**
```bash
# Verify environment variables
echo $GEMINI_API_KEY
echo $GROQ_API_KEY
```

**File Upload Errors**
- Check file size limits
- Verify supported file formats
- Ensure proper file permissions

**Memory Issues**
- Reduce batch size for large documents
- Adjust chunk size in text splitter
- Monitor system resources



