import os
import sys

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Test environment variables
TEST_ENV_VARS = {
    'GROQ_API_KEY':'your-groq-api-key',
'LLM_for_chat':'openai/gpt-oss-120b',
'GEMINI_API_KEY':'your-gemini-api-key',
'EMBEDDING_MODEL':'gemini-embedding-001',
'GOOGLE_API_KEY':'your-google-api-key',
'CHROMA_HOST':'chromadb',
'CHROMA_PORT':'8000',
'ANONYMIZED_TELEMETRY':'False'
}

def setup_test_environment():
    """Set up environment variables for testing"""
    for key, value in TEST_ENV_VARS.items():
        os.environ[key] = value

def teardown_test_environment():
    """Clean up test environment"""
    for key in TEST_ENV_VARS.keys():
        if key in os.environ:
            del os.environ[key]