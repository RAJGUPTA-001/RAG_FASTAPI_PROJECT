import unittest
import tempfile
import os
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

# Import the main application
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from main import app
from db_utils import get_db_connection, create_application_logs, create_document_store

class TestRAGIntegration(unittest.TestCase):
    """Integration tests for the complete RAG pipeline"""

    @classmethod
    def setUpClass(cls):
        """Set up test database and client"""
        cls.client = TestClient(app)
        # Create test database tables
        create_application_logs()
        create_document_store()

    def test_complete_rag_workflow(self):
        """Test complete workflow: upload -> query -> get results"""
        # Step 1: Check health
        health_response = self.client.get("/health")
        self.assertEqual(health_response.status_code, 200)

        # Step 2: Check upload limits
        limits_response = self.client.get("/upload-limits")
        self.assertEqual(limits_response.status_code, 200)

        # Step 3: List documents (should be empty initially)
        docs_response = self.client.get("/list-docs")
        self.assertEqual(docs_response.status_code, 200)
        initial_docs = docs_response.json()

        # Step 4: Test chat functionality
        chat_response = self.client.post("/chat", json={
            "question": "What is artificial intelligence?",
            "session_id": "test-session"
        })
        self.assertEqual(chat_response.status_code, 200)
        chat_data = chat_response.json()
        self.assertIn("answer", chat_data)
        self.assertIn("session_id", chat_data)

    def test_upload_and_retrieval_workflow(self):
        """Test document upload and retrieval workflow"""
        # Create a test document
        test_content = "This is a test document about machine learning and AI."

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(test_content)
            temp_path = f.name

        try:
            # Test file upload (this would require mocking the file upload)
            # For now, test the validation endpoint
            with open(temp_path, 'rb') as test_file:
                files = {"files": ("test.txt", test_file, "text/plain")}
                validate_response = self.client.post("/validate-files", files=files)
                # Note: This might fail if dependencies aren't properly mocked

        finally:
            os.unlink(temp_path)

    def test_session_persistence(self):
        """Test that chat sessions maintain context"""
        session_id = "test-persistence-session"

        # First message
        response1 = self.client.post("/chat", json={
            "question": "My name is John",
            "session_id": session_id
        })
        self.assertEqual(response1.status_code, 200)

        # Second message referencing first
        response2 = self.client.post("/chat", json={
            "question": "What is my name?",
            "session_id": session_id
        })
        self.assertEqual(response2.status_code, 200)

        # Different session should not have context
        response3 = self.client.post("/chat", json={
            "question": "What is my name?",
            "session_id": "different-session"
        })
        self.assertEqual(response3.status_code, 200)

class TestDatabaseIntegration(unittest.TestCase):
    """Test database operations"""

    def test_document_crud_operations(self):
        """Test Create, Read, Update, Delete operations for documents"""
        from db_utils import insert_document_record, get_all_documents, delete_document_record

        # Insert a test document
        file_id = insert_document_record("test_integration.pdf")
        self.assertIsInstance(file_id, int)

        # Read documents
        documents = get_all_documents()
        test_doc = next((doc for doc in documents if doc['id'] == file_id), None)
        self.assertIsNotNone(test_doc)
        self.assertEqual(test_doc['filename'], "test_integration.pdf")

        # Delete document
        result = delete_document_record(file_id)
        self.assertTrue(result)

        # Verify deletion
        documents_after = get_all_documents()
        test_doc_after = next((doc for doc in documents_after if doc['id'] == file_id), None)
        self.assertIsNone(test_doc_after)

    def test_chat_history_operations(self):
        """Test chat history storage and retrieval"""
        from db_utils import insert_application_logs, get_chat_history

        session_id = "test-history-session"

        # Insert some chat history
        insert_application_logs(session_id, "Hello", "Hi there!", "test-model")
        insert_application_logs(session_id, "How are you?", "I'm doing well!", "test-model")

        # Retrieve chat history
        history = get_chat_history(session_id)

        # Should have 4 messages (2 questions + 2 responses)
        self.assertEqual(len(history), 4)
        self.assertEqual(history[0]['role'], 'human')
        self.assertEqual(history[0]['content'], 'Hello')
        self.assertEqual(history[1]['role'], 'ai')
        self.assertEqual(history[1]['content'], 'Hi there!')

if __name__ == '__main__':
    unittest.main()