import unittest
from fastapi.testclient import TestClient
from main import app

class TestDocumentAPI(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_list_documents_empty(self):
        response = self.client.get('/list-docs')
        self.assertEqual(response.status_code, 200)
        self.assertIsInstance(response.json(), list)

    def test_health_check(self):
        response = self.client.get('/health')
        self.assertEqual(response.status_code, 200)
        json_data = response.json()
        self.assertEqual(json_data.get('status'), 'healthy')

    

class TestQueryHandling(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_chat_endpoint_no_session(self):
        test_question = {"question": "What is Artificial Intelligence?"}
        response = self.client.post('/chat', json=test_question)
        self.assertEqual(response.status_code, 200)
        json_data = response.json()
        self.assertIn('answer', json_data)
        self.assertIn('session_id', json_data)

    def test_chat_endpoint_with_session(self):
        session_id = "test-session-123"
        test_question = {"question": "Explain AI.", "session_id": session_id}
        response = self.client.post('/chat', json=test_question)
        self.assertEqual(response.status_code, 200)
        json_data = response.json()
        self.assertEqual(json_data.get('session_id'), session_id)