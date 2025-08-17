import unittest
import sqlite3
import tempfile
import os
from unittest.mock import patch, MagicMock

class TestDatabaseUtils(unittest.TestCase):

    def setUp(self):
        """Set up test database"""
        self.test_db = ":memory:"

    def test_database_table_creation(self):
        """Test that database tables are created correctly"""
        conn = sqlite3.connect(":memory:")
        cursor = conn.cursor()

        # Create application_logs table
        cursor.execute('''CREATE TABLE IF NOT EXISTS application_logs
                         (id INTEGER PRIMARY KEY AUTOINCREMENT,
                          session_id TEXT,
                          user_query TEXT,
                          gpt_response TEXT,
                          model TEXT,
                          created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')

        # Check if table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='application_logs'")
        result = cursor.fetchone()
        self.assertIsNotNone(result)

        conn.close()

    def test_document_store_operations(self):
        """Test document store CRUD operations"""
        conn = sqlite3.connect(":memory:")
        conn.execute('''CREATE TABLE IF NOT EXISTS document_store
                       (id INTEGER PRIMARY KEY AUTOINCREMENT,
                        filename TEXT,
                        upload_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')

        # Test insert
        cursor = conn.cursor()
        cursor.execute('INSERT INTO document_store (filename) VALUES (?)', ("test.pdf",))
        file_id = cursor.lastrowid
        conn.commit()

        self.assertIsInstance(file_id, int)

        # Test retrieval
        cursor.execute('SELECT id, filename FROM document_store WHERE id = ?', (file_id,))
        result = cursor.fetchone()
        self.assertIsNotNone(result)
        self.assertEqual(result[1], "test.pdf")

        # Test deletion
        cursor.execute('DELETE FROM document_store WHERE id = ?', (file_id,))
        conn.commit()

        cursor.execute('SELECT id FROM document_store WHERE id = ?', (file_id,))
        result = cursor.fetchone()
        self.assertIsNone(result)

        conn.close()

class TestChatHistory(unittest.TestCase):

    def test_chat_history_storage(self):
        """Test chat history storage and retrieval"""
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row

        # Create table
        conn.execute('''CREATE TABLE IF NOT EXISTS application_logs
                       (id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT,
                        user_query TEXT,
                        gpt_response TEXT,
                        model TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')

        session_id = "test-session"

        # Insert chat entries
        conn.execute('INSERT INTO application_logs (session_id, user_query, gpt_response, model) VALUES (?, ?, ?, ?)',
                    (session_id, "Hello", "Hi there", "test-model"))
        conn.execute('INSERT INTO application_logs (session_id, user_query, gpt_response, model) VALUES (?, ?, ?, ?)',
                    (session_id, "How are you?", "I'm fine", "test-model"))
        conn.commit()

        # Retrieve chat history
        cursor = conn.cursor()
        cursor.execute('SELECT user_query, gpt_response FROM application_logs WHERE session_id = ? ORDER BY created_at', 
                      (session_id,))
        messages = []
        for row in cursor.fetchall():
            messages.extend([
                {"role": "human", "content": row['user_query']},
                {"role": "ai", "content": row['gpt_response']}
            ])

        # Verify chat history structure
        self.assertEqual(len(messages), 4)  # 2 exchanges = 4 messages
        self.assertEqual(messages[0]['role'], 'human')
        self.assertEqual(messages[0]['content'], 'Hello')
        self.assertEqual(messages[1]['role'], 'ai')
        self.assertEqual(messages[1]['content'], 'Hi there')

        conn.close()

if __name__ == '__main__':
    unittest.main()