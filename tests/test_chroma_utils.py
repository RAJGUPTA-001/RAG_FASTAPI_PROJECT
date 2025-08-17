import unittest
import tempfile
import os
from unittest.mock import patch, MagicMock, mock_open
from pathlib import Path

# Import modules to test
from chroma_utils import (
    format_file_size, validate_single_document, validate_document_batch,
    get_document_page_count, safe_load_text_file, UploadLimits
)

class TestChromaUtils(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures"""
        self.test_limits = UploadLimits(
            max_documents=5,
            max_pages_per_doc=10,
            pdf_max_size=1024 * 1024,  # 1MB
            docx_max_size=512 * 1024,  # 512KB
            txt_max_size=256 * 1024,   # 256KB
            html_max_size=512 * 1024   # 512KB
        )

    def test_format_file_size(self):
        """Test file size formatting"""
        self.assertEqual(format_file_size(512), "512.0 B")
        self.assertEqual(format_file_size(1024), "1.0 KB")
        self.assertEqual(format_file_size(1024 * 1024), "1.0 MB")
        self.assertEqual(format_file_size(1024 * 1024 * 1024), "1.0 GB")

    def test_upload_limits_get_max_size(self):
        """Test upload limits max size calculation"""
        self.assertEqual(self.test_limits.get_max_size_for_file("test.pdf"), 1024 * 1024)
        self.assertEqual(self.test_limits.get_max_size_for_file("test.docx"), 512 * 1024)
        self.assertEqual(self.test_limits.get_max_size_for_file("test.txt"), 256 * 1024)
        self.assertEqual(self.test_limits.get_max_size_for_file("test.html"), 512 * 1024)
        # Test unknown extension defaults to PDF limit
        self.assertEqual(self.test_limits.get_max_size_for_file("test.unknown"), 1024 * 1024)

    @patch('os.path.exists')
    @patch('os.path.getsize')
    def test_validate_single_document_file_not_found(self, mock_getsize, mock_exists):
        """Test validation when file doesn't exist"""
        mock_exists.return_value = False

        result = validate_single_document("nonexistent.pdf", self.test_limits)

        self.assertFalse(result["valid"])
        self.assertIn("File not found", result["errors"][0])

    @patch('os.path.exists')
    @patch('os.path.getsize')
    def test_validate_single_document_unsupported_type(self, mock_getsize, mock_exists):
        """Test validation with unsupported file type"""
        mock_exists.return_value = True
        mock_getsize.return_value = 1024

        result = validate_single_document("test.exe", self.test_limits)

        self.assertFalse(result["valid"])
        self.assertIn("Unsupported file type", result["errors"][0])

    @patch('os.path.exists')
    @patch('os.path.getsize')
    @patch('chroma_utils.get_document_page_count')
    def test_validate_single_document_size_exceeded(self, mock_page_count, mock_getsize, mock_exists):
        """Test validation when file size exceeds limit"""
        mock_exists.return_value = True
        mock_getsize.return_value = 2 * 1024 * 1024  # 2MB (exceeds 1MB limit)
        mock_page_count.return_value = 5

        result = validate_single_document("test.pdf", self.test_limits)

        self.assertFalse(result["valid"])
        self.assertTrue(any("exceeds maximum" in error for error in result["errors"]))

    @patch('os.path.exists')
    @patch('os.path.getsize')
    @patch('chroma_utils.get_document_page_count')
    def test_validate_single_document_page_count_exceeded(self, mock_page_count, mock_getsize, mock_exists):
        """Test validation when page count exceeds limit"""
        mock_exists.return_value = True
        mock_getsize.return_value = 1024  # Small file
        mock_page_count.return_value = 15  # Exceeds limit of 10

        result = validate_single_document("test.pdf", self.test_limits)

        self.assertFalse(result["valid"])
        self.assertTrue(any("exceeding maximum" in error for error in result["errors"]))

    @patch('os.path.exists')
    @patch('os.path.getsize')
    @patch('chroma_utils.get_document_page_count')
    def test_validate_single_document_valid(self, mock_page_count, mock_getsize, mock_exists):
        """Test validation with valid document"""
        mock_exists.return_value = True
        mock_getsize.return_value = 512 * 1024  # 512KB
        mock_page_count.return_value = 5

        result = validate_single_document("test.pdf", self.test_limits)

        self.assertTrue(result["valid"])
        self.assertEqual(len(result["errors"]), 0)
        self.assertEqual(result["file_size"], 512 * 1024)
        self.assertEqual(result["page_count"], 5)

    def test_validate_document_batch_too_many_documents(self):
        """Test batch validation with too many documents"""
        file_paths = [f"test{i}.pdf" for i in range(10)]  # 10 files, limit is 5

        result = validate_document_batch(file_paths, self.test_limits)

        self.assertFalse(result["valid"])
        self.assertTrue(any("Too many documents" in error for error in result["errors"]))

    @patch('chroma_utils.validate_single_document')
    def test_validate_document_batch_valid(self, mock_validate_single):
        """Test batch validation with valid documents"""
        file_paths = ["test1.pdf", "test2.pdf"]

        # Mock successful validation for each document
        mock_validate_single.return_value = {
            "valid": True,
            "errors": [],
            "page_count": 3,
            "file_size": 1024
        }

        result = validate_document_batch(file_paths, self.test_limits)

        self.assertTrue(result["valid"])
        self.assertEqual(result["total_documents"], 2)
        self.assertEqual(result["total_pages"], 6)  # 3 pages * 2 docs
        self.assertEqual(result["total_size"], 2048)  # 1024 * 2 docs

class TestDocumentLoading(unittest.TestCase):

    def test_safe_load_text_file_utf8(self):
        """Test loading UTF-8 text file"""
        test_content = "This is a test document with UTF-8 content."

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write(test_content)
            temp_path = f.name

        try:
            documents = safe_load_text_file(temp_path)
            self.assertEqual(len(documents), 1)
            self.assertEqual(documents[0].page_content, test_content)
        finally:
            os.unlink(temp_path)

    def test_safe_load_text_file_nonexistent(self):
        """Test loading non-existent file"""
        with self.assertRaises(Exception):
            safe_load_text_file("nonexistent_file.txt")

if __name__ == '__main__':
    unittest.main()