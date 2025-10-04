"""Tests for file processing functionality."""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch

from app.core.context.file_processor import FileProcessor, FileType, ProcessingResult


class TestFileProcessor:
    """Test cases for FileProcessor class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.processor = FileProcessor()
    
    def test_detect_file_type_by_extension(self):
        """Test file type detection by extension."""
        test_cases = [
            ("document.pdf", FileType.PDF),
            ("document.docx", FileType.DOCX),
            ("spreadsheet.xlsx", FileType.XLSX),
            ("presentation.pptx", FileType.PPTX),
            ("image.png", FileType.IMAGE),
            ("text.txt", FileType.TEXT),
            ("unknown.xyz", FileType.UNKNOWN),
        ]
        
        for filename, expected_type in test_cases:
            with tempfile.NamedTemporaryFile(suffix=Path(filename).suffix, delete=False) as tmp:
                tmp.write(b"test content")
                tmp.flush()
                
                detected_type = self.processor.detect_file_type(tmp.name)
                assert detected_type == expected_type
                
                os.unlink(tmp.name)
    
    def test_process_text_file(self):
        """Test processing of plain text files."""
        test_content = "This is a test document with multiple lines.\nSecond line here.\nThird line."
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp:
            tmp.write(test_content)
            tmp.flush()
            
            result = self.processor.process_file(tmp.name)
            
            assert result.success is True
            assert result.content == test_content
            assert result.file_type == FileType.TEXT
            assert result.error is None
            
            os.unlink(tmp.name)
    
    def test_process_nonexistent_file(self):
        """Test processing of non-existent file."""
        result = self.processor.process_file("nonexistent_file.txt")
        
        assert result.success is False
        assert "File not found" in result.error
        assert result.content == ""
    
    @patch('app.core.context.file_processor.partition_pdf')
    def test_process_pdf_success(self, mock_partition):
        """Test successful PDF processing."""
        # Mock the partition_pdf function
        mock_elements = [Mock(), Mock()]
        mock_elements[0].__str__ = Mock(return_value="First paragraph")
        mock_elements[1].__str__ = Mock(return_value="Second paragraph")
        mock_partition.return_value = mock_elements
        
        # Create a dummy PDF file
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
            tmp.write(b"%PDF-1.4 dummy content")
            tmp.flush()
            
            with patch('builtins.open', create=True), \
                 patch('app.core.context.file_processor.PdfReader') as mock_reader:
                
                # Mock PDF reader for page count
                mock_pdf = Mock()
                mock_pdf.pages = [Mock(), Mock()]  # 2 pages
                mock_reader.return_value = mock_pdf
                
                result = self.processor.process_file(tmp.name)
                
                assert result.success is True
                assert "First paragraph\nSecond paragraph" in result.content
                assert result.file_type == FileType.PDF
                assert result.page_count == 2
                
            os.unlink(tmp.name)
    
    @patch('app.core.context.file_processor.partition_docx')
    def test_process_docx_success(self, mock_partition):
        """Test successful DOCX processing."""
        mock_elements = [Mock()]
        mock_elements[0].__str__ = Mock(return_value="Document content")
        mock_partition.return_value = mock_elements
        
        with tempfile.NamedTemporaryFile(suffix='.docx', delete=False) as tmp:
            tmp.write(b"dummy docx content")
            tmp.flush()
            
            result = self.processor.process_file(tmp.name)
            
            assert result.success is True
            assert result.content == "Document content"
            assert result.file_type == FileType.DOCX
            
            os.unlink(tmp.name)
    
    @patch('app.core.context.file_processor.pytesseract.image_to_string')
    @patch('app.core.context.file_processor.Image.open')
    def test_process_image_with_ocr(self, mock_image_open, mock_ocr):
        """Test image processing with OCR."""
        # Mock PIL Image
        mock_image = Mock()
        mock_image.size = (800, 600)
        mock_image_open.return_value = mock_image
        
        # Mock OCR result
        mock_ocr.return_value = "Extracted text from image"
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            tmp.write(b"dummy image content")
            tmp.flush()
            
            result = self.processor.process_file(tmp.name)
            
            assert result.success is True
            assert result.content == "Extracted text from image"
            assert result.file_type == FileType.IMAGE
            
            os.unlink(tmp.name)
    
    def test_processing_error_handling(self):
        """Test error handling during file processing."""
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as tmp:
            tmp.write(b"test content")
            tmp.flush()
            
            # Make file unreadable to trigger error
            os.chmod(tmp.name, 0o000)
            
            try:
                result = self.processor.process_file(tmp.name)
                
                # Should handle the error gracefully
                assert result.success is False
                assert result.error is not None
                
            finally:
                # Restore permissions and cleanup
                os.chmod(tmp.name, 0o644)
                os.unlink(tmp.name)