"""File processing pipeline for various document types."""

import logging
import mimetypes
import os
from pathlib import Path
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
from enum import Enum

try:
    import magic
    MAGIC_AVAILABLE = True
except ImportError:
    MAGIC_AVAILABLE = False
import pytesseract
from PIL import Image
from pypdf import PdfReader
from unstructured.partition.auto import partition
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.docx import partition_docx
from unstructured.partition.xlsx import partition_xlsx
from unstructured.partition.pptx import partition_pptx

logger = logging.getLogger(__name__)

# Log warning about missing magic after logger is defined
if not MAGIC_AVAILABLE:
    logger.warning("python-magic not available. File type detection will be limited to extensions.")


class FileType(Enum):
    """Supported file types."""
    PDF = "pdf"
    DOCX = "docx"
    XLSX = "xlsx"
    PPTX = "pptx"
    DOC = "doc"
    XLS = "xls"
    PPT = "ppt"
    IMAGE = "image"
    TEXT = "text"
    UNKNOWN = "unknown"


@dataclass
class ProcessingResult:
    """Result of file processing operation."""
    success: bool
    content: str
    metadata: Dict
    error: Optional[str] = None
    file_type: Optional[FileType] = None
    page_count: Optional[int] = None


class FileProcessor:
    """Handles file ingestion and content extraction for various document types."""
    
    def __init__(self):
        """Initialize the file processor."""
        self.supported_extensions = {
            '.pdf': FileType.PDF,
            '.docx': FileType.DOCX,
            '.xlsx': FileType.XLSX,
            '.pptx': FileType.PPTX,
            '.doc': FileType.DOC,
            '.xls': FileType.XLS,
            '.ppt': FileType.PPT,
            '.txt': FileType.TEXT,
            '.png': FileType.IMAGE,
            '.jpg': FileType.IMAGE,
            '.jpeg': FileType.IMAGE,
            '.tiff': FileType.IMAGE,
            '.bmp': FileType.IMAGE,
        }
        
        # Configure Tesseract if available
        self._configure_tesseract()
    
    def _configure_tesseract(self) -> None:
        """Configure Tesseract OCR engine."""
        try:
            # Test if Tesseract is available
            pytesseract.get_tesseract_version()
            logger.info("Tesseract OCR is available")
        except Exception as e:
            logger.warning(f"Tesseract OCR not available: {e}")
    
    def detect_file_type(self, file_path: Union[str, Path]) -> FileType:
        """
        Detect file type using multiple methods.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Detected file type
        """
        file_path = Path(file_path)
        
        # First try extension-based detection
        extension = file_path.suffix.lower()
        if extension in self.supported_extensions:
            return self.supported_extensions[extension]
        
        # Fallback to MIME type detection if available
        if MAGIC_AVAILABLE:
            try:
                mime_type = magic.from_file(str(file_path), mime=True)
                if mime_type.startswith('image/'):
                    return FileType.IMAGE
                elif mime_type == 'application/pdf':
                    return FileType.PDF
                elif mime_type in ['application/vnd.openxmlformats-officedocument.wordprocessingml.document']:
                    return FileType.DOCX
                elif mime_type in ['application/vnd.openxmlformats-officedocument.spreadsheetml.sheet']:
                    return FileType.XLSX
                elif mime_type in ['application/vnd.openxmlformats-officedocument.presentationml.presentation']:
                    return FileType.PPTX
                elif mime_type.startswith('text/'):
                    return FileType.TEXT
            except Exception as e:
                logger.warning(f"MIME type detection failed: {e}")
        
        return FileType.UNKNOWN
    
    def process_file(self, file_path: Union[str, Path]) -> ProcessingResult:
        """
        Process a file and extract its content.
        
        Args:
            file_path: Path to the file to process
            
        Returns:
            Processing result with extracted content and metadata
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            return ProcessingResult(
                success=False,
                content="",
                metadata={},
                error=f"File not found: {file_path}"
            )
        
        file_type = self.detect_file_type(file_path)
        logger.info(f"Processing file {file_path} as {file_type.value}")
        
        try:
            if file_type == FileType.PDF:
                return self._process_pdf(file_path)
            elif file_type in [FileType.DOCX, FileType.DOC]:
                return self._process_word_document(file_path)
            elif file_type in [FileType.XLSX, FileType.XLS]:
                return self._process_excel_document(file_path)
            elif file_type in [FileType.PPTX, FileType.PPT]:
                return self._process_powerpoint_document(file_path)
            elif file_type == FileType.IMAGE:
                return self._process_image(file_path)
            elif file_type == FileType.TEXT:
                return self._process_text_file(file_path)
            else:
                # Try generic unstructured processing as fallback
                return self._process_with_unstructured(file_path)
                
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            return ProcessingResult(
                success=False,
                content="",
                metadata={"file_path": str(file_path)},
                error=str(e),
                file_type=file_type
            )
    
    def _process_pdf(self, file_path: Path) -> ProcessingResult:
        """Process PDF files with text extraction and OCR fallback."""
        try:
            # First try text extraction
            elements = partition_pdf(str(file_path))
            content = "\n".join([str(element) for element in elements])
            
            # Get page count
            with open(file_path, 'rb') as file:
                pdf_reader = PdfReader(file)
                page_count = len(pdf_reader.pages)
            
            # If content is too short, might be image-based PDF - try OCR
            if len(content.strip()) < 100:
                logger.info(f"PDF appears to be image-based, attempting OCR: {file_path}")
                try:
                    # Use unstructured with OCR strategy
                    elements = partition_pdf(str(file_path), strategy="ocr_only")
                    content = "\n".join([str(element) for element in elements])
                except Exception as ocr_error:
                    logger.warning(f"OCR processing failed: {ocr_error}")
            
            return ProcessingResult(
                success=True,
                content=content,
                metadata={
                    "file_path": str(file_path),
                    "file_type": "pdf",
                    "page_count": page_count
                },
                file_type=FileType.PDF,
                page_count=page_count
            )
            
        except Exception as e:
            logger.error(f"PDF processing failed: {e}")
            raise
    
    def _process_word_document(self, file_path: Path) -> ProcessingResult:
        """Process Word documents."""
        try:
            elements = partition_docx(str(file_path))
            content = "\n".join([str(element) for element in elements])
            
            return ProcessingResult(
                success=True,
                content=content,
                metadata={
                    "file_path": str(file_path),
                    "file_type": "docx"
                },
                file_type=FileType.DOCX
            )
            
        except Exception as e:
            logger.error(f"Word document processing failed: {e}")
            raise
    
    def _process_excel_document(self, file_path: Path) -> ProcessingResult:
        """Process Excel documents."""
        try:
            elements = partition_xlsx(str(file_path))
            content = "\n".join([str(element) for element in elements])
            
            return ProcessingResult(
                success=True,
                content=content,
                metadata={
                    "file_path": str(file_path),
                    "file_type": "xlsx"
                },
                file_type=FileType.XLSX
            )
            
        except Exception as e:
            logger.error(f"Excel document processing failed: {e}")
            raise
    
    def _process_powerpoint_document(self, file_path: Path) -> ProcessingResult:
        """Process PowerPoint documents."""
        try:
            elements = partition_pptx(str(file_path))
            content = "\n".join([str(element) for element in elements])
            
            return ProcessingResult(
                success=True,
                content=content,
                metadata={
                    "file_path": str(file_path),
                    "file_type": "pptx"
                },
                file_type=FileType.PPTX
            )
            
        except Exception as e:
            logger.error(f"PowerPoint document processing failed: {e}")
            raise
    
    def _process_image(self, file_path: Path) -> ProcessingResult:
        """Process images using OCR."""
        try:
            # Open image and perform OCR
            image = Image.open(file_path)
            content = pytesseract.image_to_string(image)
            
            return ProcessingResult(
                success=True,
                content=content,
                metadata={
                    "file_path": str(file_path),
                    "file_type": "image",
                    "image_size": image.size
                },
                file_type=FileType.IMAGE
            )
            
        except Exception as e:
            logger.error(f"Image OCR processing failed: {e}")
            raise
    
    def _process_text_file(self, file_path: Path) -> ProcessingResult:
        """Process plain text files."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            return ProcessingResult(
                success=True,
                content=content,
                metadata={
                    "file_path": str(file_path),
                    "file_type": "text"
                },
                file_type=FileType.TEXT
            )
            
        except Exception as e:
            logger.error(f"Text file processing failed: {e}")
            raise
    
    def _process_with_unstructured(self, file_path: Path) -> ProcessingResult:
        """Generic processing using unstructured library."""
        try:
            elements = partition(str(file_path))
            content = "\n".join([str(element) for element in elements])
            
            return ProcessingResult(
                success=True,
                content=content,
                metadata={
                    "file_path": str(file_path),
                    "file_type": "generic"
                },
                file_type=FileType.UNKNOWN
            )
            
        except Exception as e:
            logger.error(f"Generic unstructured processing failed: {e}")
            raise