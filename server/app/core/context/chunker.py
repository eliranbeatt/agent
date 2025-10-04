"""Content chunking system for semantic text segmentation."""

import logging
import re
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import tiktoken

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """Represents a chunk of content with metadata."""
    id: str
    content: str
    source_file: str
    page_number: Optional[int] = None
    chunk_index: int = 0
    token_count: int = 0
    start_char: int = 0
    end_char: int = 0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class ContentChunker:
    pass

class ContentChunker:
    """Handles semantic chunking of content into manageable segments."""
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
        min_chunk_size: int = 100,
        max_chunk_size: int = 1200,
        encoding_name: str = "cl100k_base"
    ):
        """
        Initialize the content chunker.
        
        Args:
            chunk_size: Target chunk size in tokens (800-1200 range)
            chunk_overlap: Overlap between chunks in tokens (80-150 range)
            min_chunk_size: Minimum chunk size in tokens
            max_chunk_size: Maximum chunk size in tokens
            encoding_name: Tokenizer encoding to use
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        
        try:
            self.encoding = tiktoken.get_encoding(encoding_name)
        except Exception as e:
            logger.warning(f"Failed to load encoding {encoding_name}: {e}")
            self.encoding = tiktoken.get_encoding("cl100k_base")
        
        # Sentence boundary patterns
        self.sentence_endings = re.compile(r'[.!?]+\s+')
        self.paragraph_breaks = re.compile(r'\n\s*\n')
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text using the configured encoding."""
        try:
            return len(self.encoding.encode(text))
        except Exception as e:
            logger.warning(f"Token counting failed: {e}")
            # Fallback to rough estimation
            return len(text.split()) * 1.3
    
    def chunk_content(
        self,
        content: str,
        source_file: str,
        page_number: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Chunk]:
        """
        Chunk content into semantic segments.
        
        Args:
            content: Text content to chunk
            source_file: Source file path
            page_number: Page number if applicable
            metadata: Additional metadata for chunks
            
        Returns:
            List of content chunks
        """
        if not content or not content.strip():
            return []
        
        content = content.strip()
        
        # First try paragraph-based chunking
        chunks = self._chunk_by_paragraphs(content, source_file, page_number, metadata)
        
        # If paragraphs are too large, fall back to sentence-based chunking
        oversized_chunks = [chunk for chunk in chunks if chunk.token_count > self.max_chunk_size]
        if oversized_chunks:
            logger.info(f"Found {len(oversized_chunks)} oversized chunks, applying sentence-based chunking")
            final_chunks = []
            for chunk in chunks:
                if chunk.token_count > self.max_chunk_size:
                    sentence_chunks = self._chunk_by_sentences(
                        chunk.content, source_file, page_number, metadata, chunk.chunk_index
                    )
                    final_chunks.extend(sentence_chunks)
                else:
                    final_chunks.append(chunk)
            chunks = final_chunks
        
        # Final pass: handle any remaining oversized chunks with token-based splitting
        final_chunks = []
        for chunk in chunks:
            if chunk.token_count > self.max_chunk_size:
                token_chunks = self._chunk_by_tokens(
                    chunk.content, source_file, page_number, metadata, chunk.chunk_index
                )
                final_chunks.extend(token_chunks)
            else:
                final_chunks.append(chunk)
        
        # Update chunk indices and IDs
        for i, chunk in enumerate(final_chunks):
            chunk.chunk_index = i
            chunk.id = f"{source_file}_{i}"
        
        logger.info(f"Created {len(final_chunks)} chunks from {source_file}")
        return final_chunks
    
    def _chunk_by_paragraphs(
        self,
        content: str,
        source_file: str,
        page_number: Optional[int],
        metadata: Optional[Dict[str, Any]]
    ) -> List[Chunk]:
        """Chunk content by paragraphs."""
        paragraphs = self.paragraph_breaks.split(content)
        chunks = []
        current_chunk = ""
        current_start = 0
        chunk_index = 0
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # Check if adding this paragraph would exceed chunk size
            test_chunk = current_chunk + "\n\n" + paragraph if current_chunk else paragraph
            token_count = self.count_tokens(test_chunk)
            
            if token_count <= self.chunk_size or not current_chunk:
                # Add paragraph to current chunk
                current_chunk = test_chunk
            else:
                # Save current chunk and start new one
                if current_chunk:
                    chunk = self._create_chunk(
                        current_chunk, source_file, page_number, metadata,
                        chunk_index, current_start, current_start + len(current_chunk)
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                
                # Start new chunk with overlap
                overlap_text = self._get_overlap_text(current_chunk)
                current_chunk = overlap_text + paragraph if overlap_text else paragraph
                current_start = current_start + len(current_chunk) - len(overlap_text) if overlap_text else current_start + len(current_chunk)
        
        # Add final chunk
        if current_chunk:
            chunk = self._create_chunk(
                current_chunk, source_file, page_number, metadata,
                chunk_index, current_start, current_start + len(current_chunk)
            )
            chunks.append(chunk)
        
        return chunks
    
    def _chunk_by_sentences(
        self,
        content: str,
        source_file: str,
        page_number: Optional[int],
        metadata: Optional[Dict[str, Any]],
        base_index: int = 0
    ) -> List[Chunk]:
        """Chunk content by sentences."""
        sentences = self.sentence_endings.split(content)
        chunks = []
        current_chunk = ""
        current_start = 0
        chunk_index = base_index
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Check if adding this sentence would exceed chunk size
            test_chunk = current_chunk + " " + sentence if current_chunk else sentence
            token_count = self.count_tokens(test_chunk)
            
            if token_count <= self.chunk_size or not current_chunk:
                current_chunk = test_chunk
            else:
                # Save current chunk and start new one
                if current_chunk:
                    chunk = self._create_chunk(
                        current_chunk, source_file, page_number, metadata,
                        chunk_index, current_start, current_start + len(current_chunk)
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                
                # Start new chunk with overlap
                overlap_text = self._get_overlap_text(current_chunk)
                current_chunk = overlap_text + sentence if overlap_text else sentence
                current_start = current_start + len(current_chunk) - len(overlap_text) if overlap_text else current_start + len(current_chunk)
        
        # Add final chunk
        if current_chunk:
            chunk = self._create_chunk(
                current_chunk, source_file, page_number, metadata,
                chunk_index, current_start, current_start + len(current_chunk)
            )
            chunks.append(chunk)
        
        return chunks
    
    def _chunk_by_tokens(
        self,
        content: str,
        source_file: str,
        page_number: Optional[int],
        metadata: Optional[Dict[str, Any]],
        base_index: int = 0
    ) -> List[Chunk]:
        """Chunk content by token count as last resort."""
        tokens = self.encoding.encode(content)
        chunks = []
        chunk_index = base_index
        
        start_idx = 0
        while start_idx < len(tokens):
            # Calculate end index for this chunk
            end_idx = min(start_idx + self.chunk_size, len(tokens))
            
            # Extract chunk tokens and decode
            chunk_tokens = tokens[start_idx:end_idx]
            chunk_text = self.encoding.decode(chunk_tokens)
            
            # Create chunk
            chunk = self._create_chunk(
                chunk_text, source_file, page_number, metadata,
                chunk_index, start_idx, end_idx
            )
            chunks.append(chunk)
            chunk_index += 1
            
            # Move start index with overlap
            start_idx = end_idx - self.chunk_overlap
            if start_idx >= len(tokens):
                break
        
        return chunks
    
    def _get_overlap_text(self, text: str) -> str:
        """Get overlap text from the end of current chunk."""
        tokens = self.encoding.encode(text)
        if len(tokens) <= self.chunk_overlap:
            return text
        
        overlap_tokens = tokens[-self.chunk_overlap:]
        return self.encoding.decode(overlap_tokens)
    
    def _create_chunk(
        self,
        content: str,
        source_file: str,
        page_number: Optional[int],
        metadata: Optional[Dict[str, Any]],
        chunk_index: int,
        start_char: int,
        end_char: int
    ) -> Chunk:
        """Create a chunk object with metadata."""
        token_count = self.count_tokens(content)
        
        chunk_metadata = {
            "source_file": source_file,
            "chunk_index": chunk_index,
            "token_count": token_count,
            **(metadata or {})
        }
        
        if page_number is not None:
            chunk_metadata["page_number"] = page_number
        
        return Chunk(
            id=f"{source_file}_{chunk_index}",
            content=content,
            source_file=source_file,
            page_number=page_number,
            chunk_index=chunk_index,
            token_count=token_count,
            start_char=start_char,
            end_char=end_char,
            metadata=chunk_metadata
        )

class SemanticChunker(ContentChunker):
    pass