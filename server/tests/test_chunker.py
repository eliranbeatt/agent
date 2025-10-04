"""Tests for content chunking functionality."""

import pytest
from unittest.mock import Mock, patch

from app.core.context.chunker import ContentChunker, Chunk


class TestContentChunker:
    """Test cases for ContentChunker class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.chunker = ContentChunker(
            chunk_size=100,  # Small size for testing
            chunk_overlap=20,
            min_chunk_size=10,
            max_chunk_size=150
        )
    
    def test_token_counting(self):
        """Test token counting functionality."""
        text = "This is a test sentence with multiple words."
        token_count = self.chunker.count_tokens(text)
        
        # Should return a reasonable token count
        assert isinstance(token_count, (int, float))
        assert token_count > 0
        assert token_count < len(text)  # Tokens should be fewer than characters
    
    def test_chunk_short_content(self):
        """Test chunking of short content that fits in one chunk."""
        content = "This is a short piece of content."
        source_file = "test.txt"
        
        chunks = self.chunker.chunk_content(content, source_file)
        
        assert len(chunks) == 1
        assert chunks[0].content == content
        assert chunks[0].source_file == source_file
        assert chunks[0].chunk_index == 0
        assert chunks[0].id == f"{source_file}_0"
    
    def test_chunk_long_content(self):
        """Test chunking of long content that requires multiple chunks."""
        # Create content that will definitely need multiple chunks
        sentences = [
            "This is the first sentence of a long document.",
            "Here is the second sentence with more content.",
            "The third sentence continues the narrative.",
            "Fourth sentence adds even more text to the document.",
            "Fifth sentence ensures we exceed the chunk size limit.",
            "Sixth sentence provides additional content for testing.",
            "Seventh sentence helps create multiple chunks.",
            "Eighth sentence completes our test document."
        ]
        content = " ".join(sentences)
        source_file = "long_test.txt"
        
        chunks = self.chunker.chunk_content(content, source_file)
        
        # Should create multiple chunks
        assert len(chunks) > 1
        
        # Check chunk properties
        for i, chunk in enumerate(chunks):
            assert chunk.source_file == source_file
            assert chunk.chunk_index == i
            assert chunk.id == f"{source_file}_{i}"
            assert chunk.token_count > 0
            assert len(chunk.content.strip()) > 0
        
        # Check that all content is preserved
        combined_content = " ".join(chunk.content for chunk in chunks)
        # Due to overlap, combined content should contain all original sentences
        for sentence in sentences:
            assert sentence in combined_content
    
    def test_chunk_with_paragraphs(self):
        """Test chunking content with paragraph breaks."""
        content = """First paragraph with some content here.
This is still part of the first paragraph.

Second paragraph starts here with different content.
More content in the second paragraph.

Third paragraph has its own content.
Final sentence of the third paragraph."""
        
        source_file = "paragraphs.txt"
        chunks = self.chunker.chunk_content(content, source_file)
        
        # Should create chunks respecting paragraph boundaries when possible
        assert len(chunks) >= 1
        
        # Each chunk should have reasonable content
        for chunk in chunks:
            assert len(chunk.content.strip()) > 0
            assert chunk.token_count > 0
    
    def test_chunk_with_metadata(self):
        """Test chunking with additional metadata."""
        content = "Test content for metadata checking."
        source_file = "metadata_test.txt"
        page_number = 5
        metadata = {"document_type": "test", "author": "test_user"}
        
        chunks = self.chunker.chunk_content(
            content, source_file, page_number, metadata
        )
        
        assert len(chunks) == 1
        chunk = chunks[0]
        
        assert chunk.page_number == page_number
        assert chunk.metadata["document_type"] == "test"
        assert chunk.metadata["author"] == "test_user"
        assert chunk.metadata["source_file"] == source_file
    
    def test_empty_content(self):
        """Test handling of empty content."""
        content = ""
        source_file = "empty.txt"
        
        chunks = self.chunker.chunk_content(content, source_file)
        
        assert len(chunks) == 0
    
    def test_whitespace_only_content(self):
        """Test handling of whitespace-only content."""
        content = "   \n\n\t   \n   "
        source_file = "whitespace.txt"
        
        chunks = self.chunker.chunk_content(content, source_file)
        
        assert len(chunks) == 0
    
    def test_chunk_overlap(self):
        """Test that chunk overlap is working correctly."""
        # Create content that will need chunking
        sentences = [
            "First sentence of the document.",
            "Second sentence with more content.",
            "Third sentence continues the story.",
            "Fourth sentence adds more text.",
            "Fifth sentence ensures multiple chunks.",
            "Sixth sentence provides overlap testing.",
            "Seventh sentence completes the test."
        ]
        content = " ".join(sentences)
        source_file = "overlap_test.txt"
        
        chunks = self.chunker.chunk_content(content, source_file)
        
        if len(chunks) > 1:
            # Check that there's some overlap between consecutive chunks
            for i in range(len(chunks) - 1):
                current_chunk = chunks[i].content
                next_chunk = chunks[i + 1].content
                
                # Find common words between chunks (simple overlap check)
                current_words = set(current_chunk.split())
                next_words = set(next_chunk.split())
                overlap_words = current_words.intersection(next_words)
                
                # Should have some overlapping words
                assert len(overlap_words) > 0
    
    def test_chunk_size_limits(self):
        """Test that chunks respect size limits."""
        # Create very long content
        long_content = "Very long sentence. " * 200  # Repeat to create long content
        source_file = "size_test.txt"
        
        chunks = self.chunker.chunk_content(long_content, source_file)
        
        # Check that no chunk exceeds the maximum size
        for chunk in chunks:
            assert chunk.token_count <= self.chunker.max_chunk_size
            
        # Check that chunks meet minimum size (except possibly the last one)
        for i, chunk in enumerate(chunks[:-1]):  # All but last chunk
            assert chunk.token_count >= self.chunker.min_chunk_size
    
    @patch('app.core.context.chunker.tiktoken.get_encoding')
    def test_encoding_fallback(self, mock_get_encoding):
        """Test fallback when tiktoken encoding fails."""
        # Mock encoding failure
        mock_get_encoding.side_effect = Exception("Encoding not found")
        
        # Should still work with fallback
        chunker = ContentChunker()
        content = "Test content for encoding fallback."
        
        chunks = chunker.chunk_content(content, "test.txt")
        
        assert len(chunks) == 1
        assert chunks[0].content == content