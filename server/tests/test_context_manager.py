"""Integration tests for the complete context management system."""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from app.core.context.context_manager import ContextManager, IngestionResult, RetrievalResult
from app.core.context.file_processor import ProcessingResult, FileType
from app.core.context.chunker import Chunk
from app.core.context.embeddings import EmbeddingResult
from app.core.context.vector_store import StorageResult, SearchResult


class TestContextManager:
    """Integration test cases for ContextManager class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.api_key = "test-api-key"
        
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
    
    @patch('app.core.context.context_manager.VectorStore')
    @patch('app.core.context.context_manager.EmbeddingManager')
    @patch('app.core.context.context_manager.ContentChunker')
    @patch('app.core.context.context_manager.FileProcessor')
    def test_initialization_success(self, mock_file_proc, mock_chunker, mock_embed, mock_vector):
        """Test successful ContextManager initialization."""
        manager = ContextManager(
            openai_api_key=self.api_key,
            vector_db_path=self.temp_dir
        )
        
        assert manager.openai_api_key == self.api_key
        mock_file_proc.assert_called_once()
        mock_chunker.assert_called_once()
        mock_embed.assert_called_once()
        mock_vector.assert_called_once()
    
    @patch('app.core.context.context_manager.VectorStore')
    @patch('app.core.context.context_manager.ContentChunker')
    @patch('app.core.context.context_manager.FileProcessor')
    def test_initialization_without_api_key(self, mock_file_proc, mock_chunker, mock_vector):
        """Test initialization without OpenAI API key."""
        manager = ContextManager()
        
        assert manager.embedding_manager is None
        mock_file_proc.assert_called_once()
        mock_chunker.assert_called_once()
        mock_vector.assert_called_once()
    
    @patch('app.core.context.context_manager.VectorStore')
    @patch('app.core.context.context_manager.EmbeddingManager')
    @patch('app.core.context.context_manager.ContentChunker')
    @patch('app.core.context.context_manager.FileProcessor')
    def test_ingest_file_success(self, mock_file_proc_class, mock_chunker_class, mock_embed_class, mock_vector_class):
        """Test successful file ingestion pipeline."""
        # Mock file processor
        mock_file_proc = Mock()
        mock_file_proc.process_file.return_value = ProcessingResult(
            success=True,
            content="Test document content with multiple sentences.",
            metadata={"file_type": "text"},
            file_type=FileType.TEXT
        )
        mock_file_proc_class.return_value = mock_file_proc
        
        # Mock chunker
        mock_chunker = Mock()
        test_chunks = [
            Chunk(
                id="test_1",
                content="Test document content",
                source_file="test.txt",
                chunk_index=0,
                token_count=5
            )
        ]
        mock_chunker.chunk_content.return_value = test_chunks
        mock_chunker_class.return_value = mock_chunker
        
        # Mock embedding manager
        mock_embed = Mock()
        mock_embed.embed_chunks.return_value = EmbeddingResult(
            success=True,
            embeddings=[[0.1, 0.2, 0.3]],
            tokens_used=10
        )
        mock_embed_class.return_value = mock_embed
        
        # Mock vector store
        mock_vector = Mock()
        mock_vector.store_chunks.return_value = StorageResult(
            success=True,
            stored_count=1
        )
        mock_vector_class.return_value = mock_vector
        
        # Create test file
        test_file = Path(self.temp_dir) / "test.txt"
        test_file.write_text("Test content")
        
        manager = ContextManager(openai_api_key=self.api_key)
        result = manager.ingest_file(test_file)
        
        assert result.success is True
        assert result.chunks_created == 1
        assert result.chunks_stored == 1
        assert result.tokens_used == 10
        assert result.error is None
    
    @patch('app.core.context.context_manager.VectorStore')
    @patch('app.core.context.context_manager.EmbeddingManager')
    @patch('app.core.context.context_manager.ContentChunker')
    @patch('app.core.context.context_manager.FileProcessor')
    def test_ingest_file_processing_failure(self, mock_file_proc_class, mock_chunker_class, mock_embed_class, mock_vector_class):
        """Test file ingestion with processing failure."""
        # Mock file processor failure
        mock_file_proc = Mock()
        mock_file_proc.process_file.return_value = ProcessingResult(
            success=False,
            content="",
            metadata={},
            error="File processing failed"
        )
        mock_file_proc_class.return_value = mock_file_proc
        
        test_file = Path(self.temp_dir) / "test.txt"
        test_file.write_text("Test content")
        
        manager = ContextManager(openai_api_key=self.api_key)
        result = manager.ingest_file(test_file)
        
        assert result.success is False
        assert "File processing failed" in result.error
        assert result.chunks_created == 0
    
    @patch('app.core.context.context_manager.VectorStore')
    @patch('app.core.context.context_manager.EmbeddingManager')
    @patch('app.core.context.context_manager.ContentChunker')
    @patch('app.core.context.context_manager.FileProcessor')
    def test_ingest_file_embedding_failure(self, mock_file_proc_class, mock_chunker_class, mock_embed_class, mock_vector_class):
        """Test file ingestion with embedding failure."""
        # Mock successful file processing
        mock_file_proc = Mock()
        mock_file_proc.process_file.return_value = ProcessingResult(
            success=True,
            content="Test content",
            metadata={},
            file_type=FileType.TEXT
        )
        mock_file_proc_class.return_value = mock_file_proc
        
        # Mock successful chunking
        mock_chunker = Mock()
        mock_chunker.chunk_content.return_value = [
            Chunk(id="1", content="Test", source_file="test.txt", chunk_index=0, token_count=1)
        ]
        mock_chunker_class.return_value = mock_chunker
        
        # Mock embedding failure
        mock_embed = Mock()
        mock_embed.embed_chunks.return_value = EmbeddingResult(
            success=False,
            error="Embedding generation failed"
        )
        mock_embed_class.return_value = mock_embed
        
        test_file = Path(self.temp_dir) / "test.txt"
        test_file.write_text("Test content")
        
        manager = ContextManager(openai_api_key=self.api_key)
        result = manager.ingest_file(test_file)
        
        assert result.success is False
        assert "Embedding generation failed" in result.error
        assert result.chunks_created == 1
    
    @patch('app.core.context.context_manager.VectorStore')
    @patch('app.core.context.context_manager.EmbeddingManager')
    @patch('app.core.context.context_manager.ContentChunker')
    @patch('app.core.context.context_manager.FileProcessor')
    def test_retrieve_context_success(self, mock_file_proc_class, mock_chunker_class, mock_embed_class, mock_vector_class):
        """Test successful context retrieval."""
        # Mock embedding manager
        mock_embed = Mock()
        mock_embed.generate_embeddings.return_value = EmbeddingResult(
            success=True,
            embeddings=[[0.1, 0.2, 0.3]],
            tokens_used=5
        )
        mock_embed_class.return_value = mock_embed
        
        # Mock vector store
        mock_vector = Mock()
        test_chunk = Chunk(
            id="result_1",
            content="Relevant content",
            source_file="doc.txt",
            chunk_index=0,
            token_count=3
        )
        mock_vector.search_with_mmr.return_value = [
            SearchResult(chunk=test_chunk, score=0.9, distance=0.1)
        ]
        mock_vector_class.return_value = mock_vector
        
        manager = ContextManager(openai_api_key=self.api_key)
        result = manager.retrieve_context("test query", k=5)
        
        assert result.success is True
        assert len(result.chunks) == 1
        assert result.chunks[0].content == "Relevant content"
        assert len(result.scores) == 1
        assert result.scores[0] == 0.9
        assert result.query == "test query"
    
    @patch('app.core.context.context_manager.VectorStore')
    @patch('app.core.context.context_manager.ContentChunker')
    @patch('app.core.context.context_manager.FileProcessor')
    def test_retrieve_context_no_embedding_manager(self, mock_file_proc_class, mock_chunker_class, mock_vector_class):
        """Test context retrieval without embedding manager."""
        manager = ContextManager()  # No API key
        result = manager.retrieve_context("test query")
        
        assert result.success is False
        assert "No embedding manager available" in result.error
        assert len(result.chunks) == 0
    
    @patch('app.core.context.context_manager.VectorStore')
    @patch('app.core.context.context_manager.EmbeddingManager')
    @patch('app.core.context.context_manager.ContentChunker')
    @patch('app.core.context.context_manager.FileProcessor')
    def test_retrieve_context_embedding_failure(self, mock_file_proc_class, mock_chunker_class, mock_embed_class, mock_vector_class):
        """Test context retrieval with embedding failure."""
        # Mock embedding failure
        mock_embed = Mock()
        mock_embed.generate_embeddings.return_value = EmbeddingResult(
            success=False,
            error="Query embedding failed"
        )
        mock_embed_class.return_value = mock_embed
        
        manager = ContextManager(openai_api_key=self.api_key)
        result = manager.retrieve_context("test query")
        
        assert result.success is False
        assert "Query embedding failed" in result.error
        assert len(result.chunks) == 0
    
    @patch('app.core.context.context_manager.VectorStore')
    @patch('app.core.context.context_manager.EmbeddingManager')
    @patch('app.core.context.context_manager.ContentChunker')
    @patch('app.core.context.context_manager.FileProcessor')
    def test_delete_file_content(self, mock_file_proc_class, mock_chunker_class, mock_embed_class, mock_vector_class):
        """Test deleting file content from vector store."""
        mock_vector = Mock()
        mock_vector.delete_by_source.return_value = True
        mock_vector_class.return_value = mock_vector
        
        manager = ContextManager(openai_api_key=self.api_key)
        result = manager.delete_file_content("test.txt")
        
        assert result is True
        mock_vector.delete_by_source.assert_called_once_with("test.txt")
    
    @patch('app.core.context.context_manager.VectorStore')
    @patch('app.core.context.context_manager.EmbeddingManager')
    @patch('app.core.context.context_manager.ContentChunker')
    @patch('app.core.context.context_manager.FileProcessor')
    def test_get_stats(self, mock_file_proc_class, mock_chunker_class, mock_embed_class, mock_vector_class):
        """Test getting system statistics."""
        mock_vector = Mock()
        mock_vector.get_collection_stats.return_value = {"total_chunks": 42}
        mock_vector_class.return_value = mock_vector
        
        mock_embed = Mock()
        mock_embed.model = "test-model"
        mock_embed_class.return_value = mock_embed
        
        mock_chunker = Mock()
        mock_chunker.chunk_size = 1000
        mock_chunker.chunk_overlap = 100
        mock_chunker_class.return_value = mock_chunker
        
        manager = ContextManager(openai_api_key=self.api_key)
        stats = manager.get_stats()
        
        assert stats["vector_store"]["total_chunks"] == 42
        assert stats["embedding_model"] == "test-model"
        assert stats["chunk_size"] == 1000
        assert stats["chunk_overlap"] == 100
        assert stats["api_key_available"] is True
    
    @patch('app.core.context.context_manager.VectorStore')
    @patch('app.core.context.context_manager.EmbeddingManager')
    @patch('app.core.context.context_manager.ContentChunker')
    @patch('app.core.context.context_manager.FileProcessor')
    def test_validate_setup(self, mock_file_proc_class, mock_chunker_class, mock_embed_class, mock_vector_class):
        """Test system setup validation."""
        mock_vector = Mock()
        mock_vector.get_collection_stats.return_value = {}
        mock_vector_class.return_value = mock_vector
        
        mock_embed = Mock()
        mock_embed.validate_api_key.return_value = True
        mock_embed_class.return_value = mock_embed
        
        manager = ContextManager(openai_api_key=self.api_key)
        validation = manager.validate_setup()
        
        assert validation["file_processor"] is True
        assert validation["chunker"] is True
        assert validation["vector_store"] is True
        assert validation["embedding_manager"] is True
        assert validation["api_key"] is True
    
    @patch('app.core.context.context_manager.VectorStore')
    @patch('app.core.context.context_manager.EmbeddingManager')
    @patch('app.core.context.context_manager.ContentChunker')
    @patch('app.core.context.context_manager.FileProcessor')
    def test_process_batch_files(self, mock_file_proc_class, mock_chunker_class, mock_embed_class, mock_vector_class):
        """Test batch file processing."""
        # Mock successful processing for all components
        mock_file_proc = Mock()
        mock_file_proc.process_file.return_value = ProcessingResult(
            success=True,
            content="Test content",
            metadata={},
            file_type=FileType.TEXT
        )
        mock_file_proc_class.return_value = mock_file_proc
        
        mock_chunker = Mock()
        mock_chunker.chunk_content.return_value = [
            Chunk(id="1", content="Test", source_file="test.txt", chunk_index=0, token_count=1)
        ]
        mock_chunker_class.return_value = mock_chunker
        
        mock_embed = Mock()
        mock_embed.embed_chunks.return_value = EmbeddingResult(
            success=True,
            embeddings=[[0.1, 0.2, 0.3]],
            tokens_used=5
        )
        mock_embed_class.return_value = mock_embed
        
        mock_vector = Mock()
        mock_vector.store_chunks.return_value = StorageResult(success=True, stored_count=1)
        mock_vector_class.return_value = mock_vector
        
        # Create test files
        test_files = []
        for i in range(3):
            test_file = Path(self.temp_dir) / f"test_{i}.txt"
            test_file.write_text(f"Test content {i}")
            test_files.append(test_file)
        
        manager = ContextManager(openai_api_key=self.api_key)
        results = manager.process_batch_files(test_files)
        
        assert len(results) == 3
        assert all(result.success for result in results)
        assert all(result.chunks_created == 1 for result in results)