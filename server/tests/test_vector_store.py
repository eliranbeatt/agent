"""Tests for vector store functionality."""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from app.core.context.vector_store import VectorStore, SearchResult, StorageResult
from app.core.context.chunker import Chunk


class TestVectorStore:
    """Test cases for VectorStore class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create temporary directory for test database
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test_chroma"
        
    def teardown_method(self):
        """Clean up test fixtures."""
        # Remove temporary directory
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
    
    @patch('app.core.context.vector_store.chromadb.PersistentClient')
    def test_initialization_success(self, mock_client_class):
        """Test successful VectorStore initialization."""
        mock_client = Mock()
        mock_collection = Mock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_client_class.return_value = mock_client
        
        store = VectorStore(persist_directory=str(self.db_path))
        
        assert store.persist_directory == self.db_path
        assert store.collection_name == "document_chunks"
        mock_client_class.assert_called_once()
        mock_client.get_or_create_collection.assert_called_once()
    
    @patch('app.core.context.vector_store.chromadb.PersistentClient')
    def test_initialization_failure(self, mock_client_class):
        """Test VectorStore initialization failure."""
        mock_client_class.side_effect = Exception("ChromaDB initialization failed")
        
        with pytest.raises(Exception):
            VectorStore(persist_directory=str(self.db_path))
    
    @patch('app.core.context.vector_store.chromadb.PersistentClient')
    def test_store_chunks_success(self, mock_client_class):
        """Test successful chunk storage."""
        mock_client = Mock()
        mock_collection = Mock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_client_class.return_value = mock_client
        
        store = VectorStore(persist_directory=str(self.db_path))
        
        # Create test chunks
        chunks = [
            Chunk(
                id="chunk_1",
                content="First chunk content",
                source_file="test.txt",
                chunk_index=0,
                token_count=5,
                start_char=0,
                end_char=20,
                page_number=1
            ),
            Chunk(
                id="chunk_2",
                content="Second chunk content", 
                source_file="test.txt",
                chunk_index=1,
                token_count=6,
                start_char=20,
                end_char=42
            )
        ]
        
        embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        
        result = store.store_chunks(chunks, embeddings)
        
        assert result.success is True
        assert result.stored_count == 2
        assert result.error is None
        
        # Verify collection.add was called with correct parameters
        mock_collection.add.assert_called_once()
        call_args = mock_collection.add.call_args[1]
        
        assert len(call_args["ids"]) == 2
        assert len(call_args["documents"]) == 2
        assert len(call_args["embeddings"]) == 2
        assert len(call_args["metadatas"]) == 2
        
        # Check metadata structure
        metadata_1 = call_args["metadatas"][0]
        assert metadata_1["source_file"] == "test.txt"
        assert metadata_1["chunk_index"] == 0
        assert metadata_1["page_number"] == 1
    
    @patch('app.core.context.vector_store.chromadb.PersistentClient')
    def test_store_chunks_mismatch_error(self, mock_client_class):
        """Test error when chunks and embeddings count mismatch."""
        mock_client = Mock()
        mock_collection = Mock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_client_class.return_value = mock_client
        
        store = VectorStore(persist_directory=str(self.db_path))
        
        chunks = [Chunk(id="1", content="test", source_file="test.txt")]
        embeddings = [[0.1, 0.2], [0.3, 0.4]]  # Mismatch: 1 chunk, 2 embeddings
        
        result = store.store_chunks(chunks, embeddings)
        
        assert result.success is False
        assert "Mismatch between chunks" in result.error
    
    @patch('app.core.context.vector_store.chromadb.PersistentClient')
    def test_store_empty_chunks(self, mock_client_class):
        """Test storing empty chunks list."""
        mock_client = Mock()
        mock_collection = Mock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_client_class.return_value = mock_client
        
        store = VectorStore(persist_directory=str(self.db_path))
        
        result = store.store_chunks([], [])
        
        assert result.success is True
        assert result.stored_count == 0
    
    @patch('app.core.context.vector_store.chromadb.PersistentClient')
    def test_search_success(self, mock_client_class):
        """Test successful vector search."""
        mock_client = Mock()
        mock_collection = Mock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_client_class.return_value = mock_client
        
        # Mock search results
        mock_collection.query.return_value = {
            "documents": [["First chunk", "Second chunk"]],
            "metadatas": [[
                {"source_file": "test.txt", "chunk_index": 0, "token_count": 5},
                {"source_file": "test.txt", "chunk_index": 1, "token_count": 6}
            ]],
            "distances": [[0.1, 0.3]],
            "ids": [["chunk_1", "chunk_2"]]
        }
        
        store = VectorStore(persist_directory=str(self.db_path))
        query_embedding = [0.2, 0.3, 0.4]
        
        results = store.search(query_embedding, k=2)
        
        assert len(results) == 2
        assert isinstance(results[0], SearchResult)
        assert results[0].chunk.content == "First chunk"
        assert results[0].chunk.source_file == "test.txt"
        assert results[0].distance == 0.1
        assert results[0].score > results[1].score  # Lower distance = higher score
    
    @patch('app.core.context.vector_store.chromadb.PersistentClient')
    def test_search_with_filter(self, mock_client_class):
        """Test vector search with metadata filter."""
        mock_client = Mock()
        mock_collection = Mock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_client_class.return_value = mock_client
        
        mock_collection.query.return_value = {
            "documents": [["Filtered chunk"]],
            "metadatas": [[{"source_file": "specific.txt", "chunk_index": 0}]],
            "distances": [[0.2]],
            "ids": [["chunk_1"]]
        }
        
        store = VectorStore(persist_directory=str(self.db_path))
        query_embedding = [0.1, 0.2, 0.3]
        where_filter = {"source_file": "specific.txt"}
        
        results = store.search(query_embedding, k=5, where=where_filter)
        
        assert len(results) == 1
        assert results[0].chunk.source_file == "specific.txt"
        
        # Verify filter was passed to query
        mock_collection.query.assert_called_once()
        call_args = mock_collection.query.call_args[1]
        assert call_args["where"] == where_filter
    
    @patch('app.core.context.vector_store.chromadb.PersistentClient')
    def test_search_no_results(self, mock_client_class):
        """Test search with no results."""
        mock_client = Mock()
        mock_collection = Mock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_client_class.return_value = mock_client
        
        mock_collection.query.return_value = {
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
            "ids": [[]]
        }
        
        store = VectorStore(persist_directory=str(self.db_path))
        query_embedding = [0.1, 0.2, 0.3]
        
        results = store.search(query_embedding, k=5)
        
        assert len(results) == 0
    
    @patch('app.core.context.vector_store.chromadb.PersistentClient')
    def test_search_mmr(self, mock_client_class):
        """Test MMR search functionality."""
        mock_client = Mock()
        mock_collection = Mock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_client_class.return_value = mock_client
        
        # Mock search results for MMR
        mock_collection.query.return_value = {
            "documents": [["First chunk", "Second chunk", "Third chunk"]],
            "metadatas": [[
                {"source_file": "test.txt", "chunk_index": 0},
                {"source_file": "test.txt", "chunk_index": 1},
                {"source_file": "test.txt", "chunk_index": 2}
            ]],
            "distances": [[0.1, 0.2, 0.3]],
            "ids": [["chunk_1", "chunk_2", "chunk_3"]]
        }
        
        store = VectorStore(persist_directory=str(self.db_path))
        query_embedding = [0.1, 0.2, 0.3]
        
        results = store.search_with_mmr(query_embedding, k=2, fetch_k=3)
        
        # Should return at most k results
        assert len(results) <= 2
        assert all(isinstance(r, SearchResult) for r in results)
    
    @patch('app.core.context.vector_store.chromadb.PersistentClient')
    def test_delete_by_source(self, mock_client_class):
        """Test deleting chunks by source file."""
        mock_client = Mock()
        mock_collection = Mock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_client_class.return_value = mock_client
        
        # Mock get results for deletion
        mock_collection.get.return_value = {
            "ids": ["chunk_1", "chunk_2"]
        }
        
        store = VectorStore(persist_directory=str(self.db_path))
        
        result = store.delete_by_source("test.txt")
        
        assert result is True
        mock_collection.get.assert_called_once_with(
            where={"source_file": "test.txt"},
            include=["documents"]
        )
        mock_collection.delete.assert_called_once_with(ids=["chunk_1", "chunk_2"])
    
    @patch('app.core.context.vector_store.chromadb.PersistentClient')
    def test_get_collection_stats(self, mock_client_class):
        """Test getting collection statistics."""
        mock_client = Mock()
        mock_collection = Mock()
        mock_collection.count.return_value = 42
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_client_class.return_value = mock_client
        
        store = VectorStore(persist_directory=str(self.db_path))
        
        stats = store.get_collection_stats()
        
        assert stats["total_chunks"] == 42
        assert stats["collection_name"] == "document_chunks"
        assert str(self.db_path) in stats["persist_directory"]
    
    @patch('app.core.context.vector_store.chromadb.PersistentClient')
    def test_reset_collection(self, mock_client_class):
        """Test resetting the collection."""
        mock_client = Mock()
        mock_collection = Mock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_client.create_collection.return_value = mock_collection
        mock_client_class.return_value = mock_client
        
        store = VectorStore(persist_directory=str(self.db_path))
        
        result = store.reset_collection()
        
        assert result is True
        mock_client.delete_collection.assert_called_once_with("document_chunks")
        mock_client.create_collection.assert_called_once_with(name="document_chunks")
    
    def test_cosine_similarity(self):
        """Test cosine similarity calculation."""
        store = VectorStore(persist_directory=str(self.db_path))
        
        # Test identical vectors
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [1.0, 0.0, 0.0]
        similarity = store._cosine_similarity(vec1, vec2)
        assert abs(similarity - 1.0) < 1e-6
        
        # Test orthogonal vectors
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [0.0, 1.0, 0.0]
        similarity = store._cosine_similarity(vec1, vec2)
        assert abs(similarity - 0.0) < 1e-6
        
        # Test zero vectors
        vec1 = [0.0, 0.0, 0.0]
        vec2 = [1.0, 0.0, 0.0]
        similarity = store._cosine_similarity(vec1, vec2)
        assert similarity == 0.0