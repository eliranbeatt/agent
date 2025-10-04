"""Tests for embedding generation functionality."""

import pytest
from unittest.mock import Mock, patch, MagicMock

from app.core.context.embeddings import EmbeddingManager, EmbeddingResult
from app.core.context.chunker import Chunk


class TestEmbeddingManager:
    """Test cases for EmbeddingManager class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.api_key = "test-api-key"
        
    @patch('app.core.context.embeddings.OpenAI')
    def test_initialization_success(self, mock_openai):
        """Test successful initialization of EmbeddingManager."""
        mock_client = Mock()
        mock_openai.return_value = mock_client
        
        manager = EmbeddingManager(api_key=self.api_key)
        
        assert manager.model == "text-embedding-3-small"
        assert manager.max_retries == 3
        mock_openai.assert_called_once_with(api_key=self.api_key)
    
    @patch('app.core.context.embeddings.OpenAI')
    def test_initialization_failure(self, mock_openai):
        """Test initialization failure handling."""
        mock_openai.side_effect = Exception("API key invalid")
        
        with pytest.raises(Exception):
            EmbeddingManager(api_key="invalid-key")
    
    @patch('app.core.context.embeddings.OpenAI')
    def test_generate_embeddings_success(self, mock_openai):
        """Test successful embedding generation."""
        # Mock OpenAI client and response
        mock_client = Mock()
        mock_openai.return_value = mock_client
        
        # Mock embedding response
        mock_response = Mock()
        mock_response.data = [
            Mock(embedding=[0.1, 0.2, 0.3]),
            Mock(embedding=[0.4, 0.5, 0.6])
        ]
        mock_response.usage.total_tokens = 10
        mock_client.embeddings.create.return_value = mock_response
        
        manager = EmbeddingManager(api_key=self.api_key)
        texts = ["First text", "Second text"]
        
        result = manager.generate_embeddings(texts)
        
        assert result.success is True
        assert len(result.embeddings) == 2
        assert result.embeddings[0] == [0.1, 0.2, 0.3]
        assert result.embeddings[1] == [0.4, 0.5, 0.6]
        assert result.tokens_used == 10
        assert result.error is None
    
    @patch('app.core.context.embeddings.OpenAI')
    def test_generate_embeddings_empty_input(self, mock_openai):
        """Test embedding generation with empty input."""
        mock_client = Mock()
        mock_openai.return_value = mock_client
        
        manager = EmbeddingManager(api_key=self.api_key)
        
        result = manager.generate_embeddings([])
        
        assert result.success is True
        assert result.embeddings == []
        assert result.tokens_used == 0
    
    @patch('app.core.context.embeddings.OpenAI')
    def test_generate_embeddings_api_error(self, mock_openai):
        """Test handling of API errors during embedding generation."""
        mock_client = Mock()
        mock_openai.return_value = mock_client
        
        # Mock API error
        from openai import APIError
        mock_client.embeddings.create.side_effect = APIError("API Error")
        
        manager = EmbeddingManager(api_key=self.api_key, max_retries=1)
        texts = ["Test text"]
        
        result = manager.generate_embeddings(texts)
        
        assert result.success is False
        assert result.error is not None
        assert "API Error" in result.error
    
    @patch('app.core.context.embeddings.OpenAI')
    @patch('app.core.context.embeddings.time.sleep')
    def test_generate_embeddings_rate_limit_retry(self, mock_sleep, mock_openai):
        """Test retry logic for rate limit errors."""
        mock_client = Mock()
        mock_openai.return_value = mock_client
        
        # Mock rate limit error on first call, success on second
        from openai import RateLimitError
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1, 0.2, 0.3])]
        mock_response.usage.total_tokens = 5
        
        mock_client.embeddings.create.side_effect = [
            RateLimitError("Rate limit exceeded"),
            mock_response
        ]
        
        manager = EmbeddingManager(api_key=self.api_key, max_retries=2)
        texts = ["Test text"]
        
        result = manager.generate_embeddings(texts)
        
        assert result.success is True
        assert len(result.embeddings) == 1
        mock_sleep.assert_called()  # Should have slept for retry
    
    @patch('app.core.context.embeddings.OpenAI')
    def test_embed_chunks(self, mock_openai):
        """Test embedding generation for chunks."""
        mock_client = Mock()
        mock_openai.return_value = mock_client
        
        # Mock embedding response
        mock_response = Mock()
        mock_response.data = [
            Mock(embedding=[0.1, 0.2, 0.3]),
            Mock(embedding=[0.4, 0.5, 0.6])
        ]
        mock_response.usage.total_tokens = 15
        mock_client.embeddings.create.return_value = mock_response
        
        manager = EmbeddingManager(api_key=self.api_key)
        
        # Create test chunks
        chunks = [
            Chunk(
                id="chunk_1",
                content="First chunk content",
                source_file="test.txt",
                chunk_index=0,
                token_count=5
            ),
            Chunk(
                id="chunk_2", 
                content="Second chunk content",
                source_file="test.txt",
                chunk_index=1,
                token_count=6
            )
        ]
        
        result = manager.embed_chunks(chunks)
        
        assert result.success is True
        assert len(result.embeddings) == 2
        assert result.tokens_used == 15
    
    @patch('app.core.context.embeddings.OpenAI')
    def test_batch_processing(self, mock_openai):
        """Test batch processing of large text lists."""
        mock_client = Mock()
        mock_openai.return_value = mock_client
        
        # Mock responses for two batches
        mock_response_1 = Mock()
        mock_response_1.data = [Mock(embedding=[0.1, 0.2]) for _ in range(2)]
        mock_response_1.usage.total_tokens = 10
        
        mock_response_2 = Mock()
        mock_response_2.data = [Mock(embedding=[0.3, 0.4]) for _ in range(1)]
        mock_response_2.usage.total_tokens = 5
        
        mock_client.embeddings.create.side_effect = [mock_response_1, mock_response_2]
        
        manager = EmbeddingManager(api_key=self.api_key, batch_size=2)
        texts = ["Text 1", "Text 2", "Text 3"]  # 3 texts, batch size 2
        
        result = manager.generate_embeddings(texts)
        
        assert result.success is True
        assert len(result.embeddings) == 3
        assert result.tokens_used == 15  # 10 + 5
        assert mock_client.embeddings.create.call_count == 2
    
    def test_get_embedding_dimension(self):
        """Test getting embedding dimensions for different models."""
        with patch('app.core.context.embeddings.OpenAI'):
            manager = EmbeddingManager(api_key=self.api_key, model="text-embedding-3-small")
            assert manager.get_embedding_dimension() == 1536
            
            manager = EmbeddingManager(api_key=self.api_key, model="text-embedding-3-large")
            assert manager.get_embedding_dimension() == 3072
            
            manager = EmbeddingManager(api_key=self.api_key, model="unknown-model")
            assert manager.get_embedding_dimension() == 1536  # Default
    
    @patch('app.core.context.embeddings.OpenAI')
    def test_validate_api_key_success(self, mock_openai):
        """Test successful API key validation."""
        mock_client = Mock()
        mock_openai.return_value = mock_client
        
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1, 0.2, 0.3])]
        mock_response.usage.total_tokens = 1
        mock_client.embeddings.create.return_value = mock_response
        
        manager = EmbeddingManager(api_key=self.api_key)
        
        assert manager.validate_api_key() is True
    
    @patch('app.core.context.embeddings.OpenAI')
    def test_validate_api_key_failure(self, mock_openai):
        """Test API key validation failure."""
        mock_client = Mock()
        mock_openai.return_value = mock_client
        
        mock_client.embeddings.create.side_effect = Exception("Invalid API key")
        
        manager = EmbeddingManager(api_key=self.api_key)
        
        assert manager.validate_api_key() is False