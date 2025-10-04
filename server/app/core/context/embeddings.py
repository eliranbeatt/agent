"""Embedding generation and management using OpenAI API."""

import asyncio
import logging
import time
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import openai
from openai import OpenAI

from .chunker import Chunk

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingResult:
    """Result of embedding generation."""
    success: bool
    embeddings: Optional[List[List[float]]] = None
    error: Optional[str] = None
    tokens_used: int = 0


class EmbeddingManager:
    """Manages embedding generation with OpenAI API integration."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "text-embedding-3-small",
        max_retries: int = 3,
        retry_delay: float = 1.0,
        batch_size: int = 100,
        rate_limit_delay: float = 0.1
    ):
        """
        Initialize the embedding manager.
        
        Args:
            api_key: OpenAI API key
            model: Embedding model to use
            max_retries: Maximum number of retries for failed requests
            retry_delay: Initial delay between retries (exponential backoff)
            batch_size: Maximum number of texts to embed in one request
            rate_limit_delay: Delay between requests to avoid rate limits
        """
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.batch_size = batch_size
        self.rate_limit_delay = rate_limit_delay
        
        # Initialize OpenAI client
        try:
            self.client = OpenAI(api_key=api_key)
            logger.info(f"Initialized OpenAI client with model: {model}")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            raise
    
    def generate_embeddings(self, texts: List[str]) -> EmbeddingResult:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            Embedding result with embeddings or error information
        """
        if not texts:
            return EmbeddingResult(success=True, embeddings=[], tokens_used=0)
        
        try:
            # Process in batches to avoid API limits
            all_embeddings = []
            total_tokens = 0
            
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i + self.batch_size]
                batch_result = self._generate_batch_embeddings(batch)
                
                if not batch_result.success:
                    return batch_result
                
                all_embeddings.extend(batch_result.embeddings)
                total_tokens += batch_result.tokens_used
                
                # Rate limiting delay
                if i + self.batch_size < len(texts):
                    time.sleep(self.rate_limit_delay)
            
            return EmbeddingResult(
                success=True,
                embeddings=all_embeddings,
                tokens_used=total_tokens
            )
            
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return EmbeddingResult(
                success=False,
                error=str(e)
            )
    
    def _generate_batch_embeddings(self, texts: List[str]) -> EmbeddingResult:
        """Generate embeddings for a batch of texts with retry logic."""
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                # Clean texts - remove empty strings and excessive whitespace
                cleaned_texts = [text.strip() for text in texts if text.strip()]
                
                if not cleaned_texts:
                    return EmbeddingResult(success=True, embeddings=[], tokens_used=0)
                
                # Make API request
                response = self.client.embeddings.create(
                    model=self.model,
                    input=cleaned_texts
                )
                
                # Extract embeddings
                embeddings = [data.embedding for data in response.data]
                tokens_used = response.usage.total_tokens
                
                logger.debug(f"Generated {len(embeddings)} embeddings using {tokens_used} tokens")
                
                return EmbeddingResult(
                    success=True,
                    embeddings=embeddings,
                    tokens_used=tokens_used
                )
                
            except openai.RateLimitError as e:
                logger.warning(f"Rate limit hit on attempt {attempt + 1}: {e}")
                last_error = str(e)
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2 ** attempt)  # Exponential backoff
                    logger.info(f"Waiting {delay} seconds before retry...")
                    time.sleep(delay)
                
            except openai.APIError as e:
                logger.error(f"OpenAI API error on attempt {attempt + 1}: {e}")
                last_error = str(e)
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2 ** attempt)
                    time.sleep(delay)
                
            except Exception as e:
                logger.error(f"Unexpected error on attempt {attempt + 1}: {e}")
                last_error = str(e)
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
        
        return EmbeddingResult(
            success=False,
            error=f"Failed after {self.max_retries} attempts. Last error: {last_error}"
        )
    
    def embed_chunks(self, chunks: List[Chunk]) -> EmbeddingResult:
        """
        Generate embeddings for a list of chunks.
        
        Args:
            chunks: List of chunks to embed
            
        Returns:
            Embedding result with embeddings matching chunk order
        """
        if not chunks:
            return EmbeddingResult(success=True, embeddings=[], tokens_used=0)
        
        # Extract text content from chunks
        texts = [chunk.content for chunk in chunks]
        
        # Generate embeddings
        result = self.generate_embeddings(texts)
        
        if result.success:
            logger.info(f"Generated embeddings for {len(chunks)} chunks using {result.tokens_used} tokens")
        
        return result
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings for the current model.
        
        Returns:
            Embedding dimension
        """
        # Known dimensions for OpenAI models
        model_dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }
        
        return model_dimensions.get(self.model, 1536)  # Default to 1536
    
    def validate_api_key(self) -> bool:
        """
        Validate that the API key is working.
        
        Returns:
            True if API key is valid, False otherwise
        """
        try:
            # Test with a simple embedding request
            response = self.client.embeddings.create(
                model=self.model,
                input=["test"]
            )
            logger.info("API key validation successful")
            return True
            
        except Exception as e:
            logger.error(f"API key validation failed: {e}")
            return False