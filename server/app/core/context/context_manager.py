"""Main context manager that orchestrates file processing, chunking, embedding, and retrieval."""

import logging
import os
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
from dataclasses import dataclass

from .file_processor import FileProcessor, ProcessingResult
from .chunker import ContentChunker, Chunk
from .embeddings import EmbeddingManager, EmbeddingResult
from .vector_store import VectorStore, SearchResult, StorageResult

logger = logging.getLogger(__name__)


@dataclass
class IngestionResult:
    """Result of file ingestion process."""
    success: bool
    file_path: str
    chunks_created: int = 0
    chunks_stored: int = 0
    tokens_used: int = 0
    error: Optional[str] = None


@dataclass
class RetrievalResult:
    """Result of context retrieval."""
    success: bool
    chunks: List[Chunk]
    scores: List[float]
    query: str
    total_results: int = 0
    error: Optional[str] = None


class ContextManager:
    """
    Main context manager that handles the complete pipeline:
    file ingestion → processing → chunking → embedding → storage → retrieval
    """
    
    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        vector_db_path: str = "./data/chroma_db",
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
        embedding_model: str = "text-embedding-3-small",
        context_config: Optional[Any] = None
    ):
        """
        Initialize the context manager.
        
        Args:
            openai_api_key: OpenAI API key for embeddings
            vector_db_path: Path to vector database storage
            chunk_size: Target chunk size in tokens
            chunk_overlap: Overlap between chunks in tokens
            embedding_model: OpenAI embedding model to use
            context_config: Optional ContextConfig object for file processing settings
        """
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        
        if not self.openai_api_key:
            logger.warning("No OpenAI API key provided. Embedding functionality will be limited.")
        
        # Initialize components
        try:
            self.file_processor = FileProcessor(config=context_config)
            self.chunker = ContentChunker(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            
            if self.openai_api_key:
                self.embedding_manager = EmbeddingManager(
                    api_key=self.openai_api_key,
                    model=embedding_model
                )
            else:
                self.embedding_manager = None
            
            self.vector_store = VectorStore(persist_directory=vector_db_path)
            
            logger.info("Context manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize context manager: {e}")
            raise
    
    def ingest_file(self, file_path: Union[str, Path]) -> IngestionResult:
        """
        Complete file ingestion pipeline: process → chunk → embed → store.
        
        Args:
            file_path: Path to file to ingest
            
        Returns:
            Ingestion result with success status and metrics
        """
        file_path = Path(file_path)
        logger.info(f"Starting ingestion of file: {file_path}")
        
        try:
            # Step 1: Process file
            processing_result = self.file_processor.process_file(file_path)
            
            if not processing_result.success:
                return IngestionResult(
                    success=False,
                    file_path=str(file_path),
                    error=f"File processing failed: {processing_result.error}"
                )
            
            # Step 2: Chunk content
            chunks = self.chunker.chunk_content(
                content=processing_result.content,
                source_file=str(file_path),
                page_number=processing_result.page_count,
                metadata=processing_result.metadata
            )
            
            if not chunks:
                return IngestionResult(
                    success=False,
                    file_path=str(file_path),
                    error="No chunks created from file content"
                )
            
            # Step 3: Generate embeddings (if available)
            tokens_used = 0
            if self.embedding_manager:
                embedding_result = self.embedding_manager.embed_chunks(chunks)
                
                if not embedding_result.success:
                    return IngestionResult(
                        success=False,
                        file_path=str(file_path),
                        chunks_created=len(chunks),
                        error=f"Embedding generation failed: {embedding_result.error}"
                    )
                
                tokens_used = embedding_result.tokens_used
                
                # Step 4: Store in vector database
                storage_result = self.vector_store.store_chunks(chunks, embedding_result.embeddings)
                
                if not storage_result.success:
                    return IngestionResult(
                        success=False,
                        file_path=str(file_path),
                        chunks_created=len(chunks),
                        tokens_used=tokens_used,
                        error=f"Vector storage failed: {storage_result.error}"
                    )
                
                chunks_stored = storage_result.stored_count
            else:
                # Store without embeddings (metadata only)
                dummy_embeddings = [[0.0] * 1536] * len(chunks)  # Placeholder embeddings
                storage_result = self.vector_store.store_chunks(chunks, dummy_embeddings)
                chunks_stored = storage_result.stored_count if storage_result.success else 0
            
            logger.info(f"Successfully ingested {file_path}: {len(chunks)} chunks, {tokens_used} tokens")
            
            return IngestionResult(
                success=True,
                file_path=str(file_path),
                chunks_created=len(chunks),
                chunks_stored=chunks_stored,
                tokens_used=tokens_used
            )
            
        except Exception as e:
            logger.error(f"File ingestion failed for {file_path}: {e}")
            return IngestionResult(
                success=False,
                file_path=str(file_path),
                error=str(e)
            )
    
    def retrieve_context(
        self,
        query: str,
        k: int = 10,
        use_mmr: bool = True,
        source_filter: Optional[str] = None,
        page_filter: Optional[int] = None
    ) -> RetrievalResult:
        """
        Retrieve relevant context chunks for a query.
        
        Args:
            query: Query text to search for
            k: Number of results to return (8-12 recommended)
            use_mmr: Whether to use MMR for diverse results
            source_filter: Filter by specific source file
            page_filter: Filter by specific page number
            
        Returns:
            Retrieval result with relevant chunks and scores
        """
        if not self.embedding_manager:
            return RetrievalResult(
                success=False,
                chunks=[],
                scores=[],
                query=query,
                error="No embedding manager available (missing OpenAI API key)"
            )
        
        try:
            # Generate query embedding
            embedding_result = self.embedding_manager.generate_embeddings([query])
            
            if not embedding_result.success:
                return RetrievalResult(
                    success=False,
                    chunks=[],
                    scores=[],
                    query=query,
                    error=f"Query embedding failed: {embedding_result.error}"
                )
            
            query_embedding = embedding_result.embeddings[0]
            
            # Build metadata filter
            where_filter = {}
            if source_filter:
                where_filter["source_file"] = source_filter
            if page_filter is not None:
                where_filter["page_number"] = page_filter
            
            # Perform search
            if use_mmr:
                search_results = self.vector_store.search_with_mmr(
                    query_embedding=query_embedding,
                    k=k,
                    fetch_k=k * 2,
                    lambda_mult=0.5,
                    where=where_filter if where_filter else None
                )
            else:
                search_results = self.vector_store.search(
                    query_embedding=query_embedding,
                    k=k,
                    where=where_filter if where_filter else None
                )
            
            # Extract chunks and scores
            chunks = [result.chunk for result in search_results]
            scores = [result.score for result in search_results]
            
            logger.info(f"Retrieved {len(chunks)} relevant chunks for query: '{query[:50]}...'")
            
            return RetrievalResult(
                success=True,
                chunks=chunks,
                scores=scores,
                query=query,
                total_results=len(chunks)
            )
            
        except Exception as e:
            logger.error(f"Context retrieval failed for query '{query}': {e}")
            return RetrievalResult(
                success=False,
                chunks=[],
                scores=[],
                query=query,
                error=str(e)
            )
    
    def delete_file_content(self, file_path: Union[str, Path]) -> bool:
        """
        Delete all content associated with a file from the vector store.
        
        Args:
            file_path: Path to file whose content should be deleted
            
        Returns:
            True if deletion was successful
        """
        try:
            return self.vector_store.delete_by_source(str(file_path))
        except Exception as e:
            logger.error(f"Failed to delete content for {file_path}: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the context management system.
        
        Returns:
            Dictionary with system statistics
        """
        try:
            vector_stats = self.vector_store.get_collection_stats()
            
            stats = {
                "vector_store": vector_stats,
                "embedding_model": self.embedding_manager.model if self.embedding_manager else None,
                "chunk_size": self.chunker.chunk_size,
                "chunk_overlap": self.chunker.chunk_overlap,
                "api_key_available": self.openai_api_key is not None
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {"error": str(e)}
    
    def validate_setup(self) -> Dict[str, bool]:
        """
        Validate that all components are properly set up.
        
        Returns:
            Dictionary with validation results for each component
        """
        results = {
            "file_processor": True,  # Always available
            "chunker": True,  # Always available
            "vector_store": True,  # Always available
            "embedding_manager": False,
            "api_key": False
        }
        
        try:
            # Test vector store
            self.vector_store.get_collection_stats()
            results["vector_store"] = True
        except Exception:
            results["vector_store"] = False
        
        # Test embedding manager and API key
        if self.embedding_manager:
            results["embedding_manager"] = True
            results["api_key"] = self.embedding_manager.validate_api_key()
        
        return results
    
    def process_batch_files(self, file_paths: List[Union[str, Path]]) -> List[IngestionResult]:
        """
        Process multiple files in batch.
        
        Args:
            file_paths: List of file paths to process
            
        Returns:
            List of ingestion results for each file
        """
        results = []
        
        for file_path in file_paths:
            logger.info(f"Processing file {len(results) + 1}/{len(file_paths)}: {file_path}")
            result = self.ingest_file(file_path)
            results.append(result)
            
            if not result.success:
                logger.warning(f"Failed to process {file_path}: {result.error}")
        
        successful = sum(1 for r in results if r.success)
        logger.info(f"Batch processing complete: {successful}/{len(file_paths)} files successful")
        
        return results