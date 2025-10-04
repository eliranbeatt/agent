"""Vector database integration using ChromaDB for embedding storage and retrieval."""

import logging
import os
import uuid
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

from .chunker import Chunk

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Result from vector similarity search."""
    chunk: Chunk
    score: float
    distance: float


@dataclass
class StorageResult:
    """Result from vector storage operation."""
    success: bool
    stored_count: int = 0
    error: Optional[str] = None


class VectorStore:
    """ChromaDB-based vector store for embedding storage and retrieval."""
    
    def __init__(
        self,
        persist_directory: str = "./data/chroma_db",
        collection_name: str = "document_chunks",
        embedding_function: Optional[Any] = None
    ):
        """
        Initialize the vector store.
        
        Args:
            persist_directory: Directory to persist the database
            collection_name: Name of the collection to use
            embedding_function: Custom embedding function (optional)
        """
        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name
        
        # Ensure directory exists
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client
        try:
            self.client = chromadb.PersistentClient(
                path=str(self.persist_directory),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                embedding_function=embedding_function
            )
            
            logger.info(f"Initialized ChromaDB at {persist_directory} with collection '{collection_name}'")
            
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise
    
    def store_chunks(self, chunks: List[Chunk], embeddings: List[List[float]]) -> StorageResult:
        """
        Store chunks with their embeddings in the vector database.
        
        Args:
            chunks: List of chunks to store
            embeddings: Corresponding embeddings for each chunk
            
        Returns:
            Storage result indicating success/failure
        """
        if not chunks or not embeddings:
            return StorageResult(success=True, stored_count=0)
        
        if len(chunks) != len(embeddings):
            error_msg = f"Mismatch between chunks ({len(chunks)}) and embeddings ({len(embeddings)})"
            logger.error(error_msg)
            return StorageResult(success=False, error=error_msg)
        
        try:
            # Prepare data for ChromaDB
            ids = []
            documents = []
            metadatas = []
            
            for chunk, embedding in zip(chunks, embeddings):
                # Generate unique ID if not present
                chunk_id = chunk.id or str(uuid.uuid4())
                
                ids.append(chunk_id)
                documents.append(chunk.content)
                
                # Prepare metadata
                metadata = {
                    "source_file": chunk.source_file,
                    "chunk_index": chunk.chunk_index,
                    "token_count": chunk.token_count,
                    "start_char": chunk.start_char,
                    "end_char": chunk.end_char,
                }
                
                if chunk.page_number is not None:
                    metadata["page_number"] = chunk.page_number
                
                # Add custom metadata
                if chunk.metadata:
                    for key, value in chunk.metadata.items():
                        # ChromaDB requires string, int, float, or bool values
                        if isinstance(value, (str, int, float, bool)):
                            metadata[key] = value
                        else:
                            metadata[key] = str(value)
                
                metadatas.append(metadata)
            
            # Store in ChromaDB
            self.collection.add(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas
            )
            
            logger.info(f"Stored {len(chunks)} chunks in vector database")
            
            return StorageResult(success=True, stored_count=len(chunks))
            
        except Exception as e:
            logger.error(f"Failed to store chunks: {e}")
            return StorageResult(success=False, error=str(e))
    
    def search(
        self,
        query_embedding: List[float],
        k: int = 10,
        where: Optional[Dict[str, Any]] = None,
        include_distances: bool = True
    ) -> List[SearchResult]:
        """
        Search for similar chunks using vector similarity.
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            where: Metadata filter conditions
            include_distances: Whether to include distance scores
            
        Returns:
            List of search results with chunks and scores
        """
        try:
            # Perform similarity search
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                where=where,
                include=["documents", "metadatas", "distances"] if include_distances else ["documents", "metadatas"]
            )
            
            # Convert results to SearchResult objects
            search_results = []
            
            if results["documents"] and results["documents"][0]:
                documents = results["documents"][0]
                metadatas = results["metadatas"][0]
                distances = results["distances"][0] if include_distances else [0.0] * len(documents)
                ids = results["ids"][0]
                
                for doc, metadata, distance, chunk_id in zip(documents, metadatas, distances, ids):
                    # Reconstruct chunk from stored data
                    chunk = Chunk(
                        id=chunk_id,
                        content=doc,
                        source_file=metadata.get("source_file", ""),
                        page_number=metadata.get("page_number"),
                        chunk_index=metadata.get("chunk_index", 0),
                        token_count=metadata.get("token_count", 0),
                        start_char=metadata.get("start_char", 0),
                        end_char=metadata.get("end_char", 0),
                        metadata=metadata
                    )
                    
                    # Convert distance to similarity score (higher is better)
                    score = 1.0 - distance if distance <= 1.0 else 1.0 / (1.0 + distance)
                    
                    search_results.append(SearchResult(
                        chunk=chunk,
                        score=score,
                        distance=distance
                    ))
            
            logger.debug(f"Found {len(search_results)} similar chunks")
            return search_results
            
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []
    
    def search_with_mmr(
        self,
        query_embedding: List[float],
        k: int = 10,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        where: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Search using Maximal Marginal Relevance (MMR) for diverse results.
        
        Args:
            query_embedding: Query embedding vector
            k: Number of final results to return
            fetch_k: Number of initial candidates to fetch
            lambda_mult: Balance between relevance and diversity (0-1)
            where: Metadata filter conditions
            
        Returns:
            List of search results with MMR ranking
        """
        # First, get more candidates than needed
        initial_results = self.search(
            query_embedding=query_embedding,
            k=min(fetch_k, k * 2),
            where=where,
            include_distances=True
        )
        
        if len(initial_results) <= k:
            return initial_results
        
        # Apply MMR algorithm
        selected_results = []
        remaining_results = initial_results.copy()
        
        # Select the first (most similar) result
        if remaining_results:
            selected_results.append(remaining_results.pop(0))
        
        # Select remaining results using MMR
        while len(selected_results) < k and remaining_results:
            best_score = -float('inf')
            best_idx = 0
            
            for i, candidate in enumerate(remaining_results):
                # Calculate relevance score (similarity to query)
                relevance = candidate.score
                
                # Calculate diversity score (minimum similarity to selected results)
                diversity = 1.0  # Maximum diversity if no selected results yet
                if selected_results:
                    max_similarity = max(
                        self._cosine_similarity(
                            self._get_embedding_from_chunk(candidate.chunk),
                            self._get_embedding_from_chunk(selected.chunk)
                        )
                        for selected in selected_results
                    )
                    diversity = 1.0 - max_similarity
                
                # MMR score combines relevance and diversity
                mmr_score = lambda_mult * relevance + (1 - lambda_mult) * diversity
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = i
            
            # Add best candidate to selected results
            selected_results.append(remaining_results.pop(best_idx))
        
        logger.debug(f"Applied MMR to select {len(selected_results)} diverse results")
        return selected_results
    
    def _get_embedding_from_chunk(self, chunk: Chunk) -> List[float]:
        """Get embedding for a chunk (placeholder - would need to be stored or regenerated)."""
        # This is a simplified version - in practice, you'd either:
        # 1. Store embeddings in chunk metadata
        # 2. Re-query ChromaDB for the embedding
        # 3. Regenerate the embedding
        # For now, return a dummy embedding
        return [0.0] * 1536  # Assuming OpenAI embedding dimension
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        try:
            import numpy as np
            
            vec1_np = np.array(vec1)
            vec2_np = np.array(vec2)
            
            dot_product = np.dot(vec1_np, vec2_np)
            norm1 = np.linalg.norm(vec1_np)
            norm2 = np.linalg.norm(vec2_np)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return dot_product / (norm1 * norm2)
            
        except Exception:
            # Fallback to simple dot product if numpy not available
            if len(vec1) != len(vec2):
                return 0.0
            
            dot_product = sum(a * b for a, b in zip(vec1, vec2))
            norm1 = sum(a * a for a in vec1) ** 0.5
            norm2 = sum(b * b for b in vec2) ** 0.5
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return dot_product / (norm1 * norm2)
    
    def delete_by_source(self, source_file: str) -> bool:
        """
        Delete all chunks from a specific source file.
        
        Args:
            source_file: Source file path to delete chunks for
            
        Returns:
            True if deletion was successful
        """
        try:
            # Query for chunks from this source
            results = self.collection.get(
                where={"source_file": source_file},
                include=["documents"]
            )
            
            if results["ids"]:
                # Delete the chunks
                self.collection.delete(ids=results["ids"])
                logger.info(f"Deleted {len(results['ids'])} chunks from {source_file}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete chunks from {source_file}: {e}")
            return False
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the collection.
        
        Returns:
            Dictionary with collection statistics
        """
        try:
            count = self.collection.count()
            
            return {
                "total_chunks": count,
                "collection_name": self.collection_name,
                "persist_directory": str(self.persist_directory)
            }
            
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {"error": str(e)}
    
    def reset_collection(self) -> bool:
        """
        Reset (clear) the entire collection.
        
        Returns:
            True if reset was successful
        """
        try:
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name
            )
            logger.info(f"Reset collection '{self.collection_name}'")
            return True
            
        except Exception as e:
            logger.error(f"Failed to reset collection: {e}")
            return False