"""Context management module for file processing and retrieval."""

# Import modules with graceful handling of missing dependencies
__all__ = []

try:
    from .chunker import ContentChunker, Chunk
    __all__.extend(["ContentChunker", "Chunk"])
except ImportError as e:
    print(f"Warning: Could not import chunker: {e}")

try:
    from .file_processor import FileProcessor
    __all__.append("FileProcessor")
except ImportError as e:
    print(f"Warning: Could not import file_processor: {e}")

try:
    from .embeddings import EmbeddingManager
    __all__.append("EmbeddingManager")
except ImportError as e:
    print(f"Warning: Could not import embeddings: {e}")

try:
    from .vector_store import VectorStore
    __all__.append("VectorStore")
except ImportError as e:
    print(f"Warning: Could not import vector_store: {e}")

try:
    from .context_manager import ContextManager
    __all__.append("ContextManager")
except ImportError as e:
    print(f"Warning: Could not import context_manager: {e}")