# Memory Management System

This module implements a comprehensive memory management system for Local Agent Studio using Mem0 for persistent memory storage.

## Components

### 1. Memory Manager (`memory_manager.py`)
The main interface for memory operations, integrating Mem0 with local storage.

**Features:**
- User profile management
- Fact storage with confidence scores and timestamps
- Conversation summaries with topic extraction
- RAG trace storage for retrieval pattern analysis
- Automatic retention policy application
- Cross-session persistence

**Key Methods:**
- `store_profile(profile)` - Store user profile
- `get_profile(user_id)` - Retrieve user profile
- `add_fact(content, source, ...)` - Add a fact to memory
- `update_conversation(session_id, summary, ...)` - Update conversation summary
- `store_rag_trace(query, chunks, ...)` - Store RAG retrieval trace
- `retrieve_relevant_memory(context, user_id)` - Retrieve relevant memories
- `search_facts(query, user_id)` - Search facts by query
- `analyze_retrieval_patterns(user_id, days)` - Analyze RAG patterns
- `apply_all_retention_policies()` - Apply retention policies

### 2. Conversation Manager (`conversation_manager.py`)
Manages conversation history with rolling windows and topic extraction.

**Features:**
- Message-level conversation tracking
- Rolling window for recent messages
- Topic extraction from conversations
- Automatic conversation summarization
- Session-based storage
- Cross-session persistence

**Key Methods:**
- `add_message(session_id, role, content)` - Add message to conversation
- `get_rolling_window(session_id, window_size)` - Get recent messages
- `get_full_history(session_id)` - Get all messages
- `extract_topics(messages)` - Extract key topics
- `generate_summary(session_id)` - Generate conversation summary
- `clear_session(session_id)` - Clear conversation session

### 3. Retention Policies (`retention_policies.py`)
Flexible retention policy system for managing memory lifecycle.

**Policy Types:**
- **TTLPolicy** - Time-to-live based retention
- **LRUPolicy** - Least Recently Used retention
- **FIFOPolicy** - First In First Out retention
- **SizeLimitPolicy** - Size-based retention
- **CompositePolicy** - Combine multiple policies

**Features:**
- Configurable TTL (time-to-live)
- Maximum item limits
- Size-based limits
- Policy composition
- Automatic cleanup

### 4. Data Models (`models.py`)
Structured data models for memory entities.

**Models:**
- `UserProfile` - User profile with preferences
- `Fact` - Stored fact with metadata
- `ConversationSummary` - Conversation summary with topics
- `RAGTrace` - RAG retrieval trace
- `MemoryContext` - Retrieved memory context
- `MemoryType` - Enum for memory types

## Configuration

Memory configuration is defined in `config/memory.yaml`:

```yaml
memory:
  mem0_enabled: true
  memory_db_path: "data/memory"
  
  # TTL settings
  profile_ttl_days: 365
  facts_ttl_days: 180
  conversation_ttl_days: 30
  
  # Retention policies
  max_memory_size_mb: 500
  retention_policy: "lru"  # lru, fifo, ttl, composite
  
  # Retrieval settings
  max_relevant_memories: 10
  similarity_threshold: 0.7
  context_window_size: 5
```

## Usage Example

```python
from app.core.memory import MemoryManager, UserProfile

# Initialize memory manager
memory_manager = MemoryManager(
    api_key="your-openai-api-key",
    storage_path="./data/memory",
    ttl_days=90,
)

# Store user profile
profile = UserProfile(
    user_id="user123",
    name="John Doe",
    preferences={"theme": "dark"},
)
memory_manager.store_profile(profile)

# Add a fact
memory_manager.add_fact(
    content="User prefers Python for backend development",
    source="conversation",
    user_id="user123",
    confidence=0.95,
)

# Add conversation message
memory_manager.add_conversation_message(
    session_id="session123",
    role="user",
    content="Tell me about machine learning",
    user_id="user123",
)

# Retrieve relevant memory
context = memory_manager.retrieve_relevant_memory(
    context="machine learning discussion",
    user_id="user123",
)

# Get conversation window
messages = memory_manager.get_conversation_window(
    session_id="session123",
    window_size=10,
)

# Apply retention policies
removed = memory_manager.apply_all_retention_policies()
```

## Testing

Comprehensive unit tests are provided:

- `test_memory_manager.py` - Memory manager tests (16 tests)
- `test_conversation_manager.py` - Conversation manager tests (14 tests)
- `test_retention_policies.py` - Retention policy tests (20 tests)

Run tests:
```bash
pytest tests/test_memory_manager.py -v
pytest tests/test_conversation_manager.py -v
pytest tests/test_retention_policies.py -v
```

## Requirements

The memory system requires the following dependencies:

- `mem0ai>=0.1.0` - Mem0 memory framework
- `openai>=1.0.0` - OpenAI API for embeddings
- `chromadb>=0.4.0` - Vector database for embeddings

Install with:
```bash
pip install mem0ai openai chromadb
```

## Architecture

The memory system follows a layered architecture:

1. **Storage Layer** - Local file system + ChromaDB for vectors
2. **Memory Layer** - Mem0 integration for semantic memory
3. **Management Layer** - Memory manager with retention policies
4. **Application Layer** - Conversation and context management

All data is stored locally with only OpenAI API calls for embeddings, ensuring privacy and control.
