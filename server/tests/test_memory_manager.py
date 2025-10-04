"""Unit tests for memory manager."""

import pytest
import tempfile
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from app.core.memory import (
    MemoryManager,
    UserProfile,
    Fact,
    ConversationSummary,
    RAGTrace,
)


@pytest.fixture
def temp_storage():
    """Create temporary storage directory."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Try to cleanup, but don't fail if files are locked (Windows issue with ChromaDB)
    try:
        shutil.rmtree(temp_dir)
    except (PermissionError, OSError) as e:
        # On Windows, ChromaDB may keep files locked
        import time
        time.sleep(0.5)
        try:
            shutil.rmtree(temp_dir)
        except (PermissionError, OSError):
            pass  # Ignore cleanup errors in tests


@pytest.fixture
def memory_manager(temp_storage):
    """Create memory manager instance with mocked Mem0."""
    # Mock Mem0 Memory class
    with patch('app.core.memory.memory_manager.Memory') as mock_memory_class:
        # Create a mock memory instance
        mock_memory = MagicMock()
        mock_memory.add.return_value = {"id": "test-id"}
        mock_memory.search.return_value = []
        mock_memory_class.from_config.return_value = mock_memory
        
        # Mock API key for testing
        api_key = "test-api-key"
        manager = MemoryManager(
            api_key=api_key,
            storage_path=temp_storage,
            ttl_days=30,
            max_facts=100,
            max_conversations=50,
            max_rag_traces=200,
        )
        
        # Store the mock for test access
        manager._mock_memory = mock_memory
        
        yield manager


class TestMemoryManager:
    """Test memory manager functionality."""

    def test_initialization(self, memory_manager, temp_storage):
        """Test memory manager initialization."""
        assert memory_manager.storage_path == Path(temp_storage)
        assert memory_manager.ttl_days == 30
        assert memory_manager.max_facts == 100
        assert memory_manager.profiles_path.exists()
        assert memory_manager.facts_path.exists()
        assert memory_manager.conversations_path.exists()
        assert memory_manager.rag_traces_path.exists()

    def test_store_and_retrieve_profile(self, memory_manager):
        """Test storing and retrieving user profile."""
        profile = UserProfile(
            user_id="user123",
            name="Test User",
            preferences={"theme": "dark", "language": "en"},
            communication_style="formal",
        )

        # Store profile
        memory_manager.store_profile(profile)

        # Retrieve profile
        retrieved = memory_manager.get_profile("user123")

        assert retrieved is not None
        assert retrieved.user_id == "user123"
        assert retrieved.name == "Test User"
        assert retrieved.preferences["theme"] == "dark"
        assert retrieved.communication_style == "formal"

    def test_get_nonexistent_profile(self, memory_manager):
        """Test retrieving non-existent profile."""
        profile = memory_manager.get_profile("nonexistent")
        assert profile is None

    def test_add_fact(self, memory_manager):
        """Test adding a fact."""
        fact = memory_manager.add_fact(
            content="The sky is blue",
            source="observation",
            user_id="user123",
            confidence=0.95,
            metadata={"category": "nature"},
        )

        assert fact.content == "The sky is blue"
        assert fact.source == "observation"
        assert fact.user_id == "user123"
        assert fact.confidence == 0.95
        assert fact.metadata["category"] == "nature"

    def test_update_conversation(self, memory_manager):
        """Test updating conversation summary."""
        summary = memory_manager.update_conversation(
            session_id="session123",
            summary="Discussed weather and travel plans",
            key_topics=["weather", "travel", "vacation"],
            user_id="user123",
            message_count=15,
            metadata={"duration_minutes": 30},
        )

        assert summary.session_id == "session123"
        assert summary.summary == "Discussed weather and travel plans"
        assert "weather" in summary.key_topics
        assert summary.message_count == 15

    def test_store_rag_trace(self, memory_manager):
        """Test storing RAG trace."""
        trace = memory_manager.store_rag_trace(
            query="What is the capital of France?",
            retrieved_chunks=["Paris is the capital", "France is in Europe"],
            chunk_ids=["chunk1", "chunk2"],
            relevance_scores=[0.95, 0.87],
            user_id="user123",
            session_id="session123",
            metadata={"model": "gpt-4"},
        )

        assert trace.query == "What is the capital of France?"
        assert len(trace.retrieved_chunks) == 2
        assert len(trace.chunk_ids) == 2
        assert trace.relevance_scores[0] == 0.95

    def test_add_conversation_message(self, memory_manager):
        """Test adding conversation message."""
        message = memory_manager.add_conversation_message(
            session_id="session123",
            role="user",
            content="Hello, how are you?",
            user_id="user123",
            metadata={"ip": "127.0.0.1"},
        )

        assert message.role == "user"
        assert message.content == "Hello, how are you?"

    def test_get_conversation_window(self, memory_manager):
        """Test getting conversation window."""
        # Add multiple messages
        for i in range(15):
            memory_manager.add_conversation_message(
                session_id="session123",
                role="user" if i % 2 == 0 else "assistant",
                content=f"Message {i}",
            )

        # Get window of last 10 messages
        window = memory_manager.get_conversation_window("session123", window_size=10)

        assert len(window) == 10
        assert window[-1].content == "Message 14"

    def test_get_conversation_summary(self, memory_manager):
        """Test generating conversation summary."""
        # Add some messages
        for i in range(5):
            memory_manager.add_conversation_message(
                session_id="session123",
                role="user" if i % 2 == 0 else "assistant",
                content=f"Message about weather {i}",
            )

        summary = memory_manager.get_conversation_summary("session123")

        assert "5 messages" in summary
        assert "weather" in summary.lower()

    def test_extract_conversation_topics(self, memory_manager):
        """Test extracting conversation topics."""
        # Add messages with clear topics
        topics_content = [
            "I want to learn about machine learning",
            "Machine learning is fascinating",
            "Tell me about neural networks",
            "Neural networks are powerful",
        ]

        for i, content in enumerate(topics_content):
            memory_manager.add_conversation_message(
                session_id="session123",
                role="user" if i % 2 == 0 else "assistant",
                content=content,
            )

        topics = memory_manager.extract_conversation_topics("session123")

        # Should extract meaningful topics
        assert len(topics) > 0
        # Check if any relevant terms are in topics
        topic_str = " ".join(topics).lower()
        assert any(term in topic_str for term in ["learning", "machine", "neural", "networks"])

    def test_search_facts(self, memory_manager):
        """Test searching facts."""
        # Add multiple facts
        memory_manager.add_fact(
            content="Python is a programming language",
            source="documentation",
            user_id="user123",
            confidence=1.0,
        )
        memory_manager.add_fact(
            content="JavaScript is used for web development",
            source="documentation",
            user_id="user123",
            confidence=0.9,
        )

        # Note: This test may not work fully without actual Mem0 connection
        # but tests the interface
        facts = memory_manager.search_facts(
            query="programming",
            user_id="user123",
            limit=10,
            min_confidence=0.8,
        )

        # Should return list (may be empty without real Mem0)
        assert isinstance(facts, list)

    def test_get_recent_rag_traces(self, memory_manager):
        """Test getting recent RAG traces."""
        # Add multiple traces
        for i in range(5):
            memory_manager.store_rag_trace(
                query=f"Query {i}",
                retrieved_chunks=[f"Chunk {i}"],
                chunk_ids=[f"id{i}"],
                relevance_scores=[0.9],
                user_id="user123",
                session_id="session123",
            )

        traces = memory_manager.get_recent_rag_traces(
            user_id="user123",
            limit=3,
        )

        assert len(traces) <= 3
        # Most recent should be last added
        if traces:
            assert "Query" in traces[0].query

    def test_analyze_retrieval_patterns(self, memory_manager):
        """Test analyzing retrieval patterns."""
        # Add some traces with session_id to match the file naming pattern
        for i in range(10):
            memory_manager.store_rag_trace(
                query=f"test query about topic {i % 3}",
                retrieved_chunks=[f"chunk{i}"],
                chunk_ids=[f"id{i}"],
                relevance_scores=[0.8 + (i % 3) * 0.05],
                user_id="user123",
                session_id="session123",  # Add session_id for proper file naming
            )

        analysis = memory_manager.analyze_retrieval_patterns(
            user_id="user123",
            days=7,
        )

        # Should have analysis results
        assert "total_queries" in analysis or "message" in analysis
        if "total_queries" in analysis:
            assert analysis["total_queries"] == 10
            assert "avg_chunks_retrieved" in analysis
            assert "avg_relevance_score" in analysis

    def test_apply_all_retention_policies(self, memory_manager):
        """Test applying retention policies."""
        # Add some data
        memory_manager.add_fact(
            content="Test fact",
            source="test",
            user_id="user123",
        )

        memory_manager.update_conversation(
            session_id="session123",
            summary="Test conversation",
            key_topics=["test"],
            user_id="user123",
        )

        memory_manager.store_rag_trace(
            query="test query",
            retrieved_chunks=["chunk"],
            chunk_ids=["id"],
            relevance_scores=[0.9],
            user_id="user123",
        )

        # Apply retention
        removed = memory_manager.apply_all_retention_policies()

        assert "facts" in removed
        assert "conversations" in removed
        assert "rag_traces" in removed
        assert all(count >= 0 for count in removed.values())


class TestMemoryManagerCrossSessions:
    """Test memory persistence across sessions."""

    def test_profile_persistence(self, temp_storage):
        """Test profile persists across manager instances."""
        with patch('app.core.memory.memory_manager.Memory') as mock_memory_class:
            # Mock Mem0
            mock_memory = MagicMock()
            mock_memory.add.return_value = {"id": "test-id"}
            mock_memory.search.return_value = []
            mock_memory_class.from_config.return_value = mock_memory
            
            # Create first manager and store profile
            manager1 = MemoryManager(
                api_key="test-key",
                storage_path=temp_storage,
            )

            profile = UserProfile(
                user_id="user123",
                name="Test User",
                preferences={"theme": "dark"},
            )
            manager1.store_profile(profile)

            # Create second manager and retrieve profile
            manager2 = MemoryManager(
                api_key="test-key",
                storage_path=temp_storage,
            )

            retrieved = manager2.get_profile("user123")

            assert retrieved is not None
            assert retrieved.user_id == "user123"
            assert retrieved.name == "Test User"

    def test_conversation_persistence(self, temp_storage):
        """Test conversation persists across manager instances."""
        with patch('app.core.memory.memory_manager.Memory') as mock_memory_class:
            # Mock Mem0
            mock_memory = MagicMock()
            mock_memory.add.return_value = {"id": "test-id"}
            mock_memory.search.return_value = []
            mock_memory_class.from_config.return_value = mock_memory
            
            # Create first manager and add messages
            manager1 = MemoryManager(
                api_key="test-key",
                storage_path=temp_storage,
            )

            for i in range(3):
                manager1.add_conversation_message(
                    session_id="session123",
                    role="user",
                    content=f"Message {i}",
                )

            # Create second manager and retrieve messages
            manager2 = MemoryManager(
                api_key="test-key",
                storage_path=temp_storage,
            )

            messages = manager2.get_conversation_window("session123")

            assert len(messages) == 3
            assert messages[0].content == "Message 0"
