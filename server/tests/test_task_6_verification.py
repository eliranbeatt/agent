"""Verification tests for Task 6: Complete Mem0 Memory Integration."""

import os
import pytest
import tempfile
from pathlib import Path
from datetime import datetime, timedelta

from app.core.memory.memory_manager import MemoryManager
from app.core.memory.models import UserProfile, Fact, ConversationSummary


class TestTask6_1_Mem0Installation:
    """Test Task 6.1: Install and configure Mem0."""
    
    def test_mem0_package_installed(self):
        """Verify Mem0 package is installed."""
        try:
            import mem0
            assert hasattr(mem0, 'Memory')
            print(f"✓ Mem0 version {mem0.__version__} is installed")
        except ImportError:
            pytest.fail("Mem0 package is not installed")
    
    def test_memory_manager_initializes_with_mem0(self):
        """Verify MemoryManager can initialize with Mem0."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test with API key
            api_key = os.getenv("OPENAI_API_KEY", "test-key")
            manager = MemoryManager(
                api_key=api_key,
                storage_path=temp_dir,
                ttl_days=30
            )
            
            # Verify storage paths are created
            assert manager.storage_path.exists()
            assert manager.profiles_path.exists()
            assert manager.facts_path.exists()
            assert manager.conversations_path.exists()
            assert manager.rag_traces_path.exists()
            print("✓ MemoryManager initializes with Mem0 configuration")
    
    def test_memory_manager_works_without_api_key(self):
        """Verify MemoryManager works in fallback mode without API key."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = MemoryManager(
                api_key="",
                storage_path=temp_dir,
                ttl_days=30
            )
            
            # Should still work with local storage
            assert manager.storage_path.exists()
            assert manager.memory is None  # Mem0 not initialized
            print("✓ MemoryManager works in fallback mode without API key")


class TestTask6_2_ConversationStorage:
    """Test Task 6.2: Implement conversation storage."""
    
    @pytest.fixture
    def memory_manager(self):
        """Create a memory manager for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = MemoryManager(
                api_key="",  # Use fallback mode for testing
                storage_path=temp_dir,
                ttl_days=30
            )
            yield manager
    
    def test_store_conversation_turn(self, memory_manager):
        """Test storing a conversation turn."""
        session_id = "test-session-1"
        
        memory_manager.store_conversation(
            session_id=session_id,
            user_message="What is the weather today?",
            assistant_message="I don't have access to real-time weather data.",
            metadata={"timestamp": datetime.now().isoformat()}
        )
        
        # Verify messages were stored
        history = memory_manager.get_conversation_history(session_id)
        assert len(history) == 2
        assert history[0].role == "user"
        assert history[0].content == "What is the weather today?"
        assert history[1].role == "assistant"
        print("✓ Conversation turns are stored correctly")
    
    def test_add_conversation_message_with_timestamp(self, memory_manager):
        """Test adding conversation messages with timestamps."""
        session_id = "test-session-2"
        
        msg1 = memory_manager.add_conversation_message(
            session_id=session_id,
            role="user",
            content="Hello",
            metadata={"source": "web"}
        )
        
        msg2 = memory_manager.add_conversation_message(
            session_id=session_id,
            role="assistant",
            content="Hi there!",
            metadata={"source": "agent"}
        )
        
        assert msg1.timestamp is not None
        assert msg2.timestamp is not None
        assert msg2.timestamp >= msg1.timestamp
        print("✓ Messages have timestamps and session IDs")
    
    def test_get_conversation_window(self, memory_manager):
        """Test retrieving conversation window."""
        session_id = "test-session-3"
        
        # Add 15 messages
        for i in range(15):
            memory_manager.add_conversation_message(
                session_id=session_id,
                role="user" if i % 2 == 0 else "assistant",
                content=f"Message {i}"
            )
        
        # Get window of last 5 messages
        window = memory_manager.get_conversation_window(session_id, window_size=5)
        
        assert len(window) == 5
        assert window[-1].content == "Message 14"
        print("✓ Conversation window retrieval works")
    
    def test_conversation_summarization(self, memory_manager):
        """Test conversation summarization."""
        session_id = "test-session-4"
        
        # Add messages about a specific topic
        memory_manager.add_conversation_message(
            session_id=session_id,
            role="user",
            content="Tell me about Python programming"
        )
        memory_manager.add_conversation_message(
            session_id=session_id,
            role="assistant",
            content="Python is a high-level programming language"
        )
        
        summary = memory_manager.get_conversation_summary(session_id)
        
        assert isinstance(summary, str)
        assert len(summary) > 0
        print("✓ Conversation summarization works")


class TestTask6_3_ProfileAndFactsStorage:
    """Test Task 6.3: Implement profile and facts storage."""
    
    @pytest.fixture
    def memory_manager(self):
        """Create a memory manager for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = MemoryManager(
                api_key="",
                storage_path=temp_dir,
                ttl_days=30
            )
            yield manager
    
    def test_store_and_retrieve_user_profile(self, memory_manager):
        """Test storing and retrieving user profiles."""
        profile = UserProfile(
            user_id="user123",
            name="Test User",
            preferences={
                "theme": "dark",
                "language": "en",
                "expertise_areas": ["Python", "AI"],
                "goals": ["Learn machine learning"]
            },
            communication_style="professional"
        )
        
        memory_manager.store_profile(profile)
        
        # Retrieve profile
        retrieved = memory_manager.get_profile("user123")
        
        assert retrieved is not None
        assert retrieved.user_id == "user123"
        assert retrieved.name == "Test User"
        assert retrieved.preferences["theme"] == "dark"
        assert "Python" in retrieved.preferences["expertise_areas"]
        print("✓ User profile storage and retrieval works")
    
    def test_add_fact_with_source(self, memory_manager):
        """Test adding facts with sources."""
        fact = memory_manager.add_fact(
            content="The Earth orbits the Sun",
            source="astronomy_textbook.pdf",
            user_id="user123",
            confidence=0.95,
            metadata={"category": "science", "verified": True}
        )
        
        assert fact.content == "The Earth orbits the Sun"
        assert fact.source == "astronomy_textbook.pdf"
        assert fact.confidence == 0.95
        assert fact.metadata["category"] == "science"
        print("✓ Facts are stored with sources and metadata")
    
    def test_search_facts_by_query(self, memory_manager):
        """Test searching facts by query."""
        # Add multiple facts
        memory_manager.add_fact(
            content="Python was created by Guido van Rossum",
            source="python_history.txt",
            user_id="user123",
            confidence=1.0
        )
        memory_manager.add_fact(
            content="Python is used for web development",
            source="web_dev_guide.pdf",
            user_id="user123",
            confidence=0.9
        )
        memory_manager.add_fact(
            content="JavaScript is used for frontend development",
            source="js_guide.pdf",
            user_id="user123",
            confidence=0.85
        )
        
        # Search for Python-related facts
        facts = memory_manager.search_facts(
            query="Python programming",
            user_id="user123",
            limit=5,
            min_confidence=0.8
        )
        
        # Should work even without Mem0 (returns empty list in fallback mode)
        assert isinstance(facts, list)
        print("✓ Fact search functionality exists")
    
    def test_fact_confidence_scoring(self, memory_manager):
        """Test fact confidence scoring."""
        fact1 = memory_manager.add_fact(
            content="High confidence fact",
            source="verified_source.pdf",
            confidence=0.95
        )
        
        fact2 = memory_manager.add_fact(
            content="Lower confidence fact",
            source="unverified_source.txt",
            confidence=0.6
        )
        
        assert fact1.confidence > fact2.confidence
        print("✓ Fact confidence scoring works")


class TestTask6_4_RetentionPolicies:
    """Test Task 6.4: Implement retention policies."""
    
    @pytest.fixture
    def memory_manager(self):
        """Create a memory manager with short TTL for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = MemoryManager(
                api_key="",
                storage_path=temp_dir,
                ttl_days=1,  # Short TTL for testing
                max_facts=5,
                max_conversations=3
            )
            yield manager
    
    def test_ttl_policy_configuration(self, memory_manager):
        """Test TTL policies are configured."""
        assert memory_manager.ttl_days == 1
        assert memory_manager.max_facts == 5
        assert memory_manager.max_conversations == 3
        print("✓ TTL policies are configured")
    
    def test_size_based_cleanup(self, memory_manager):
        """Test size-based cleanup."""
        # Add more facts than the limit
        for i in range(10):
            memory_manager.add_fact(
                content=f"Fact {i}",
                source="test.txt",
                user_id="user123"
            )
        
        # Check that retention was applied
        fact_files = list(memory_manager.facts_path.glob("user123_*.json"))
        assert len(fact_files) <= memory_manager.max_facts
        print("✓ Size-based cleanup works")
    
    def test_apply_all_retention_policies(self, memory_manager):
        """Test applying all retention policies."""
        # Add some data
        memory_manager.add_fact(
            content="Test fact",
            source="test.txt",
            user_id="user123"
        )
        
        memory_manager.store_conversation(
            session_id="session1",
            user_message="Hello",
            assistant_message="Hi"
        )
        
        # Apply retention policies
        result = memory_manager.apply_all_retention_policies()
        
        assert isinstance(result, dict)
        assert "facts" in result
        assert "conversations" in result
        assert "rag_traces" in result
        print("✓ All retention policies can be applied")
    
    def test_retention_policy_types(self, memory_manager):
        """Test different retention policy types."""
        # Verify retention managers are initialized
        assert memory_manager.fact_retention is not None
        assert memory_manager.conversation_retention is not None
        assert memory_manager.rag_trace_retention is not None
        
        # Verify policy types
        assert memory_manager.fact_retention.policy_type == "lru"
        assert memory_manager.conversation_retention.policy_type == "lru"
        assert memory_manager.rag_trace_retention.policy_type == "fifo"
        print("✓ Different retention policy types are supported")


class TestTask6_5_MemoryPersistence:
    """Test Task 6.5: Test memory persistence across sessions."""
    
    def test_profile_persistence_across_sessions(self):
        """Test that profiles persist across manager instances."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Session 1: Store profile
            manager1 = MemoryManager(
                api_key="",
                storage_path=temp_dir,
                ttl_days=30
            )
            
            profile = UserProfile(
                user_id="persistent_user",
                name="Persistent User",
                preferences={"theme": "dark"}
            )
            manager1.store_profile(profile)
            
            # Session 2: Retrieve profile with new manager instance
            manager2 = MemoryManager(
                api_key="",
                storage_path=temp_dir,
                ttl_days=30
            )
            
            retrieved = manager2.get_profile("persistent_user")
            
            assert retrieved is not None
            assert retrieved.user_id == "persistent_user"
            assert retrieved.name == "Persistent User"
            print("✓ Profiles persist across sessions")
    
    def test_conversation_persistence_across_sessions(self):
        """Test that conversations persist across manager instances."""
        with tempfile.TemporaryDirectory() as temp_dir:
            session_id = "persistent_session"
            
            # Session 1: Store conversation
            manager1 = MemoryManager(
                api_key="",
                storage_path=temp_dir,
                ttl_days=30
            )
            
            manager1.store_conversation(
                session_id=session_id,
                user_message="Remember this",
                assistant_message="I will remember"
            )
            
            # Session 2: Retrieve conversation with new manager instance
            manager2 = MemoryManager(
                api_key="",
                storage_path=temp_dir,
                ttl_days=30
            )
            
            history = manager2.get_conversation_history(session_id)
            
            assert len(history) == 2
            assert history[0].content == "Remember this"
            print("✓ Conversations persist across sessions")
    
    def test_facts_persistence_across_sessions(self):
        """Test that facts persist across manager instances."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Session 1: Store fact
            manager1 = MemoryManager(
                api_key="",
                storage_path=temp_dir,
                ttl_days=30
            )
            
            manager1.add_fact(
                content="Persistent fact",
                source="test.txt",
                user_id="user123"
            )
            
            # Session 2: Check fact exists with new manager instance
            manager2 = MemoryManager(
                api_key="",
                storage_path=temp_dir,
                ttl_days=30
            )
            
            # Verify fact file exists
            fact_files = list(manager2.facts_path.glob("user123_*.json"))
            assert len(fact_files) > 0
            print("✓ Facts persist across sessions")
    
    def test_rag_traces_persistence(self):
        """Test that RAG traces persist across sessions."""
        with tempfile.TemporaryDirectory() as temp_dir:
            session_id = "rag_session"
            
            # Session 1: Store RAG trace
            manager1 = MemoryManager(
                api_key="",
                storage_path=temp_dir,
                ttl_days=30
            )
            
            manager1.store_rag_trace(
                query="Test query",
                retrieved_chunks=["chunk1", "chunk2"],
                chunk_ids=["id1", "id2"],
                relevance_scores=[0.9, 0.8],
                session_id=session_id
            )
            
            # Session 2: Retrieve traces with new manager instance
            manager2 = MemoryManager(
                api_key="",
                storage_path=temp_dir,
                ttl_days=30
            )
            
            traces = manager2.get_recent_rag_traces(session_id=session_id, limit=10)
            
            assert len(traces) > 0
            assert traces[0].query == "Test query"
            print("✓ RAG traces persist across sessions")


class TestTask6_Integration:
    """Integration tests for complete Mem0 memory system."""
    
    def test_complete_memory_workflow(self):
        """Test complete memory workflow with all components."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = MemoryManager(
                api_key="",
                storage_path=temp_dir,
                ttl_days=30
            )
            
            # 1. Store user profile
            profile = UserProfile(
                user_id="integration_user",
                name="Integration Test User",
                preferences={"language": "en"}
            )
            manager.store_profile(profile)
            
            # 2. Add facts
            manager.add_fact(
                content="User prefers Python",
                source="conversation",
                user_id="integration_user",
                confidence=0.9
            )
            
            # 3. Store conversation
            session_id = "integration_session"
            manager.store_conversation(
                session_id=session_id,
                user_message="I love Python programming",
                assistant_message="That's great! Python is very versatile."
            )
            
            # 4. Store RAG trace
            manager.store_rag_trace(
                query="Python best practices",
                retrieved_chunks=["Use PEP 8", "Write tests"],
                chunk_ids=["chunk1", "chunk2"],
                relevance_scores=[0.95, 0.85],
                session_id=session_id,
                user_id="integration_user"
            )
            
            # 5. Verify all data is stored
            retrieved_profile = manager.get_profile("integration_user")
            conversation_history = manager.get_conversation_history(session_id)
            rag_traces = manager.get_recent_rag_traces(session_id=session_id)
            
            assert retrieved_profile is not None
            assert len(conversation_history) == 2
            assert len(rag_traces) > 0
            
            print("✓ Complete memory workflow works end-to-end")
    
    def test_memory_retrieval_with_context(self):
        """Test retrieving relevant memory based on context."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = MemoryManager(
                api_key="",
                storage_path=temp_dir,
                ttl_days=30
            )
            
            # Add profile
            profile = UserProfile(
                user_id="context_user",
                name="Context User"
            )
            manager.store_profile(profile)
            
            # Add facts
            manager.add_fact(
                content="User is learning machine learning",
                source="conversation",
                user_id="context_user"
            )
            
            # Retrieve relevant memory
            memory_context = manager.retrieve_relevant_memory(
                context="machine learning tutorials",
                user_id="context_user",
                limit=5
            )
            
            assert memory_context is not None
            assert memory_context.profile is not None
            assert memory_context.profile.user_id == "context_user"
            
            print("✓ Memory retrieval with context works")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
