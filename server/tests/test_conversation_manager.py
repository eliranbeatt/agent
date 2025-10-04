"""Unit tests for conversation manager."""

import pytest
import tempfile
import shutil
from datetime import datetime

from app.core.memory.conversation_manager import ConversationManager, Message


@pytest.fixture
def temp_storage():
    """Create temporary storage directory."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def conversation_manager(temp_storage):
    """Create conversation manager instance."""
    return ConversationManager(
        storage_path=temp_storage,
        window_size=5,
        max_messages_per_session=50,
        summarization_threshold=10,
    )


class TestConversationManager:
    """Test conversation manager functionality."""

    def test_initialization(self, conversation_manager, temp_storage):
        """Test conversation manager initialization."""
        assert conversation_manager.window_size == 5
        assert conversation_manager.max_messages_per_session == 50
        assert conversation_manager.summarization_threshold == 10

    def test_add_message(self, conversation_manager):
        """Test adding a message."""
        message = conversation_manager.add_message(
            session_id="session1",
            role="user",
            content="Hello, world!",
            metadata={"source": "web"},
        )

        assert message.role == "user"
        assert message.content == "Hello, world!"
        assert message.metadata["source"] == "web"
        assert isinstance(message.timestamp, datetime)

    def test_add_multiple_messages(self, conversation_manager):
        """Test adding multiple messages."""
        for i in range(10):
            conversation_manager.add_message(
                session_id="session1",
                role="user" if i % 2 == 0 else "assistant",
                content=f"Message {i}",
            )

        messages = conversation_manager.get_full_history("session1")
        assert len(messages) == 10

    def test_get_rolling_window(self, conversation_manager):
        """Test getting rolling window of messages."""
        # Add 10 messages
        for i in range(10):
            conversation_manager.add_message(
                session_id="session1",
                role="user",
                content=f"Message {i}",
            )

        # Get window of 5 messages
        window = conversation_manager.get_rolling_window("session1", window_size=5)

        assert len(window) == 5
        assert window[0].content == "Message 5"
        assert window[-1].content == "Message 9"

    def test_get_rolling_window_default_size(self, conversation_manager):
        """Test getting rolling window with default size."""
        # Add 10 messages
        for i in range(10):
            conversation_manager.add_message(
                session_id="session1",
                role="user",
                content=f"Message {i}",
            )

        # Get window with default size (5)
        window = conversation_manager.get_rolling_window("session1")

        assert len(window) == 5

    def test_get_rolling_window_fewer_messages(self, conversation_manager):
        """Test rolling window when fewer messages than window size."""
        # Add only 3 messages
        for i in range(3):
            conversation_manager.add_message(
                session_id="session1",
                role="user",
                content=f"Message {i}",
            )

        window = conversation_manager.get_rolling_window("session1", window_size=5)

        # Should return all 3 messages
        assert len(window) == 3

    def test_get_full_history(self, conversation_manager):
        """Test getting full conversation history."""
        # Add messages
        for i in range(15):
            conversation_manager.add_message(
                session_id="session1",
                role="user",
                content=f"Message {i}",
            )

        history = conversation_manager.get_full_history("session1")

        assert len(history) == 15
        assert history[0].content == "Message 0"
        assert history[-1].content == "Message 14"

    def test_extract_topics(self, conversation_manager):
        """Test extracting topics from messages."""
        messages = [
            Message(
                role="user",
                content="I want to learn about machine learning and artificial intelligence",
                timestamp=datetime.now(),
                metadata={},
            ),
            Message(
                role="assistant",
                content="Machine learning is a subset of artificial intelligence",
                timestamp=datetime.now(),
                metadata={},
            ),
            Message(
                role="user",
                content="Tell me more about neural networks and deep learning",
                timestamp=datetime.now(),
                metadata={},
            ),
        ]

        topics = conversation_manager.extract_topics(messages)

        # Should extract meaningful topics
        assert len(topics) > 0
        # Check for relevant terms (case-insensitive)
        topic_str = " ".join(topics).lower()
        assert any(
            term in topic_str
            for term in ["learning", "machine", "artificial", "intelligence", "neural", "networks"]
        )

    def test_generate_summary(self, conversation_manager):
        """Test generating conversation summary."""
        # Add messages
        for i in range(8):
            role = "user" if i % 2 == 0 else "assistant"
            conversation_manager.add_message(
                session_id="session1",
                role=role,
                content=f"Message about weather and climate {i}",
            )

        summary = conversation_manager.generate_summary("session1")

        assert "8 messages" in summary
        assert "4 questions" in summary or "User asked 4" in summary
        assert "weather" in summary.lower() or "climate" in summary.lower()

    def test_generate_summary_empty_session(self, conversation_manager):
        """Test generating summary for empty session."""
        summary = conversation_manager.generate_summary("empty_session")

        assert "No messages" in summary

    def test_clear_session(self, conversation_manager):
        """Test clearing a session."""
        # Add messages
        for i in range(5):
            conversation_manager.add_message(
                session_id="session1",
                role="user",
                content=f"Message {i}",
            )

        # Clear session
        conversation_manager.clear_session("session1")

        # Should be empty
        history = conversation_manager.get_full_history("session1")
        assert len(history) == 0

    def test_multiple_sessions(self, conversation_manager):
        """Test managing multiple sessions."""
        # Add messages to different sessions
        for session_num in range(3):
            for msg_num in range(5):
                conversation_manager.add_message(
                    session_id=f"session{session_num}",
                    role="user",
                    content=f"Session {session_num} Message {msg_num}",
                )

        # Check each session
        for session_num in range(3):
            history = conversation_manager.get_full_history(f"session{session_num}")
            assert len(history) == 5
            assert f"Session {session_num}" in history[0].content

    def test_message_persistence(self, temp_storage):
        """Test message persistence across manager instances."""
        # Create first manager and add messages
        manager1 = ConversationManager(storage_path=temp_storage)

        for i in range(5):
            manager1.add_message(
                session_id="session1",
                role="user",
                content=f"Message {i}",
            )

        # Create second manager and retrieve messages
        manager2 = ConversationManager(storage_path=temp_storage)

        history = manager2.get_full_history("session1")

        assert len(history) == 5
        assert history[0].content == "Message 0"
        assert history[-1].content == "Message 4"

    def test_message_metadata(self, conversation_manager):
        """Test message metadata handling."""
        metadata = {
            "ip_address": "192.168.1.1",
            "user_agent": "Mozilla/5.0",
            "session_data": {"key": "value"},
        }

        message = conversation_manager.add_message(
            session_id="session1",
            role="user",
            content="Test message",
            metadata=metadata,
        )

        assert message.metadata["ip_address"] == "192.168.1.1"
        assert message.metadata["session_data"]["key"] == "value"

        # Retrieve and verify
        history = conversation_manager.get_full_history("session1")
        assert history[0].metadata["ip_address"] == "192.168.1.1"
