"""Conversation history management with rolling windows and topic summaries."""

import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Message:
    """A single message in a conversation."""
    role: str  # user, assistant, system
    content: str
    timestamp: datetime
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """Create from dictionary."""
        return cls(
            role=data["role"],
            content=data["content"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metadata=data.get("metadata", {}),
        )


class ConversationManager:
    """Manages conversation history with rolling windows and topic extraction."""

    def __init__(
        self,
        storage_path: str,
        window_size: int = 10,
        max_messages_per_session: int = 100,
        summarization_threshold: int = 20,
    ):
        """
        Initialize conversation manager.

        Args:
            storage_path: Path for conversation storage
            window_size: Number of recent messages to keep in active window
            max_messages_per_session: Maximum messages before summarization
            summarization_threshold: Trigger summarization after this many messages
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.window_size = window_size
        self.max_messages_per_session = max_messages_per_session
        self.summarization_threshold = summarization_threshold

        # In-memory cache for active sessions
        self.active_sessions: Dict[str, List[Message]] = {}

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Message:
        """
        Add a message to conversation history.

        Args:
            session_id: Session ID
            role: Message role (user, assistant, system)
            content: Message content
            metadata: Additional metadata

        Returns:
            Created message
        """
        message = Message(
            role=role,
            content=content,
            timestamp=datetime.now(),
            metadata=metadata or {},
        )

        # Add to active session
        if session_id not in self.active_sessions:
            self.active_sessions[session_id] = self._load_session(session_id)

        self.active_sessions[session_id].append(message)

        # Save to disk
        self._save_session(session_id)

        # Check if summarization is needed
        if len(self.active_sessions[session_id]) >= self.summarization_threshold:
            logger.info(f"Session {session_id} reached summarization threshold")

        logger.debug(f"Added message to session {session_id}: {role}")

        return message

    def get_rolling_window(
        self,
        session_id: str,
        window_size: Optional[int] = None,
    ) -> List[Message]:
        """
        Get rolling window of recent messages.

        Args:
            session_id: Session ID
            window_size: Window size (uses default if not specified)

        Returns:
            List of recent messages
        """
        if session_id not in self.active_sessions:
            self.active_sessions[session_id] = self._load_session(session_id)

        messages = self.active_sessions[session_id]
        size = window_size or self.window_size

        return messages[-size:] if len(messages) > size else messages

    def get_full_history(self, session_id: str) -> List[Message]:
        """
        Get full conversation history for a session.

        Args:
            session_id: Session ID

        Returns:
            All messages in the session
        """
        if session_id not in self.active_sessions:
            self.active_sessions[session_id] = self._load_session(session_id)

        return self.active_sessions[session_id]

    def extract_topics(self, messages: List[Message]) -> List[str]:
        """
        Extract key topics from messages.

        Args:
            messages: List of messages

        Returns:
            List of extracted topics
        """
        # Simple keyword-based topic extraction
        # In production, this could use NLP or LLM-based extraction
        topics = set()
        keywords = []

        for message in messages:
            # Extract potential topics (simple word frequency approach)
            words = message.content.lower().split()
            # Filter out common words and keep meaningful terms
            meaningful_words = [
                w for w in words
                if len(w) > 4 and w.isalpha()
            ]
            keywords.extend(meaningful_words)

        # Count frequency and take top topics
        from collections import Counter
        word_counts = Counter(keywords)
        topics = [word for word, count in word_counts.most_common(5)]

        return topics

    def generate_summary(
        self,
        session_id: str,
        messages: Optional[List[Message]] = None,
    ) -> str:
        """
        Generate a summary of conversation messages.

        Args:
            session_id: Session ID
            messages: Messages to summarize (uses full history if not specified)

        Returns:
            Summary text
        """
        if messages is None:
            messages = self.get_full_history(session_id)

        if not messages:
            return "No messages to summarize."

        # Simple summary generation
        # In production, this could use an LLM for better summaries
        user_messages = [m for m in messages if m.role == "user"]
        assistant_messages = [m for m in messages if m.role == "assistant"]

        topics = self.extract_topics(messages)

        summary = f"Conversation with {len(messages)} messages. "
        summary += f"User asked {len(user_messages)} questions. "
        summary += f"Assistant provided {len(assistant_messages)} responses. "

        if topics:
            summary += f"Main topics: {', '.join(topics)}."

        return summary

    def clear_session(self, session_id: str) -> None:
        """
        Clear a conversation session.

        Args:
            session_id: Session ID to clear
        """
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]

        session_file = self.storage_path / f"{session_id}.json"
        if session_file.exists():
            session_file.unlink()

        logger.info(f"Cleared session {session_id}")

    def _load_session(self, session_id: str) -> List[Message]:
        """Load session from disk."""
        session_file = self.storage_path / f"{session_id}.json"
        if not session_file.exists():
            return []

        try:
            with open(session_file, "r") as f:
                data = json.load(f)
            return [Message.from_dict(m) for m in data.get("messages", [])]
        except Exception as e:
            logger.error(f"Error loading session {session_id}: {e}")
            return []

    def _save_session(self, session_id: str) -> None:
        """Save session to disk."""
        if session_id not in self.active_sessions:
            return

        session_file = self.storage_path / f"{session_id}.json"

        try:
            data = {
                "session_id": session_id,
                "messages": [m.to_dict() for m in self.active_sessions[session_id]],
                "updated_at": datetime.now().isoformat(),
            }

            with open(session_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving session {session_id}: {e}")
