"""Data models for memory management."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional
from enum import Enum


class MemoryType(str, Enum):
    """Types of memory stored in the system."""
    PROFILE = "profile"
    FACT = "fact"
    CONVERSATION = "conversation"
    RAG_TRACE = "rag_trace"


@dataclass
class UserProfile:
    """User profile information."""
    user_id: str
    name: Optional[str] = None
    preferences: Dict[str, Any] = field(default_factory=dict)
    communication_style: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "user_id": self.user_id,
            "name": self.name,
            "preferences": self.preferences,
            "communication_style": self.communication_style,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UserProfile":
        """Create from dictionary."""
        return cls(
            user_id=data["user_id"],
            name=data.get("name"),
            preferences=data.get("preferences", {}),
            communication_style=data.get("communication_style"),
            created_at=datetime.fromisoformat(data.get("created_at", datetime.now().isoformat())),
            updated_at=datetime.fromisoformat(data.get("updated_at", datetime.now().isoformat())),
        )


@dataclass
class Fact:
    """A stored fact with metadata."""
    content: str
    source: str
    timestamp: datetime = field(default_factory=datetime.now)
    confidence: float = 1.0
    user_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "content": self.content,
            "source": self.source,
            "timestamp": self.timestamp.isoformat(),
            "confidence": self.confidence,
            "user_id": self.user_id,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Fact":
        """Create from dictionary."""
        return cls(
            content=data["content"],
            source=data["source"],
            timestamp=datetime.fromisoformat(data.get("timestamp", datetime.now().isoformat())),
            confidence=data.get("confidence", 1.0),
            user_id=data.get("user_id"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class ConversationSummary:
    """Summary of a conversation session."""
    session_id: str
    summary: str
    key_topics: List[str]
    timestamp: datetime = field(default_factory=datetime.now)
    user_id: Optional[str] = None
    message_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "session_id": self.session_id,
            "summary": self.summary,
            "key_topics": self.key_topics,
            "timestamp": self.timestamp.isoformat(),
            "user_id": self.user_id,
            "message_count": self.message_count,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConversationSummary":
        """Create from dictionary."""
        return cls(
            session_id=data["session_id"],
            summary=data["summary"],
            key_topics=data.get("key_topics", []),
            timestamp=datetime.fromisoformat(data.get("timestamp", datetime.now().isoformat())),
            user_id=data.get("user_id"),
            message_count=data.get("message_count", 0),
            metadata=data.get("metadata", {}),
        )


@dataclass
class RAGTrace:
    """Trace of a RAG retrieval operation."""
    query: str
    retrieved_chunks: List[str]
    chunk_ids: List[str]
    relevance_scores: List[float]
    timestamp: datetime = field(default_factory=datetime.now)
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "query": self.query,
            "retrieved_chunks": self.retrieved_chunks,
            "chunk_ids": self.chunk_ids,
            "relevance_scores": self.relevance_scores,
            "timestamp": self.timestamp.isoformat(),
            "user_id": self.user_id,
            "session_id": self.session_id,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RAGTrace":
        """Create from dictionary."""
        return cls(
            query=data["query"],
            retrieved_chunks=data.get("retrieved_chunks", []),
            chunk_ids=data.get("chunk_ids", []),
            relevance_scores=data.get("relevance_scores", []),
            timestamp=datetime.fromisoformat(data.get("timestamp", datetime.now().isoformat())),
            user_id=data.get("user_id"),
            session_id=data.get("session_id"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class MemoryContext:
    """Context retrieved from memory for a request."""
    profile: Optional[UserProfile] = None
    relevant_facts: List[Fact] = field(default_factory=list)
    conversation_history: List[ConversationSummary] = field(default_factory=list)
    rag_traces: List[RAGTrace] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "profile": self.profile.to_dict() if self.profile else None,
            "relevant_facts": [f.to_dict() for f in self.relevant_facts],
            "conversation_history": [c.to_dict() for c in self.conversation_history],
            "rag_traces": [r.to_dict() for r in self.rag_traces],
            "metadata": self.metadata,
        }
