"""Memory manager implementation using Mem0."""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path

from mem0 import Memory

from .models import (
    UserProfile,
    Fact,
    ConversationSummary,
    MemoryContext,
    RAGTrace,
    MemoryType,
)
from .conversation_manager import ConversationManager, Message
from .retention_policies import RetentionPolicyManager

logger = logging.getLogger(__name__)


class MemoryManager:
    """Manages persistent memory using Mem0."""

    def __init__(
        self,
        api_key: str,
        storage_path: Optional[str] = None,
        ttl_days: int = 90,
        max_facts: int = 1000,
        max_conversations: int = 100,
        max_rag_traces: int = 500,
    ):
        """
        Initialize memory manager.

        Args:
            api_key: OpenAI API key for Mem0
            storage_path: Path for local storage (default: ./data/memory)
            ttl_days: Time-to-live for memories in days
            max_facts: Maximum number of facts to store
            max_conversations: Maximum number of conversation summaries
            max_rag_traces: Maximum number of RAG traces
        """
        self.storage_path = Path(storage_path or "./data/memory")
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Initialize Mem0 with OpenAI integration
        config = {
            "llm": {
                "provider": "openai",
                "config": {
                    "model": "gpt-4o-mini",
                    "api_key": api_key,
                }
            },
            "embedder": {
                "provider": "openai",
                "config": {
                    "model": "text-embedding-3-small",
                    "api_key": api_key,
                }
            },
            "vector_store": {
                "provider": "chroma",
                "config": {
                    "path": str(self.storage_path / "chroma"),
                }
            },
        }

        self.memory = Memory.from_config(config)
        self.ttl_days = ttl_days
        self.max_facts = max_facts
        self.max_conversations = max_conversations
        self.max_rag_traces = max_rag_traces

        # Local storage for structured data
        self.profiles_path = self.storage_path / "profiles"
        self.facts_path = self.storage_path / "facts"
        self.conversations_path = self.storage_path / "conversations"
        self.rag_traces_path = self.storage_path / "rag_traces"

        for path in [self.profiles_path, self.facts_path, self.conversations_path, self.rag_traces_path]:
            path.mkdir(parents=True, exist_ok=True)

        # Initialize conversation manager
        self.conversation_manager = ConversationManager(
            storage_path=str(self.conversations_path / "sessions"),
            window_size=10,
            max_messages_per_session=100,
            summarization_threshold=20,
        )

        # Initialize retention policy managers
        self.fact_retention = RetentionPolicyManager(
            ttl_days=ttl_days,
            max_items=max_facts,
            policy_type="lru",
        )
        self.conversation_retention = RetentionPolicyManager(
            ttl_days=ttl_days,
            max_items=max_conversations,
            policy_type="lru",
        )
        self.rag_trace_retention = RetentionPolicyManager(
            ttl_days=ttl_days,
            max_items=max_rag_traces,
            policy_type="fifo",
        )

        logger.info(f"Memory manager initialized with storage at {self.storage_path}")

    def store_profile(self, profile: UserProfile) -> None:
        """
        Store user profile.

        Args:
            profile: User profile to store
        """
        try:
            profile.updated_at = datetime.now()
            profile_path = self.profiles_path / f"{profile.user_id}.json"

            with open(profile_path, "w") as f:
                json.dump(profile.to_dict(), f, indent=2)

            # Also store in Mem0 for semantic retrieval
            self.memory.add(
                messages=f"User profile: {json.dumps(profile.to_dict())}",
                user_id=profile.user_id,
                metadata={"type": MemoryType.PROFILE.value}
            )

            logger.info(f"Stored profile for user {profile.user_id}")
        except Exception as e:
            logger.error(f"Error storing profile: {e}")
            raise

    def get_profile(self, user_id: str) -> Optional[UserProfile]:
        """
        Retrieve user profile.

        Args:
            user_id: User ID

        Returns:
            User profile or None if not found
        """
        try:
            profile_path = self.profiles_path / f"{user_id}.json"
            if not profile_path.exists():
                return None

            with open(profile_path, "r") as f:
                data = json.load(f)

            return UserProfile.from_dict(data)
        except Exception as e:
            logger.error(f"Error retrieving profile: {e}")
            return None

    def add_fact(
        self,
        content: str,
        source: str,
        user_id: Optional[str] = None,
        confidence: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Fact:
        """
        Add a fact to memory.

        Args:
            content: Fact content
            source: Source of the fact
            user_id: Associated user ID
            confidence: Confidence score (0-1)
            metadata: Additional metadata

        Returns:
            Created fact
        """
        try:
            fact = Fact(
                content=content,
                source=source,
                user_id=user_id,
                confidence=confidence,
                metadata=metadata or {},
            )

            # Store in Mem0
            mem_metadata = {
                "type": MemoryType.FACT.value,
                "source": source,
                "confidence": confidence,
                **(metadata or {}),
            }

            self.memory.add(
                messages=content,
                user_id=user_id or "system",
                metadata=mem_metadata
            )

            # Store locally
            fact_file = self.facts_path / f"{user_id or 'system'}_{fact.timestamp.timestamp()}.json"
            with open(fact_file, "w") as f:
                json.dump(fact.to_dict(), f, indent=2)

            logger.info(f"Added fact: {content[:50]}...")

            # Apply retention policies
            self._apply_fact_retention(user_id)

            return fact
        except Exception as e:
            logger.error(f"Error adding fact: {e}")
            raise

    def update_conversation(
        self,
        session_id: str,
        summary: str,
        key_topics: List[str],
        user_id: Optional[str] = None,
        message_count: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ConversationSummary:
        """
        Update conversation summary.

        Args:
            session_id: Session ID
            summary: Conversation summary
            key_topics: Key topics discussed
            user_id: Associated user ID
            message_count: Number of messages
            metadata: Additional metadata

        Returns:
            Created conversation summary
        """
        try:
            conv_summary = ConversationSummary(
                session_id=session_id,
                summary=summary,
                key_topics=key_topics,
                user_id=user_id,
                message_count=message_count,
                metadata=metadata or {},
            )

            # Store in Mem0
            mem_metadata = {
                "type": MemoryType.CONVERSATION.value,
                "session_id": session_id,
                "topics": ",".join(key_topics),
                **(metadata or {}),
            }

            self.memory.add(
                messages=f"Conversation summary: {summary}. Topics: {', '.join(key_topics)}",
                user_id=user_id or "system",
                metadata=mem_metadata
            )

            # Store locally
            conv_file = self.conversations_path / f"{session_id}.json"
            with open(conv_file, "w") as f:
                json.dump(conv_summary.to_dict(), f, indent=2)

            logger.info(f"Updated conversation {session_id}")

            # Apply retention policies
            self._apply_conversation_retention(user_id)

            return conv_summary
        except Exception as e:
            logger.error(f"Error updating conversation: {e}")
            raise

    def store_rag_trace(
        self,
        query: str,
        retrieved_chunks: List[str],
        chunk_ids: List[str],
        relevance_scores: List[float],
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> RAGTrace:
        """
        Store RAG retrieval trace.

        Args:
            query: Query text
            retrieved_chunks: Retrieved chunk contents
            chunk_ids: Chunk IDs
            relevance_scores: Relevance scores
            user_id: Associated user ID
            session_id: Associated session ID
            metadata: Additional metadata

        Returns:
            Created RAG trace
        """
        try:
            rag_trace = RAGTrace(
                query=query,
                retrieved_chunks=retrieved_chunks,
                chunk_ids=chunk_ids,
                relevance_scores=relevance_scores,
                user_id=user_id,
                session_id=session_id,
                metadata=metadata or {},
            )

            # Store in Mem0
            mem_metadata = {
                "type": MemoryType.RAG_TRACE.value,
                "session_id": session_id or "",
                "num_chunks": len(chunk_ids),
                **(metadata or {}),
            }

            self.memory.add(
                messages=f"RAG query: {query}. Retrieved {len(chunk_ids)} chunks.",
                user_id=user_id or "system",
                metadata=mem_metadata
            )

            # Store locally
            trace_file = self.rag_traces_path / f"{session_id or 'system'}_{rag_trace.timestamp.timestamp()}.json"
            with open(trace_file, "w") as f:
                json.dump(rag_trace.to_dict(), f, indent=2)

            logger.info(f"Stored RAG trace for query: {query[:50]}...")

            # Apply retention policies
            self._apply_rag_trace_retention(user_id)

            return rag_trace
        except Exception as e:
            logger.error(f"Error storing RAG trace: {e}")
            raise

    def retrieve_relevant_memory(
        self,
        context: str,
        user_id: Optional[str] = None,
        limit: int = 10,
    ) -> MemoryContext:
        """
        Retrieve relevant memory based on context.

        Args:
            context: Context string for retrieval
            user_id: User ID to filter by
            limit: Maximum number of items to retrieve

        Returns:
            Memory context with relevant information
        """
        try:
            # Get profile
            profile = self.get_profile(user_id) if user_id else None

            # Search Mem0 for relevant memories
            memories = self.memory.search(
                query=context,
                user_id=user_id or "system",
                limit=limit
            )

            # Separate by type
            relevant_facts = []
            conversation_history = []
            rag_traces = []

            for mem in memories:
                mem_type = mem.get("metadata", {}).get("type")
                if mem_type == MemoryType.FACT.value:
                    # Load full fact from local storage
                    fact_data = self._find_fact_by_content(mem["memory"], user_id)
                    if fact_data:
                        relevant_facts.append(Fact.from_dict(fact_data))

                elif mem_type == MemoryType.CONVERSATION.value:
                    session_id = mem.get("metadata", {}).get("session_id")
                    if session_id:
                        conv_data = self._load_conversation(session_id)
                        if conv_data:
                            conversation_history.append(ConversationSummary.from_dict(conv_data))

            memory_context = MemoryContext(
                profile=profile,
                relevant_facts=relevant_facts[:limit],
                conversation_history=conversation_history[:limit],
                rag_traces=rag_traces,
            )

            logger.info(f"Retrieved memory context with {len(relevant_facts)} facts, {len(conversation_history)} conversations")

            return memory_context
        except Exception as e:
            logger.error(f"Error retrieving relevant memory: {e}")
            return MemoryContext()

    def _apply_fact_retention(self, user_id: Optional[str]) -> None:
        """Apply retention policies to facts."""
        try:
            pattern = f"{user_id or 'system'}_*.json"
            fact_files = sorted(self.facts_path.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)

            # Remove old facts beyond TTL
            cutoff_date = datetime.now() - timedelta(days=self.ttl_days)
            for fact_file in fact_files:
                with open(fact_file, "r") as f:
                    fact_data = json.load(f)
                timestamp = datetime.fromisoformat(fact_data["timestamp"])
                if timestamp < cutoff_date:
                    fact_file.unlink()
                    logger.debug(f"Removed old fact: {fact_file.name}")

            # Remove excess facts beyond max limit
            fact_files = sorted(self.facts_path.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
            if len(fact_files) > self.max_facts:
                for fact_file in fact_files[self.max_facts:]:
                    fact_file.unlink()
                    logger.debug(f"Removed excess fact: {fact_file.name}")
        except Exception as e:
            logger.error(f"Error applying fact retention: {e}")

    def _apply_conversation_retention(self, user_id: Optional[str]) -> None:
        """Apply retention policies to conversations."""
        try:
            conv_files = sorted(self.conversations_path.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)

            # Filter by user if specified
            if user_id:
                conv_files = [f for f in conv_files if self._conversation_belongs_to_user(f, user_id)]

            # Remove old conversations beyond TTL
            cutoff_date = datetime.now() - timedelta(days=self.ttl_days)
            for conv_file in conv_files:
                with open(conv_file, "r") as f:
                    conv_data = json.load(f)
                timestamp = datetime.fromisoformat(conv_data["timestamp"])
                if timestamp < cutoff_date:
                    conv_file.unlink()
                    logger.debug(f"Removed old conversation: {conv_file.name}")

            # Remove excess conversations beyond max limit
            conv_files = sorted(self.conversations_path.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
            if user_id:
                conv_files = [f for f in conv_files if self._conversation_belongs_to_user(f, user_id)]

            if len(conv_files) > self.max_conversations:
                for conv_file in conv_files[self.max_conversations:]:
                    conv_file.unlink()
                    logger.debug(f"Removed excess conversation: {conv_file.name}")
        except Exception as e:
            logger.error(f"Error applying conversation retention: {e}")

    def _apply_rag_trace_retention(self, user_id: Optional[str]) -> None:
        """Apply retention policies to RAG traces."""
        try:
            pattern = f"{user_id or 'system'}_*.json"
            trace_files = sorted(self.rag_traces_path.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)

            # Remove old traces beyond TTL
            cutoff_date = datetime.now() - timedelta(days=self.ttl_days)
            for trace_file in trace_files:
                with open(trace_file, "r") as f:
                    trace_data = json.load(f)
                timestamp = datetime.fromisoformat(trace_data["timestamp"])
                if timestamp < cutoff_date:
                    trace_file.unlink()
                    logger.debug(f"Removed old RAG trace: {trace_file.name}")

            # Remove excess traces beyond max limit
            trace_files = sorted(self.rag_traces_path.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
            if len(trace_files) > self.max_rag_traces:
                for trace_file in trace_files[self.max_rag_traces:]:
                    trace_file.unlink()
                    logger.debug(f"Removed excess RAG trace: {trace_file.name}")
        except Exception as e:
            logger.error(f"Error applying RAG trace retention: {e}")

    def _find_fact_by_content(self, content: str, user_id: Optional[str]) -> Optional[Dict[str, Any]]:
        """Find fact by content match."""
        try:
            pattern = f"{user_id or 'system'}_*.json"
            for fact_file in self.facts_path.glob(pattern):
                with open(fact_file, "r") as f:
                    fact_data = json.load(f)
                if content in fact_data["content"]:
                    return fact_data
            return None
        except Exception as e:
            logger.error(f"Error finding fact: {e}")
            return None

    def _load_conversation(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Load conversation by session ID."""
        try:
            conv_file = self.conversations_path / f"{session_id}.json"
            if not conv_file.exists():
                return None

            with open(conv_file, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading conversation: {e}")
            return None

    def _conversation_belongs_to_user(self, conv_file: Path, user_id: str) -> bool:
        """Check if conversation belongs to user."""
        try:
            with open(conv_file, "r") as f:
                conv_data = json.load(f)
            return conv_data.get("user_id") == user_id
        except Exception as e:
            logger.error(f"Error checking conversation ownership: {e}")
            return False

    # Advanced retrieval methods for task 7.2

    def add_conversation_message(
        self,
        session_id: str,
        role: str,
        content: str,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Message:
        """
        Add a message to conversation history.

        Args:
            session_id: Session ID
            role: Message role (user, assistant, system)
            content: Message content
            user_id: Associated user ID
            metadata: Additional metadata

        Returns:
            Created message
        """
        return self.conversation_manager.add_message(
            session_id=session_id,
            role=role,
            content=content,
            metadata=metadata or {},
        )

    def store_conversation(
        self,
        session_id: str,
        user_message: str,
        assistant_message: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Store a conversation turn.

        Args:
            session_id: Session ID
            user_message: User's message
            assistant_message: Assistant's message
            metadata: Additional metadata
        """
        self.add_conversation_message(
            session_id=session_id,
            role="user",
            content=user_message,
            metadata=metadata,
        )
        self.add_conversation_message(
            session_id=session_id,
            role="assistant",
            content=assistant_message,
            metadata=metadata,
        )

    def get_conversation_history(self, session_id: str) -> List[Message]:
        """
        Get the full conversation history for a session.

        Args:
            session_id: The ID of the session.

        Returns:
            A list of messages in the conversation.
        """
        return self.conversation_manager.get_full_history(session_id)

    def get_conversation_window(
        self,
        session_id: str,
        window_size: Optional[int] = None,
    ) -> List[Message]:
        """
        Get rolling window of recent conversation messages.

        Args:
            session_id: Session ID
            window_size: Window size (uses default if not specified)

        Returns:
            List of recent messages
        """
        return self.conversation_manager.get_rolling_window(
            session_id=session_id,
            window_size=window_size,
        )

    def get_conversation_summary(self, session_id: str) -> str:
        """
        Generate summary of conversation.

        Args:
            session_id: Session ID

        Returns:
            Conversation summary
        """
        return self.conversation_manager.generate_summary(session_id)

    def extract_conversation_topics(self, session_id: str) -> List[str]:
        """
        Extract key topics from conversation.

        Args:
            session_id: Session ID

        Returns:
            List of key topics
        """
        messages = self.conversation_manager.get_full_history(session_id)
        return self.conversation_manager.extract_topics(messages)

    def search_facts(
        self,
        query: str,
        user_id: Optional[str] = None,
        limit: int = 10,
        min_confidence: float = 0.5,
    ) -> List[Fact]:
        """
        Search facts by query with confidence filtering.

        Args:
            query: Search query
            user_id: Filter by user ID
            limit: Maximum results
            min_confidence: Minimum confidence threshold

        Returns:
            List of matching facts
        """
        try:
            # Search in Mem0
            memories = self.memory.search(
                query=query,
                user_id=user_id or "system",
                limit=limit * 2,  # Get more to filter by confidence
            )

            facts = []
            for mem in memories:
                if mem.get("metadata", {}).get("type") == MemoryType.FACT.value:
                    fact_data = self._find_fact_by_content(mem["memory"], user_id)
                    if fact_data and fact_data.get("confidence", 0) >= min_confidence:
                        facts.append(Fact.from_dict(fact_data))

            return facts[:limit]
        except Exception as e:
            logger.error(f"Error searching facts: {e}")
            return []

    def get_recent_rag_traces(
        self,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        limit: int = 10,
    ) -> List[RAGTrace]:
        """
        Get recent RAG traces for pattern analysis.

        Args:
            user_id: Filter by user ID
            session_id: Filter by session ID
            limit: Maximum results

        Returns:
            List of recent RAG traces
        """
        try:
            pattern = f"{user_id or 'system'}_*.json" if user_id else "*.json"
            trace_files = sorted(
                self.rag_traces_path.glob(pattern),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )

            traces = []
            for trace_file in trace_files[:limit]:
                with open(trace_file, "r") as f:
                    trace_data = json.load(f)

                # Filter by session if specified
                if session_id and trace_data.get("session_id") != session_id:
                    continue

                traces.append(RAGTrace.from_dict(trace_data))

            return traces
        except Exception as e:
            logger.error(f"Error getting RAG traces: {e}")
            return []

    def analyze_retrieval_patterns(
        self,
        user_id: Optional[str] = None,
        days: int = 7,
    ) -> Dict[str, Any]:
        """
        Analyze RAG retrieval patterns for optimization.

        Args:
            user_id: Filter by user ID
            days: Number of days to analyze

        Returns:
            Analysis results with patterns and insights
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            traces = self.get_recent_rag_traces(user_id=user_id, limit=1000)

            # Filter by date
            recent_traces = [
                t for t in traces
                if t.timestamp >= cutoff_date
            ]

            if not recent_traces:
                return {"message": "No recent traces found"}

            # Analyze patterns
            total_queries = len(recent_traces)
            avg_chunks = sum(len(t.chunk_ids) for t in recent_traces) / total_queries
            avg_relevance = sum(
                sum(t.relevance_scores) / len(t.relevance_scores) if t.relevance_scores else 0
                for t in recent_traces
            ) / total_queries

            # Find common query patterns
            query_words = []
            for trace in recent_traces:
                query_words.extend(trace.query.lower().split())

            from collections import Counter
            common_terms = Counter(query_words).most_common(10)

            return {
                "total_queries": total_queries,
                "avg_chunks_retrieved": avg_chunks,
                "avg_relevance_score": avg_relevance,
                "common_query_terms": [term for term, count in common_terms],
                "period_days": days,
            }
        except Exception as e:
            logger.error(f"Error analyzing retrieval patterns: {e}")
            return {"error": str(e)}

    def apply_all_retention_policies(self) -> Dict[str, int]:
        """
        Apply retention policies to all memory types.

        Returns:
            Dictionary with counts of removed items per type
        """
        removed_counts = {
            "facts": 0,
            "conversations": 0,
            "rag_traces": 0,
        }

        try:
            # Apply fact retention
            fact_files = list(self.facts_path.glob("*.json"))
            fact_items = []
            for fact_file in fact_files:
                with open(fact_file, "r") as f:
                    fact_items.append({"file": fact_file, **json.load(f)})

            facts_to_remove = self.fact_retention.apply_retention(fact_items, self.max_facts)
            for fact in facts_to_remove:
                fact["file"].unlink()
                removed_counts["facts"] += 1

            # Apply conversation retention
            conv_files = list(self.conversations_path.glob("*.json"))
            conv_items = []
            for conv_file in conv_files:
                with open(conv_file, "r") as f:
                    conv_items.append({"file": conv_file, **json.load(f)})

            convs_to_remove = self.conversation_retention.apply_retention(conv_items, self.max_conversations)
            for conv in convs_to_remove:
                conv["file"].unlink()
                removed_counts["conversations"] += 1

            # Apply RAG trace retention
            trace_files = list(self.rag_traces_path.glob("*.json"))
            trace_items = []
            for trace_file in trace_files:
                with open(trace_file, "r") as f:
                    trace_items.append({"file": trace_file, **json.load(f)})

            traces_to_remove = self.rag_trace_retention.apply_retention(trace_items, self.max_rag_traces)
            for trace in traces_to_remove:
                trace["file"].unlink()
                removed_counts["rag_traces"] += 1

            logger.info(f"Applied retention policies: {removed_counts}")

            return removed_counts
        except Exception as e:
            logger.error(f"Error applying retention policies: {e}")
            return removed_counts
