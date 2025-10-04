"""Memory retention policies for managing memory lifecycle."""

import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from pathlib import Path
from abc import ABC, abstractmethod
import json

logger = logging.getLogger(__name__)


class RetentionPolicy(ABC):
    """Base class for retention policies."""

    @abstractmethod
    def should_retain(self, item: Dict[str, Any], current_time: datetime) -> bool:
        """
        Determine if an item should be retained.

        Args:
            item: Item to check
            current_time: Current timestamp

        Returns:
            True if item should be retained, False otherwise
        """
        pass

    @abstractmethod
    def get_items_to_remove(
        self,
        items: List[Dict[str, Any]],
        max_count: int,
        current_time: datetime,
    ) -> List[Dict[str, Any]]:
        """
        Get items that should be removed based on policy.

        Args:
            items: List of items
            max_count: Maximum number of items to keep
            current_time: Current timestamp

        Returns:
            List of items to remove
        """
        pass


class TTLPolicy(RetentionPolicy):
    """Time-to-live based retention policy."""

    def __init__(self, ttl_days: int):
        """
        Initialize TTL policy.

        Args:
            ttl_days: Number of days to retain items
        """
        self.ttl_days = ttl_days

    def should_retain(self, item: Dict[str, Any], current_time: datetime) -> bool:
        """Check if item is within TTL."""
        timestamp_str = item.get("timestamp")
        if not timestamp_str:
            return True  # Keep items without timestamp

        try:
            timestamp = datetime.fromisoformat(timestamp_str)
            age = current_time - timestamp
            return age.days < self.ttl_days
        except Exception as e:
            logger.error(f"Error checking TTL: {e}")
            return True

    def get_items_to_remove(
        self,
        items: List[Dict[str, Any]],
        max_count: int,
        current_time: datetime,
    ) -> List[Dict[str, Any]]:
        """Get items that exceed TTL."""
        return [item for item in items if not self.should_retain(item, current_time)]


class LRUPolicy(RetentionPolicy):
    """Least Recently Used retention policy."""

    def __init__(self, max_items: int):
        """
        Initialize LRU policy.

        Args:
            max_items: Maximum number of items to keep
        """
        self.max_items = max_items

    def should_retain(self, item: Dict[str, Any], current_time: datetime) -> bool:
        """LRU doesn't make individual decisions."""
        return True

    def get_items_to_remove(
        self,
        items: List[Dict[str, Any]],
        max_count: int,
        current_time: datetime,
    ) -> List[Dict[str, Any]]:
        """Get least recently used items beyond max count."""
        if len(items) <= max_count:
            return []

        # Sort by timestamp (oldest first)
        sorted_items = sorted(
            items,
            key=lambda x: datetime.fromisoformat(x.get("timestamp", "1970-01-01T00:00:00")),
        )

        # Return items beyond max count
        return sorted_items[: len(items) - max_count]


class FIFOPolicy(RetentionPolicy):
    """First In First Out retention policy."""

    def __init__(self, max_items: int):
        """
        Initialize FIFO policy.

        Args:
            max_items: Maximum number of items to keep
        """
        self.max_items = max_items

    def should_retain(self, item: Dict[str, Any], current_time: datetime) -> bool:
        """FIFO doesn't make individual decisions."""
        return True

    def get_items_to_remove(
        self,
        items: List[Dict[str, Any]],
        max_count: int,
        current_time: datetime,
    ) -> List[Dict[str, Any]]:
        """Get oldest items beyond max count."""
        if len(items) <= max_count:
            return []

        # Sort by creation time (oldest first)
        sorted_items = sorted(
            items,
            key=lambda x: datetime.fromisoformat(x.get("timestamp", "1970-01-01T00:00:00")),
        )

        # Return oldest items beyond max count
        return sorted_items[: len(items) - max_count]


class SizeLimitPolicy(RetentionPolicy):
    """Size-based retention policy."""

    def __init__(self, max_size_mb: int):
        """
        Initialize size limit policy.

        Args:
            max_size_mb: Maximum total size in MB
        """
        self.max_size_mb = max_size_mb
        self.max_size_bytes = max_size_mb * 1024 * 1024

    def should_retain(self, item: Dict[str, Any], current_time: datetime) -> bool:
        """Size policy doesn't make individual decisions."""
        return True

    def get_items_to_remove(
        self,
        items: List[Dict[str, Any]],
        max_count: int,
        current_time: datetime,
    ) -> List[Dict[str, Any]]:
        """Get items to remove based on total size."""
        # Calculate total size
        total_size = sum(self._get_item_size(item) for item in items)

        if total_size <= self.max_size_bytes:
            return []

        # Sort by timestamp (oldest first)
        sorted_items = sorted(
            items,
            key=lambda x: datetime.fromisoformat(x.get("timestamp", "1970-01-01T00:00:00")),
        )

        # Remove oldest items until under size limit
        items_to_remove = []
        current_size = total_size

        for item in sorted_items:
            if current_size <= self.max_size_bytes:
                break
            items_to_remove.append(item)
            current_size -= self._get_item_size(item)

        return items_to_remove

    def _get_item_size(self, item: Dict[str, Any]) -> int:
        """Estimate item size in bytes."""
        # Simple estimation based on JSON string length
        return len(json.dumps(item))


class CompositePolicy(RetentionPolicy):
    """Composite policy that combines multiple policies."""

    def __init__(self, policies: List[RetentionPolicy]):
        """
        Initialize composite policy.

        Args:
            policies: List of policies to apply
        """
        self.policies = policies

    def should_retain(self, item: Dict[str, Any], current_time: datetime) -> bool:
        """Item must pass all policies to be retained."""
        return all(policy.should_retain(item, current_time) for policy in self.policies)

    def get_items_to_remove(
        self,
        items: List[Dict[str, Any]],
        max_count: int,
        current_time: datetime,
    ) -> List[Dict[str, Any]]:
        """Get items to remove based on all policies."""
        # Use a list to track items to remove (can't use set with dicts)
        items_to_remove = []
        seen_ids = set()

        for policy in self.policies:
            policy_removals = policy.get_items_to_remove(items, max_count, current_time)
            for item in policy_removals:
                # Use a unique identifier to avoid duplicates
                item_id = id(item) if "id" not in item else item.get("id")
                if item_id not in seen_ids:
                    items_to_remove.append(item)
                    seen_ids.add(item_id)

        return items_to_remove


class RetentionPolicyManager:
    """Manages retention policies for different memory types."""

    def __init__(
        self,
        ttl_days: int = 90,
        max_items: int = 1000,
        max_size_mb: int = 500,
        policy_type: str = "lru",
    ):
        """
        Initialize retention policy manager.

        Args:
            ttl_days: Default TTL in days
            max_items: Maximum number of items
            max_size_mb: Maximum size in MB
            policy_type: Policy type (lru, fifo, ttl, composite)
        """
        self.ttl_days = ttl_days
        self.max_items = max_items
        self.max_size_mb = max_size_mb
        self.policy_type = policy_type

        self.policy = self._create_policy(policy_type)

    def _create_policy(self, policy_type: str) -> RetentionPolicy:
        """Create retention policy based on type."""
        if policy_type == "lru":
            return CompositePolicy([
                TTLPolicy(self.ttl_days),
                LRUPolicy(self.max_items),
            ])
        elif policy_type == "fifo":
            return CompositePolicy([
                TTLPolicy(self.ttl_days),
                FIFOPolicy(self.max_items),
            ])
        elif policy_type == "ttl":
            return TTLPolicy(self.ttl_days)
        elif policy_type == "composite":
            return CompositePolicy([
                TTLPolicy(self.ttl_days),
                LRUPolicy(self.max_items),
                SizeLimitPolicy(self.max_size_mb),
            ])
        else:
            logger.warning(f"Unknown policy type: {policy_type}, using LRU")
            return LRUPolicy(self.max_items)

    def apply_retention(
        self,
        items: List[Dict[str, Any]],
        max_count: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Apply retention policy to items.

        Args:
            items: List of items to check
            max_count: Maximum count override

        Returns:
            List of items to remove
        """
        current_time = datetime.now()
        max_count = max_count or self.max_items

        items_to_remove = self.policy.get_items_to_remove(
            items, max_count, current_time
        )

        logger.info(f"Retention policy identified {len(items_to_remove)} items to remove")

        return items_to_remove

    def should_retain_item(self, item: Dict[str, Any]) -> bool:
        """
        Check if an individual item should be retained.

        Args:
            item: Item to check

        Returns:
            True if item should be retained
        """
        return self.policy.should_retain(item, datetime.now())
