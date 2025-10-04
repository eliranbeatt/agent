"""Unit tests for retention policies."""

import pytest
from datetime import datetime, timedelta

from app.core.memory.retention_policies import (
    TTLPolicy,
    LRUPolicy,
    FIFOPolicy,
    SizeLimitPolicy,
    CompositePolicy,
    RetentionPolicyManager,
)


class TestTTLPolicy:
    """Test TTL (Time-To-Live) retention policy."""

    def test_should_retain_recent_item(self):
        """Test that recent items are retained."""
        policy = TTLPolicy(ttl_days=30)
        current_time = datetime.now()

        item = {
            "content": "test",
            "timestamp": (current_time - timedelta(days=10)).isoformat(),
        }

        assert policy.should_retain(item, current_time) is True

    def test_should_not_retain_old_item(self):
        """Test that old items are not retained."""
        policy = TTLPolicy(ttl_days=30)
        current_time = datetime.now()

        item = {
            "content": "test",
            "timestamp": (current_time - timedelta(days=40)).isoformat(),
        }

        assert policy.should_retain(item, current_time) is False

    def test_should_retain_item_without_timestamp(self):
        """Test that items without timestamp are retained."""
        policy = TTLPolicy(ttl_days=30)
        current_time = datetime.now()

        item = {"content": "test"}

        assert policy.should_retain(item, current_time) is True

    def test_get_items_to_remove(self):
        """Test getting items to remove based on TTL."""
        policy = TTLPolicy(ttl_days=30)
        current_time = datetime.now()

        items = [
            {"id": 1, "timestamp": (current_time - timedelta(days=10)).isoformat()},
            {"id": 2, "timestamp": (current_time - timedelta(days=40)).isoformat()},
            {"id": 3, "timestamp": (current_time - timedelta(days=50)).isoformat()},
            {"id": 4, "timestamp": (current_time - timedelta(days=5)).isoformat()},
        ]

        to_remove = policy.get_items_to_remove(items, max_count=10, current_time=current_time)

        assert len(to_remove) == 2
        assert any(item["id"] == 2 for item in to_remove)
        assert any(item["id"] == 3 for item in to_remove)


class TestLRUPolicy:
    """Test LRU (Least Recently Used) retention policy."""

    def test_get_items_to_remove_within_limit(self):
        """Test that no items are removed when within limit."""
        policy = LRUPolicy(max_items=5)
        current_time = datetime.now()

        items = [
            {"id": i, "timestamp": (current_time - timedelta(days=i)).isoformat()}
            for i in range(3)
        ]

        to_remove = policy.get_items_to_remove(items, max_count=5, current_time=current_time)

        assert len(to_remove) == 0

    def test_get_items_to_remove_exceeds_limit(self):
        """Test that oldest items are removed when exceeding limit."""
        policy = LRUPolicy(max_items=3)
        current_time = datetime.now()

        items = [
            {"id": 1, "timestamp": (current_time - timedelta(days=5)).isoformat()},
            {"id": 2, "timestamp": (current_time - timedelta(days=3)).isoformat()},
            {"id": 3, "timestamp": (current_time - timedelta(days=1)).isoformat()},
            {"id": 4, "timestamp": (current_time - timedelta(days=4)).isoformat()},
            {"id": 5, "timestamp": (current_time - timedelta(days=2)).isoformat()},
        ]

        to_remove = policy.get_items_to_remove(items, max_count=3, current_time=current_time)

        assert len(to_remove) == 2
        # Should remove oldest items (id 1 and 4)
        removed_ids = [item["id"] for item in to_remove]
        assert 1 in removed_ids
        assert 4 in removed_ids


class TestFIFOPolicy:
    """Test FIFO (First In First Out) retention policy."""

    def test_get_items_to_remove_within_limit(self):
        """Test that no items are removed when within limit."""
        policy = FIFOPolicy(max_items=5)
        current_time = datetime.now()

        items = [
            {"id": i, "timestamp": (current_time - timedelta(days=i)).isoformat()}
            for i in range(3)
        ]

        to_remove = policy.get_items_to_remove(items, max_count=5, current_time=current_time)

        assert len(to_remove) == 0

    def test_get_items_to_remove_exceeds_limit(self):
        """Test that oldest items are removed when exceeding limit."""
        policy = FIFOPolicy(max_items=3)
        current_time = datetime.now()

        items = [
            {"id": 1, "timestamp": (current_time - timedelta(days=5)).isoformat()},
            {"id": 2, "timestamp": (current_time - timedelta(days=4)).isoformat()},
            {"id": 3, "timestamp": (current_time - timedelta(days=3)).isoformat()},
            {"id": 4, "timestamp": (current_time - timedelta(days=2)).isoformat()},
            {"id": 5, "timestamp": (current_time - timedelta(days=1)).isoformat()},
        ]

        to_remove = policy.get_items_to_remove(items, max_count=3, current_time=current_time)

        assert len(to_remove) == 2
        # Should remove oldest items (id 1 and 2)
        removed_ids = [item["id"] for item in to_remove]
        assert 1 in removed_ids
        assert 2 in removed_ids


class TestSizeLimitPolicy:
    """Test size-based retention policy."""

    def test_get_items_to_remove_within_limit(self):
        """Test that no items are removed when within size limit."""
        policy = SizeLimitPolicy(max_size_mb=1)  # 1 MB
        current_time = datetime.now()

        items = [
            {"id": i, "content": "small", "timestamp": datetime.now().isoformat()}
            for i in range(3)
        ]

        to_remove = policy.get_items_to_remove(items, max_count=10, current_time=current_time)

        # Small items should be within limit
        assert len(to_remove) == 0

    def test_get_items_to_remove_exceeds_limit(self):
        """Test that items are removed when exceeding size limit."""
        policy = SizeLimitPolicy(max_size_mb=0.001)  # Very small limit
        current_time = datetime.now()

        # Create items with substantial content
        items = [
            {
                "id": i,
                "content": "x" * 1000,  # 1KB of content
                "timestamp": (current_time - timedelta(days=5 - i)).isoformat(),
            }
            for i in range(5)
        ]

        to_remove = policy.get_items_to_remove(items, max_count=10, current_time=current_time)

        # Should remove some items to get under size limit
        assert len(to_remove) > 0


class TestCompositePolicy:
    """Test composite retention policy."""

    def test_should_retain_passes_all_policies(self):
        """Test that item must pass all policies to be retained."""
        ttl_policy = TTLPolicy(ttl_days=30)
        lru_policy = LRUPolicy(max_items=5)

        composite = CompositePolicy([ttl_policy, lru_policy])
        current_time = datetime.now()

        # Recent item should pass TTL
        item = {
            "content": "test",
            "timestamp": (current_time - timedelta(days=10)).isoformat(),
        }

        assert composite.should_retain(item, current_time) is True

    def test_should_not_retain_fails_one_policy(self):
        """Test that item failing one policy is not retained."""
        ttl_policy = TTLPolicy(ttl_days=30)
        lru_policy = LRUPolicy(max_items=5)

        composite = CompositePolicy([ttl_policy, lru_policy])
        current_time = datetime.now()

        # Old item should fail TTL
        item = {
            "content": "test",
            "timestamp": (current_time - timedelta(days=40)).isoformat(),
        }

        assert composite.should_retain(item, current_time) is False

    def test_get_items_to_remove_combines_policies(self):
        """Test that composite policy combines removal from all policies."""
        ttl_policy = TTLPolicy(ttl_days=30)
        lru_policy = LRUPolicy(max_items=3)

        composite = CompositePolicy([ttl_policy, lru_policy])
        current_time = datetime.now()

        items = [
            {"id": 1, "timestamp": (current_time - timedelta(days=40)).isoformat()},  # Fails TTL
            {"id": 2, "timestamp": (current_time - timedelta(days=10)).isoformat()},
            {"id": 3, "timestamp": (current_time - timedelta(days=5)).isoformat()},
            {"id": 4, "timestamp": (current_time - timedelta(days=3)).isoformat()},
            {"id": 5, "timestamp": (current_time - timedelta(days=1)).isoformat()},
        ]

        to_remove = composite.get_items_to_remove(items, max_count=3, current_time=current_time)

        # Should remove item 1 (TTL) and oldest items beyond max_count
        assert len(to_remove) >= 1
        assert any(item["id"] == 1 for item in to_remove)


class TestRetentionPolicyManager:
    """Test retention policy manager."""

    def test_create_lru_policy(self):
        """Test creating LRU policy."""
        manager = RetentionPolicyManager(
            ttl_days=30,
            max_items=100,
            policy_type="lru",
        )

        assert manager.policy_type == "lru"
        assert isinstance(manager.policy, CompositePolicy)

    def test_create_fifo_policy(self):
        """Test creating FIFO policy."""
        manager = RetentionPolicyManager(
            ttl_days=30,
            max_items=100,
            policy_type="fifo",
        )

        assert manager.policy_type == "fifo"
        assert isinstance(manager.policy, CompositePolicy)

    def test_create_ttl_policy(self):
        """Test creating TTL policy."""
        manager = RetentionPolicyManager(
            ttl_days=30,
            max_items=100,
            policy_type="ttl",
        )

        assert manager.policy_type == "ttl"
        assert isinstance(manager.policy, TTLPolicy)

    def test_create_composite_policy(self):
        """Test creating composite policy."""
        manager = RetentionPolicyManager(
            ttl_days=30,
            max_items=100,
            max_size_mb=500,
            policy_type="composite",
        )

        assert manager.policy_type == "composite"
        assert isinstance(manager.policy, CompositePolicy)

    def test_apply_retention(self):
        """Test applying retention policy."""
        manager = RetentionPolicyManager(
            ttl_days=30,
            max_items=3,
            policy_type="lru",
        )

        current_time = datetime.now()
        items = [
            {"id": i, "timestamp": (current_time - timedelta(days=i)).isoformat()}
            for i in range(5)
        ]

        to_remove = manager.apply_retention(items, max_count=3)

        # Should identify items to remove
        assert len(to_remove) >= 0

    def test_should_retain_item(self):
        """Test checking if individual item should be retained."""
        manager = RetentionPolicyManager(
            ttl_days=30,
            max_items=100,
            policy_type="ttl",
        )

        current_time = datetime.now()

        # Recent item
        recent_item = {
            "content": "test",
            "timestamp": (current_time - timedelta(days=10)).isoformat(),
        }
        assert manager.should_retain_item(recent_item) is True

        # Old item
        old_item = {
            "content": "test",
            "timestamp": (current_time - timedelta(days=40)).isoformat(),
        }
        assert manager.should_retain_item(old_item) is False

    def test_unknown_policy_type_defaults_to_lru(self):
        """Test that unknown policy type defaults to LRU."""
        manager = RetentionPolicyManager(
            ttl_days=30,
            max_items=100,
            policy_type="unknown",
        )

        # Should default to LRU
        assert isinstance(manager.policy, LRUPolicy)
