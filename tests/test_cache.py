"""Tests for caching functionality."""

import time

import pytest

from src.cache import CacheEntry, SimpleCache


class TestCacheEntry:
    """Test CacheEntry dataclass."""

    def test_cache_entry_creation(self):
        """Test creating a cache entry."""
        entry = CacheEntry(value="test", expires_at=time.time() + 100)
        assert entry.value == "test"
        assert entry.expires_at > time.time()


class TestSimpleCache:
    """Test SimpleCache functionality."""

    @pytest.fixture
    def cache(self):
        """Create a cache with short TTL."""
        return SimpleCache(ttl_seconds=1, max_size=3)

    def test_cache_set_and_get(self, cache):
        """Test setting and getting values."""
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

    def test_cache_get_nonexistent(self, cache):
        """Test getting non-existent key returns None."""
        assert cache.get("nonexistent") is None

    def test_cache_expiration(self, cache):
        """Test that entries expire after TTL."""
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

        # Wait for expiration
        time.sleep(1.1)
        assert cache.get("key1") is None

    def test_cache_max_size_eviction(self, cache):
        """Test that cache evicts entries when at max size."""
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")
        assert len(cache._cache) == 3

        # Adding 4th entry should evict oldest
        cache.set("key4", "value4")
        assert len(cache._cache) == 3
        assert cache.get("key4") == "value4"

    def test_cache_clear(self, cache):
        """Test clearing the cache."""
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        assert len(cache._cache) == 2

        cache.clear()
        assert len(cache._cache) == 0
        assert cache.get("key1") is None

    def test_generate_key(self, cache):
        """Test key generation from arguments."""
        key1 = cache._generate_key("arg1", "arg2", param1="value1")
        key2 = cache._generate_key("arg1", "arg2", param1="value1")
        key3 = cache._generate_key("arg1", "arg2", param1="value2")

        assert key1 == key2  # Same args = same key
        assert key1 != key3  # Different args = different key

    def test_cached_get_miss(self, cache):
        """Test cached_get with cache miss."""
        value, key = cache.cached_get("query", top_k=4)
        assert value is None
        assert isinstance(key, str)
        assert len(key) > 0

    def test_cached_get_hit(self, cache):
        """Test cached_get with cache hit."""
        _, key = cache.cached_get("query", top_k=4)
        cache.set(key, {"result": "data"})

        value, key2 = cache.cached_get("query", top_k=4)
        assert value == {"result": "data"}
        assert key == key2

    def test_evict_expired(self, cache):
        """Test manual eviction of expired entries."""
        cache.set("key1", "value1")
        time.sleep(1.1)  # Wait for expiration

        cache._evict_expired()
        assert cache.get("key1") is None
        assert len(cache._cache) == 0
