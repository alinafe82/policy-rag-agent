"""Simple in-memory cache with TTL support."""

import hashlib
import time
from dataclasses import dataclass
from typing import Any


@dataclass
class CacheEntry:
    """Cache entry with value and expiration."""

    value: Any
    expires_at: float


class SimpleCache:
    """Thread-safe in-memory cache with TTL and size limits."""

    def __init__(self, ttl_seconds: int = 300, max_size: int = 1000):
        """Initialize cache with TTL and max size."""
        self.ttl_seconds = ttl_seconds
        self.max_size = max_size
        self._cache: dict[str, CacheEntry] = {}

    def _generate_key(self, *args: Any, **kwargs: Any) -> str:
        """Generate cache key from arguments."""
        key_parts = [str(arg) for arg in args]
        key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
        key_str = "|".join(key_parts)
        return hashlib.md5(key_str.encode()).hexdigest()

    def get(self, key: str) -> Any | None:
        """Get value from cache if not expired."""
        if key not in self._cache:
            return None

        entry = self._cache[key]
        if time.time() > entry.expires_at:
            del self._cache[key]
            return None

        return entry.value

    def set(self, key: str, value: Any) -> None:
        """Set value in cache with TTL."""
        # Evict oldest entries if at max size
        if len(self._cache) >= self.max_size:
            # Simple eviction: remove expired first, then oldest
            self._evict_expired()
            if len(self._cache) >= self.max_size:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]

        expires_at = time.time() + self.ttl_seconds
        self._cache[key] = CacheEntry(value=value, expires_at=expires_at)

    def _evict_expired(self) -> None:
        """Remove expired entries."""
        now = time.time()
        expired_keys = [
            key for key, entry in self._cache.items() if now > entry.expires_at
        ]
        for key in expired_keys:
            del self._cache[key]

    def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()

    def cached_get(self, *args: Any, **kwargs: Any) -> tuple[Any | None, str]:
        """Get with automatic key generation."""
        key = self._generate_key(*args, **kwargs)
        return self.get(key), key
