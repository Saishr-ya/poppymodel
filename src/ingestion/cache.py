"""
src/ingestion/cache.py

Redis-backed caching decorator for all external API calls.

Why this matters operationally:
  - ChEMBL, DisGeNET, PubMed all have rate limits.
  - A batch run of 120,000 pairs without caching = 240,000+ API calls = banned.
  - With caching: first run is slow; every subsequent run is instant.
  - Default TTL: 30 days. Gene-disease data changes rarely. Drug data almost never.

Setup:
  Docker Compose includes Redis on port 6379.
  No configuration needed beyond docker-compose up -d.

Fallback behavior:
  If Redis is unavailable (e.g., during unit tests), the decorator
  runs the function directly without caching. This ensures tests
  don't require a running Redis instance.

Usage:
    from src.ingestion.cache import cached_api_call

    class MyClient:
        @cached_api_call(ttl_seconds=86400 * 30)
        def fetch_data(self, some_id: str) -> dict:
            # This HTTP call is cached automatically
            r = requests.get(f"https://api.example.com/{some_id}")
            return r.json()

    # Also usable as a standalone function decorator:
    @cached_api_call(ttl_seconds=3600)
    def get_pubmed_articles(query: str) -> list:
        ...
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from functools import wraps
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# Lazy Redis connection — only established when first cache call is made
_redis_client = None
_redis_available = None   # None = not yet checked; True/False = checked


def _get_redis():
    """Return Redis client, or None if Redis is unavailable."""
    global _redis_client, _redis_available

    if _redis_available is False:
        return None

    if _redis_client is not None:
        return _redis_client

    try:
        import redis
        import os

        host = os.getenv("REDIS_HOST", "localhost")
        port = int(os.getenv("REDIS_PORT", 6379))

        client = redis.Redis(
            host=host,
            port=port,
            decode_responses=True,
            socket_connect_timeout=2,
            socket_timeout=5,
        )
        client.ping()   # Test connection
        _redis_client = client
        _redis_available = True
        logger.info(f"Redis cache connected: {host}:{port}")
        return _redis_client

    except ImportError:
        logger.warning("redis package not installed. Run: pip install redis. Caching disabled.")
        _redis_available = False
        return None

    except Exception as e:
        logger.warning(f"Redis unavailable ({e}). API caching disabled — expect slower runs.")
        _redis_available = False
        return None


def _make_cache_key(func_name: str, args: tuple, kwargs: dict) -> str:
    """
    Build a deterministic cache key from function name + arguments.

    Uses MD5 of JSON-serialized args — fast, collision-resistant enough for cache keys.
    Keys are prefixed with 'repurposing:' to namespace from other Redis users.
    """
    try:
        key_data = f"{func_name}:{json.dumps(args, sort_keys=True, default=str)}:{json.dumps(kwargs, sort_keys=True, default=str)}"
    except (TypeError, ValueError):
        # If args aren't JSON-serializable, use repr
        key_data = f"{func_name}:{repr(args)}:{repr(kwargs)}"

    key_hash = hashlib.md5(key_data.encode()).hexdigest()
    return f"repurposing:{func_name}:{key_hash}"


def cached_api_call(ttl_seconds: int = 86400 * 30):
    """
    Decorator: cache the return value of any API-calling function in Redis.

    TTL recommendations:
        Drug/target data (ChEMBL, DrugBank):     86400 * 90  (90 days — stable)
        Disease/gene data (DisGeNET, Orphanet):   86400 * 30  (30 days)
        Literature (PubMed, ClinicalTrials):      86400 * 7   (7 days — updates weekly)
        Safety data (FAERS, FDA):                 86400 * 14  (14 days)

    Usage:
        @cached_api_call(ttl_seconds=86400 * 30)
        def my_api_function(self, drug_id: str) -> dict:
            ...

    Works on both instance methods and module-level functions.
    The 'self' argument (for methods) is excluded from the cache key
    since we don't want object identity to affect cache hits.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            redis = _get_redis()

            if redis is None:
                # No cache available — run function directly
                return func(*args, **kwargs)

            # Exclude 'self' from cache key (first arg of instance methods)
            # This allows different client instances to share cache entries
            cache_args = args[1:] if args and hasattr(args[0], "__class__") else args
            cache_key = _make_cache_key(func.__qualname__, cache_args, kwargs)

            # ── Cache read ──────────────────────────────────────────────
            try:
                cached_value = redis.get(cache_key)
                if cached_value is not None:
                    logger.debug(f"Cache HIT: {func.__name__} {cache_args[:1]}")
                    return json.loads(cached_value)
            except Exception as e:
                logger.debug(f"Cache read error for {func.__name__}: {e}")

            # ── Polite rate limiting: 500ms between API calls ────────────
            time.sleep(0.5)

            # ── Execute function ─────────────────────────────────────────
            logger.debug(f"Cache MISS: {func.__name__} {cache_args[:1]}")
            result = func(*args, **kwargs)

            # ── Cache write ─────────────────────────────────────────────
            if result is not None:
                try:
                    redis.setex(cache_key, ttl_seconds, json.dumps(result, default=str))
                except Exception as e:
                    logger.debug(f"Cache write error for {func.__name__}: {e}")

            return result

        # Attach cache management utilities to the function
        wrapper.cache_key = lambda *a, **kw: _make_cache_key(
            func.__qualname__, a[1:] if a else a, kw
        )
        wrapper.invalidate = lambda *a, **kw: _invalidate_key(
            _make_cache_key(func.__qualname__, a[1:] if a else a, kw)
        )

        return wrapper
    return decorator


def _invalidate_key(cache_key: str) -> bool:
    """Delete a specific cache entry."""
    redis = _get_redis()
    if redis:
        try:
            redis.delete(cache_key)
            return True
        except Exception:
            pass
    return False


def clear_cache_for_function(func_name: str) -> int:
    """
    Delete all cached entries for a specific function.
    Useful when a data source has been updated and you want fresh data.

    Returns number of keys deleted.

    Usage:
        from src.ingestion.cache import clear_cache_for_function
        clear_cache_for_function("ChEMBLClient.get_molecule")
    """
    redis = _get_redis()
    if not redis:
        return 0
    pattern = f"repurposing:{func_name}:*"
    try:
        keys = list(redis.scan_iter(match=pattern))
        if keys:
            redis.delete(*keys)
        logger.info(f"Cleared {len(keys)} cache entries for {func_name}")
        return len(keys)
    except Exception as e:
        logger.error(f"Cache clear failed for {func_name}: {e}")
        return 0


def cache_stats() -> dict:
    """
    Return cache statistics: total keys, memory usage, hit/miss counters.
    Useful for monitoring cache health during long batch runs.
    """
    redis = _get_redis()
    if not redis:
        return {"available": False}

    try:
        info = redis.info("memory")
        total_keys = redis.dbsize()
        repurposing_keys = len(list(redis.scan_iter(match="repurposing:*")))
        return {
            "available": True,
            "total_redis_keys": total_keys,
            "repurposing_keys": repurposing_keys,
            "memory_used_mb": round(info.get("used_memory", 0) / 1024 / 1024, 1),
            "memory_peak_mb": round(info.get("used_memory_peak", 0) / 1024 / 1024, 1),
        }
    except Exception as e:
        return {"available": True, "error": str(e)}