"""
src/ingestion/cache.py

Redis-backed caching decorator for all external API calls.
Build this Day 1 — prevents rate-limit bans and expensive re-downloads.

Usage:
    from src.ingestion.cache import cached_api_call

    @cached_api_call(ttl_seconds=86400 * 7)   # 7-day cache
    def get_chembl_targets(chembl_id: str) -> list:
        ...

If Redis is unavailable, falls back to in-memory dict cache (dev mode).
"""

from __future__ import annotations
import hashlib
import json
import time
import logging
from functools import wraps
from typing import Callable, Any, Optional

logger = logging.getLogger(__name__)

# ── Redis connection (optional) ────────────────────────────────────────────────
_redis_client = None
_memory_cache: dict = {}   # fallback when Redis is down


def _get_redis():
    global _redis_client
    if _redis_client is not None:
        return _redis_client
    try:
        import redis
        import os
        host = os.getenv("REDIS_HOST", "localhost")
        port = int(os.getenv("REDIS_PORT", "6379"))
        _redis_client = redis.Redis(host=host, port=port, decode_responses=True, socket_timeout=2)
        _redis_client.ping()
        logger.info(f"Redis cache connected at {host}:{port}")
        return _redis_client
    except Exception as e:
        logger.warning(f"Redis unavailable ({e}) — using in-memory cache")
        return None


def _make_cache_key(func_name: str, args: tuple, kwargs: dict) -> str:
    raw = f"{func_name}:{json.dumps(args, sort_keys=True, default=str)}:{json.dumps(kwargs, sort_keys=True, default=str)}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


def cached_api_call(
    ttl_seconds: int = 86400 * 30,      # 30-day default
    rate_limit_delay: float = 0.5,       # seconds between live API calls
    skip_cache: bool = False,
) -> Callable:
    """
    Decorator that caches external API call results.

    Args:
        ttl_seconds:       How long to keep the cached result.
        rate_limit_delay:  Sleep this many seconds before each live API call.
        skip_cache:        If True, always hits the API (use for debugging).
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            cache_key = _make_cache_key(func.__name__, args, kwargs)
            r = _get_redis()

            if not skip_cache:
                # Try Redis first
                if r is not None:
                    try:
                        cached = r.get(cache_key)
                        if cached:
                            logger.debug(f"Cache HIT [{func.__name__}] key={cache_key[:8]}…")
                            return json.loads(cached)
                    except Exception as e:
                        logger.warning(f"Redis read error: {e}")

                # Fall back to memory cache
                if cache_key in _memory_cache:
                    entry = _memory_cache[cache_key]
                    if time.time() < entry["expires"]:
                        logger.debug(f"Memory cache HIT [{func.__name__}]")
                        return entry["value"]

            # Cache miss — call the real API
            logger.debug(f"Cache MISS [{func.__name__}] — calling API")
            time.sleep(rate_limit_delay)
            result = func(*args, **kwargs)
            serialized = json.dumps(result, default=str)

            # Write to Redis
            if r is not None:
                try:
                    r.setex(cache_key, ttl_seconds, serialized)
                except Exception as e:
                    logger.warning(f"Redis write error: {e}")

            # Write to memory fallback
            _memory_cache[cache_key] = {
                "value": result,
                "expires": time.time() + ttl_seconds,
            }

            return result
        return wrapper
    return decorator


def invalidate_cache(func_name: str, *args, **kwargs):
    """Manually invalidate a specific cached call."""
    cache_key = _make_cache_key(func_name, args, kwargs)
    r = _get_redis()
    if r:
        r.delete(cache_key)
    _memory_cache.pop(cache_key, None)
    logger.info(f"Cache invalidated for {func_name} key={cache_key[:8]}…")
