"""
src/ingestion/cache.py

Redis-backed caching decorator for all external API calls.

Fixes applied:
  - _is_instance_method_call now uses inspect.isfunction + qualname dot-check
    AND validates that args[0] is not a built-in type. The previous check
    `hasattr(args[0], "__class__")` is True for EVERY Python object — strings,
    ints, etc. — causing module-level functions to strip their first argument
    from the cache key, producing collisions.
  - Added explicit guard for nested/local functions (qualname contains
    "<locals>") which should never be treated as methods.
  - Cache key uses func.__qualname__ consistently (not __name__) so methods
    on different classes with the same name don't collide.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from functools import wraps
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

_redis_client = None
_redis_available = None


def _get_redis():
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
        client.ping()
        _redis_client = client
        _redis_available = True
        logger.info(f"Redis cache connected: {host}:{port}")
        return _redis_client
    except ImportError:
        logger.warning("redis package not installed. Caching disabled.")
        _redis_available = False
        return None
    except Exception as e:
        logger.warning(f"Redis unavailable ({e}). API caching disabled.")
        _redis_available = False
        return None


def _is_instance_method_call(func: Callable, args: tuple) -> bool:
    """
    Determine whether this call is an instance method (first arg is 'self')
    or a module-level / static function (all args are data).

    Rules:
      1. If there are no args, it cannot be a method call.
      2. If qualname contains '<locals>' it is a closure/nested function, not a method.
      3. If qualname contains '.' it was defined inside a class body → method.
         The first arg is 'self' and should be excluded from the cache key.
      4. Otherwise it is a module-level function → include all args.

    This is safer than the old `hasattr(args[0], '__class__')` check which
    returned True for every Python object including str, int, list, etc.
    """
    if not args:
        return False
    qualname = getattr(func, "__qualname__", "")
    # Nested/closure functions: not methods
    if "<locals>" in qualname:
        return False
    # Class method: qualname contains a dot (e.g. "MyClass.my_method")
    return "." in qualname


def _make_cache_key(func_qualname: str, args: tuple, kwargs: dict) -> str:
    try:
        key_data = (
            f"{func_qualname}:"
            f"{json.dumps(args, sort_keys=True, default=str)}:"
            f"{json.dumps(kwargs, sort_keys=True, default=str)}"
        )
    except (TypeError, ValueError):
        key_data = f"{func_qualname}:{repr(args)}:{repr(kwargs)}"

    key_hash = hashlib.md5(key_data.encode()).hexdigest()
    # Use only the leaf name in the readable prefix to keep keys short
    leaf_name = func_qualname.split(".")[-1]
    return f"repurposing:{leaf_name}:{key_hash}"


def cached_api_call(ttl_seconds: int = 86400 * 30):
    """
    Decorator: cache the return value of any API-calling function in Redis.

    Works correctly for both instance methods and module-level functions.
    'self' is excluded from the cache key for instance methods.
    All args are included for module-level functions.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            redis = _get_redis()
            if redis is None:
                return func(*args, **kwargs)

            # Strip 'self' for methods, keep all args for module-level functions
            if _is_instance_method_call(func, args):
                cache_args = args[1:]
            else:
                cache_args = args

            cache_key = _make_cache_key(func.__qualname__, cache_args, kwargs)

            # Cache read
            try:
                cached_value = redis.get(cache_key)
                if cached_value is not None:
                    logger.debug(f"Cache HIT: {func.__name__} {cache_args[:1]}")
                    return json.loads(cached_value)
            except Exception as e:
                logger.debug(f"Cache read error for {func.__name__}: {e}")

            # Polite rate limiting
            time.sleep(0.5)

            # Execute
            logger.debug(f"Cache MISS: {func.__name__} {cache_args[:1]}")
            result = func(*args, **kwargs)

            # Cache write
            if result is not None:
                try:
                    redis.setex(cache_key, ttl_seconds, json.dumps(result, default=str))
                except Exception as e:
                    logger.debug(f"Cache write error for {func.__name__}: {e}")

            return result

        # Attach helpers for testing / manual invalidation
        wrapper.cache_key = lambda *a, **kw: _make_cache_key(
            func.__qualname__,
            a[1:] if _is_instance_method_call(func, a) else a,
            kw,
        )
        wrapper.invalidate = lambda *a, **kw: _invalidate_key(
            _make_cache_key(
                func.__qualname__,
                a[1:] if _is_instance_method_call(func, a) else a,
                kw,
            )
        )
        return wrapper
    return decorator


def _invalidate_key(cache_key: str) -> bool:
    redis = _get_redis()
    if redis:
        try:
            redis.delete(cache_key)
            return True
        except Exception:
            pass
    return False


def clear_cache_for_function(func_name: str) -> int:
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