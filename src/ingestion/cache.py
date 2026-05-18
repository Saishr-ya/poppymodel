"""
src/ingestion/cache.py

Redis-backed caching decorator for all external API calls.

BUG FIX: The decorator previously stripped the first argument from the cache
key assuming it was always 'self' (an instance method receiver). The check
`hasattr(args[0], "__class__")` is True for EVERY Python object including
strings, ints, etc. — so for module-level functions like
_resolve_chembl_parent("CHEMBL192"), the first arg "CHEMBL192" was being
stripped and all calls produced the same empty cache key.

Fix: use `inspect.ismethod` or check whether the first arg is an instance
of a user-defined class (not a built-in type) to detect 'self'. The safest
heuristic: if the function's __qualname__ contains a '.' (e.g.
"ChEMBLClient.get_molecule") it's a method and we strip arg[0]; if it
doesn't contain '.' (e.g. "_resolve_chembl_parent") it's a module-level
function and we include all args in the key.
"""

from __future__ import annotations

import hashlib
import inspect
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
        import redis, os
        host = os.getenv("REDIS_HOST", "localhost")
        port = int(os.getenv("REDIS_PORT", 6379))
        client = redis.Redis(
            host=host, port=port,
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


def _is_instance_method_call(func, args) -> bool:
    """
    Determine whether a cached function is being called as an instance method
    (first arg is 'self') or as a module-level function (first arg is data).

    Heuristic: if the function's qualified name contains a dot, it was defined
    inside a class body and args[0] is 'self'. Module-level functions have a
    flat qualname with no dot.

    Examples:
        ChEMBLClient.get_molecule       → qualname has dot → is method → strip args[0]
        _resolve_chembl_parent          → no dot → module function → keep all args
        OpenTargetsClient._batch_lookup → has dot → is method → strip args[0]
    """
    if not args:
        return False
    qualname = getattr(func, "__qualname__", "")
    # A dot in qualname means the function was defined inside a class
    return "." in qualname and "<locals>" not in qualname


def _make_cache_key(func_name: str, args: tuple, kwargs: dict) -> str:
    try:
        key_data = (
            f"{func_name}:"
            f"{json.dumps(args, sort_keys=True, default=str)}:"
            f"{json.dumps(kwargs, sort_keys=True, default=str)}"
        )
    except (TypeError, ValueError):
        key_data = f"{func_name}:{repr(args)}:{repr(kwargs)}"

    key_hash = hashlib.md5(key_data.encode()).hexdigest()
    return f"repurposing:{func_name}:{key_hash}"


def cached_api_call(ttl_seconds: int = 86400 * 30):
    """
    Decorator: cache the return value of any API-calling function in Redis.

    Works correctly for both instance methods and module-level functions.
    The 'self' argument is excluded from the cache key for instance methods;
    for module-level functions all arguments are included.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            redis = _get_redis()
            if redis is None:
                return func(*args, **kwargs)

            # Determine cache key args — strip 'self' for methods, keep all for functions
            if _is_instance_method_call(func, args):
                cache_args = args[1:]   # skip 'self'
            else:
                cache_args = args       # module-level: include everything

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