import functools
import threading
from typing import Callable
from datetime import datetime


def cache_result(ttl_seconds: int = 3600):
    """
    Decorator to cache function results with a time-to-live.

    Parameters:
    - ttl_seconds: Time-to-live for cache in seconds
    """

    def decorator(func: Callable):
        cache = {}
        cache_lock = threading.RLock()

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create a cache key from the function arguments
            key = str(args) + str(sorted(kwargs.items()))

            with cache_lock:
                # Check if result is cached and not expired
                if key in cache:
                    result, timestamp = cache[key]
                    if (datetime.now() - timestamp).total_seconds() < ttl_seconds:
                        return result

            # Call the function and cache the result
            result = func(*args, **kwargs)

            with cache_lock:
                cache[key] = (result, datetime.now())

                # Clean expired cache entries periodically
                if len(cache) > 100:  # Arbitrary threshold to avoid constant cleanup
                    expired_keys = []
                    now = datetime.now()
                    for k, (_, ts) in cache.items():
                        if (now - ts).total_seconds() >= ttl_seconds:
                            expired_keys.append(k)

                    for k in expired_keys:
                        del cache[k]

            return result

        return wrapper

    return decorator
