"""Process-local NCBI pacing shared by ingestion, agents, and verification."""

import asyncio
import hashlib
import time
import weakref
from typing import Awaitable, Callable, Optional


class NcbiRateLimiter:
    """Serialize request starts using a monotonic clock, not wall-clock time."""

    def __init__(
        self,
        requests_per_second: float = 3.0,
        clock: Callable[[], float] = time.monotonic,
        delay: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        if not 0 < requests_per_second <= 10:
            raise ValueError("NCBI request rate must be between 0 and 10")
        self._interval = 1.0 / requests_per_second
        self._clock = clock
        self._delay = delay
        self._next_request = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Reserve a rate-limited request start; cancellation consumes no slot."""
        async with self._lock:
            remaining = self._next_request - self._clock()
            if remaining > 0:
                await self._delay(remaining)
            self._next_request = self._clock() + self._interval


_LIMITERS: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def get_ncbi_limiter(api_key: Optional[str] = None) -> NcbiRateLimiter:
    """Share pacing per running loop and credential without retaining raw keys.

    Args:
        api_key: NCBI credential, or None for the shared anonymous IP budget.

    Returns:
        Rate limiter shared by clients using this credential in the same loop.
    """
    loop = asyncio.get_running_loop()
    identity = hashlib.sha256((api_key or "").encode()).hexdigest()
    limiters = _LIMITERS.setdefault(loop, {})
    if identity not in limiters:
        limiters[identity] = NcbiRateLimiter(10.0 if api_key else 3.0)
    return limiters[identity]
