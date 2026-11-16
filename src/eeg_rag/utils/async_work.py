"""Bound blocking work without releasing capacity before cancellation completes."""

import asyncio
from typing import Any, Callable


class BoundedWorker:
    """Run blocking functions with a fixed number of concurrent workers."""

    def __init__(self, concurrency: int = 1) -> None:
        if (
            isinstance(concurrency, bool)
            or not isinstance(concurrency, int)
            or concurrency < 1
        ):
            raise ValueError("concurrency must be a positive integer")
        self._capacity = asyncio.Semaphore(concurrency)

    async def run(self, function: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Offload work and retain capacity until its thread actually finishes.

        Args:
            function: Blocking callable.
            args: Positional arguments for the callable.
            kwargs: Keyword arguments for the callable.

        Returns:
            The callable's result.

        Raises:
            asyncio.CancelledError: After in-flight blocking work has completed.
            Exception: The callable's original exception.
        """
        async with self._capacity:
            task = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
            try:
                return await asyncio.shield(task)
            except asyncio.CancelledError:
                try:
                    await asyncio.shield(task)
                except Exception:
                    pass
                raise
