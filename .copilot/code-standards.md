# Code Quality and Standards Instructions

## Code Standards for EEG-RAG

### Python Style
- Use Python 3.9+ features including type hints for all function signatures
- Follow Google-style docstrings with Args, Returns, Raises sections
- Maximum line length 88 characters (Black formatter standard)
- Use dataclasses or Pydantic models for data structures, not plain dicts
- Prefer composition over inheritance for agent classes

### Async Patterns
- All I/O operations (API calls, file reads, database queries) must be async
- Use asyncio.gather for parallel agent execution
- Implement proper timeout handling with asyncio.wait_for
- Always use async context managers for HTTP clients

### Error Handling
- Create domain-specific exceptions in src/eeg_rag/exceptions.py
- Never catch bare Exception, always specify exception types
- Log errors with full context before re-raising
- Implement retry logic with exponential backoff for external APIs

### Example Pattern
```python
from dataclasses import dataclass, field
from typing import Optional
import asyncio
import logging

logger = logging.getLogger(__name__)

@dataclass
class RetrievalResult:
    """Result from document retrieval."""
    document_id: str
    content: str
    score: float
    pmid: Optional[str] = None
    metadata: dict = field(default_factory=dict)

async def fetch_with_retry(
    url: str,
    max_retries: int = 3,
    timeout: float = 10.0
) -> dict:
    """Fetch URL with exponential backoff retry.
    
    Args:
        url: Target URL to fetch
        max_retries: Maximum retry attempts
        timeout: Request timeout in seconds
        
    Returns:
        Parsed JSON response
        
    Raises:
        RetrievalError: If all retries fail
    """
    for attempt in range(max_retries):
        try:
            async with asyncio.timeout(timeout):
                # implementation
                pass
        except asyncio.TimeoutError:
            wait_time = 2 ** attempt
            logger.warning(f"Timeout on attempt {attempt + 1}, retrying in {wait_time}s")
            await asyncio.sleep(wait_time)
    raise RetrievalError(f"Failed after {max_retries} attempts")
```