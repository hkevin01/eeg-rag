# Performance Optimization Guidelines

## Latency Targets
- Local retrieval: < 100ms for 10K documents
- End-to-end query: < 2 seconds (p95)
- Cache hit response: < 50ms

## Caching Strategy
- Cache embeddings indefinitely (they don't change)
- Cache PubMed API responses for 24 hours
- Cache generated answers for 1 hour
- Use LRU eviction with 10K entry limit

## Batch Processing
- Embed documents in batches of 32
- Fetch PubMed records in batches of 200 (API limit)
- Process Neo4j writes in transactions of 1000 nodes

## Memory Management
- Stream large files instead of loading entirely
- Use generators for document iteration
- Clear FAISS index from memory when not in use
- Implement connection pooling for databases

## Example Optimization Pattern
```python
from functools import lru_cache
from typing import Iterator
import asyncio

class OptimizedRetriever:
    def __init__(self, cache_size: int = 10000):
        self._embedding_cache = {}
        self._result_cache = TTLCache(maxsize=cache_size, ttl=3600)
        
    @lru_cache(maxsize=50000)
    def get_embedding(self, text: str) -> np.ndarray:
        """Cache embeddings - they never change."""
        return self.model.encode(text)
    
    async def batch_embed(
        self, 
        texts: list[str], 
        batch_size: int = 32
    ) -> list[np.ndarray]:
        """Embed texts in batches for efficiency."""
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = await asyncio.to_thread(
                self.model.encode, batch
            )
            embeddings.extend(batch_embeddings)
        return embeddings
    
    def stream_documents(self, path: str) -> Iterator[dict]:
        """Stream large document files without loading all into memory."""
        with open(path, 'r') as f:
            for line in f:
                yield json.loads(line)
```

## Profiling Requirements
- Add timing decorators to all agent execute methods
- Log slow queries (> 3 seconds) with full context
- Track embedding generation time separately from search time
- Monitor cache hit rates