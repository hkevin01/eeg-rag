#!/usr/bin/env python3
"""
Redis Caching Layer for EEG-RAG

Provides caching for:
- Query results
- PubMed API responses
- Embedding vectors
- Search results
"""

import json
import pickle
import hashlib
from typing import Any, Optional, Dict, List, Union
from datetime import timedelta
import asyncio
import logging
from dataclasses import asdict

try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None

import numpy as np

logger = logging.getLogger(__name__)


class RedisCacheManager:
    """Redis-based caching manager for EEG-RAG system."""
    
    def __init__(
        self,
        host: str = 'localhost',
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        prefix: str = 'eeg_rag',
        default_ttl: int = 3600,  # 1 hour
        max_connections: int = 10
    ):
        """Initialize Redis cache manager.
        
        Args:
            host: Redis server host.
            port: Redis server port.
            db: Redis database number.
            password: Redis password if required.
            prefix: Key prefix for namespacing.
            default_ttl: Default time-to-live in seconds.
            max_connections: Maximum connection pool size.
        """
        if not REDIS_AVAILABLE:
            logger.warning("Redis not available. Install with: pip install redis")
            self.enabled = False
            return
        
        self.host = host
        self.port = port
        self.db = db
        self.password = password
        self.prefix = prefix
        self.default_ttl = default_ttl
        
        # Connection pool
        self.pool = redis.ConnectionPool(
            host=host,
            port=port,
            db=db,
            password=password,
            max_connections=max_connections,
            decode_responses=False  # Keep binary for pickle
        )
        
        self.client = redis.Redis(connection_pool=self.pool)
        self.enabled = True
        
        # Cache statistics
        self.hits = 0
        self.misses = 0
        self.errors = 0
    
    async def __aenter__(self):
        """Async context manager entry."""
        if self.enabled:
            await self._test_connection()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.enabled:
            await self.close()
    
    async def _test_connection(self):
        """Test Redis connection."""
        try:
            await self.client.ping()
            logger.info("Redis connection established")
        except Exception as e:
            logger.error(f"Redis connection failed: {str(e)}")
            self.enabled = False
    
    def _make_key(self, key: str, namespace: Optional[str] = None) -> str:
        """Create namespaced cache key."""
        if namespace:
            return f"{self.prefix}:{namespace}:{key}"
        return f"{self.prefix}:{key}"
    
    def _hash_key(self, data: Union[str, Dict, List]) -> str:
        """Create hash of complex data for use as cache key."""
        if isinstance(data, str):
            content = data
        else:
            content = json.dumps(data, sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()
    
    async def get(self, key: str, namespace: Optional[str] = None) -> Optional[Any]:
        """Get value from cache.
        
        Args:
            key: Cache key.
            namespace: Optional namespace.
            
        Returns:
            Cached value or None if not found.
        """
        if not self.enabled:
            return None
        
        try:
            cache_key = self._make_key(key, namespace)
            data = await self.client.get(cache_key)
            
            if data is not None:
                self.hits += 1
                return pickle.loads(data)
            else:
                self.misses += 1
                return None
                
        except Exception as e:
            self.errors += 1
            logger.error(f"Cache get error for key {key}: {str(e)}")
            return None
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        namespace: Optional[str] = None
    ) -> bool:
        """Set value in cache.
        
        Args:
            key: Cache key.
            value: Value to cache.
            ttl: Time-to-live in seconds.
            namespace: Optional namespace.
            
        Returns:
            True if successful, False otherwise.
        """
        if not self.enabled:
            return False
        
        try:
            cache_key = self._make_key(key, namespace)
            data = pickle.dumps(value)
            
            ttl = ttl or self.default_ttl
            await self.client.setex(cache_key, ttl, data)
            
            return True
            
        except Exception as e:
            self.errors += 1
            logger.error(f"Cache set error for key {key}: {str(e)}")
            return False
    
    async def delete(self, key: str, namespace: Optional[str] = None) -> bool:
        """Delete key from cache."""
        if not self.enabled:
            return False
        
        try:
            cache_key = self._make_key(key, namespace)
            result = await self.client.delete(cache_key)
            return result > 0
            
        except Exception as e:
            self.errors += 1
            logger.error(f"Cache delete error for key {key}: {str(e)}")
            return False
    
    async def exists(self, key: str, namespace: Optional[str] = None) -> bool:
        """Check if key exists in cache."""
        if not self.enabled:
            return False
        
        try:
            cache_key = self._make_key(key, namespace)
            result = await self.client.exists(cache_key)
            return result > 0
            
        except Exception as e:
            self.errors += 1
            logger.error(f"Cache exists error for key {key}: {str(e)}")
            return False
    
    async def clear_namespace(self, namespace: str) -> int:
        """Clear all keys in a namespace.
        
        Args:
            namespace: Namespace to clear.
            
        Returns:
            Number of keys deleted.
        """
        if not self.enabled:
            return 0
        
        try:
            pattern = self._make_key("*", namespace)
            keys = await self.client.keys(pattern)
            
            if keys:
                deleted = await self.client.delete(*keys)
                logger.info(f"Cleared {deleted} keys from namespace {namespace}")
                return deleted
            
            return 0
            
        except Exception as e:
            self.errors += 1
            logger.error(f"Cache clear namespace error for {namespace}: {str(e)}")
            return 0
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        stats = {
            'enabled': self.enabled,
            'hits': self.hits,
            'misses': self.misses,
            'errors': self.errors,
            'hit_rate': self.hits / (self.hits + self.misses) if (self.hits + self.misses) > 0 else 0
        }
        
        if self.enabled:
            try:
                info = await self.client.info()
                stats.update({
                    'connected_clients': info.get('connected_clients', 0),
                    'used_memory': info.get('used_memory_human', '0B'),
                    'keyspace_hits': info.get('keyspace_hits', 0),
                    'keyspace_misses': info.get('keyspace_misses', 0)
                })
            except Exception as e:
                logger.error(f"Error getting Redis stats: {str(e)}")
        
        return stats
    
    async def close(self):
        """Close Redis connections."""
        if self.enabled and self.client:
            try:
                await self.client.close()
                logger.info("Redis connections closed")
            except Exception as e:
                logger.error(f"Error closing Redis connections: {str(e)}")


class CachedQueryManager:
    """Manages caching for different query types."""
    
    def __init__(self, cache_manager: RedisCacheManager):
        self.cache = cache_manager
        
        # Different TTLs for different data types
        self.ttls = {
            'query_results': 3600,      # 1 hour
            'pubmed_results': 86400,    # 24 hours
            'embeddings': None,         # Never expire
            'search_results': 1800,     # 30 minutes
            'graph_results': 7200,      # 2 hours
        }
    
    async def get_query_result(self, query: str, agent_type: str) -> Optional[Dict]:
        """Get cached query result."""
        key = self._make_query_key(query, agent_type)
        return await self.cache.get(key, namespace='query_results')
    
    async def cache_query_result(
        self, 
        query: str, 
        agent_type: str, 
        result: Dict
    ) -> bool:
        """Cache query result."""
        key = self._make_query_key(query, agent_type)
        return await self.cache.set(
            key, result, 
            ttl=self.ttls['query_results'], 
            namespace='query_results'
        )
    
    async def get_pubmed_result(self, pmid: str) -> Optional[Dict]:
        """Get cached PubMed result."""
        return await self.cache.get(pmid, namespace='pubmed_results')
    
    async def cache_pubmed_result(self, pmid: str, result: Dict) -> bool:
        """Cache PubMed result."""
        return await self.cache.set(
            pmid, result,
            ttl=self.ttls['pubmed_results'],
            namespace='pubmed_results'
        )
    
    async def get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get cached embedding."""
        key = self.cache._hash_key(text)
        data = await self.cache.get(key, namespace='embeddings')
        return np.array(data) if data is not None else None
    
    async def cache_embedding(self, text: str, embedding: np.ndarray) -> bool:
        """Cache embedding (no expiration)."""
        key = self.cache._hash_key(text)
        return await self.cache.set(
            key, embedding.tolist(),
            ttl=self.ttls['embeddings'],
            namespace='embeddings'
        )
    
    async def get_search_results(
        self, 
        query: str, 
        search_type: str,
        top_k: int = 10
    ) -> Optional[List]:
        """Get cached search results."""
        key = self._make_search_key(query, search_type, top_k)
        return await self.cache.get(key, namespace='search_results')
    
    async def cache_search_results(
        self,
        query: str,
        search_type: str,
        results: List,
        top_k: int = 10
    ) -> bool:
        """Cache search results."""
        key = self._make_search_key(query, search_type, top_k)
        return await self.cache.set(
            key, results,
            ttl=self.ttls['search_results'],
            namespace='search_results'
        )
    
    def _make_query_key(self, query: str, agent_type: str) -> str:
        """Create cache key for query result."""
        return self.cache._hash_key(f"{query}:{agent_type}")
    
    def _make_search_key(self, query: str, search_type: str, top_k: int) -> str:
        """Create cache key for search result."""
        return self.cache._hash_key(f"{query}:{search_type}:{top_k}")
    
    async def clear_expired_cache(self):
        """Clear expired cache entries (Redis handles this automatically)."""
        # Redis automatically handles TTL expiration
        # This method is for manual cleanup if needed
        logger.info("Cache expiration handled automatically by Redis TTL")
    
    async def warm_cache(self, common_queries: List[str]):
        """Pre-warm cache with common queries."""
        logger.info(f"Warming cache with {len(common_queries)} common queries")
        
        for query in common_queries:
            # This would typically involve running the queries
            # and caching the results, but depends on the system being ready
            logger.debug(f"Would warm cache for query: {query}")


# Global cache instance (initialized as needed)
_cache_manager: Optional[RedisCacheManager] = None
_query_cache: Optional[CachedQueryManager] = None


async def get_cache_manager(**kwargs) -> RedisCacheManager:
    """Get global cache manager instance."""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = RedisCacheManager(**kwargs)
        async with _cache_manager:
            pass  # Test connection
    return _cache_manager


async def get_query_cache(**kwargs) -> CachedQueryManager:
    """Get global query cache instance."""
    global _query_cache
    if _query_cache is None:
        cache_manager = await get_cache_manager(**kwargs)
        _query_cache = CachedQueryManager(cache_manager)
    return _query_cache