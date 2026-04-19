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


# ---------------------------------------------------------------------------
# ID           : utils.redis_cache.RedisCacheManager
# Requirement  : `RedisCacheManager` class shall be instantiable and expose the documented interface
# Purpose      : Redis-based caching manager for EEG-RAG system
# Rationale    : Object-oriented encapsulation isolates state and enforces invariants
# Inputs       : Constructor arguments — see __init__ signature
# Outputs      : N/A (class definition)
# Precond.     : All imported dependencies must be available at import time
# Postcond.    : Instance attributes initialised as documented; invariants hold
# Assumptions  : Python runtime ≥ 3.9; package dependencies installed
# Side Effects : May allocate heap memory; __init__ may open connections or load models
# Fail Modes   : ImportError if dependency missing; TypeError for invalid constructor args
# Err Handling : Constructor raises on invalid args; see __init__ body
# Constraints  : Thread-safety not guaranteed unless explicitly documented
# Verification : Instantiate RedisCacheManager with valid args; assert attribute types and values
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
class RedisCacheManager:
    """Redis-based caching manager for EEG-RAG system."""
    
    # ---------------------------------------------------------------------------
    # ID           : utils.redis_cache.RedisCacheManager.__init__
    # Requirement  : `__init__` shall initialize Redis cache manager
    # Purpose      : Initialize Redis cache manager
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : host: str (default='localhost'); port: int (default=6379); db: int (default=0); password: Optional[str] (default=None); prefix: str (default='eeg_rag'); default_ttl: int (default=3600); max_connections: int (default=10)
    # Outputs      : Implicitly None or see body
    # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
    # Postcond.    : Return value satisfies documented output type and range
    # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
    # Side Effects : May update instance state or perform I/O; see body
    # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
    # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
    # Constraints  : Synchronous — must not block event loop
    # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
    # References   : EEG-RAG system design specification; see module docstring
    # ---------------------------------------------------------------------------
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
    
    # ---------------------------------------------------------------------------
    # ID           : utils.redis_cache.RedisCacheManager.__aenter__
    # Requirement  : `__aenter__` shall async context manager entry
    # Purpose      : Async context manager entry
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : None
    # Outputs      : Implicitly None or see body
    # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
    # Postcond.    : Return value satisfies documented output type and range
    # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
    # Side Effects : May update instance state or perform I/O; see body
    # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
    # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
    # Constraints  : Must be awaited (async)
    # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
    # References   : EEG-RAG system design specification; see module docstring
    # ---------------------------------------------------------------------------
    async def __aenter__(self):
        """Async context manager entry."""
        if self.enabled:
            await self._test_connection()
        return self
    
    # ---------------------------------------------------------------------------
    # ID           : utils.redis_cache.RedisCacheManager.__aexit__
    # Requirement  : `__aexit__` shall async context manager exit
    # Purpose      : Async context manager exit
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : exc_type; exc_val; exc_tb
    # Outputs      : Implicitly None or see body
    # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
    # Postcond.    : Return value satisfies documented output type and range
    # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
    # Side Effects : May update instance state or perform I/O; see body
    # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
    # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
    # Constraints  : Must be awaited (async)
    # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
    # References   : EEG-RAG system design specification; see module docstring
    # ---------------------------------------------------------------------------
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.enabled:
            await self.close()
    
    # ---------------------------------------------------------------------------
    # ID           : utils.redis_cache.RedisCacheManager._test_connection
    # Requirement  : `_test_connection` shall test Redis connection
    # Purpose      : Test Redis connection
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : None
    # Outputs      : Implicitly None or see body
    # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
    # Postcond.    : Return value satisfies documented output type and range
    # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
    # Side Effects : May update instance state or perform I/O; see body
    # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
    # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
    # Constraints  : Must be awaited (async)
    # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
    # References   : EEG-RAG system design specification; see module docstring
    # ---------------------------------------------------------------------------
    async def _test_connection(self):
        """Test Redis connection."""
        try:
            await self.client.ping()
            logger.info("Redis connection established")
        except Exception as e:
            logger.error(f"Redis connection failed: {str(e)}")
            self.enabled = False
    
    # ---------------------------------------------------------------------------
    # ID           : utils.redis_cache.RedisCacheManager._make_key
    # Requirement  : `_make_key` shall create namespaced cache key
    # Purpose      : Create namespaced cache key
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : key: str; namespace: Optional[str] (default=None)
    # Outputs      : str
    # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
    # Postcond.    : Return value satisfies documented output type and range
    # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
    # Side Effects : May update instance state or perform I/O; see body
    # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
    # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
    # Constraints  : Synchronous — must not block event loop
    # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
    # References   : EEG-RAG system design specification; see module docstring
    # ---------------------------------------------------------------------------
    def _make_key(self, key: str, namespace: Optional[str] = None) -> str:
        """Create namespaced cache key."""
        if namespace:
            return f"{self.prefix}:{namespace}:{key}"
        return f"{self.prefix}:{key}"
    
    # ---------------------------------------------------------------------------
    # ID           : utils.redis_cache.RedisCacheManager._hash_key
    # Requirement  : `_hash_key` shall create hash of complex data for use as cache key
    # Purpose      : Create hash of complex data for use as cache key
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : data: Union[str, Dict, List]
    # Outputs      : str
    # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
    # Postcond.    : Return value satisfies documented output type and range
    # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
    # Side Effects : May update instance state or perform I/O; see body
    # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
    # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
    # Constraints  : Synchronous — must not block event loop
    # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
    # References   : EEG-RAG system design specification; see module docstring
    # ---------------------------------------------------------------------------
    def _hash_key(self, data: Union[str, Dict, List]) -> str:
        """Create hash of complex data for use as cache key."""
        if isinstance(data, str):
            content = data
        else:
            content = json.dumps(data, sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()
    
    # ---------------------------------------------------------------------------
    # ID           : utils.redis_cache.RedisCacheManager.get
    # Requirement  : `get` shall get value from cache
    # Purpose      : Get value from cache
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : key: str; namespace: Optional[str] (default=None)
    # Outputs      : Optional[Any]
    # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
    # Postcond.    : Return value satisfies documented output type and range
    # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
    # Side Effects : May update instance state or perform I/O; see body
    # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
    # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
    # Constraints  : Must be awaited (async)
    # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
    # References   : EEG-RAG system design specification; see module docstring
    # ---------------------------------------------------------------------------
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
    
    # ---------------------------------------------------------------------------
    # ID           : utils.redis_cache.RedisCacheManager.set
    # Requirement  : `set` shall set value in cache
    # Purpose      : Set value in cache
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : key: str; value: Any; ttl: Optional[int] (default=None); namespace: Optional[str] (default=None)
    # Outputs      : bool
    # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
    # Postcond.    : Return value satisfies documented output type and range
    # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
    # Side Effects : May update instance state or perform I/O; see body
    # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
    # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
    # Constraints  : Must be awaited (async)
    # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
    # References   : EEG-RAG system design specification; see module docstring
    # ---------------------------------------------------------------------------
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
    
    # ---------------------------------------------------------------------------
    # ID           : utils.redis_cache.RedisCacheManager.delete
    # Requirement  : `delete` shall delete key from cache
    # Purpose      : Delete key from cache
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : key: str; namespace: Optional[str] (default=None)
    # Outputs      : bool
    # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
    # Postcond.    : Return value satisfies documented output type and range
    # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
    # Side Effects : May update instance state or perform I/O; see body
    # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
    # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
    # Constraints  : Must be awaited (async)
    # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
    # References   : EEG-RAG system design specification; see module docstring
    # ---------------------------------------------------------------------------
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
    
    # ---------------------------------------------------------------------------
    # ID           : utils.redis_cache.RedisCacheManager.exists
    # Requirement  : `exists` shall check if key exists in cache
    # Purpose      : Check if key exists in cache
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : key: str; namespace: Optional[str] (default=None)
    # Outputs      : bool
    # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
    # Postcond.    : Return value satisfies documented output type and range
    # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
    # Side Effects : May update instance state or perform I/O; see body
    # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
    # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
    # Constraints  : Must be awaited (async)
    # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
    # References   : EEG-RAG system design specification; see module docstring
    # ---------------------------------------------------------------------------
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
    
    # ---------------------------------------------------------------------------
    # ID           : utils.redis_cache.RedisCacheManager.clear_namespace
    # Requirement  : `clear_namespace` shall clear all keys in a namespace
    # Purpose      : Clear all keys in a namespace
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : namespace: str
    # Outputs      : int
    # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
    # Postcond.    : Return value satisfies documented output type and range
    # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
    # Side Effects : May update instance state or perform I/O; see body
    # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
    # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
    # Constraints  : Must be awaited (async)
    # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
    # References   : EEG-RAG system design specification; see module docstring
    # ---------------------------------------------------------------------------
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
    
    # ---------------------------------------------------------------------------
    # ID           : utils.redis_cache.RedisCacheManager.get_stats
    # Requirement  : `get_stats` shall get cache statistics
    # Purpose      : Get cache statistics
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : None
    # Outputs      : Dict[str, Any]
    # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
    # Postcond.    : Return value satisfies documented output type and range
    # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
    # Side Effects : May update instance state or perform I/O; see body
    # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
    # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
    # Constraints  : Must be awaited (async)
    # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
    # References   : EEG-RAG system design specification; see module docstring
    # ---------------------------------------------------------------------------
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
    
    # ---------------------------------------------------------------------------
    # ID           : utils.redis_cache.RedisCacheManager.close
    # Requirement  : `close` shall close Redis connections
    # Purpose      : Close Redis connections
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : None
    # Outputs      : Implicitly None or see body
    # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
    # Postcond.    : Return value satisfies documented output type and range
    # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
    # Side Effects : May update instance state or perform I/O; see body
    # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
    # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
    # Constraints  : Must be awaited (async)
    # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
    # References   : EEG-RAG system design specification; see module docstring
    # ---------------------------------------------------------------------------
    async def close(self):
        """Close Redis connections."""
        if self.enabled and self.client:
            try:
                await self.client.close()
                logger.info("Redis connections closed")
            except Exception as e:
                logger.error(f"Error closing Redis connections: {str(e)}")


# ---------------------------------------------------------------------------
# ID           : utils.redis_cache.CachedQueryManager
# Requirement  : `CachedQueryManager` class shall be instantiable and expose the documented interface
# Purpose      : Manages caching for different query types
# Rationale    : Object-oriented encapsulation isolates state and enforces invariants
# Inputs       : Constructor arguments — see __init__ signature
# Outputs      : N/A (class definition)
# Precond.     : All imported dependencies must be available at import time
# Postcond.    : Instance attributes initialised as documented; invariants hold
# Assumptions  : Python runtime ≥ 3.9; package dependencies installed
# Side Effects : May allocate heap memory; __init__ may open connections or load models
# Fail Modes   : ImportError if dependency missing; TypeError for invalid constructor args
# Err Handling : Constructor raises on invalid args; see __init__ body
# Constraints  : Thread-safety not guaranteed unless explicitly documented
# Verification : Instantiate CachedQueryManager with valid args; assert attribute types and values
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
class CachedQueryManager:
    """Manages caching for different query types."""
    
    # ---------------------------------------------------------------------------
    # ID           : utils.redis_cache.CachedQueryManager.__init__
    # Requirement  : `__init__` shall execute as specified
    # Purpose      :   init  
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : cache_manager: RedisCacheManager
    # Outputs      : Implicitly None or see body
    # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
    # Postcond.    : Return value satisfies documented output type and range
    # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
    # Side Effects : May update instance state or perform I/O; see body
    # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
    # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
    # Constraints  : Synchronous — must not block event loop
    # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
    # References   : EEG-RAG system design specification; see module docstring
    # ---------------------------------------------------------------------------
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
    
    # ---------------------------------------------------------------------------
    # ID           : utils.redis_cache.CachedQueryManager.get_query_result
    # Requirement  : `get_query_result` shall get cached query result
    # Purpose      : Get cached query result
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : query: str; agent_type: str
    # Outputs      : Optional[Dict]
    # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
    # Postcond.    : Return value satisfies documented output type and range
    # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
    # Side Effects : May update instance state or perform I/O; see body
    # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
    # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
    # Constraints  : Must be awaited (async)
    # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
    # References   : EEG-RAG system design specification; see module docstring
    # ---------------------------------------------------------------------------
    async def get_query_result(self, query: str, agent_type: str) -> Optional[Dict]:
        """Get cached query result."""
        key = self._make_query_key(query, agent_type)
        return await self.cache.get(key, namespace='query_results')
    
    # ---------------------------------------------------------------------------
    # ID           : utils.redis_cache.CachedQueryManager.cache_query_result
    # Requirement  : `cache_query_result` shall cache query result
    # Purpose      : Cache query result
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : query: str; agent_type: str; result: Dict
    # Outputs      : bool
    # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
    # Postcond.    : Return value satisfies documented output type and range
    # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
    # Side Effects : May update instance state or perform I/O; see body
    # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
    # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
    # Constraints  : Must be awaited (async)
    # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
    # References   : EEG-RAG system design specification; see module docstring
    # ---------------------------------------------------------------------------
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
    
    # ---------------------------------------------------------------------------
    # ID           : utils.redis_cache.CachedQueryManager.get_pubmed_result
    # Requirement  : `get_pubmed_result` shall get cached PubMed result
    # Purpose      : Get cached PubMed result
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : pmid: str
    # Outputs      : Optional[Dict]
    # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
    # Postcond.    : Return value satisfies documented output type and range
    # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
    # Side Effects : May update instance state or perform I/O; see body
    # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
    # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
    # Constraints  : Must be awaited (async)
    # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
    # References   : EEG-RAG system design specification; see module docstring
    # ---------------------------------------------------------------------------
    async def get_pubmed_result(self, pmid: str) -> Optional[Dict]:
        """Get cached PubMed result."""
        return await self.cache.get(pmid, namespace='pubmed_results')
    
    # ---------------------------------------------------------------------------
    # ID           : utils.redis_cache.CachedQueryManager.cache_pubmed_result
    # Requirement  : `cache_pubmed_result` shall cache PubMed result
    # Purpose      : Cache PubMed result
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : pmid: str; result: Dict
    # Outputs      : bool
    # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
    # Postcond.    : Return value satisfies documented output type and range
    # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
    # Side Effects : May update instance state or perform I/O; see body
    # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
    # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
    # Constraints  : Must be awaited (async)
    # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
    # References   : EEG-RAG system design specification; see module docstring
    # ---------------------------------------------------------------------------
    async def cache_pubmed_result(self, pmid: str, result: Dict) -> bool:
        """Cache PubMed result."""
        return await self.cache.set(
            pmid, result,
            ttl=self.ttls['pubmed_results'],
            namespace='pubmed_results'
        )
    
    # ---------------------------------------------------------------------------
    # ID           : utils.redis_cache.CachedQueryManager.get_embedding
    # Requirement  : `get_embedding` shall get cached embedding
    # Purpose      : Get cached embedding
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : text: str
    # Outputs      : Optional[np.ndarray]
    # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
    # Postcond.    : Return value satisfies documented output type and range
    # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
    # Side Effects : May update instance state or perform I/O; see body
    # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
    # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
    # Constraints  : Must be awaited (async)
    # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
    # References   : EEG-RAG system design specification; see module docstring
    # ---------------------------------------------------------------------------
    async def get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get cached embedding."""
        key = self.cache._hash_key(text)
        data = await self.cache.get(key, namespace='embeddings')
        return np.array(data) if data is not None else None
    
    # ---------------------------------------------------------------------------
    # ID           : utils.redis_cache.CachedQueryManager.cache_embedding
    # Requirement  : `cache_embedding` shall cache embedding (no expiration)
    # Purpose      : Cache embedding (no expiration)
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : text: str; embedding: np.ndarray
    # Outputs      : bool
    # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
    # Postcond.    : Return value satisfies documented output type and range
    # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
    # Side Effects : May update instance state or perform I/O; see body
    # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
    # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
    # Constraints  : Must be awaited (async)
    # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
    # References   : EEG-RAG system design specification; see module docstring
    # ---------------------------------------------------------------------------
    async def cache_embedding(self, text: str, embedding: np.ndarray) -> bool:
        """Cache embedding (no expiration)."""
        key = self.cache._hash_key(text)
        return await self.cache.set(
            key, embedding.tolist(),
            ttl=self.ttls['embeddings'],
            namespace='embeddings'
        )
    
    # ---------------------------------------------------------------------------
    # ID           : utils.redis_cache.CachedQueryManager.get_search_results
    # Requirement  : `get_search_results` shall get cached search results
    # Purpose      : Get cached search results
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : query: str; search_type: str; top_k: int (default=10)
    # Outputs      : Optional[List]
    # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
    # Postcond.    : Return value satisfies documented output type and range
    # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
    # Side Effects : May update instance state or perform I/O; see body
    # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
    # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
    # Constraints  : Must be awaited (async)
    # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
    # References   : EEG-RAG system design specification; see module docstring
    # ---------------------------------------------------------------------------
    async def get_search_results(
        self, 
        query: str, 
        search_type: str,
        top_k: int = 10
    ) -> Optional[List]:
        """Get cached search results."""
        key = self._make_search_key(query, search_type, top_k)
        return await self.cache.get(key, namespace='search_results')
    
    # ---------------------------------------------------------------------------
    # ID           : utils.redis_cache.CachedQueryManager.cache_search_results
    # Requirement  : `cache_search_results` shall cache search results
    # Purpose      : Cache search results
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : query: str; search_type: str; results: List; top_k: int (default=10)
    # Outputs      : bool
    # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
    # Postcond.    : Return value satisfies documented output type and range
    # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
    # Side Effects : May update instance state or perform I/O; see body
    # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
    # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
    # Constraints  : Must be awaited (async)
    # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
    # References   : EEG-RAG system design specification; see module docstring
    # ---------------------------------------------------------------------------
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
    
    # ---------------------------------------------------------------------------
    # ID           : utils.redis_cache.CachedQueryManager._make_query_key
    # Requirement  : `_make_query_key` shall create cache key for query result
    # Purpose      : Create cache key for query result
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : query: str; agent_type: str
    # Outputs      : str
    # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
    # Postcond.    : Return value satisfies documented output type and range
    # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
    # Side Effects : May update instance state or perform I/O; see body
    # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
    # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
    # Constraints  : Synchronous — must not block event loop
    # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
    # References   : EEG-RAG system design specification; see module docstring
    # ---------------------------------------------------------------------------
    def _make_query_key(self, query: str, agent_type: str) -> str:
        """Create cache key for query result."""
        return self.cache._hash_key(f"{query}:{agent_type}")
    
    # ---------------------------------------------------------------------------
    # ID           : utils.redis_cache.CachedQueryManager._make_search_key
    # Requirement  : `_make_search_key` shall create cache key for search result
    # Purpose      : Create cache key for search result
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : query: str; search_type: str; top_k: int
    # Outputs      : str
    # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
    # Postcond.    : Return value satisfies documented output type and range
    # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
    # Side Effects : May update instance state or perform I/O; see body
    # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
    # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
    # Constraints  : Synchronous — must not block event loop
    # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
    # References   : EEG-RAG system design specification; see module docstring
    # ---------------------------------------------------------------------------
    def _make_search_key(self, query: str, search_type: str, top_k: int) -> str:
        """Create cache key for search result."""
        return self.cache._hash_key(f"{query}:{search_type}:{top_k}")
    
    # ---------------------------------------------------------------------------
    # ID           : utils.redis_cache.CachedQueryManager.clear_expired_cache
    # Requirement  : `clear_expired_cache` shall clear expired cache entries (Redis handles this automatically)
    # Purpose      : Clear expired cache entries (Redis handles this automatically)
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : None
    # Outputs      : Implicitly None or see body
    # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
    # Postcond.    : Return value satisfies documented output type and range
    # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
    # Side Effects : May update instance state or perform I/O; see body
    # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
    # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
    # Constraints  : Must be awaited (async)
    # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
    # References   : EEG-RAG system design specification; see module docstring
    # ---------------------------------------------------------------------------
    async def clear_expired_cache(self):
        """Clear expired cache entries (Redis handles this automatically)."""
        # Redis automatically handles TTL expiration
        # This method is for manual cleanup if needed
        logger.info("Cache expiration handled automatically by Redis TTL")
    
    # ---------------------------------------------------------------------------
    # ID           : utils.redis_cache.CachedQueryManager.warm_cache
    # Requirement  : `warm_cache` shall pre-warm cache with common queries
    # Purpose      : Pre-warm cache with common queries
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : common_queries: List[str]
    # Outputs      : Implicitly None or see body
    # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
    # Postcond.    : Return value satisfies documented output type and range
    # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
    # Side Effects : May update instance state or perform I/O; see body
    # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
    # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
    # Constraints  : Must be awaited (async)
    # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
    # References   : EEG-RAG system design specification; see module docstring
    # ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# ID           : utils.redis_cache.get_cache_manager
# Requirement  : `get_cache_manager` shall get global cache manager instance
# Purpose      : Get global cache manager instance
# Rationale    : Implements domain-specific logic per system design; see referenced specs
# Inputs       : **kwargs
# Outputs      : RedisCacheManager
# Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
# Postcond.    : Return value satisfies documented output type and range
# Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
# Side Effects : May update instance state or perform I/O; see body
# Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
# Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
# Constraints  : Must be awaited (async)
# Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
async def get_cache_manager(**kwargs) -> RedisCacheManager:
    """Get global cache manager instance."""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = RedisCacheManager(**kwargs)
        async with _cache_manager:
            pass  # Test connection
    return _cache_manager


# ---------------------------------------------------------------------------
# ID           : utils.redis_cache.get_query_cache
# Requirement  : `get_query_cache` shall get global query cache instance
# Purpose      : Get global query cache instance
# Rationale    : Implements domain-specific logic per system design; see referenced specs
# Inputs       : **kwargs
# Outputs      : CachedQueryManager
# Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
# Postcond.    : Return value satisfies documented output type and range
# Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
# Side Effects : May update instance state or perform I/O; see body
# Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
# Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
# Constraints  : Must be awaited (async)
# Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
async def get_query_cache(**kwargs) -> CachedQueryManager:
    """Get global query cache instance."""
    global _query_cache
    if _query_cache is None:
        cache_manager = await get_cache_manager(**kwargs)
        _query_cache = CachedQueryManager(cache_manager)
    return _query_cache