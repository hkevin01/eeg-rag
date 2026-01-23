# EEG-RAG Troubleshooting Guide

> **Comprehensive guide to diagnosing and resolving common issues with the EEG-RAG system**

This guide provides step-by-step troubleshooting procedures for common issues encountered when using the EEG-RAG system.

## Quick Reference

| Issue Type               | Quick Fix                 | See Section                                    |
| ------------------------ | ------------------------- | ---------------------------------------------- |
| 🚨 System health warnings | Check resource usage      | [System Health](#system-health-issues)         |
| 🔄 Circuit breaker open   | Reset or wait for timeout | [Circuit Breakers](#circuit-breaker-issues)    |
| 🐌 Slow performance       | Check agent statistics    | [Performance Issues](#performance-issues)      |
| ❌ Validation errors      | Fix input format          | [Input Validation](#input-validation-errors)   |
| 🔌 Connection failures    | Check network/services    | [Connectivity Issues](#connectivity-issues)    |
| 💾 Memory issues          | Clear caches, restart     | [Memory Management](#memory-management-issues) |

## System Health Issues

### Warning: High CPU Usage

**Symptoms:**
- System health reports `WARNING` or `CRITICAL` status
- CPU usage >80%
- Slow response times

**Diagnosis:**
```python
from eeg_rag.utils.common_utils import check_system_health

health = check_system_health()
print(f"CPU Usage: {health.cpu_percent:.1f}%")
print(f"Status: {health.status.value}")
for warning in health.warnings:
    print(f"⚠️  {warning}")
```

**Solutions:**

1. **Reduce concurrent operations:**
```python
# Limit number of parallel agents
orchestrator.config["max_concurrent_agents"] = 2

# Reduce batch sizes
agent.config["batch_size"] = 10
```

2. **Optimize query patterns:**
```python
# Use more specific queries to reduce processing
query = AgentQuery(
    text="specific EEG biomarker for epilepsy",  # Good
    # text="tell me everything about EEG"       # Bad - too broad
)
```

3. **Check for CPU-intensive operations:**
```bash
# Monitor CPU usage by process
top -p $(pgrep -d, python)

# Check for blocking operations in logs
grep "blocking" logs/eeg-rag.log
```

### Warning: High Memory Usage

**Symptoms:**
- Memory usage >85%
- `MemoryError` exceptions
- System becoming unresponsive

**Diagnosis:**
```python
import psutil
import gc

# Check memory details
memory = psutil.virtual_memory()
print(f"Memory usage: {memory.percent:.1f}%")
print(f"Available: {memory.available / (1024**3):.2f} GB")

# Check for memory leaks
gc.collect()  # Force garbage collection
print(f"Objects in memory: {len(gc.get_objects())}")
```

**Solutions:**

1. **Clear caches and reset agents:**
```python
# Clear agent caches
for agent in orchestrator.agents:
    if hasattr(agent, 'clear_cache'):
        agent.clear_cache()

# Reset memory manager
memory_manager.clear_short_term_memory()

# Force garbage collection
import gc
gc.collect()
```

2. **Reduce memory footprint:**
```python
# Reduce memory limits
memory_manager = MemoryManager(
    short_term_max_entries=100,  # Reduced from default
    long_term_max_size_mb=50     # Reduced from default
)

# Use smaller embedding dimensions
faiss_config = {
    "dimension": 384,  # Instead of 768
    "index_type": "IndexIVFFlat"  # More memory efficient
}
```

3. **Monitor memory usage over time:**
```python
import tracemalloc
import asyncio

tracemalloc.start()

async def memory_monitor():
    while True:
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')[:3]
        
        print("Top memory consumers:")
        for stat in top_stats:
            print(f"  {stat}")
        
        await asyncio.sleep(60)  # Check every minute

asyncio.create_task(memory_monitor())
```

### Critical: Disk Space Low

**Symptoms:**
- Disk usage >95%
- Cannot save files
- Database write errors

**Immediate Actions:**

1. **Clean temporary files:**
```bash
# Clean Python cache
find . -type d -name __pycache__ -exec rm -rf {} +
find . -name "*.pyc" -delete

# Clean logs (keep last 7 days)
find logs/ -name "*.log" -mtime +7 -delete

# Clean temporary embeddings
rm -rf data/embeddings/cache/*
```

2. **Compress or archive old data:**
```bash
# Archive old processed data
tar -czf data/archive/processed_$(date +%Y%m%d).tar.gz data/processed/
rm -rf data/processed/*

# Clean old vector indices
find data/embeddings/ -name "*.faiss" -mtime +30 -delete
```

3. **Check disk usage:**
```python
import shutil

# Check available space
total, used, free = shutil.disk_usage("/")
print(f"Disk usage: {used / total:.1%}")
print(f"Free space: {free / (1024**3):.2f} GB")

# Check directory sizes
import os
def get_dir_size(path):
    total = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total += os.path.getsize(fp)
    return total

print(f"Data directory: {get_dir_size('data') / (1024**3):.2f} GB")
print(f"Logs directory: {get_dir_size('logs') / (1024**3):.2f} GB")
```

## Circuit Breaker Issues

### Circuit Breaker Open Error

**Symptoms:**
- `CircuitBreakerOpenError` exceptions
- External service calls being blocked
- Logs show "Circuit breaker is now OPEN"

**Diagnosis:**
```python
from eeg_rag.utils.common_utils import CircuitBreaker

# Check circuit breaker status
cb = get_circuit_breaker("pubmed_api")  # Example service
print(f"State: {cb.state}")
print(f"Failure count: {cb.failure_count}")
print(f"Last failure: {cb.last_failure_time}")
```

**Solutions:**

1. **Manual reset (immediate but risky):**
```python
# Reset circuit breaker manually
cb.reset()
print("Circuit breaker reset")
```

2. **Wait for automatic recovery (safer):**
```python
import asyncio
from datetime import datetime, timedelta

# Calculate time until automatic reset
if cb.last_failure_time:
    reset_time = cb.last_failure_time + timedelta(seconds=cb.timeout_seconds)
    wait_seconds = (reset_time - datetime.now()).total_seconds()
    print(f"Circuit breaker will auto-reset in {wait_seconds:.0f} seconds")
    
    # Wait for auto-reset
    if wait_seconds > 0:
        await asyncio.sleep(wait_seconds)
```

3. **Use fallback mechanisms:**
```python
async def robust_api_call():
    try:
        return await cb.call(primary_service)
    except CircuitBreakerOpenError:
        logger.warning("Primary service unavailable, using fallback")
        return await fallback_service()
    except Exception as e:
        logger.error(f"All services failed: {e}")
        return default_response()
```

4. **Investigate root cause:**
```bash
# Check service logs for failure patterns
grep "Circuit breaker" logs/eeg-rag.log | tail -20

# Check external service status
curl -I https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi

# Monitor network connectivity
ping -c 5 eutils.ncbi.nlm.nih.gov
```

### Frequent Circuit Breaker Trips

**Symptoms:**
- Circuit breaker opening frequently
- Many timeout or connection errors
- Inconsistent external service responses

**Analysis:**
```python
# Analyze failure patterns
failures = []
with open("logs/eeg-rag.log", "r") as f:
    for line in f:
        if "circuit breaker" in line.lower() and "failure" in line.lower():
            failures.append(line.strip())

print(f"Recent failures ({len(failures)} total):")
for failure in failures[-10:]:
    print(f"  {failure}")
```

**Solutions:**

1. **Adjust circuit breaker thresholds:**
```python
# More lenient settings for unreliable services
cb = CircuitBreaker(
    name="unreliable_service",
    failure_threshold=10,      # Allow more failures
    timeout_seconds=120.0      # Longer timeout before retry
)
```

2. **Implement better retry logic:**
```python
@retry_with_backoff(
    max_retries=5,           # More retry attempts
    initial_delay=2.0,       # Longer initial delay
    backoff_factor=1.5,      # Gentler backoff
    max_delay=30.0           # Reasonable max delay
)
async def stable_external_call():
    return await external_api()
```

## Performance Issues

### Slow Agent Execution

**Symptoms:**
- Agent execution time >2 seconds
- Low throughput (<10 requests/second)
- Users reporting slow responses

**Diagnosis:**
```python
# Check agent performance statistics
agent = get_agent("local_data")
stats = agent.get_statistics()

print(f"Average execution time: {stats['average_execution_time_seconds']:.3f}s")
print(f"Success rate: {stats['success_rate']:.1%}")
print(f"Total executions: {stats['total_executions']}")

# Identify slowest operations
if stats['max_execution_time_seconds'] > 5.0:
    print(f"⚠️  Slowest execution: {stats['max_execution_time_seconds']:.3f}s")
```

**Solutions:**

1. **Enable caching:**
```python
# Enable result caching
agent.config.update({
    "cache_enabled": True,
    "cache_ttl": 300,  # 5 minutes
    "cache_size": 1000  # Number of entries
})
```

2. **Optimize batch processing:**
```python
# Increase batch size for bulk operations
agent.config["batch_size"] = 100

# Use parallel processing
agent.config["max_workers"] = 4
```

3. **Profile slow operations:**
```python
import cProfile
import pstats

# Profile agent execution
profiler = cProfile.Profile()
profiler.enable()

# Run slow operation
result = await agent.run(query)

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('tottime').print_stats(10)
```

### High Memory Usage During Processing

**Symptoms:**
- Memory usage spikes during query processing
- Out of memory errors
- Performance degrades over time

**Solutions:**

1. **Implement streaming processing:**
```python
async def stream_large_dataset(data_iterator):
    for batch in chunked(data_iterator, batch_size=100):
        # Process smaller batches
        results = await process_batch(batch)
        yield results
        
        # Clear memory after each batch
        import gc
        gc.collect()
```

2. **Use memory-mapped files for large data:**
```python
import numpy as np

# Use memory mapping for large arrays
large_array = np.memmap(
    'data/large_embeddings.dat',
    dtype='float32',
    mode='r',
    shape=(1000000, 768)
)
```

3. **Implement lazy loading:**
```python
class LazyEmbeddings:
    def __init__(self, file_path):
        self.file_path = file_path
        self._embeddings = None
    
    @property
    def embeddings(self):
        if self._embeddings is None:
            self._embeddings = load_embeddings(self.file_path)
        return self._embeddings
```

## Input Validation Errors

### Common Validation Failures

**Error:** `ValueError: query text Value cannot be empty or contain only whitespace`

**Cause:** Empty or whitespace-only query text
**Solution:**
```python
# Fix empty queries
query_text = query_text.strip()
if not query_text:
    query_text = "default search query"

query = AgentQuery(text=query_text)
```

**Error:** `ValueError: max_results must be a positive number`

**Cause:** Invalid numeric parameters
**Solution:**
```python
# Validate numeric inputs
max_results = max(1, int(max_results))  # Ensure positive
confidence_threshold = max(0.0, min(1.0, float(confidence_threshold)))  # Clamp to [0,1]
```

**Error:** `ValueError: parameters must be a dictionary`

**Cause:** Incorrect parameter type
**Solution:**
```python
# Ensure parameters are dictionaries
if not isinstance(parameters, dict):
    parameters = {}

query = AgentQuery(
    text="valid query",
    parameters=parameters
)
```

### Input Sanitization

```python
from eeg_rag.utils.common_utils import (
    validate_non_empty_string,
    validate_positive_number,
    validate_range
)

def sanitize_query_input(text, max_results=None, confidence=None):
    """Sanitize and validate user input"""
    # Clean text
    text = text.strip()
    if not text:
        raise ValueError("Query text cannot be empty")
    
    # Remove harmful characters
    import re
    text = re.sub(r'[^\w\s\-\.,?!]', '', text)
    
    # Limit length
    if len(text) > 1000:
        text = text[:997] + "..."
    
    # Validate numbers
    if max_results is not None:
        max_results = validate_positive_number(max_results, "max_results")
        max_results = min(max_results, 100)  # Cap at reasonable limit
    
    if confidence is not None:
        confidence = validate_range(confidence, 0.0, 1.0, "confidence")
    
    return text, max_results, confidence
```

## Connectivity Issues

### External API Failures

**Error:** `ConnectionError: Cannot connect to PubMed API`

**Diagnosis:**
```bash
# Test network connectivity
ping -c 3 eutils.ncbi.nlm.nih.gov
curl -I "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"

# Check DNS resolution
nslookup eutils.ncbi.nlm.nih.gov

# Test with detailed curl
curl -v "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term=EEG&retmax=1"
```

**Solutions:**

1. **Configure proxy if needed:**
```python
import os
os.environ['HTTP_PROXY'] = 'http://proxy.company.com:8080'
os.environ['HTTPS_PROXY'] = 'http://proxy.company.com:8080'
```

2. **Add retry with exponential backoff:**
```python
import aiohttp
import asyncio

async def robust_api_request(url, max_retries=3):
    for attempt in range(max_retries):
        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30)
            ) as session:
                async with session.get(url) as response:
                    return await response.json()
        except Exception as e:
            wait_time = 2 ** attempt  # Exponential backoff
            logger.warning(f"API request failed (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(wait_time)
            else:
                raise
```

3. **Implement fallback data sources:**
```python
async def get_papers_with_fallback(query):
    try:
        # Primary: PubMed API
        return await pubmed_search(query)
    except ConnectionError:
        # Fallback: Local cache
        logger.warning("PubMed unavailable, using cache")
        return await search_local_cache(query)
    except Exception:
        # Last resort: Default results
        logger.error("All data sources failed")
        return default_search_results()
```

### Database Connection Issues

**Error:** `sqlite3.OperationalError: database is locked`

**Cause:** Concurrent database access
**Solutions:**

1. **Use connection pooling:**
```python
import sqlite3
from contextlib import contextmanager

@contextmanager
def get_db_connection():
    conn = sqlite3.connect(
        "database.db",
        timeout=30.0,  # Wait up to 30 seconds for lock
        check_same_thread=False
    )
    try:
        yield conn
    finally:
        conn.close()
```

2. **Implement database retry logic:**
```python
@retry_with_backoff(
    max_retries=3,
    initial_delay=0.1,
    exceptions=(sqlite3.OperationalError,)
)
def safe_database_operation(operation):
    with get_db_connection() as conn:
        return operation(conn)
```

## Memory Management Issues

### Memory Leaks

**Symptoms:**
- Memory usage increases continuously
- No corresponding increase in data processing
- System becomes sluggish over time

**Detection:**
```python
import tracemalloc
import gc

# Enable memory tracking
tracemalloc.start()

# Run your application for a while, then check:
snapshot = tracemalloc.take_snapshot()
top_stats = snapshot.statistics('lineno')

print("Top 10 memory consumers:")
for index, stat in enumerate(top_stats[:10], 1):
    print(f"{index}. {stat}")

# Check for uncollected objects
print(f"Garbage collection stats: {gc.get_stats()}")
print(f"Uncollectable objects: {len(gc.garbage)}")
```

**Solutions:**

1. **Explicit cleanup:**
```python
class AgentWithCleanup(BaseAgent):
    def __init__(self):
        super().__init__(AgentType.LOCAL_DATA, "cleanup_agent")
        self._large_objects = []
    
    async def execute(self, query):
        try:
            result = await self._process_query(query)
            return result
        finally:
            # Explicit cleanup
            self._large_objects.clear()
            gc.collect()
```

2. **Use weak references for caches:**
```python
import weakref

class WeakCache:
    def __init__(self):
        self._cache = weakref.WeakValueDictionary()
    
    def get(self, key):
        return self._cache.get(key)
    
    def set(self, key, value):
        self._cache[key] = value
```

### Cache Overflow

**Error:** `MemoryError` or very high memory usage from caches

**Solutions:**

1. **Implement LRU cache with size limits:**
```python
from functools import lru_cache

@lru_cache(maxsize=1000)  # Limit cache size
def cached_embedding_computation(text):
    return compute_embedding(text)
```

2. **Use time-based cache expiration:**
```python
import time
from collections import defaultdict

class TTLCache:
    def __init__(self, ttl_seconds=300):
        self._cache = {}
        self._timestamps = {}
        self.ttl = ttl_seconds
    
    def get(self, key):
        if key in self._cache:
            if time.time() - self._timestamps[key] < self.ttl:
                return self._cache[key]
            else:
                # Expired, remove
                del self._cache[key]
                del self._timestamps[key]
        return None
    
    def set(self, key, value):
        self._cache[key] = value
        self._timestamps[key] = time.time()
```

## Logging and Debugging

### Enable Debug Logging

```python
import logging

# Enable detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('debug.log'),
        logging.StreamHandler()
    ]
)

# Enable specific module debug logging
logging.getLogger('eeg_rag.agents').setLevel(logging.DEBUG)
logging.getLogger('eeg_rag.memory').setLevel(logging.DEBUG)
```

### Structured Logging

```python
import json
import logging

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_data = {
            'timestamp': self.formatTime(record),
            'level': record.levelname,
            'module': record.name,
            'message': record.getMessage(),
            'function': record.funcName,
            'line': record.lineno
        }
        
        if hasattr(record, 'agent_name'):
            log_data['agent'] = record.agent_name
        if hasattr(record, 'query_id'):
            log_data['query_id'] = record.query_id
            
        return json.dumps(log_data)

# Use structured logging
handler = logging.StreamHandler()
handler.setFormatter(JSONFormatter())
logger = logging.getLogger('eeg_rag')
logger.addHandler(handler)
```

### Performance Profiling

```python
import cProfile
import pstats
import io

def profile_function(func):
    """Decorator to profile function execution"""
    def wrapper(*args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()
        
        result = func(*args, **kwargs)
        
        pr.disable()
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
        ps.print_stats()
        
        print(f"Profile for {func.__name__}:")
        print(s.getvalue())
        
        return result
    return wrapper

# Use profiling
@profile_function
async def slow_operation(query):
    return await agent.run(query)
```

## Emergency Procedures

### System Recovery

1. **Immediate steps for system failure:**
```bash
# Stop all processes
pkill -f "python.*eeg.*rag"

# Check system resources
df -h  # Disk space
free -h  # Memory
top  # CPU usage

# Clear temporary files
rm -rf /tmp/eeg_rag_*
find . -name "*.tmp" -delete

# Restart with minimal configuration
python -c "
from eeg_rag.utils.common_utils import check_system_health
health = check_system_health()
print(f'System status: {health.status.value}')
"
```

2. **Reset to known good state:**
```python
# Clear all caches and reset agents
from eeg_rag.agents import AgentRegistry

registry = AgentRegistry()
for agent in registry.get_all_agents():
    agent.reset_statistics()
    if hasattr(agent, 'clear_cache'):
        agent.clear_cache()

# Reset memory manager
memory_manager.clear_all_memory()

print("System reset complete")
```

### Data Recovery

```bash
# Backup current state
timestamp=$(date +%Y%m%d_%H%M%S)
tar -czf "backup_${timestamp}.tar.gz" data/ logs/ config/

# Restore from backup
tar -xzf backup_20241122_143000.tar.gz

# Verify data integrity
python scripts/verify_data_integrity.py
```

## Getting Help

### Collect Diagnostic Information

```python
def generate_diagnostic_report():
    """Generate comprehensive diagnostic information"""
    import platform
    import sys
    from datetime import datetime
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "system": {
            "platform": platform.platform(),
            "python_version": sys.version,
            "cpu_count": psutil.cpu_count(),
            "memory_total": psutil.virtual_memory().total
        },
        "health": check_system_health().to_dict(),
        "agents": {}
    }
    
    # Add agent statistics
    registry = AgentRegistry()
    for agent in registry.get_all_agents():
        report["agents"][agent.name] = agent.get_statistics()
    
    # Save report
    with open(f"diagnostic_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    
    return report

# Generate and save diagnostic report
report = generate_diagnostic_report()
print(f"Diagnostic report saved: {report['timestamp']}")
```

### Support Checklist

Before requesting support, please provide:

- [ ] Diagnostic report (generated above)
- [ ] System health status
- [ ] Recent log files (last 24 hours)
- [ ] Steps to reproduce the issue
- [ ] Expected vs actual behavior
- [ ] System configuration details

### Contact Information

- **Documentation**: `docs/TECHNICAL_REFERENCE.md`
- **Issue Tracking**: GitHub Issues
- **Development Team**: See `CONTRIBUTORS.md`

---

*This troubleshooting guide is a living document. Please contribute improvements based on your experience with the system.*