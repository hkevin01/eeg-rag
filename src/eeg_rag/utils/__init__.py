"""
Utility modules for EEG-RAG.
"""

from .deduplication import PaperDeduplicator, deduplicate_papers
from .citations import CitationGenerator, generate_citations
from .quality_badges import (
    get_code_badge,
    get_data_badge,
    get_reproducibility_badge,
    get_citation_count_badge,
    get_all_badges,
    get_quality_score,
)
from .time_utils import (
    TimeUnits,
    convert_time,
    Timer,
    TimingStats,
    format_duration,
)
from .error_handling import (
    ErrorCode,
    EEGRAGError,
    ValidationError,
    safe_execute,
    with_retry,
    validate_not_empty,
)
from .memory_utils import (
    MemoryUsage,
    MemoryProfile,
    MemoryMonitor,
    MemoryLeakDetector,
    MemoryPool,
    check_memory_health,
)
from .persistence_utils import (
    PersistenceResult,
    BackupInfo,
    with_persistence_retry,
    BackupManager,
    atomic_write,
    JsonPersistence,
    ConnectionPool,
    WriteAheadLog,
)
from .resilience_utils import (
    CircuitState,
    CircuitStats,
    CircuitBreaker,
    CircuitOpenError,
    circuit_breaker,
    HealthStatus,
    HealthCheckResult,
    HealthChecker,
    RateLimiter,
    rate_limit,
    RateLimitExceeded,
    DegradationLevel,
    GracefulDegradation,
    feature_gate,
    FeatureDisabledError,
)

__all__ = [
    # Deduplication
    'PaperDeduplicator',
    'deduplicate_papers',
    # Citations
    'CitationGenerator',
    'generate_citations',
    # Quality badges
    'get_code_badge',
    'get_data_badge',
    'get_reproducibility_badge',
    'get_citation_count_badge',
    'get_all_badges',
    'get_quality_score',
    # Time utilities
    'TimeUnits',
    'convert_time',
    'Timer',
    'TimingStats',
    'format_duration',
    # Error handling
    'ErrorCode',
    'EEGRAGError',
    'ValidationError',
    'safe_execute',
    'with_retry',
    'validate_not_empty',
    # Memory utilities
    'MemoryUsage',
    'MemoryProfile',
    'MemoryMonitor',
    'MemoryLeakDetector',
    'MemoryPool',
    'check_memory_health',
    # Persistence utilities
    'PersistenceResult',
    'BackupInfo',
    'with_persistence_retry',
    'BackupManager',
    'atomic_write',
    'JsonPersistence',
    'ConnectionPool',
    'WriteAheadLog',
    # Resilience utilities
    'CircuitState',
    'CircuitStats',
    'CircuitBreaker',
    'CircuitOpenError',
    'circuit_breaker',
    'HealthStatus',
    'HealthCheckResult',
    'HealthChecker',
    'RateLimiter',
    'rate_limit',
    'RateLimitExceeded',
    'DegradationLevel',
    'GracefulDegradation',
    'feature_gate',
    'FeatureDisabledError',
]
