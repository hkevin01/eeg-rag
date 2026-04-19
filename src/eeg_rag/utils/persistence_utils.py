"""
Persistence Utilities for EEG-RAG

Provides robust persistence handling with retry logic, backup mechanisms,
and data integrity verification.

Requirements Implemented:
- REQ-DAT-001: Data persistence operations
- REQ-DAT-002: Database retry logic
- REQ-DAT-003: Backup and recovery
- REQ-REL-003: Data integrity verification
"""

import asyncio
import hashlib
import json
import logging
import os
import shutil
import time
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, Generic, List, Optional, TypeVar, Union

logger = logging.getLogger(__name__)

T = TypeVar("T")


# REQ-DAT-001: Persistence operation status codes
# ---------------------------------------------------------------------------
# ID           : utils.persistence_utils.PersistenceStatus
# Requirement  : `PersistenceStatus` class shall be instantiable and expose the documented interface
# Purpose      : Status codes for persistence operations
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
# Verification : Instantiate PersistenceStatus with valid args; assert attribute types and values
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
class PersistenceStatus:
    """Status codes for persistence operations."""
    SUCCESS = "success"
    FAILED = "failed"
    PENDING = "pending"
    RETRYING = "retrying"
    RECOVERED = "recovered"
    CORRUPTED = "corrupted"


# ---------------------------------------------------------------------------
# ID           : utils.persistence_utils.PersistenceResult
# Requirement  : `PersistenceResult` class shall be instantiable and expose the documented interface
# Purpose      : REQ-DAT-001: Result of a persistence operation
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
# Verification : Instantiate PersistenceResult with valid args; assert attribute types and values
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
@dataclass
class PersistenceResult(Generic[T]):
    """
    REQ-DAT-001: Result of a persistence operation.
    
    Attributes:
        success: Whether the operation succeeded
        data: The result data (if successful)
        error: Error message (if failed)
        attempts: Number of attempts made
        duration_ms: Total duration in milliseconds
    """
    success: bool
    data: Optional[T] = None
    error: Optional[str] = None
    attempts: int = 1
    duration_ms: float = 0.0
    recovered: bool = False
    
    # ---------------------------------------------------------------------------
    # ID           : utils.persistence_utils.PersistenceResult.to_dict
    # Requirement  : `to_dict` shall serialize to dictionary
    # Purpose      : Serialize to dictionary
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : None
    # Outputs      : Dict[str, Any]
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
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "success": self.success,
            "error": self.error,
            "attempts": self.attempts,
            "duration_ms": self.duration_ms,
            "recovered": self.recovered,
        }


# ---------------------------------------------------------------------------
# ID           : utils.persistence_utils.BackupInfo
# Requirement  : `BackupInfo` class shall be instantiable and expose the documented interface
# Purpose      : REQ-DAT-003: Information about a backup
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
# Verification : Instantiate BackupInfo with valid args; assert attribute types and values
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
@dataclass
class BackupInfo:
    """
    REQ-DAT-003: Information about a backup.
    
    Attributes:
        path: Path to the backup file
        timestamp: When the backup was created
        size_bytes: Size of the backup in bytes
        checksum: SHA256 checksum for integrity
    """
    path: Path
    timestamp: datetime
    size_bytes: int
    checksum: str
    
    # ---------------------------------------------------------------------------
    # ID           : utils.persistence_utils.BackupInfo.to_dict
    # Requirement  : `to_dict` shall serialize to dictionary
    # Purpose      : Serialize to dictionary
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : None
    # Outputs      : Dict[str, Any]
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
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "path": str(self.path),
            "timestamp": self.timestamp.isoformat(),
            "size_bytes": self.size_bytes,
            "checksum": self.checksum,
        }


# REQ-REL-003: Checksum calculation
# ---------------------------------------------------------------------------
# ID           : utils.persistence_utils.calculate_checksum
# Requirement  : `calculate_checksum` shall calculate SHA256 checksum for data integrity verification
# Purpose      : Calculate SHA256 checksum for data integrity verification
# Rationale    : Implements domain-specific logic per system design; see referenced specs
# Inputs       : data: Union[bytes, str, Path]
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
def calculate_checksum(data: Union[bytes, str, Path]) -> str:
    """
    Calculate SHA256 checksum for data integrity verification.
    
    Args:
        data: Bytes, string, or file path to checksum
        
    Returns:
        Hex-encoded SHA256 checksum
    """
    hasher = hashlib.sha256()
    
    if isinstance(data, Path):
        with open(data, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                hasher.update(chunk)
    elif isinstance(data, str):
        hasher.update(data.encode('utf-8'))
    else:
        hasher.update(data)
    
    return hasher.hexdigest()


# ---------------------------------------------------------------------------
# ID           : utils.persistence_utils.verify_checksum
# Requirement  : `verify_checksum` shall verify data integrity by comparing checksums
# Purpose      : Verify data integrity by comparing checksums
# Rationale    : Implements domain-specific logic per system design; see referenced specs
# Inputs       : data: Union[bytes, str, Path]; expected: str
# Outputs      : bool
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
def verify_checksum(data: Union[bytes, str, Path], expected: str) -> bool:
    """
    Verify data integrity by comparing checksums.
    
    Args:
        data: Data to verify
        expected: Expected checksum
        
    Returns:
        True if checksums match
    """
    actual = calculate_checksum(data)
    return actual == expected


# REQ-DAT-002: Retry decorator for persistence operations
# ---------------------------------------------------------------------------
# ID           : utils.persistence_utils.with_persistence_retry
# Requirement  : `with_persistence_retry` shall decorator that adds retry logic to persistence operations
# Purpose      : Decorator that adds retry logic to persistence operations
# Rationale    : Implements domain-specific logic per system design; see referenced specs
# Inputs       : max_attempts: int (default=3); delay_seconds: float (default=1.0); backoff_multiplier: float (default=2.0); retryable_exceptions: tuple (default=(IOError, OSError, ConnectionError)); on_retry: Optional[Callable[[int, Exception], None]] (default=None)
# Outputs      : Callable
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
def with_persistence_retry(
    max_attempts: int = 3,
    delay_seconds: float = 1.0,
    backoff_multiplier: float = 2.0,
    retryable_exceptions: tuple = (IOError, OSError, ConnectionError),
    on_retry: Optional[Callable[[int, Exception], None]] = None,
) -> Callable:
    """
    Decorator that adds retry logic to persistence operations.
    
    Args:
        max_attempts: Maximum number of attempts
        delay_seconds: Initial delay between retries
        backoff_multiplier: Multiplier for exponential backoff
        retryable_exceptions: Exceptions that trigger retry
        on_retry: Callback called on each retry with (attempt, exception)
        
    Returns:
        Decorated function
    """
    # ---------------------------------------------------------------------------
    # ID           : utils.persistence_utils.decorator
    # Requirement  : `decorator` shall execute as specified
    # Purpose      : Decorator
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : func: Callable
    # Outputs      : Callable
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
    def decorator(func: Callable) -> Callable:
        # ---------------------------------------------------------------------------
        # ID           : utils.persistence_utils.wrapper
        # Requirement  : `wrapper` shall execute as specified
        # Purpose      : Wrapper
        # Rationale    : Implements domain-specific logic per system design; see referenced specs
        # Inputs       : *args; **kwargs
        # Outputs      : PersistenceResult
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
        @wraps(func)
        def wrapper(*args, **kwargs) -> PersistenceResult:
            last_error: Optional[Exception] = None
            delay = delay_seconds
            start_time = time.time()
            
            for attempt in range(1, max_attempts + 1):
                try:
                    result = func(*args, **kwargs)
                    duration = (time.time() - start_time) * 1000
                    return PersistenceResult(
                        success=True,
                        data=result,
                        attempts=attempt,
                        duration_ms=duration,
                    )
                except retryable_exceptions as e:
                    last_error = e
                    if on_retry:
                        on_retry(attempt, e)
                    
                    if attempt < max_attempts:
                        logger.warning(
                            f"Persistence operation failed (attempt {attempt}/{max_attempts}): {e}. "
                            f"Retrying in {delay:.1f}s..."
                        )
                        time.sleep(delay)
                        delay *= backoff_multiplier
                    else:
                        logger.error(
                            f"Persistence operation failed after {max_attempts} attempts: {e}"
                        )
            
            duration = (time.time() - start_time) * 1000
            return PersistenceResult(
                success=False,
                error=str(last_error),
                attempts=max_attempts,
                duration_ms=duration,
            )
        
        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# ID           : utils.persistence_utils.with_async_persistence_retry
# Requirement  : `with_async_persistence_retry` shall async version of persistence retry decorator
# Purpose      : Async version of persistence retry decorator
# Rationale    : Implements domain-specific logic per system design; see referenced specs
# Inputs       : max_attempts: int (default=3); delay_seconds: float (default=1.0); backoff_multiplier: float (default=2.0); retryable_exceptions: tuple (default=(IOError, OSError, ConnectionError))
# Outputs      : Callable
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
def with_async_persistence_retry(
    max_attempts: int = 3,
    delay_seconds: float = 1.0,
    backoff_multiplier: float = 2.0,
    retryable_exceptions: tuple = (IOError, OSError, ConnectionError),
) -> Callable:
    """
    Async version of persistence retry decorator.
    
    Args:
        max_attempts: Maximum number of attempts
        delay_seconds: Initial delay between retries
        backoff_multiplier: Multiplier for exponential backoff
        retryable_exceptions: Exceptions that trigger retry
        
    Returns:
        Decorated async function
    """
    # ---------------------------------------------------------------------------
    # ID           : utils.persistence_utils.decorator
    # Requirement  : `decorator` shall execute as specified
    # Purpose      : Decorator
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : func: Callable
    # Outputs      : Callable
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
    def decorator(func: Callable) -> Callable:
        # ---------------------------------------------------------------------------
        # ID           : utils.persistence_utils.wrapper
        # Requirement  : `wrapper` shall execute as specified
        # Purpose      : Wrapper
        # Rationale    : Implements domain-specific logic per system design; see referenced specs
        # Inputs       : *args; **kwargs
        # Outputs      : PersistenceResult
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
        @wraps(func)
        async def wrapper(*args, **kwargs) -> PersistenceResult:
            last_error: Optional[Exception] = None
            delay = delay_seconds
            start_time = time.time()
            
            for attempt in range(1, max_attempts + 1):
                try:
                    result = await func(*args, **kwargs)
                    duration = (time.time() - start_time) * 1000
                    return PersistenceResult(
                        success=True,
                        data=result,
                        attempts=attempt,
                        duration_ms=duration,
                    )
                except retryable_exceptions as e:
                    last_error = e
                    
                    if attempt < max_attempts:
                        logger.warning(
                            f"Async persistence operation failed (attempt {attempt}/{max_attempts}): {e}. "
                            f"Retrying in {delay:.1f}s..."
                        )
                        await asyncio.sleep(delay)
                        delay *= backoff_multiplier
                    else:
                        logger.error(
                            f"Async persistence operation failed after {max_attempts} attempts: {e}"
                        )
            
            duration = (time.time() - start_time) * 1000
            return PersistenceResult(
                success=False,
                error=str(last_error),
                attempts=max_attempts,
                duration_ms=duration,
            )
        
        return wrapper
    return decorator


# REQ-DAT-003: Backup management
# ---------------------------------------------------------------------------
# ID           : utils.persistence_utils.BackupManager
# Requirement  : `BackupManager` class shall be instantiable and expose the documented interface
# Purpose      : Manages file backups with rotation and integrity verification
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
# Verification : Instantiate BackupManager with valid args; assert attribute types and values
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
class BackupManager:
    """
    Manages file backups with rotation and integrity verification.
    
    Attributes:
        backup_dir: Directory for storing backups
        max_backups: Maximum number of backups to retain
    """
    
    # ---------------------------------------------------------------------------
    # ID           : utils.persistence_utils.BackupManager.__init__
    # Requirement  : `__init__` shall initialize backup manager
    # Purpose      : Initialize backup manager
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : backup_dir: Union[str, Path]; max_backups: int (default=5)
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
        backup_dir: Union[str, Path],
        max_backups: int = 5,
    ):
        """
        Initialize backup manager.
        
        Args:
            backup_dir: Directory for storing backups
            max_backups: Maximum backups to retain (oldest deleted first)
        """
        self.backup_dir = Path(backup_dir)
        self.max_backups = max_backups
        self._ensure_dir()
    
    # ---------------------------------------------------------------------------
    # ID           : utils.persistence_utils.BackupManager._ensure_dir
    # Requirement  : `_ensure_dir` shall ensure backup directory exists
    # Purpose      : Ensure backup directory exists
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : None
    # Outputs      : None
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
    def _ensure_dir(self) -> None:
        """Ensure backup directory exists."""
        self.backup_dir.mkdir(parents=True, exist_ok=True)
    
    # ---------------------------------------------------------------------------
    # ID           : utils.persistence_utils.BackupManager.create_backup
    # Requirement  : `create_backup` shall rEQ-DAT-003: Create a backup of a file
    # Purpose      : REQ-DAT-003: Create a backup of a file
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : source: Union[str, Path]; prefix: str (default='backup')
    # Outputs      : BackupInfo
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
    def create_backup(
        self,
        source: Union[str, Path],
        prefix: str = "backup",
    ) -> BackupInfo:
        """
        REQ-DAT-003: Create a backup of a file.
        
        Args:
            source: Path to file to backup
            prefix: Prefix for backup filename
            
        Returns:
            BackupInfo with details about the backup
            
        Raises:
            FileNotFoundError: If source doesn't exist
        """
        source = Path(source)
        if not source.exists():
            raise FileNotFoundError(f"Source file not found: {source}")
        
        timestamp = datetime.now()
        backup_name = f"{prefix}_{timestamp.strftime('%Y%m%d_%H%M%S')}_{source.name}"
        backup_path = self.backup_dir / backup_name
        
        # Copy file
        shutil.copy2(source, backup_path)
        
        # Calculate checksum
        checksum = calculate_checksum(backup_path)
        
        # Get size
        size = backup_path.stat().st_size
        
        backup_info = BackupInfo(
            path=backup_path,
            timestamp=timestamp,
            size_bytes=size,
            checksum=checksum,
        )
        
        logger.info(f"Created backup: {backup_path} (checksum: {checksum[:8]}...)")
        
        # Rotate old backups
        self._rotate_backups(prefix)
        
        return backup_info
    
    # ---------------------------------------------------------------------------
    # ID           : utils.persistence_utils.BackupManager._rotate_backups
    # Requirement  : `_rotate_backups` shall remove old backups exceeding max_backups
    # Purpose      : Remove old backups exceeding max_backups
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : prefix: str
    # Outputs      : None
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
    def _rotate_backups(self, prefix: str) -> None:
        """Remove old backups exceeding max_backups."""
        backups = sorted(
            self.backup_dir.glob(f"{prefix}_*"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        
        for old_backup in backups[self.max_backups:]:
            logger.info(f"Rotating out old backup: {old_backup}")
            old_backup.unlink()
    
    # ---------------------------------------------------------------------------
    # ID           : utils.persistence_utils.BackupManager.list_backups
    # Requirement  : `list_backups` shall rEQ-DAT-003: List all backups for a prefix
    # Purpose      : REQ-DAT-003: List all backups for a prefix
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : prefix: str (default='backup')
    # Outputs      : List[BackupInfo]
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
    def list_backups(self, prefix: str = "backup") -> List[BackupInfo]:
        """
        REQ-DAT-003: List all backups for a prefix.
        
        Args:
            prefix: Prefix to filter backups
            
        Returns:
            List of BackupInfo sorted by timestamp (newest first)
        """
        backups = []
        for path in self.backup_dir.glob(f"{prefix}_*"):
            try:
                stat = path.stat()
                backups.append(BackupInfo(
                    path=path,
                    timestamp=datetime.fromtimestamp(stat.st_mtime),
                    size_bytes=stat.st_size,
                    checksum=calculate_checksum(path),
                ))
            except Exception as e:
                logger.warning(f"Failed to read backup {path}: {e}")
        
        return sorted(backups, key=lambda b: b.timestamp, reverse=True)
    
    # ---------------------------------------------------------------------------
    # ID           : utils.persistence_utils.BackupManager.restore_backup
    # Requirement  : `restore_backup` shall rEQ-DAT-003: Restore a file from backup
    # Purpose      : REQ-DAT-003: Restore a file from backup
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : backup: Union[BackupInfo, Path]; destination: Union[str, Path]; verify: bool (default=True)
    # Outputs      : PersistenceResult[Path]
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
    def restore_backup(
        self,
        backup: Union[BackupInfo, Path],
        destination: Union[str, Path],
        verify: bool = True,
    ) -> PersistenceResult[Path]:
        """
        REQ-DAT-003: Restore a file from backup.
        
        Args:
            backup: BackupInfo or path to backup file
            destination: Where to restore the file
            verify: Whether to verify checksum after restore
            
        Returns:
            PersistenceResult with restored path
        """
        start_time = time.time()
        
        if isinstance(backup, BackupInfo):
            backup_path = backup.path
            expected_checksum = backup.checksum
        else:
            backup_path = Path(backup)
            expected_checksum = calculate_checksum(backup_path)
        
        destination = Path(destination)
        
        try:
            # Verify backup integrity
            if verify:
                if not verify_checksum(backup_path, expected_checksum):
                    return PersistenceResult(
                        success=False,
                        error="Backup file corrupted (checksum mismatch)",
                        duration_ms=(time.time() - start_time) * 1000,
                    )
            
            # Restore
            shutil.copy2(backup_path, destination)
            
            # Verify restored file
            if verify:
                if not verify_checksum(destination, expected_checksum):
                    return PersistenceResult(
                        success=False,
                        error="Restored file corrupted",
                        duration_ms=(time.time() - start_time) * 1000,
                    )
            
            logger.info(f"Restored backup {backup_path} to {destination}")
            
            return PersistenceResult(
                success=True,
                data=destination,
                duration_ms=(time.time() - start_time) * 1000,
                recovered=True,
            )
            
        except Exception as e:
            return PersistenceResult(
                success=False,
                error=str(e),
                duration_ms=(time.time() - start_time) * 1000,
            )
    
    # ---------------------------------------------------------------------------
    # ID           : utils.persistence_utils.BackupManager.cleanup
    # Requirement  : `cleanup` shall remove all backups
    # Purpose      : Remove all backups
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : None
    # Outputs      : int
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
    def cleanup(self) -> int:
        """
        Remove all backups.
        
        Returns:
            Number of backups removed
        """
        count = 0
        for path in self.backup_dir.glob("*"):
            if path.is_file():
                path.unlink()
                count += 1
        return count


# REQ-DAT-001: Atomic file write
# ---------------------------------------------------------------------------
# ID           : utils.persistence_utils.atomic_write
# Requirement  : `atomic_write` shall context manager for atomic file writes
# Purpose      : Context manager for atomic file writes
# Rationale    : Implements domain-specific logic per system design; see referenced specs
# Inputs       : path: Union[str, Path]; mode: str (default='w'); encoding: str (default='utf-8')
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
@contextmanager
def atomic_write(
    path: Union[str, Path],
    mode: str = "w",
    encoding: str = "utf-8",
):
    """
    Context manager for atomic file writes.
    
    Writes to a temporary file first, then renames to target path.
    This prevents partial writes on crash/error.
    
    Args:
        path: Target file path
        mode: Write mode ('w' for text, 'wb' for binary)
        encoding: Text encoding (ignored for binary mode)
        
    Yields:
        File handle for writing
    """
    path = Path(path)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    
    try:
        if 'b' in mode:
            f = open(temp_path, mode)
        else:
            f = open(temp_path, mode, encoding=encoding)
        
        yield f
        f.close()
        
        # Atomic rename
        temp_path.replace(path)
        
    except Exception:
        f.close()
        if temp_path.exists():
            temp_path.unlink()
        raise


# ---------------------------------------------------------------------------
# ID           : utils.persistence_utils.async_atomic_write
# Requirement  : `async_atomic_write` shall async context manager for atomic file writes
# Purpose      : Async context manager for atomic file writes
# Rationale    : Implements domain-specific logic per system design; see referenced specs
# Inputs       : path: Union[str, Path]; mode: str (default='w'); encoding: str (default='utf-8')
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
@asynccontextmanager
async def async_atomic_write(
    path: Union[str, Path],
    mode: str = "w",
    encoding: str = "utf-8",
):
    """
    Async context manager for atomic file writes.
    
    Args:
        path: Target file path
        mode: Write mode
        encoding: Text encoding
        
    Yields:
        File handle for writing
    """
    import aiofiles
    
    path = Path(path)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    
    try:
        if 'b' in mode:
            f = await aiofiles.open(temp_path, mode)
        else:
            f = await aiofiles.open(temp_path, mode, encoding=encoding)
        
        yield f
        await f.close()
        
        # Atomic rename (run in executor)
        await asyncio.get_event_loop().run_in_executor(
            None, temp_path.replace, path
        )
        
    except Exception:
        await f.close()
        if temp_path.exists():
            await asyncio.get_event_loop().run_in_executor(
                None, temp_path.unlink
            )
        raise


# REQ-DAT-001: JSON persistence with integrity
# ---------------------------------------------------------------------------
# ID           : utils.persistence_utils.JsonPersistence
# Requirement  : `JsonPersistence` class shall be instantiable and expose the documented interface
# Purpose      : JSON file persistence with integrity verification and backup
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
# Verification : Instantiate JsonPersistence with valid args; assert attribute types and values
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
class JsonPersistence:
    """
    JSON file persistence with integrity verification and backup.
    
    Provides safe read/write operations with automatic checksums.
    """
    
    # ---------------------------------------------------------------------------
    # ID           : utils.persistence_utils.JsonPersistence.__init__
    # Requirement  : `__init__` shall initialize JSON persistence
    # Purpose      : Initialize JSON persistence
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : path: Union[str, Path]; backup_manager: Optional[BackupManager] (default=None); pretty: bool (default=True)
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
        path: Union[str, Path],
        backup_manager: Optional[BackupManager] = None,
        pretty: bool = True,
    ):
        """
        Initialize JSON persistence.
        
        Args:
            path: Path to JSON file
            backup_manager: Optional backup manager for automatic backups
            pretty: Whether to format JSON with indentation
        """
        self.path = Path(path)
        self.backup_manager = backup_manager
        self.pretty = pretty
        self._checksum_file = self.path.with_suffix(".checksum")
    
    # ---------------------------------------------------------------------------
    # ID           : utils.persistence_utils.JsonPersistence.save
    # Requirement  : `save` shall rEQ-DAT-001: Save data to JSON file with integrity
    # Purpose      : REQ-DAT-001: Save data to JSON file with integrity
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : data: Dict[str, Any]
    # Outputs      : Dict[str, Any]
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
    @with_persistence_retry(max_attempts=3)
    def save(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        REQ-DAT-001: Save data to JSON file with integrity.
        
        Args:
            data: Data to save
            
        Returns:
            The saved data
        """
        # Create backup if manager exists
        if self.backup_manager and self.path.exists():
            self.backup_manager.create_backup(
                self.path,
                prefix=self.path.stem,
            )
        
        # Serialize
        content = json.dumps(
            data,
            indent=2 if self.pretty else None,
            ensure_ascii=False,
            default=str,
        )
        
        # Write atomically
        with atomic_write(self.path) as f:
            f.write(content)
        
        # Save checksum
        checksum = calculate_checksum(content)
        self._checksum_file.write_text(checksum)
        
        logger.debug(f"Saved JSON to {self.path} (checksum: {checksum[:8]}...)")
        
        return data
    
    # ---------------------------------------------------------------------------
    # ID           : utils.persistence_utils.JsonPersistence.load
    # Requirement  : `load` shall rEQ-DAT-001: Load data from JSON file with integrity check
    # Purpose      : REQ-DAT-001: Load data from JSON file with integrity check
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : verify: bool (default=True)
    # Outputs      : Dict[str, Any]
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
    @with_persistence_retry(max_attempts=3)
    def load(self, verify: bool = True) -> Dict[str, Any]:
        """
        REQ-DAT-001: Load data from JSON file with integrity check.
        
        Args:
            verify: Whether to verify checksum
            
        Returns:
            Loaded data
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If checksum verification fails
        """
        if not self.path.exists():
            raise FileNotFoundError(f"JSON file not found: {self.path}")
        
        content = self.path.read_text(encoding="utf-8")
        
        # Verify checksum if available
        if verify and self._checksum_file.exists():
            expected = self._checksum_file.read_text().strip()
            if not verify_checksum(content, expected):
                raise ValueError(f"JSON file corrupted: {self.path}")
        
        return json.loads(content)
    
    # ---------------------------------------------------------------------------
    # ID           : utils.persistence_utils.JsonPersistence.exists
    # Requirement  : `exists` shall check if the JSON file exists
    # Purpose      : Check if the JSON file exists
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : None
    # Outputs      : bool
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
    def exists(self) -> bool:
        """Check if the JSON file exists."""
        return self.path.exists()
    
    # ---------------------------------------------------------------------------
    # ID           : utils.persistence_utils.JsonPersistence.delete
    # Requirement  : `delete` shall delete the JSON file and its checksum
    # Purpose      : Delete the JSON file and its checksum
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : None
    # Outputs      : bool
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
    def delete(self) -> bool:
        """Delete the JSON file and its checksum."""
        deleted = False
        if self.path.exists():
            self.path.unlink()
            deleted = True
        if self._checksum_file.exists():
            self._checksum_file.unlink()
        return deleted


# REQ-DAT-002: Connection pool for database connections
# ---------------------------------------------------------------------------
# ID           : utils.persistence_utils.ConnectionPool
# Requirement  : `ConnectionPool` class shall be instantiable and expose the documented interface
# Purpose      : Generic connection pool for database connections
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
# Verification : Instantiate ConnectionPool with valid args; assert attribute types and values
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
class ConnectionPool(Generic[T]):
    """
    Generic connection pool for database connections.
    
    Manages a pool of reusable connections with health checking.
    """
    
    # ---------------------------------------------------------------------------
    # ID           : utils.persistence_utils.ConnectionPool.__init__
    # Requirement  : `__init__` shall initialize connection pool
    # Purpose      : Initialize connection pool
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : create_fn: Callable[[], T]; close_fn: Callable[[T], None]; validate_fn: Optional[Callable[[T], bool]] (default=None); min_size: int (default=1); max_size: int (default=10)
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
        create_fn: Callable[[], T],
        close_fn: Callable[[T], None],
        validate_fn: Optional[Callable[[T], bool]] = None,
        min_size: int = 1,
        max_size: int = 10,
    ):
        """
        Initialize connection pool.
        
        Args:
            create_fn: Function to create a new connection
            close_fn: Function to close a connection
            validate_fn: Function to validate connection health
            min_size: Minimum pool size
            max_size: Maximum pool size
        """
        self.create_fn = create_fn
        self.close_fn = close_fn
        self.validate_fn = validate_fn or (lambda c: True)
        self.min_size = min_size
        self.max_size = max_size
        
        self._pool: List[T] = []
        self._in_use: List[T] = []
        
        # Thread-safe lock (always available)
        import threading
        self._thread_lock = threading.Lock()
        
        # Async lock (created lazily when needed)
        self._async_lock: Optional[asyncio.Lock] = None
        
        # Initialize minimum connections
        self._initialize()
    
    # ---------------------------------------------------------------------------
    # ID           : utils.persistence_utils.ConnectionPool._initialize
    # Requirement  : `_initialize` shall create initial connections
    # Purpose      : Create initial connections
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : None
    # Outputs      : None
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
    def _initialize(self) -> None:
        """Create initial connections."""
        for _ in range(self.min_size):
            try:
                conn = self.create_fn()
                self._pool.append(conn)
            except Exception as e:
                logger.warning(f"Failed to create initial connection: {e}")
    
    # ---------------------------------------------------------------------------
    # ID           : utils.persistence_utils.ConnectionPool.acquire
    # Requirement  : `acquire` shall acquire a connection from the pool
    # Purpose      : Acquire a connection from the pool
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : None
    # Outputs      : T
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
    def acquire(self) -> T:
        """
        Acquire a connection from the pool.
        
        Returns:
            A valid connection
        """
        with self._thread_lock:
            # Try to get from pool
            while self._pool:
                conn = self._pool.pop()
                if self.validate_fn(conn):
                    self._in_use.append(conn)
                    return conn
                else:
                    # Connection invalid, close it
                    try:
                        self.close_fn(conn)
                    except Exception:
                        pass
            
            # Create new if under max
            if len(self._in_use) < self.max_size:
                conn = self.create_fn()
                self._in_use.append(conn)
                return conn
            
            raise RuntimeError("Connection pool exhausted")
    
    # ---------------------------------------------------------------------------
    # ID           : utils.persistence_utils.ConnectionPool.release
    # Requirement  : `release` shall release a connection back to the pool
    # Purpose      : Release a connection back to the pool
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : conn: T
    # Outputs      : None
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
    def release(self, conn: T) -> None:
        """
        Release a connection back to the pool.
        
        Args:
            conn: Connection to release
        """
        with self._thread_lock:
            if conn in self._in_use:
                self._in_use.remove(conn)
                
                if self.validate_fn(conn) and len(self._pool) < self.max_size:
                    self._pool.append(conn)
                else:
                    try:
                        self.close_fn(conn)
                    except Exception:
                        pass
    
    # ---------------------------------------------------------------------------
    # ID           : utils.persistence_utils.ConnectionPool.connection
    # Requirement  : `connection` shall context manager for acquiring a connection
    # Purpose      : Context manager for acquiring a connection
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : None
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
    @contextmanager
    def connection(self):
        """
        Context manager for acquiring a connection.
        
        Yields:
            Connection from pool
        """
        conn = self.acquire()
        try:
            yield conn
        finally:
            self.release(conn)
    
    # ---------------------------------------------------------------------------
    # ID           : utils.persistence_utils.ConnectionPool.close_all
    # Requirement  : `close_all` shall close all connections in the pool
    # Purpose      : Close all connections in the pool
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : None
    # Outputs      : None
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
    def close_all(self) -> None:
        """Close all connections in the pool."""
        with self._thread_lock:
            for conn in self._pool + self._in_use:
                try:
                    self.close_fn(conn)
                except Exception:
                    pass
            self._pool.clear()
            self._in_use.clear()
    
    # ---------------------------------------------------------------------------
    # ID           : utils.persistence_utils.ConnectionPool.stats
    # Requirement  : `stats` shall get pool statistics
    # Purpose      : Get pool statistics
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : None
    # Outputs      : Dict[str, int]
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
    def stats(self) -> Dict[str, int]:
        """Get pool statistics."""
        return {
            "available": len(self._pool),
            "in_use": len(self._in_use),
            "total": len(self._pool) + len(self._in_use),
            "max_size": self.max_size,
        }


# REQ-DAT-001: Write-ahead logging for crash recovery
# ---------------------------------------------------------------------------
# ID           : utils.persistence_utils.WriteAheadLog
# Requirement  : `WriteAheadLog` class shall be instantiable and expose the documented interface
# Purpose      : Simple write-ahead log for crash recovery
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
# Verification : Instantiate WriteAheadLog with valid args; assert attribute types and values
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
class WriteAheadLog:
    """
    Simple write-ahead log for crash recovery.
    
    Records operations before they are applied, allowing
    recovery of incomplete operations after a crash.
    """
    
    # ---------------------------------------------------------------------------
    # ID           : utils.persistence_utils.WriteAheadLog.__init__
    # Requirement  : `__init__` shall initialize write-ahead log
    # Purpose      : Initialize write-ahead log
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : log_dir: Union[str, Path]
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
    def __init__(self, log_dir: Union[str, Path]):
        """
        Initialize write-ahead log.
        
        Args:
            log_dir: Directory for log files
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._current_log: Optional[Path] = None
    
    # ---------------------------------------------------------------------------
    # ID           : utils.persistence_utils.WriteAheadLog._get_log_path
    # Requirement  : `_get_log_path` shall get current log file path
    # Purpose      : Get current log file path
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : None
    # Outputs      : Path
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
    def _get_log_path(self) -> Path:
        """Get current log file path."""
        return self.log_dir / f"wal_{datetime.now().strftime('%Y%m%d')}.log"
    
    # ---------------------------------------------------------------------------
    # ID           : utils.persistence_utils.WriteAheadLog.begin_transaction
    # Requirement  : `begin_transaction` shall begin a new transaction
    # Purpose      : Begin a new transaction
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : transaction_id: str
    # Outputs      : None
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
    def begin_transaction(self, transaction_id: str) -> None:
        """
        Begin a new transaction.
        
        Args:
            transaction_id: Unique identifier for the transaction
        """
        self._append_entry({
            "type": "BEGIN",
            "transaction_id": transaction_id,
            "timestamp": datetime.now().isoformat(),
        })
    
    # ---------------------------------------------------------------------------
    # ID           : utils.persistence_utils.WriteAheadLog.log_operation
    # Requirement  : `log_operation` shall log an operation within a transaction
    # Purpose      : Log an operation within a transaction
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : transaction_id: str; operation: str; data: Dict[str, Any]
    # Outputs      : None
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
    def log_operation(
        self,
        transaction_id: str,
        operation: str,
        data: Dict[str, Any],
    ) -> None:
        """
        Log an operation within a transaction.
        
        Args:
            transaction_id: Transaction identifier
            operation: Operation type (e.g., "INSERT", "UPDATE")
            data: Operation data
        """
        self._append_entry({
            "type": "OPERATION",
            "transaction_id": transaction_id,
            "operation": operation,
            "data": data,
            "timestamp": datetime.now().isoformat(),
        })
    
    # ---------------------------------------------------------------------------
    # ID           : utils.persistence_utils.WriteAheadLog.commit_transaction
    # Requirement  : `commit_transaction` shall commit a transaction
    # Purpose      : Commit a transaction
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : transaction_id: str
    # Outputs      : None
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
    def commit_transaction(self, transaction_id: str) -> None:
        """
        Commit a transaction.
        
        Args:
            transaction_id: Transaction identifier
        """
        self._append_entry({
            "type": "COMMIT",
            "transaction_id": transaction_id,
            "timestamp": datetime.now().isoformat(),
        })
    
    # ---------------------------------------------------------------------------
    # ID           : utils.persistence_utils.WriteAheadLog.rollback_transaction
    # Requirement  : `rollback_transaction` shall rollback a transaction
    # Purpose      : Rollback a transaction
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : transaction_id: str
    # Outputs      : None
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
    def rollback_transaction(self, transaction_id: str) -> None:
        """
        Rollback a transaction.
        
        Args:
            transaction_id: Transaction identifier
        """
        self._append_entry({
            "type": "ROLLBACK",
            "transaction_id": transaction_id,
            "timestamp": datetime.now().isoformat(),
        })
    
    # ---------------------------------------------------------------------------
    # ID           : utils.persistence_utils.WriteAheadLog._append_entry
    # Requirement  : `_append_entry` shall append an entry to the log
    # Purpose      : Append an entry to the log
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : entry: Dict[str, Any]
    # Outputs      : None
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
    def _append_entry(self, entry: Dict[str, Any]) -> None:
        """Append an entry to the log."""
        log_path = self._get_log_path()
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    
    # ---------------------------------------------------------------------------
    # ID           : utils.persistence_utils.WriteAheadLog.get_uncommitted_transactions
    # Requirement  : `get_uncommitted_transactions` shall get list of uncommitted transactions for recovery
    # Purpose      : Get list of uncommitted transactions for recovery
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : None
    # Outputs      : List[str]
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
    def get_uncommitted_transactions(self) -> List[str]:
        """
        Get list of uncommitted transactions for recovery.
        
        Returns:
            List of transaction IDs that were started but not committed
        """
        begun: set = set()
        committed: set = set()
        rolledback: set = set()
        
        for log_path in sorted(self.log_dir.glob("wal_*.log")):
            try:
                with open(log_path, "r", encoding="utf-8") as f:
                    for line in f:
                        entry = json.loads(line.strip())
                        tid = entry.get("transaction_id")
                        
                        if entry["type"] == "BEGIN":
                            begun.add(tid)
                        elif entry["type"] == "COMMIT":
                            committed.add(tid)
                        elif entry["type"] == "ROLLBACK":
                            rolledback.add(tid)
            except Exception as e:
                logger.warning(f"Failed to read log {log_path}: {e}")
        
        return list(begun - committed - rolledback)
    
    # ---------------------------------------------------------------------------
    # ID           : utils.persistence_utils.WriteAheadLog.cleanup_old_logs
    # Requirement  : `cleanup_old_logs` shall remove log files older than specified days
    # Purpose      : Remove log files older than specified days
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : days: int (default=7)
    # Outputs      : int
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
    def cleanup_old_logs(self, days: int = 7) -> int:
        """
        Remove log files older than specified days.
        
        Args:
            days: Number of days to retain
            
        Returns:
            Number of files removed
        """
        cutoff = time.time() - (days * 86400)
        removed = 0
        
        for log_path in self.log_dir.glob("wal_*.log"):
            if log_path.stat().st_mtime < cutoff:
                log_path.unlink()
                removed += 1
        
        return removed


# Convenience exports
__all__ = [
    "PersistenceStatus",
    "PersistenceResult",
    "BackupInfo",
    "calculate_checksum",
    "verify_checksum",
    "with_persistence_retry",
    "with_async_persistence_retry",
    "BackupManager",
    "atomic_write",
    "async_atomic_write",
    "JsonPersistence",
    "ConnectionPool",
    "WriteAheadLog",
]
