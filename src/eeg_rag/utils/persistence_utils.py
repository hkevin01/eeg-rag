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
class PersistenceStatus:
    """Status codes for persistence operations."""
    SUCCESS = "success"
    FAILED = "failed"
    PENDING = "pending"
    RETRYING = "retrying"
    RECOVERED = "recovered"
    CORRUPTED = "corrupted"


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
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "success": self.success,
            "error": self.error,
            "attempts": self.attempts,
            "duration_ms": self.duration_ms,
            "recovered": self.recovered,
        }


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
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "path": str(self.path),
            "timestamp": self.timestamp.isoformat(),
            "size_bytes": self.size_bytes,
            "checksum": self.checksum,
        }


# REQ-REL-003: Checksum calculation
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
    def decorator(func: Callable) -> Callable:
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
    def decorator(func: Callable) -> Callable:
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
class BackupManager:
    """
    Manages file backups with rotation and integrity verification.
    
    Attributes:
        backup_dir: Directory for storing backups
        max_backups: Maximum number of backups to retain
    """
    
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
    
    def _ensure_dir(self) -> None:
        """Ensure backup directory exists."""
        self.backup_dir.mkdir(parents=True, exist_ok=True)
    
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
class JsonPersistence:
    """
    JSON file persistence with integrity verification and backup.
    
    Provides safe read/write operations with automatic checksums.
    """
    
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
    
    def exists(self) -> bool:
        """Check if the JSON file exists."""
        return self.path.exists()
    
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
class ConnectionPool(Generic[T]):
    """
    Generic connection pool for database connections.
    
    Manages a pool of reusable connections with health checking.
    """
    
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
    
    def _initialize(self) -> None:
        """Create initial connections."""
        for _ in range(self.min_size):
            try:
                conn = self.create_fn()
                self._pool.append(conn)
            except Exception as e:
                logger.warning(f"Failed to create initial connection: {e}")
    
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
    
    def stats(self) -> Dict[str, int]:
        """Get pool statistics."""
        return {
            "available": len(self._pool),
            "in_use": len(self._in_use),
            "total": len(self._pool) + len(self._in_use),
            "max_size": self.max_size,
        }


# REQ-DAT-001: Write-ahead logging for crash recovery
class WriteAheadLog:
    """
    Simple write-ahead log for crash recovery.
    
    Records operations before they are applied, allowing
    recovery of incomplete operations after a crash.
    """
    
    def __init__(self, log_dir: Union[str, Path]):
        """
        Initialize write-ahead log.
        
        Args:
            log_dir: Directory for log files
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._current_log: Optional[Path] = None
    
    def _get_log_path(self) -> Path:
        """Get current log file path."""
        return self.log_dir / f"wal_{datetime.now().strftime('%Y%m%d')}.log"
    
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
    
    def _append_entry(self, entry: Dict[str, Any]) -> None:
        """Append an entry to the log."""
        log_path = self._get_log_path()
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    
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
