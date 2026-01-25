"""
Unit Tests for Persistence Utilities

Tests backup, retry logic, atomic writes, and data integrity features.

Requirements Tested:
- REQ-DAT-001: Data persistence operations
- REQ-DAT-002: Database retry logic
- REQ-DAT-003: Backup and recovery
- REQ-REL-003: Data integrity verification
"""

import pytest
import json
import time
import tempfile
import threading
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch

from eeg_rag.utils.persistence_utils import (
    PersistenceStatus,
    PersistenceResult,
    BackupInfo,
    calculate_checksum,
    verify_checksum,
    with_persistence_retry,
    BackupManager,
    atomic_write,
    JsonPersistence,
    ConnectionPool,
    WriteAheadLog,
)


class TestPersistenceStatus:
    """Test PersistenceStatus class"""
    
    def test_status_values(self):
        """REQ-DAT-001: Status values exist"""
        assert PersistenceStatus.SUCCESS == "success"
        assert PersistenceStatus.FAILED == "failed"
        assert PersistenceStatus.PENDING == "pending"
        assert PersistenceStatus.RETRYING == "retrying"
        assert PersistenceStatus.RECOVERED == "recovered"
        assert PersistenceStatus.CORRUPTED == "corrupted"


class TestPersistenceResult:
    """Test PersistenceResult dataclass"""
    
    def test_success_result(self):
        """REQ-DAT-001: Create success result"""
        result = PersistenceResult(
            success=True,
            data={"key": "value"},
            attempts=1,
            duration_ms=10.5,
        )
        assert result.success is True
        assert result.data == {"key": "value"}
        assert result.error is None
    
    def test_failure_result(self):
        """REQ-DAT-001: Create failure result"""
        result = PersistenceResult(
            success=False,
            error="Connection timeout",
            attempts=3,
            duration_ms=5000.0,
        )
        assert result.success is False
        assert result.error == "Connection timeout"
        assert result.attempts == 3
    
    def test_to_dict(self):
        """REQ-DAT-001: Serialize result"""
        result = PersistenceResult(
            success=True,
            attempts=2,
            duration_ms=100.0,
            recovered=True,
        )
        d = result.to_dict()
        assert d['success'] is True
        assert d['attempts'] == 2
        assert d['recovered'] is True


class TestBackupInfo:
    """Test BackupInfo dataclass"""
    
    def test_backup_info_creation(self):
        """REQ-DAT-003: Create backup info"""
        info = BackupInfo(
            path=Path("/tmp/backup.json"),
            timestamp=datetime.now(),
            size_bytes=1024,
            checksum="abc123",
        )
        assert info.size_bytes == 1024
        assert info.checksum == "abc123"
    
    def test_to_dict(self):
        """REQ-DAT-003: Serialize backup info"""
        now = datetime.now()
        info = BackupInfo(
            path=Path("/tmp/backup.json"),
            timestamp=now,
            size_bytes=1024,
            checksum="abc123",
        )
        d = info.to_dict()
        assert d['path'] == "/tmp/backup.json"
        assert d['size_bytes'] == 1024


class TestChecksum:
    """Test checksum functions"""
    
    def test_checksum_string(self):
        """REQ-REL-003: Calculate checksum for string"""
        checksum = calculate_checksum("hello world")
        assert len(checksum) == 64  # SHA256 hex
        assert checksum.isalnum()
    
    def test_checksum_bytes(self):
        """REQ-REL-003: Calculate checksum for bytes"""
        checksum = calculate_checksum(b"hello world")
        assert len(checksum) == 64
    
    def test_checksum_file(self, tmp_path):
        """REQ-REL-003: Calculate checksum for file"""
        file_path = tmp_path / "test.txt"
        file_path.write_text("hello world")
        checksum = calculate_checksum(file_path)
        assert len(checksum) == 64
    
    def test_checksum_consistency(self):
        """REQ-REL-003: Checksums are consistent"""
        data = "test data 12345"
        cs1 = calculate_checksum(data)
        cs2 = calculate_checksum(data)
        assert cs1 == cs2
    
    def test_verify_checksum_valid(self):
        """REQ-REL-003: Verify valid checksum"""
        data = "test data"
        checksum = calculate_checksum(data)
        assert verify_checksum(data, checksum) is True
    
    def test_verify_checksum_invalid(self):
        """REQ-REL-003: Detect invalid checksum"""
        data = "test data"
        assert verify_checksum(data, "wrong_checksum") is False


class TestPersistenceRetry:
    """Test with_persistence_retry decorator"""
    
    def test_success_on_first_try(self):
        """REQ-DAT-002: Returns result on success"""
        @with_persistence_retry(max_attempts=3)
        def successful_op():
            return "result"
        
        result = successful_op()
        assert result.success is True
        assert result.data == "result"
        assert result.attempts == 1
    
    def test_retry_on_failure(self):
        """REQ-DAT-002: Retries on retryable exception"""
        call_count = 0
        
        @with_persistence_retry(max_attempts=3, delay_seconds=0.01)
        def failing_then_success():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise IOError("Temporary failure")
            return "success"
        
        result = failing_then_success()
        assert result.success is True
        assert result.attempts == 3
        assert call_count == 3
    
    def test_max_attempts_exceeded(self):
        """REQ-DAT-002: Fails after max attempts"""
        @with_persistence_retry(max_attempts=2, delay_seconds=0.01)
        def always_fails():
            raise IOError("Persistent failure")
        
        result = always_fails()
        assert result.success is False
        assert result.attempts == 2
        assert "Persistent failure" in result.error
    
    def test_on_retry_callback(self):
        """REQ-DAT-002: Callback called on retry"""
        retry_calls = []
        
        def on_retry(attempt, exc):
            retry_calls.append((attempt, str(exc)))
        
        @with_persistence_retry(
            max_attempts=3,
            delay_seconds=0.01,
            on_retry=on_retry,
        )
        def fails_twice():
            if len(retry_calls) < 2:
                raise IOError("Temporary error")
            return "ok"
        
        fails_twice()
        assert len(retry_calls) == 2
    
    def test_non_retryable_exception(self):
        """REQ-DAT-002: Non-retryable exception not caught"""
        @with_persistence_retry(
            max_attempts=3,
            retryable_exceptions=(IOError,),
        )
        def raises_value_error():
            raise ValueError("Not retryable")
        
        with pytest.raises(ValueError):
            raises_value_error()


class TestAtomicWrite:
    """Test atomic_write context manager"""
    
    def test_atomic_write_text(self, tmp_path):
        """REQ-DAT-001: Atomic text write"""
        file_path = tmp_path / "test.txt"
        
        with atomic_write(file_path) as f:
            f.write("Hello, World!")
        
        assert file_path.exists()
        assert file_path.read_text() == "Hello, World!"
    
    def test_atomic_write_binary(self, tmp_path):
        """REQ-DAT-001: Atomic binary write"""
        file_path = tmp_path / "test.bin"
        
        with atomic_write(file_path, mode="wb") as f:
            f.write(b"binary data")
        
        assert file_path.read_bytes() == b"binary data"
    
    def test_atomic_write_cleanup_on_error(self, tmp_path):
        """REQ-DAT-001: Cleanup temp file on error"""
        file_path = tmp_path / "test.txt"
        temp_path = file_path.with_suffix(".txt.tmp")
        
        with pytest.raises(ValueError):
            with atomic_write(file_path) as f:
                f.write("partial")
                raise ValueError("Error during write")
        
        assert not file_path.exists()
        assert not temp_path.exists()


class TestBackupManager:
    """Test BackupManager class"""
    
    @pytest.fixture
    def backup_manager(self, tmp_path):
        """Create backup manager with temp directory"""
        backup_dir = tmp_path / "backups"
        return BackupManager(backup_dir, max_backups=3)
    
    @pytest.fixture
    def source_file(self, tmp_path):
        """Create a source file to backup"""
        source = tmp_path / "source.json"
        source.write_text('{"key": "value"}')
        return source
    
    def test_create_backup(self, backup_manager, source_file):
        """REQ-DAT-003: Create backup"""
        info = backup_manager.create_backup(source_file, prefix="test")
        
        assert info.path.exists()
        assert info.size_bytes > 0
        assert len(info.checksum) == 64
    
    def test_backup_rotation(self, backup_manager, source_file):
        """REQ-DAT-003: Rotate old backups"""
        for i in range(5):
            backup_manager.create_backup(source_file, prefix="test")
            time.sleep(0.01)  # Ensure different timestamps
        
        backups = backup_manager.list_backups("test")
        assert len(backups) <= 3  # max_backups=3
    
    def test_list_backups(self, backup_manager, source_file):
        """REQ-DAT-003: List backups"""
        backup_manager.create_backup(source_file, prefix="test")
        time.sleep(1.01)  # Ensure different timestamp (second precision)
        backup_manager.create_backup(source_file, prefix="test")
        
        backups = backup_manager.list_backups("test")
        assert len(backups) == 2
        # Should be sorted newest first
        assert backups[0].timestamp >= backups[1].timestamp
    
    def test_restore_backup(self, backup_manager, source_file, tmp_path):
        """REQ-DAT-003: Restore from backup"""
        info = backup_manager.create_backup(source_file, prefix="test")
        
        # Delete original
        source_file.unlink()
        assert not source_file.exists()
        
        # Restore
        result = backup_manager.restore_backup(info, source_file)
        
        assert result.success is True
        assert result.recovered is True
        assert source_file.exists()
        assert source_file.read_text() == '{"key": "value"}'
    
    def test_restore_corrupted_backup(self, backup_manager, source_file, tmp_path):
        """REQ-DAT-003: Detect corrupted backup on restore"""
        info = backup_manager.create_backup(source_file, prefix="test")
        
        # Corrupt the backup
        info.path.write_text("corrupted")
        
        # Restore should fail
        dest = tmp_path / "restored.json"
        result = backup_manager.restore_backup(info, dest)
        
        assert result.success is False
        assert "corrupted" in result.error.lower()
    
    def test_cleanup_backups(self, backup_manager, source_file):
        """REQ-DAT-003: Cleanup all backups"""
        backup_manager.create_backup(source_file, prefix="test")
        time.sleep(1.01)  # Ensure different timestamp
        backup_manager.create_backup(source_file, prefix="test")
        
        count = backup_manager.cleanup()
        assert count == 2
    
    def test_backup_missing_source(self, backup_manager):
        """REQ-DAT-003: Backup of missing file raises error"""
        with pytest.raises(FileNotFoundError):
            backup_manager.create_backup(Path("/nonexistent.txt"))


class TestJsonPersistence:
    """Test JsonPersistence class"""
    
    @pytest.fixture
    def json_persistence(self, tmp_path):
        """Create JSON persistence with temp file"""
        json_path = tmp_path / "data.json"
        return JsonPersistence(json_path)
    
    def test_save_and_load(self, json_persistence):
        """REQ-DAT-001: Save and load JSON data"""
        data = {"name": "test", "values": [1, 2, 3]}
        
        result = json_persistence.save(data)
        assert result.success is True
        
        loaded_result = json_persistence.load()
        assert loaded_result.success is True
        assert loaded_result.data == data
    
    def test_checksum_verification(self, json_persistence):
        """REQ-DAT-001: Checksum verification on load"""
        data = {"key": "value"}
        json_persistence.save(data)
        
        # Corrupt the file
        json_persistence.path.write_text('{"key": "modified"}')
        
        # Load should fail verification (ValueError raised, caught by retry decorator)
        # The ValueError is not a retryable exception, so it propagates
        with pytest.raises(ValueError, match="corrupted"):
            json_persistence.load(verify=True)
    
    def test_load_nonexistent(self, json_persistence):
        """REQ-DAT-001: Load nonexistent file fails"""
        result = json_persistence.load()
        assert result.success is False
    
    def test_exists(self, json_persistence):
        """REQ-DAT-001: Check file existence"""
        assert json_persistence.exists() is False
        
        json_persistence.save({"key": "value"})
        assert json_persistence.exists() is True
    
    def test_delete(self, json_persistence):
        """REQ-DAT-001: Delete file"""
        json_persistence.save({"key": "value"})
        assert json_persistence.path.exists()
        
        deleted = json_persistence.delete()
        assert deleted is True
        assert not json_persistence.path.exists()
    
    def test_with_backup_manager(self, tmp_path):
        """REQ-DAT-001: JSON persistence with backup"""
        backup_dir = tmp_path / "backups"
        backup_manager = BackupManager(backup_dir, max_backups=3)
        
        json_path = tmp_path / "data.json"
        persistence = JsonPersistence(json_path, backup_manager=backup_manager)
        
        # First save
        persistence.save({"version": 1})
        
        # Second save should create backup
        persistence.save({"version": 2})
        
        backups = backup_manager.list_backups("data")
        assert len(backups) >= 1


class TestConnectionPool:
    """Test ConnectionPool class"""
    
    def test_pool_creation(self):
        """REQ-DAT-002: Create connection pool"""
        pool = ConnectionPool(
            create_fn=lambda: "connection",
            close_fn=lambda c: None,
            min_size=2,
            max_size=5,
        )
        
        stats = pool.stats()
        assert stats['available'] == 2  # min_size
        assert stats['in_use'] == 0
    
    def test_acquire_and_release(self):
        """REQ-DAT-002: Acquire and release connections"""
        pool = ConnectionPool(
            create_fn=lambda: {"id": id(object())},
            close_fn=lambda c: None,
            min_size=1,
            max_size=5,
        )
        
        conn1 = pool.acquire()
        assert conn1 is not None
        
        stats = pool.stats()
        assert stats['in_use'] == 1
        
        pool.release(conn1)
        stats = pool.stats()
        assert stats['available'] >= 1
        assert stats['in_use'] == 0
    
    def test_connection_context_manager(self):
        """REQ-DAT-002: Connection context manager"""
        pool = ConnectionPool(
            create_fn=lambda: "conn",
            close_fn=lambda c: None,
            min_size=1,
            max_size=5,
        )
        
        with pool.connection() as conn:
            assert conn == "conn"
            stats = pool.stats()
            assert stats['in_use'] == 1
        
        stats = pool.stats()
        assert stats['in_use'] == 0
    
    def test_pool_exhaustion(self):
        """REQ-DAT-002: Pool exhaustion raises error"""
        pool = ConnectionPool(
            create_fn=lambda: "conn",
            close_fn=lambda c: None,
            min_size=0,
            max_size=2,
        )
        
        conn1 = pool.acquire()
        conn2 = pool.acquire()
        
        with pytest.raises(RuntimeError, match="exhausted"):
            pool.acquire()
        
        pool.release(conn1)
        pool.release(conn2)
    
    def test_close_all(self):
        """REQ-DAT-002: Close all connections"""
        close_calls = []
        
        pool = ConnectionPool(
            create_fn=lambda: "conn",
            close_fn=lambda c: close_calls.append(c),
            min_size=2,
            max_size=5,
        )
        
        pool.close_all()
        assert len(close_calls) == 2
    
    def test_validation_function(self):
        """REQ-DAT-002: Validation function filters connections"""
        conn_count = [0]
        
        def create():
            conn_count[0] += 1
            return {"id": conn_count[0]}
        
        # Invalidate connections with id < 2
        pool = ConnectionPool(
            create_fn=create,
            close_fn=lambda c: None,
            validate_fn=lambda c: c["id"] >= 2,
            min_size=1,
            max_size=5,
        )
        
        # First connection is invalid, should create new
        conn = pool.acquire()
        assert conn["id"] >= 2


class TestWriteAheadLog:
    """Test WriteAheadLog class"""
    
    @pytest.fixture
    def wal(self, tmp_path):
        """Create WAL with temp directory"""
        return WriteAheadLog(tmp_path / "wal")
    
    def test_transaction_lifecycle(self, wal):
        """REQ-DAT-001: Transaction begin, log, commit"""
        wal.begin_transaction("tx1")
        wal.log_operation("tx1", "INSERT", {"key": "value"})
        wal.commit_transaction("tx1")
        
        uncommitted = wal.get_uncommitted_transactions()
        assert "tx1" not in uncommitted
    
    def test_uncommitted_transactions(self, wal):
        """REQ-DAT-001: Detect uncommitted transactions"""
        wal.begin_transaction("tx1")
        wal.log_operation("tx1", "INSERT", {"key": "value"})
        # No commit
        
        uncommitted = wal.get_uncommitted_transactions()
        assert "tx1" in uncommitted
    
    def test_rollback_transaction(self, wal):
        """REQ-DAT-001: Rollback removes from uncommitted"""
        wal.begin_transaction("tx1")
        wal.log_operation("tx1", "INSERT", {"key": "value"})
        wal.rollback_transaction("tx1")
        
        uncommitted = wal.get_uncommitted_transactions()
        assert "tx1" not in uncommitted
    
    def test_multiple_transactions(self, wal):
        """REQ-DAT-001: Multiple concurrent transactions"""
        wal.begin_transaction("tx1")
        wal.begin_transaction("tx2")
        wal.begin_transaction("tx3")
        
        wal.commit_transaction("tx1")
        wal.rollback_transaction("tx2")
        # tx3 left uncommitted
        
        uncommitted = wal.get_uncommitted_transactions()
        assert "tx1" not in uncommitted
        assert "tx2" not in uncommitted
        assert "tx3" in uncommitted


class TestThreadSafety:
    """Test thread safety of persistence utilities"""
    
    def test_connection_pool_thread_safe(self):
        """REQ-DAT-002: Connection pool thread safety"""
        pool = ConnectionPool(
            create_fn=lambda: {"id": id(object())},
            close_fn=lambda c: None,
            min_size=2,
            max_size=10,
        )
        errors = []
        
        def worker():
            try:
                for _ in range(10):
                    conn = pool.acquire()
                    time.sleep(0.001)
                    pool.release(conn)
            except Exception as e:
                errors.append(e)
        
        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(errors) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
