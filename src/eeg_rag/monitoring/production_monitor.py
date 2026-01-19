#!/usr/bin/env python3
"""
Production Monitoring for EEG-RAG

Comprehensive monitoring including:
- Performance metrics
- Health checks
- Error tracking
- Resource utilization
- Business metrics
"""

import time
import psutil
import asyncio
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from pathlib import Path
import json
from collections import defaultdict, deque

try:
    from prometheus_client import Counter, Histogram, Gauge, start_http_server, CollectorRegistry
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    Counter = Histogram = Gauge = None

try:
    import sentry_sdk
    SENTRY_AVAILABLE = True
except ImportError:
    SENTRY_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class SystemMetrics:
    """System resource metrics."""
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_available_mb: float
    disk_usage_percent: float
    disk_free_gb: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ApplicationMetrics:
    """Application-specific metrics."""
    active_connections: int
    total_requests: int
    requests_per_second: float
    average_response_time: float
    error_rate: float
    cache_hit_rate: float
    query_success_rate: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class HealthStatus:
    """System health status."""
    status: str  # healthy, degraded, unhealthy
    checks: Dict[str, Dict[str, Any]]
    overall_score: float
    timestamp: datetime = field(default_factory=datetime.now)


class ProductionMonitor:
    """Production monitoring system for EEG-RAG."""
    
    def __init__(
        self,
        metrics_retention_hours: int = 24,
        health_check_interval: int = 30,
        prometheus_port: Optional[int] = None,
        sentry_dsn: Optional[str] = None
    ):
        """Initialize production monitor.
        
        Args:
            metrics_retention_hours: How long to keep metrics in memory.
            health_check_interval: Interval between health checks in seconds.
            prometheus_port: Port for Prometheus metrics server.
            sentry_dsn: Sentry DSN for error tracking.
        """
        self.retention_hours = metrics_retention_hours
        self.health_check_interval = health_check_interval
        
        # Metrics storage
        self.system_metrics: deque = deque(maxlen=metrics_retention_hours * 120)  # 30s intervals
        self.app_metrics: deque = deque(maxlen=metrics_retention_hours * 120)
        self.health_history: deque = deque(maxlen=metrics_retention_hours * 120)
        
        # Request tracking
        self.request_times: deque = deque(maxlen=1000)
        self.error_count = Counter()
        self.request_count = 0
        self.last_request_time = time.time()
        
        # Cache metrics
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Query metrics
        self.query_success = 0
        self.query_failures = 0
        
        # Health check registry
        self.health_checks = {}
        
        # Initialize Prometheus if available
        self.prometheus_registry = None
        if PROMETHEUS_AVAILABLE and prometheus_port:
            self._setup_prometheus(prometheus_port)
        
        # Initialize Sentry if available
        if SENTRY_AVAILABLE and sentry_dsn:
            self._setup_sentry(sentry_dsn)
        
        # Monitoring task
        self._monitoring_task = None
        self._running = False
    
    def _setup_prometheus(self, port: int):
        """Setup Prometheus metrics collection."""
        try:
            self.prometheus_registry = CollectorRegistry()
            
            # System metrics
            self.prometheus_cpu = Gauge('eeg_rag_cpu_percent', 'CPU usage percentage', registry=self.prometheus_registry)
            self.prometheus_memory = Gauge('eeg_rag_memory_percent', 'Memory usage percentage', registry=self.prometheus_registry)
            self.prometheus_disk = Gauge('eeg_rag_disk_percent', 'Disk usage percentage', registry=self.prometheus_registry)
            
            # Application metrics
            self.prometheus_requests = Counter('eeg_rag_requests_total', 'Total requests', registry=self.prometheus_registry)
            self.prometheus_errors = Counter('eeg_rag_errors_total', 'Total errors', ['error_type'], registry=self.prometheus_registry)
            self.prometheus_response_time = Histogram('eeg_rag_response_time_seconds', 'Response time', registry=self.prometheus_registry)
            self.prometheus_cache_hits = Counter('eeg_rag_cache_hits_total', 'Cache hits', registry=self.prometheus_registry)
            self.prometheus_cache_misses = Counter('eeg_rag_cache_misses_total', 'Cache misses', registry=self.prometheus_registry)
            
            # Start metrics server
            start_http_server(port, registry=self.prometheus_registry)
            logger.info(f"Prometheus metrics server started on port {port}")
            
        except Exception as e:
            logger.error(f"Failed to setup Prometheus: {str(e)}")
    
    def _setup_sentry(self, dsn: str):
        """Setup Sentry error tracking."""
        try:
            sentry_sdk.init(
                dsn=dsn,
                traces_sample_rate=0.1,
                environment='production'
            )
            logger.info("Sentry error tracking initialized")
        except Exception as e:
            logger.error(f"Failed to setup Sentry: {str(e)}")
    
    async def start_monitoring(self):
        """Start the monitoring background task."""
        if self._running:
            return
        
        self._running = True
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Production monitoring started")
    
    async def stop_monitoring(self):
        """Stop the monitoring background task."""
        self._running = False
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        logger.info("Production monitoring stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop."""
        while self._running:
            try:
                # Collect metrics
                system_metrics = await self._collect_system_metrics()
                app_metrics = await self._collect_app_metrics()
                health_status = await self._run_health_checks()
                
                # Store metrics
                self.system_metrics.append(system_metrics)
                self.app_metrics.append(app_metrics)
                self.health_history.append(health_status)
                
                # Update Prometheus if available
                if self.prometheus_registry:
                    self._update_prometheus_metrics(system_metrics, app_metrics)
                
                # Log critical issues
                if health_status.status == 'unhealthy':
                    logger.warning(f"System unhealthy: {health_status.checks}")
                
                await asyncio.sleep(self.health_check_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {str(e)}")
                await asyncio.sleep(self.health_check_interval)
    
    async def _collect_system_metrics(self) -> SystemMetrics:
        """Collect system resource metrics."""
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=None)
        
        # Memory usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_used_mb = memory.used / (1024 * 1024)
        memory_available_mb = memory.available / (1024 * 1024)
        
        # Disk usage
        disk = psutil.disk_usage('/')
        disk_usage_percent = (disk.used / disk.total) * 100
        disk_free_gb = disk.free / (1024 * 1024 * 1024)
        
        return SystemMetrics(
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            memory_used_mb=memory_used_mb,
            memory_available_mb=memory_available_mb,
            disk_usage_percent=disk_usage_percent,
            disk_free_gb=disk_free_gb
        )
    
    async def _collect_app_metrics(self) -> ApplicationMetrics:
        """Collect application metrics."""
        current_time = time.time()
        
        # Calculate requests per second
        time_window = 60  # 1 minute window
        recent_requests = sum(1 for req_time in self.request_times 
                            if current_time - req_time < time_window)
        requests_per_second = recent_requests / time_window
        
        # Calculate average response time
        recent_response_times = [rt for rt in self.request_times 
                               if current_time - rt < time_window]
        avg_response_time = (sum(recent_response_times) / len(recent_response_times) 
                           if recent_response_times else 0)
        
        # Calculate error rate
        total_recent_requests = len(recent_response_times)
        recent_errors = sum(count for error_type, count in self.error_count.items())
        error_rate = (recent_errors / max(total_recent_requests, 1)) * 100
        
        # Calculate cache hit rate
        total_cache_requests = self.cache_hits + self.cache_misses
        cache_hit_rate = (self.cache_hits / max(total_cache_requests, 1)) * 100
        
        # Calculate query success rate
        total_queries = self.query_success + self.query_failures
        query_success_rate = (self.query_success / max(total_queries, 1)) * 100
        
        return ApplicationMetrics(
            active_connections=0,  # Would need web server integration
            total_requests=self.request_count,
            requests_per_second=requests_per_second,
            average_response_time=avg_response_time,
            error_rate=error_rate,
            cache_hit_rate=cache_hit_rate,
            query_success_rate=query_success_rate
        )
    
    async def _run_health_checks(self) -> HealthStatus:
        """Run all registered health checks."""
        checks = {}
        total_score = 0
        max_score = 0
        
        for name, check_func in self.health_checks.items():
            try:
                result = await check_func()
                checks[name] = result
                total_score += result.get('score', 0)
                max_score += result.get('max_score', 100)
            except Exception as e:
                checks[name] = {
                    'status': 'error',
                    'message': str(e),
                    'score': 0,
                    'max_score': 100
                }
                max_score += 100
        
        # Calculate overall score
        overall_score = (total_score / max(max_score, 1)) * 100
        
        # Determine status
        if overall_score >= 90:
            status = 'healthy'
        elif overall_score >= 70:
            status = 'degraded'
        else:
            status = 'unhealthy'
        
        return HealthStatus(
            status=status,
            checks=checks,
            overall_score=overall_score
        )
    
    def _update_prometheus_metrics(self, system_metrics: SystemMetrics, app_metrics: ApplicationMetrics):
        """Update Prometheus metrics."""
        if not self.prometheus_registry:
            return
        
        # System metrics
        self.prometheus_cpu.set(system_metrics.cpu_percent)
        self.prometheus_memory.set(system_metrics.memory_percent)
        self.prometheus_disk.set(system_metrics.disk_usage_percent)
    
    def register_health_check(self, name: str, check_func):
        """Register a health check function."""
        self.health_checks[name] = check_func
        logger.info(f"Registered health check: {name}")
    
    def record_request(self, response_time: float):
        """Record a request with its response time."""
        self.request_count += 1
        self.request_times.append(response_time)
        self.last_request_time = time.time()
        
        # Update Prometheus
        if self.prometheus_registry:
            self.prometheus_requests.inc()
            self.prometheus_response_time.observe(response_time)
    
    def record_error(self, error_type: str):
        """Record an error occurrence."""
        self.error_count[error_type] += 1
        
        # Update Prometheus
        if self.prometheus_registry:
            self.prometheus_errors.labels(error_type=error_type).inc()
    
    def record_cache_hit(self):
        """Record a cache hit."""
        self.cache_hits += 1
        
        if self.prometheus_registry:
            self.prometheus_cache_hits.inc()
    
    def record_cache_miss(self):
        """Record a cache miss."""
        self.cache_misses += 1
        
        if self.prometheus_registry:
            self.prometheus_cache_misses.inc()
    
    def record_query_success(self):
        """Record a successful query."""
        self.query_success += 1
    
    def record_query_failure(self):
        """Record a failed query."""
        self.query_failures += 1
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current system metrics."""
        if not self.system_metrics or not self.app_metrics:
            return {}
        
        latest_system = self.system_metrics[-1]
        latest_app = self.app_metrics[-1]
        latest_health = self.health_history[-1] if self.health_history else None
        
        return {
            'system': {
                'cpu_percent': latest_system.cpu_percent,
                'memory_percent': latest_system.memory_percent,
                'memory_used_mb': latest_system.memory_used_mb,
                'disk_usage_percent': latest_system.disk_usage_percent,
                'disk_free_gb': latest_system.disk_free_gb
            },
            'application': {
                'total_requests': latest_app.total_requests,
                'requests_per_second': latest_app.requests_per_second,
                'average_response_time': latest_app.average_response_time,
                'error_rate': latest_app.error_rate,
                'cache_hit_rate': latest_app.cache_hit_rate,
                'query_success_rate': latest_app.query_success_rate
            },
            'health': {
                'status': latest_health.status if latest_health else 'unknown',
                'overall_score': latest_health.overall_score if latest_health else 0
            },
            'timestamp': datetime.now().isoformat()
        }
    
    def get_metrics_summary(self, hours: int = 1) -> Dict[str, Any]:
        """Get metrics summary for the last N hours."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        # Filter recent metrics
        recent_system = [m for m in self.system_metrics if m.timestamp >= cutoff_time]
        recent_app = [m for m in self.app_metrics if m.timestamp >= cutoff_time]
        recent_health = [h for h in self.health_history if h.timestamp >= cutoff_time]
        
        if not recent_system:
            return {}
        
        # Calculate averages
        avg_cpu = sum(m.cpu_percent for m in recent_system) / len(recent_system)
        avg_memory = sum(m.memory_percent for m in recent_system) / len(recent_system)
        avg_response_time = sum(m.average_response_time for m in recent_app) / len(recent_app) if recent_app else 0
        
        # Health score trend
        health_scores = [h.overall_score for h in recent_health]
        avg_health_score = sum(health_scores) / len(health_scores) if health_scores else 0
        
        return {
            'period_hours': hours,
            'system_averages': {
                'cpu_percent': avg_cpu,
                'memory_percent': avg_memory
            },
            'application_averages': {
                'response_time': avg_response_time
            },
            'health_average_score': avg_health_score,
            'data_points': len(recent_system)
        }


# Global monitor instance
_monitor: Optional[ProductionMonitor] = None


def get_monitor(**kwargs) -> ProductionMonitor:
    """Get global monitor instance."""
    global _monitor
    if _monitor is None:
        _monitor = ProductionMonitor(**kwargs)
    return _monitor