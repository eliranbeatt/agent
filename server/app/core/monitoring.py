"""
Monitoring and logging system for Local Agent Studio.

This module provides structured logging, performance monitoring, resource tracking,
error reporting, and execution tracing capabilities.
"""

import logging
import time
import psutil
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from functools import wraps
from contextlib import contextmanager
import threading
from collections import defaultdict, deque

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


class LogLevel(Enum):
    """Log levels for structured logging."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class MetricType(Enum):
    """Types of metrics to track."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


@dataclass
class LogEntry:
    """Structured log entry."""
    timestamp: datetime
    level: LogLevel
    component: str
    message: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    trace_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "level": self.level.value,
            "component": self.component,
            "message": self.message,
            "metadata": self.metadata,
            "trace_id": self.trace_id
        }
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())


@dataclass
class PerformanceMetric:
    """Performance metric data."""
    name: str
    metric_type: MetricType
    value: float
    timestamp: datetime = field(default_factory=datetime.now)
    tags: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "type": self.metric_type.value,
            "value": self.value,
            "timestamp": self.timestamp.isoformat(),
            "tags": self.tags
        }


@dataclass
class ResourceUsage:
    """System resource usage snapshot."""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_mb: float
    disk_usage_percent: float
    active_threads: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class ExecutionTrace:
    """Execution trace for debugging and optimization."""
    trace_id: str
    component: str
    operation: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_ms: Optional[float] = None
    status: str = "running"
    metadata: Dict[str, Any] = field(default_factory=dict)
    parent_trace_id: Optional[str] = None
    
    def complete(self, status: str = "success"):
        """Mark trace as complete."""
        self.end_time = datetime.now()
        self.duration_ms = (self.end_time - self.start_time).total_seconds() * 1000
        self.status = status
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "trace_id": self.trace_id,
            "component": self.component,
            "operation": self.operation,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_ms": self.duration_ms,
            "status": self.status,
            "metadata": self.metadata,
            "parent_trace_id": self.parent_trace_id
        }


class StructuredLogger:
    """
    Structured logging system that outputs JSON-formatted logs
    with consistent structure across all components.
    """
    
    def __init__(self, component: str):
        self.component = component
        self.logger = logging.getLogger(component)
        self.log_buffer: deque = deque(maxlen=1000)
    
    def _log(self, level: LogLevel, message: str, metadata: Optional[Dict[str, Any]] = None, trace_id: Optional[str] = None):
        """Internal logging method."""
        entry = LogEntry(
            timestamp=datetime.now(),
            level=level,
            component=self.component,
            message=message,
            metadata=metadata or {},
            trace_id=trace_id
        )
        
        self.log_buffer.append(entry)
        
        # Log to standard logger
        log_method = getattr(self.logger, level.value)
        log_method(entry.to_json())
    
    def debug(self, message: str, metadata: Optional[Dict[str, Any]] = None, trace_id: Optional[str] = None):
        """Log debug message."""
        self._log(LogLevel.DEBUG, message, metadata, trace_id)
    
    def info(self, message: str, metadata: Optional[Dict[str, Any]] = None, trace_id: Optional[str] = None):
        """Log info message."""
        self._log(LogLevel.INFO, message, metadata, trace_id)
    
    def warning(self, message: str, metadata: Optional[Dict[str, Any]] = None, trace_id: Optional[str] = None):
        """Log warning message."""
        self._log(LogLevel.WARNING, message, metadata, trace_id)
    
    def error(self, message: str, metadata: Optional[Dict[str, Any]] = None, trace_id: Optional[str] = None):
        """Log error message."""
        self._log(LogLevel.ERROR, message, metadata, trace_id)
    
    def critical(self, message: str, metadata: Optional[Dict[str, Any]] = None, trace_id: Optional[str] = None):
        """Log critical message."""
        self._log(LogLevel.CRITICAL, message, metadata, trace_id)
    
    def get_recent_logs(self, count: int = 100) -> List[LogEntry]:
        """Get recent log entries."""
        return list(self.log_buffer)[-count:]


class PerformanceMonitor:
    """
    Performance monitoring system that tracks metrics, resource usage,
    and execution times across all components.
    """
    
    def __init__(self):
        self.metrics: Dict[str, List[PerformanceMetric]] = defaultdict(list)
        self.resource_history: deque = deque(maxlen=1000)
        self._lock = threading.Lock()
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
    
    def record_metric(self, name: str, value: float, metric_type: MetricType = MetricType.GAUGE, tags: Optional[Dict[str, str]] = None):
        """Record a performance metric."""
        metric = PerformanceMetric(
            name=name,
            metric_type=metric_type,
            value=value,
            tags=tags or {}
        )
        
        with self._lock:
            self.metrics[name].append(metric)
            
            # Keep only recent metrics (last 1000 per metric)
            if len(self.metrics[name]) > 1000:
                self.metrics[name] = self.metrics[name][-1000:]
    
    def increment_counter(self, name: str, value: float = 1.0, tags: Optional[Dict[str, str]] = None):
        """Increment a counter metric."""
        self.record_metric(name, value, MetricType.COUNTER, tags)
    
    def record_gauge(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Record a gauge metric."""
        self.record_metric(name, value, MetricType.GAUGE, tags)
    
    def record_histogram(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Record a histogram metric."""
        self.record_metric(name, value, MetricType.HISTOGRAM, tags)
    
    def get_metric_stats(self, name: str) -> Dict[str, float]:
        """Get statistics for a metric."""
        with self._lock:
            if name not in self.metrics or not self.metrics[name]:
                return {}
            
            values = [m.value for m in self.metrics[name]]
            return {
                "count": len(values),
                "sum": sum(values),
                "min": min(values),
                "max": max(values),
                "avg": sum(values) / len(values)
            }
    
    def capture_resource_usage(self) -> ResourceUsage:
        """Capture current system resource usage."""
        process = psutil.Process()
        
        usage = ResourceUsage(
            timestamp=datetime.now(),
            cpu_percent=process.cpu_percent(),
            memory_percent=process.memory_percent(),
            memory_mb=process.memory_info().rss / 1024 / 1024,
            disk_usage_percent=psutil.disk_usage('/').percent,
            active_threads=threading.active_count()
        )
        
        self.resource_history.append(usage)
        return usage
    
    def start_monitoring(self, interval: float = 5.0):
        """Start continuous resource monitoring."""
        if self._monitoring:
            return
        
        self._monitoring = True
        
        def monitor_loop():
            while self._monitoring:
                self.capture_resource_usage()
                time.sleep(interval)
        
        self._monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self._monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop continuous resource monitoring."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1.0)
    
    def get_resource_history(self, count: int = 100) -> List[ResourceUsage]:
        """Get recent resource usage history."""
        return list(self.resource_history)[-count:]
    
    def get_all_metrics(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get all recorded metrics."""
        with self._lock:
            return {
                name: [m.to_dict() for m in metrics]
                for name, metrics in self.metrics.items()
            }


class ExecutionTracer:
    """
    Execution tracing system for debugging and optimization.
    Tracks execution flow across components with timing information.
    """
    
    def __init__(self):
        self.traces: Dict[str, ExecutionTrace] = {}
        self.completed_traces: deque = deque(maxlen=1000)
        self._lock = threading.Lock()
        self._trace_counter = 0
    
    def start_trace(self, component: str, operation: str, metadata: Optional[Dict[str, Any]] = None, parent_trace_id: Optional[str] = None) -> str:
        """Start a new execution trace."""
        with self._lock:
            self._trace_counter += 1
            trace_id = f"{component}_{operation}_{self._trace_counter}_{int(time.time() * 1000)}"
        
        trace = ExecutionTrace(
            trace_id=trace_id,
            component=component,
            operation=operation,
            start_time=datetime.now(),
            metadata=metadata or {},
            parent_trace_id=parent_trace_id
        )
        
        with self._lock:
            self.traces[trace_id] = trace
        
        return trace_id
    
    def end_trace(self, trace_id: str, status: str = "success", metadata: Optional[Dict[str, Any]] = None):
        """End an execution trace."""
        with self._lock:
            if trace_id not in self.traces:
                return
            
            trace = self.traces[trace_id]
            trace.complete(status)
            
            if metadata:
                trace.metadata.update(metadata)
            
            self.completed_traces.append(trace)
            del self.traces[trace_id]
    
    def get_trace(self, trace_id: str) -> Optional[ExecutionTrace]:
        """Get a trace by ID."""
        with self._lock:
            return self.traces.get(trace_id)
    
    def get_active_traces(self) -> List[ExecutionTrace]:
        """Get all active traces."""
        with self._lock:
            return list(self.traces.values())
    
    def get_completed_traces(self, count: int = 100) -> List[ExecutionTrace]:
        """Get recent completed traces."""
        return list(self.completed_traces)[-count:]
    
    def get_trace_tree(self, root_trace_id: str) -> Dict[str, Any]:
        """Get trace tree starting from root trace."""
        traces = list(self.completed_traces) + list(self.traces.values())
        trace_map = {t.trace_id: t for t in traces}
        
        if root_trace_id not in trace_map:
            return {}
        
        def build_tree(trace_id: str) -> Dict[str, Any]:
            trace = trace_map[trace_id]
            children = [
                build_tree(t.trace_id)
                for t in traces
                if t.parent_trace_id == trace_id
            ]
            
            return {
                **trace.to_dict(),
                "children": children
            }
        
        return build_tree(root_trace_id)


class AlertManager:
    """
    Alert management system for error reporting and notifications.
    """
    
    def __init__(self):
        self.alerts: deque = deque(maxlen=1000)
        self.alert_handlers: List[Callable] = []
    
    def register_handler(self, handler: Callable):
        """Register an alert handler."""
        self.alert_handlers.append(handler)
    
    def send_alert(self, severity: str, component: str, message: str, metadata: Optional[Dict[str, Any]] = None):
        """Send an alert."""
        alert = {
            "timestamp": datetime.now().isoformat(),
            "severity": severity,
            "component": component,
            "message": message,
            "metadata": metadata or {}
        }
        
        self.alerts.append(alert)
        
        # Call registered handlers
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                logging.error(f"Alert handler failed: {e}")
    
    def get_recent_alerts(self, count: int = 100) -> List[Dict[str, Any]]:
        """Get recent alerts."""
        return list(self.alerts)[-count:]


# Global monitoring instances
performance_monitor = PerformanceMonitor()
execution_tracer = ExecutionTracer()
alert_manager = AlertManager()


# Decorators for easy monitoring integration

def monitor_performance(metric_name: Optional[str] = None):
    """Decorator to monitor function performance."""
    def decorator(func: Callable) -> Callable:
        name = metric_name or f"{func.__module__}.{func.__name__}"
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                duration = (time.time() - start_time) * 1000
                performance_monitor.record_histogram(f"{name}.duration_ms", duration)
                performance_monitor.increment_counter(f"{name}.calls")
                return result
            except Exception as e:
                performance_monitor.increment_counter(f"{name}.errors")
                raise e
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = (time.time() - start_time) * 1000
                performance_monitor.record_histogram(f"{name}.duration_ms", duration)
                performance_monitor.increment_counter(f"{name}.calls")
                return result
            except Exception as e:
                performance_monitor.increment_counter(f"{name}.errors")
                raise e
        
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


def trace_execution(component: Optional[str] = None, operation: Optional[str] = None):
    """Decorator to trace function execution."""
    def decorator(func: Callable) -> Callable:
        comp = component or func.__module__
        op = operation or func.__name__
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            trace_id = execution_tracer.start_trace(comp, op)
            try:
                result = await func(*args, **kwargs)
                execution_tracer.end_trace(trace_id, "success")
                return result
            except Exception as e:
                execution_tracer.end_trace(trace_id, "error", {"error": str(e)})
                raise e
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            trace_id = execution_tracer.start_trace(comp, op)
            try:
                result = func(*args, **kwargs)
                execution_tracer.end_trace(trace_id, "success")
                return result
            except Exception as e:
                execution_tracer.end_trace(trace_id, "error", {"error": str(e)})
                raise e
        
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


@contextmanager
def trace_context(component: str, operation: str, metadata: Optional[Dict[str, Any]] = None):
    """Context manager for tracing code blocks."""
    trace_id = execution_tracer.start_trace(component, operation, metadata)
    try:
        yield trace_id
        execution_tracer.end_trace(trace_id, "success")
    except Exception as e:
        execution_tracer.end_trace(trace_id, "error", {"error": str(e)})
        raise


# Initialize monitoring on module load
performance_monitor.start_monitoring()
