"""
Comprehensive tests for monitoring and logging system.
"""

import pytest
import asyncio
import time
from datetime import datetime
from unittest.mock import Mock, patch

from app.core.monitoring import (
    LogLevel,
    MetricType,
    LogEntry,
    PerformanceMetric,
    ResourceUsage,
    ExecutionTrace,
    StructuredLogger,
    PerformanceMonitor,
    ExecutionTracer,
    AlertManager,
    monitor_performance,
    trace_execution,
    trace_context,
    performance_monitor,
    execution_tracer,
    alert_manager
)


class TestStructuredLogger:
    """Test structured logging system."""
    
    def test_logger_creation(self):
        """Test logger creation with component name."""
        logger = StructuredLogger("TestComponent")
        assert logger.component == "TestComponent"
    
    def test_log_levels(self):
        """Test all log levels."""
        logger = StructuredLogger("TestComponent")
        
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")
        logger.critical("Critical message")
        
        logs = logger.get_recent_logs()
        assert len(logs) == 5
        assert logs[0].level == LogLevel.DEBUG
        assert logs[4].level == LogLevel.CRITICAL
    
    def test_log_with_metadata(self):
        """Test logging with metadata."""
        logger = StructuredLogger("TestComponent")
        
        metadata = {"user_id": "123", "action": "test"}
        logger.info("Test message", metadata=metadata)
        
        logs = logger.get_recent_logs()
        assert logs[0].metadata == metadata
    
    def test_log_with_trace_id(self):
        """Test logging with trace ID."""
        logger = StructuredLogger("TestComponent")
        
        logger.info("Test message", trace_id="trace_123")
        
        logs = logger.get_recent_logs()
        assert logs[0].trace_id == "trace_123"
    
    def test_log_buffer_limit(self):
        """Test log buffer maintains size limit."""
        logger = StructuredLogger("TestComponent")
        
        # Log more than buffer size
        for i in range(1500):
            logger.info(f"Message {i}")
        
        logs = logger.get_recent_logs(count=2000)
        assert len(logs) == 1000  # Buffer max size
    
    def test_log_entry_serialization(self):
        """Test log entry JSON serialization."""
        entry = LogEntry(
            timestamp=datetime.now(),
            level=LogLevel.INFO,
            component="TestComponent",
            message="Test message",
            metadata={"key": "value"}
        )
        
        json_str = entry.to_json()
        assert "TestComponent" in json_str
        assert "Test message" in json_str


class TestPerformanceMonitor:
    """Test performance monitoring system."""
    
    def test_record_metric(self):
        """Test recording metrics."""
        monitor = PerformanceMonitor()
        
        monitor.record_metric("test_metric", 42.0, MetricType.GAUGE)
        
        stats = monitor.get_metric_stats("test_metric")
        assert stats["count"] == 1
        assert stats["avg"] == 42.0
    
    def test_increment_counter(self):
        """Test counter increment."""
        monitor = PerformanceMonitor()
        
        monitor.increment_counter("test_counter", 1.0)
        monitor.increment_counter("test_counter", 2.0)
        monitor.increment_counter("test_counter", 3.0)
        
        stats = monitor.get_metric_stats("test_counter")
        assert stats["count"] == 3
        assert stats["sum"] == 6.0
    
    def test_record_gauge(self):
        """Test gauge recording."""
        monitor = PerformanceMonitor()
        
        monitor.record_gauge("cpu_usage", 45.5)
        monitor.record_gauge("cpu_usage", 50.2)
        
        stats = monitor.get_metric_stats("cpu_usage")
        assert stats["count"] == 2
        assert stats["min"] == 45.5
        assert stats["max"] == 50.2
    
    def test_record_histogram(self):
        """Test histogram recording."""
        monitor = PerformanceMonitor()
        
        values = [10, 20, 30, 40, 50]
        for v in values:
            monitor.record_histogram("response_time", v)
        
        stats = monitor.get_metric_stats("response_time")
        assert stats["count"] == 5
        assert stats["avg"] == 30.0
        assert stats["min"] == 10
        assert stats["max"] == 50
    
    def test_metric_with_tags(self):
        """Test metrics with tags."""
        monitor = PerformanceMonitor()
        
        monitor.record_metric(
            "api_calls",
            1.0,
            MetricType.COUNTER,
            tags={"endpoint": "/api/chat", "method": "POST"}
        )
        
        metrics = monitor.get_all_metrics()
        assert "api_calls" in metrics
        assert metrics["api_calls"][0]["tags"]["endpoint"] == "/api/chat"
    
    def test_capture_resource_usage(self):
        """Test resource usage capture."""
        monitor = PerformanceMonitor()
        
        usage = monitor.capture_resource_usage()
        
        assert isinstance(usage, ResourceUsage)
        assert usage.cpu_percent >= 0
        assert usage.memory_percent >= 0
        assert usage.memory_mb >= 0
        assert usage.active_threads >= 1
    
    def test_resource_monitoring_start_stop(self):
        """Test continuous resource monitoring."""
        monitor = PerformanceMonitor()
        
        monitor.start_monitoring(interval=0.1)
        time.sleep(0.3)
        monitor.stop_monitoring()
        
        history = monitor.get_resource_history()
        assert len(history) >= 2  # Should have captured multiple snapshots
    
    def test_get_all_metrics(self):
        """Test getting all metrics."""
        monitor = PerformanceMonitor()
        
        monitor.record_gauge("metric1", 10.0)
        monitor.record_gauge("metric2", 20.0)
        
        all_metrics = monitor.get_all_metrics()
        assert "metric1" in all_metrics
        assert "metric2" in all_metrics


class TestExecutionTracer:
    """Test execution tracing system."""
    
    def test_start_trace(self):
        """Test starting a trace."""
        tracer = ExecutionTracer()
        
        trace_id = tracer.start_trace("TestComponent", "test_operation")
        
        assert trace_id is not None
        trace = tracer.get_trace(trace_id)
        assert trace.component == "TestComponent"
        assert trace.operation == "test_operation"
        assert trace.status == "running"
    
    def test_end_trace(self):
        """Test ending a trace."""
        tracer = ExecutionTracer()
        
        trace_id = tracer.start_trace("TestComponent", "test_operation")
        time.sleep(0.1)
        tracer.end_trace(trace_id, "success")
        
        # Trace should be moved to completed
        assert tracer.get_trace(trace_id) is None
        completed = tracer.get_completed_traces()
        assert len(completed) >= 1
        assert completed[-1].status == "success"
        assert completed[-1].duration_ms is not None
    
    def test_trace_with_metadata(self):
        """Test trace with metadata."""
        tracer = ExecutionTracer()
        
        trace_id = tracer.start_trace(
            "TestComponent",
            "test_operation",
            metadata={"user_id": "123"}
        )
        
        trace = tracer.get_trace(trace_id)
        assert trace.metadata["user_id"] == "123"
    
    def test_trace_hierarchy(self):
        """Test parent-child trace relationships."""
        tracer = ExecutionTracer()
        
        parent_id = tracer.start_trace("Parent", "parent_op")
        child_id = tracer.start_trace("Child", "child_op", parent_trace_id=parent_id)
        
        child_trace = tracer.get_trace(child_id)
        assert child_trace.parent_trace_id == parent_id
    
    def test_get_active_traces(self):
        """Test getting active traces."""
        tracer = ExecutionTracer()
        
        trace_id1 = tracer.start_trace("Component1", "op1")
        trace_id2 = tracer.start_trace("Component2", "op2")
        
        active = tracer.get_active_traces()
        assert len(active) >= 2
    
    def test_trace_tree(self):
        """Test building trace tree."""
        tracer = ExecutionTracer()
        
        root_id = tracer.start_trace("Root", "root_op")
        child1_id = tracer.start_trace("Child1", "child1_op", parent_trace_id=root_id)
        child2_id = tracer.start_trace("Child2", "child2_op", parent_trace_id=root_id)
        
        tracer.end_trace(child1_id)
        tracer.end_trace(child2_id)
        tracer.end_trace(root_id)
        
        tree = tracer.get_trace_tree(root_id)
        assert tree["trace_id"] == root_id
        assert len(tree["children"]) == 2


class TestAlertManager:
    """Test alert management system."""
    
    def test_send_alert(self):
        """Test sending alerts."""
        manager = AlertManager()
        
        manager.send_alert(
            "high",
            "TestComponent",
            "Test alert message",
            metadata={"key": "value"}
        )
        
        alerts = manager.get_recent_alerts()
        assert len(alerts) == 1
        assert alerts[0]["severity"] == "high"
        assert alerts[0]["component"] == "TestComponent"
    
    def test_alert_handler_registration(self):
        """Test registering alert handlers."""
        manager = AlertManager()
        handler_called = False
        
        def test_handler(alert):
            nonlocal handler_called
            handler_called = True
        
        manager.register_handler(test_handler)
        manager.send_alert("low", "Test", "Test message")
        
        assert handler_called
    
    def test_multiple_alert_handlers(self):
        """Test multiple alert handlers."""
        manager = AlertManager()
        call_counts = {"handler1": 0, "handler2": 0}
        
        def handler1(alert):
            call_counts["handler1"] += 1
        
        def handler2(alert):
            call_counts["handler2"] += 1
        
        manager.register_handler(handler1)
        manager.register_handler(handler2)
        manager.send_alert("medium", "Test", "Test message")
        
        assert call_counts["handler1"] == 1
        assert call_counts["handler2"] == 1
    
    def test_alert_buffer_limit(self):
        """Test alert buffer maintains size limit."""
        manager = AlertManager()
        
        for i in range(1500):
            manager.send_alert("low", "Test", f"Alert {i}")
        
        alerts = manager.get_recent_alerts(count=2000)
        assert len(alerts) == 1000  # Buffer max size


class TestMonitoringDecorators:
    """Test monitoring decorators."""
    
    def test_monitor_performance_decorator_sync(self):
        """Test performance monitoring decorator with sync function."""
        monitor = PerformanceMonitor()
        
        @monitor_performance("test_function")
        def test_func():
            time.sleep(0.01)
            return "result"
        
        result = test_func()
        assert result == "result"
        
        # Check metrics were recorded
        stats = performance_monitor.get_metric_stats("test_function.duration_ms")
        assert stats["count"] >= 1
    
    @pytest.mark.asyncio
    async def test_monitor_performance_decorator_async(self):
        """Test performance monitoring decorator with async function."""
        @monitor_performance("async_test_function")
        async def async_test_func():
            await asyncio.sleep(0.01)
            return "result"
        
        result = await async_test_func()
        assert result == "result"
    
    def test_trace_execution_decorator_sync(self):
        """Test execution tracing decorator with sync function."""
        @trace_execution("TestComponent", "test_operation")
        def test_func():
            return "result"
        
        result = test_func()
        assert result == "result"
        
        # Check trace was recorded
        completed = execution_tracer.get_completed_traces()
        assert len(completed) >= 1
    
    @pytest.mark.asyncio
    async def test_trace_execution_decorator_async(self):
        """Test execution tracing decorator with async function."""
        @trace_execution("TestComponent", "async_operation")
        async def async_test_func():
            await asyncio.sleep(0.01)
            return "result"
        
        result = await async_test_func()
        assert result == "result"
    
    def test_trace_context_manager(self):
        """Test trace context manager."""
        with trace_context("TestComponent", "context_operation") as trace_id:
            assert trace_id is not None
            time.sleep(0.01)
        
        # Trace should be completed
        completed = execution_tracer.get_completed_traces()
        assert any(t.trace_id == trace_id for t in completed)
    
    def test_trace_context_manager_with_error(self):
        """Test trace context manager handles errors."""
        try:
            with trace_context("TestComponent", "failing_operation") as trace_id:
                raise ValueError("Test error")
        except ValueError:
            pass
        
        # Trace should be marked as error
        completed = execution_tracer.get_completed_traces()
        error_trace = next((t for t in completed if t.trace_id == trace_id), None)
        assert error_trace is not None
        assert error_trace.status == "error"


class TestIntegration:
    """Integration tests for monitoring system."""
    
    @pytest.mark.asyncio
    async def test_complete_monitoring_flow(self):
        """Test complete monitoring flow with logging, metrics, and tracing."""
        logger = StructuredLogger("IntegrationTest")
        monitor = PerformanceMonitor()
        tracer = ExecutionTracer()
        
        # Start trace
        trace_id = tracer.start_trace("IntegrationTest", "complete_flow")
        
        # Log some messages
        logger.info("Starting operation", trace_id=trace_id)
        
        # Record metrics
        monitor.record_gauge("test_metric", 100.0)
        monitor.increment_counter("test_counter")
        
        # Simulate work
        await asyncio.sleep(0.01)
        
        # End trace
        tracer.end_trace(trace_id, "success")
        
        # Verify everything was recorded
        logs = logger.get_recent_logs()
        assert len(logs) >= 1
        
        stats = monitor.get_metric_stats("test_metric")
        assert stats["count"] >= 1
        
        completed = tracer.get_completed_traces()
        assert any(t.trace_id == trace_id for t in completed)
    
    def test_concurrent_monitoring(self):
        """Test monitoring under concurrent load."""
        logger = StructuredLogger("ConcurrentTest")
        monitor = PerformanceMonitor()
        
        def worker():
            for i in range(10):
                logger.info(f"Worker message {i}")
                monitor.increment_counter("worker_counter")
        
        import threading
        threads = [threading.Thread(target=worker) for _ in range(5)]
        
        for t in threads:
            t.start()
        
        for t in threads:
            t.join()
        
        # Verify metrics from all threads
        stats = monitor.get_metric_stats("worker_counter")
        assert stats["count"] == 50  # 5 threads * 10 increments


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
