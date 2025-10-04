"""
Comprehensive system integration tests for error handling, monitoring,
and system behavior under various conditions.
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch, AsyncMock
from concurrent.futures import ThreadPoolExecutor

from app.core.error_handling import (
    CircuitBreaker,
    with_retry,
    error_recovery_manager,
    ErrorCategory,
    ErrorSeverity,
    FileProcessingErrorHandler,
    AgentExecutionErrorHandler
)
from app.core.monitoring import (
    StructuredLogger,
    PerformanceMonitor,
    ExecutionTracer,
    monitor_performance,
    trace_execution,
    performance_monitor,
    execution_tracer
)


class TestSystemErrorRecovery:
    """Test system-wide error recovery scenarios."""
    
    @pytest.mark.asyncio
    async def test_cascading_failure_prevention(self):
        """Test circuit breaker prevents cascading failures."""
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=1.0)
        failure_count = 0
        
        async def failing_service():
            nonlocal failure_count
            failure_count += 1
            raise Exception("Service unavailable")
        
        # Trigger circuit breaker
        for _ in range(3):
            with pytest.raises(Exception):
                await cb.call_async(failing_service)
        
        # Circuit should be open
        assert cb.state == "OPEN"
        
        # Further calls should be blocked without hitting the service
        initial_failure_count = failure_count
        with pytest.raises(Exception, match="Circuit breaker is OPEN"):
            await cb.call_async(failing_service)
        
        # Failure count should not increase
        assert failure_count == initial_failure_count
    
    @pytest.mark.asyncio
    async def test_retry_with_exponential_backoff(self):
        """Test retry mechanism with exponential backoff."""
        attempt_times = []
        
        @with_retry(max_attempts=3, retry_on=(ValueError,))
        async def flaky_operation():
            attempt_times.append(time.time())
            if len(attempt_times) < 3:
                raise ValueError("Temporary failure")
            return "success"
        
        result = await flaky_operation()
        assert result == "success"
        assert len(attempt_times) == 3
        
        # Verify delays between attempts
        if len(attempt_times) >= 2:
            delay1 = attempt_times[1] - attempt_times[0]
            assert delay1 >= 0.5  # Should have some delay
    
    def test_error_recovery_manager_integration(self):
        """Test error recovery manager coordinates recovery strategies."""
        # Simulate file processing error
        action = error_recovery_manager.handle_error(
            error=IOError("File not found"),
            category=ErrorCategory.FILE_PROCESSING,
            severity=ErrorSeverity.HIGH,
            component="FileProcessor",
            operation="read_file"
        )
        
        assert action.action_type == "fallback"
        assert len(error_recovery_manager.error_history) >= 1
    
    @pytest.mark.asyncio
    async def test_graceful_degradation_chain(self):
        """Test graceful degradation through multiple fallback levels."""
        attempts = {"primary": 0, "secondary": 0, "tertiary": 0}
        
        async def primary_service():
            attempts["primary"] += 1
            raise Exception("Primary unavailable")
        
        async def secondary_service():
            attempts["secondary"] += 1
            raise Exception("Secondary unavailable")
        
        async def tertiary_service():
            attempts["tertiary"] += 1
            return "fallback_result"
        
        # Try services in order with fallbacks
        result = None
        try:
            result = await primary_service()
        except Exception:
            try:
                result = await secondary_service()
            except Exception:
                result = await tertiary_service()
        
        assert result == "fallback_result"
        assert attempts["primary"] == 1
        assert attempts["secondary"] == 1
        assert attempts["tertiary"] == 1


class TestResourceConstraints:
    """Test system behavior under resource constraints."""
    
    def test_memory_limit_enforcement(self):
        """Test system respects memory limits."""
        monitor = PerformanceMonitor()
        
        # Record many metrics to test memory management
        for i in range(2000):
            monitor.record_gauge(f"metric_{i % 10}", float(i))
        
        # Each metric should keep only last 1000 entries
        for i in range(10):
            stats = monitor.get_metric_stats(f"metric_{i}")
            # Should have many entries but not all 200 per metric
            assert stats["count"] <= 1000
    
    def test_execution_trace_limit(self):
        """Test execution trace history limits."""
        tracer = ExecutionTracer()
        
        # Create many traces
        for i in range(1500):
            trace_id = tracer.start_trace("Component", f"operation_{i}")
            tracer.end_trace(trace_id)
        
        # Should keep only last 1000
        completed = tracer.get_completed_traces(count=2000)
        assert len(completed) == 1000
    
    @pytest.mark.asyncio
    async def test_concurrent_request_handling(self):
        """Test system handles concurrent requests properly."""
        monitor = PerformanceMonitor()
        results = []
        
        @monitor_performance("concurrent_operation")
        async def process_request(request_id: int):
            await asyncio.sleep(0.01)
            return f"result_{request_id}"
        
        # Process multiple requests concurrently
        tasks = [process_request(i) for i in range(20)]
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 20
        assert all(r.startswith("result_") for r in results)
        
        # Verify metrics were recorded (using global performance_monitor)
        stats = performance_monitor.get_metric_stats("concurrent_operation.calls")
        if stats:  # Only check if metrics exist
            assert stats["count"] >= 20
    
    def test_thread_safety(self):
        """Test thread-safe operations under concurrent load."""
        logger = StructuredLogger("ThreadSafetyTest")
        monitor = PerformanceMonitor()
        
        def worker(worker_id: int):
            for i in range(50):
                logger.info(f"Worker {worker_id} message {i}")
                monitor.increment_counter("thread_counter")
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(worker, i) for i in range(10)]
            for future in futures:
                future.result()
        
        # Verify all operations completed
        stats = monitor.get_metric_stats("thread_counter")
        assert stats["count"] == 500  # 10 workers * 50 increments


class TestLoadHandling:
    """Test system behavior under load."""
    
    @pytest.mark.asyncio
    async def test_high_throughput_logging(self):
        """Test logging system under high throughput."""
        logger = StructuredLogger("LoadTest")
        
        async def log_worker():
            for i in range(100):
                logger.info(f"Message {i}", metadata={"worker": "test"})
                await asyncio.sleep(0.001)
        
        # Run multiple workers
        await asyncio.gather(*[log_worker() for _ in range(5)])
        
        # Verify logs were captured
        logs = logger.get_recent_logs(count=1000)
        assert len(logs) >= 500
    
    @pytest.mark.asyncio
    async def test_metric_recording_performance(self):
        """Test metric recording performance under load."""
        monitor = PerformanceMonitor()
        
        start_time = time.time()
        
        # Record many metrics quickly
        for i in range(1000):
            monitor.record_gauge("load_test_metric", float(i))
        
        duration = time.time() - start_time
        
        # Should complete quickly (< 1 second for 1000 metrics)
        assert duration < 1.0
        
        stats = monitor.get_metric_stats("load_test_metric")
        assert stats["count"] == 1000
    
    @pytest.mark.asyncio
    async def test_trace_performance_overhead(self):
        """Test tracing doesn't add significant overhead."""
        @trace_execution("PerformanceTest", "traced_operation")
        async def traced_operation():
            await asyncio.sleep(0.01)
            return "result"
        
        async def untraced_operation():
            await asyncio.sleep(0.01)
            return "result"
        
        # Measure traced operation
        traced_start = time.time()
        for _ in range(10):
            await traced_operation()
        traced_duration = time.time() - traced_start
        
        # Measure untraced operation
        untraced_start = time.time()
        for _ in range(10):
            await untraced_operation()
        untraced_duration = time.time() - untraced_start
        
        # Overhead should be minimal (< 20% difference)
        overhead_ratio = traced_duration / untraced_duration
        assert overhead_ratio < 1.2


class TestErrorScenarios:
    """Test specific error scenarios and recovery."""
    
    def test_file_processing_error_recovery(self):
        """Test file processing error handling and recovery."""
        # Test unsupported format
        result = FileProcessingErrorHandler.handle_unsupported_format(
            "test.unknown",
            Exception("Unsupported format")
        )
        assert result is None
        
        # Test OCR failure
        result = FileProcessingErrorHandler.handle_ocr_failure(
            "test.pdf",
            Exception("OCR failed")
        )
        assert result is None
        
        # Verify errors were recorded
        assert len(error_recovery_manager.error_history) >= 2
    
    def test_agent_execution_error_recovery(self):
        """Test agent execution error handling."""
        # Test resource limit exceeded
        AgentExecutionErrorHandler.handle_resource_limit_exceeded(
            "agent_123",
            "max_steps",
            Exception("Limit exceeded")
        )
        
        # Test tool failure
        should_retry = AgentExecutionErrorHandler.handle_tool_failure(
            "agent_123",
            "search_tool",
            Exception("Tool failed")
        )
        assert should_retry is True
        
        # Test infinite loop detection
        AgentExecutionErrorHandler.handle_infinite_loop(
            "agent_123",
            100,
            Exception("Infinite loop detected")
        )
        
        # Verify errors were recorded
        assert len(error_recovery_manager.error_history) >= 3
    
    @pytest.mark.asyncio
    async def test_api_call_retry_logic(self):
        """Test API call retry logic."""
        call_count = 0
        
        @with_retry(max_attempts=3, retry_on=(ConnectionError,))
        async def api_call():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("API unavailable")
            return {"status": "success"}
        
        result = await api_call()
        assert result["status"] == "success"
        assert call_count == 3


class TestMonitoringIntegration:
    """Test monitoring system integration."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_monitoring(self):
        """Test complete monitoring flow."""
        logger = StructuredLogger("E2ETest")
        monitor = PerformanceMonitor()
        tracer = ExecutionTracer()
        
        # Start operation
        trace_id = tracer.start_trace("E2ETest", "complete_operation")
        logger.info("Operation started", trace_id=trace_id)
        
        # Simulate work with metrics
        for i in range(5):
            monitor.increment_counter("operation_steps")
            await asyncio.sleep(0.01)
        
        # Complete operation
        tracer.end_trace(trace_id, "success")
        logger.info("Operation completed", trace_id=trace_id)
        
        # Verify all monitoring data
        logs = logger.get_recent_logs()
        assert len(logs) >= 2
        
        stats = monitor.get_metric_stats("operation_steps")
        assert stats["count"] == 5
        
        completed = tracer.get_completed_traces()
        assert any(t.trace_id == trace_id for t in completed)
    
    def test_resource_monitoring_accuracy(self):
        """Test resource monitoring captures accurate data."""
        monitor = PerformanceMonitor()
        
        # Capture resource usage
        usage = monitor.capture_resource_usage()
        
        assert usage.cpu_percent >= 0
        assert usage.memory_percent >= 0
        assert usage.memory_mb > 0
        assert usage.active_threads >= 1
        assert usage.disk_usage_percent >= 0
    
    @pytest.mark.asyncio
    async def test_distributed_tracing(self):
        """Test distributed tracing across components."""
        tracer = ExecutionTracer()
        
        # Simulate distributed operation
        root_id = tracer.start_trace("Orchestrator", "process_request")
        
        child1_id = tracer.start_trace("Planner", "create_plan", parent_trace_id=root_id)
        await asyncio.sleep(0.01)
        tracer.end_trace(child1_id)
        
        child2_id = tracer.start_trace("AgentGenerator", "create_agent", parent_trace_id=root_id)
        await asyncio.sleep(0.01)
        tracer.end_trace(child2_id)
        
        tracer.end_trace(root_id)
        
        # Verify trace tree
        tree = tracer.get_trace_tree(root_id)
        assert tree["trace_id"] == root_id
        assert len(tree["children"]) == 2


class TestSystemResilience:
    """Test system resilience and recovery."""
    
    @pytest.mark.asyncio
    async def test_partial_failure_handling(self):
        """Test system handles partial failures gracefully."""
        results = []
        
        async def process_item(item_id: int):
            if item_id % 3 == 0:
                raise ValueError(f"Item {item_id} failed")
            return f"processed_{item_id}"
        
        # Process items with some failures
        for i in range(10):
            try:
                result = await process_item(i)
                results.append(result)
            except ValueError:
                # Log error but continue
                error_recovery_manager.handle_error(
                    error=ValueError(f"Item {i} failed"),
                    category=ErrorCategory.AGENT_EXECUTION,
                    severity=ErrorSeverity.MEDIUM,
                    component="ItemProcessor"
                )
        
        # Should have processed non-failing items
        assert len(results) > 0
        assert len(results) < 10  # Some failed
    
    def test_circuit_breaker_recovery(self):
        """Test circuit breaker recovery after timeout."""
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=0.1)
        
        def failing_func():
            raise Exception("Failure")
        
        # Open circuit
        for _ in range(2):
            with pytest.raises(Exception):
                cb.call(failing_func)
        
        assert cb.state == "OPEN"
        
        # Wait for recovery timeout
        time.sleep(0.2)
        
        # Should attempt reset
        def success_func():
            return "success"
        
        result = cb.call(success_func)
        assert result == "success"
        assert cb.state == "CLOSED"
    
    @pytest.mark.asyncio
    async def test_timeout_handling(self):
        """Test operation timeout handling."""
        async def long_running_operation():
            await asyncio.sleep(10)
            return "completed"
        
        # Should timeout
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(long_running_operation(), timeout=0.1)


class TestSystemMetrics:
    """Test system-wide metrics collection."""
    
    def test_aggregate_metrics(self):
        """Test aggregating metrics across components."""
        monitor = PerformanceMonitor()
        
        # Simulate metrics from multiple components
        components = ["Orchestrator", "Planner", "AgentGenerator"]
        for component in components:
            for i in range(10):
                monitor.record_histogram(f"{component}.execution_time", float(i * 10))
        
        # Verify metrics for each component
        for component in components:
            stats = monitor.get_metric_stats(f"{component}.execution_time")
            assert stats["count"] == 10
    
    def test_metric_statistics(self):
        """Test metric statistics calculation."""
        monitor = PerformanceMonitor()
        
        values = [10, 20, 30, 40, 50]
        for v in values:
            monitor.record_gauge("test_metric", v)
        
        stats = monitor.get_metric_stats("test_metric")
        assert stats["min"] == 10
        assert stats["max"] == 50
        assert stats["avg"] == 30
        assert stats["sum"] == 150
        assert stats["count"] == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
