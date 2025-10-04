"""
Comprehensive tests for error handling and recovery mechanisms.
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock

from app.core.error_handling import (
    ErrorCategory,
    ErrorSeverity,
    ErrorContext,
    RecoveryAction,
    CircuitBreaker,
    RetryStrategy,
    with_retry,
    ErrorRecoveryManager,
    FileProcessingErrorHandler,
    AgentExecutionErrorHandler,
    MemoryStorageErrorHandler,
    SystemErrorHandler,
    error_recovery_manager
)


class TestCircuitBreaker:
    """Test circuit breaker pattern implementation."""
    
    def test_circuit_breaker_closed_state(self):
        """Test circuit breaker in closed state allows requests."""
        cb = CircuitBreaker(failure_threshold=3)
        
        def success_func():
            return "success"
        
        result = cb.call(success_func)
        assert result == "success"
        assert cb.state == "CLOSED"
        assert cb.failure_count == 0
    
    def test_circuit_breaker_opens_after_failures(self):
        """Test circuit breaker opens after threshold failures."""
        cb = CircuitBreaker(failure_threshold=3)
        
        def failing_func():
            raise Exception("Test failure")
        
        # Trigger failures
        for _ in range(3):
            with pytest.raises(Exception):
                cb.call(failing_func)
        
        assert cb.state == "OPEN"
        assert cb.failure_count == 3
    
    def test_circuit_breaker_blocks_when_open(self):
        """Test circuit breaker blocks requests when open."""
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=60.0)
        
        def failing_func():
            raise Exception("Test failure")
        
        # Open the circuit
        for _ in range(2):
            with pytest.raises(Exception):
                cb.call(failing_func)
        
        # Should block now
        with pytest.raises(Exception, match="Circuit breaker is OPEN"):
            cb.call(failing_func)
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_async(self):
        """Test circuit breaker with async functions."""
        cb = CircuitBreaker(failure_threshold=2)
        
        async def async_success():
            return "success"
        
        result = await cb.call_async(async_success)
        assert result == "success"
        assert cb.state == "CLOSED"
    
    def test_circuit_breaker_resets_on_success(self):
        """Test circuit breaker resets failure count on success."""
        cb = CircuitBreaker(failure_threshold=3)
        
        def failing_func():
            raise Exception("Test failure")
        
        def success_func():
            return "success"
        
        # Partial failures
        with pytest.raises(Exception):
            cb.call(failing_func)
        
        assert cb.failure_count == 1
        
        # Success resets
        cb.call(success_func)
        assert cb.failure_count == 0
        assert cb.state == "CLOSED"


class TestRetryStrategy:
    """Test retry strategy and exponential backoff."""
    
    def test_retry_strategy_delay_calculation(self):
        """Test exponential backoff delay calculation."""
        strategy = RetryStrategy(
            initial_delay=1.0,
            exponential_base=2.0,
            jitter=False
        )
        
        assert strategy.get_delay(0) == 1.0
        assert strategy.get_delay(1) == 2.0
        assert strategy.get_delay(2) == 4.0
    
    def test_retry_strategy_max_delay(self):
        """Test max delay cap."""
        strategy = RetryStrategy(
            initial_delay=1.0,
            max_delay=5.0,
            exponential_base=2.0,
            jitter=False
        )
        
        assert strategy.get_delay(10) == 5.0
    
    def test_retry_strategy_with_jitter(self):
        """Test jitter adds randomness."""
        strategy = RetryStrategy(
            initial_delay=1.0,
            exponential_base=2.0,
            jitter=True
        )
        
        delays = [strategy.get_delay(1) for _ in range(10)]
        # With jitter, delays should vary
        assert len(set(delays)) > 1
    
    @pytest.mark.asyncio
    async def test_with_retry_decorator_async(self):
        """Test retry decorator with async functions."""
        call_count = 0
        
        @with_retry(max_attempts=3, retry_on=(ValueError,))
        async def failing_then_success():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Not yet")
            return "success"
        
        result = await failing_then_success()
        assert result == "success"
        assert call_count == 3
    
    def test_with_retry_decorator_sync(self):
        """Test retry decorator with sync functions."""
        call_count = 0
        
        @with_retry(max_attempts=3, retry_on=(ValueError,))
        def failing_then_success():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Not yet")
            return "success"
        
        result = failing_then_success()
        assert result == "success"
        assert call_count == 3
    
    def test_with_retry_exhausts_attempts(self):
        """Test retry exhausts all attempts before failing."""
        call_count = 0
        
        @with_retry(max_attempts=3, retry_on=(ValueError,))
        def always_fails():
            nonlocal call_count
            call_count += 1
            raise ValueError("Always fails")
        
        with pytest.raises(ValueError):
            always_fails()
        
        assert call_count == 3


class TestErrorRecoveryManager:
    """Test error recovery manager."""
    
    def test_error_recording(self):
        """Test error recording and history."""
        manager = ErrorRecoveryManager()
        
        error_context = ErrorContext(
            category=ErrorCategory.FILE_PROCESSING,
            severity=ErrorSeverity.HIGH,
            error=Exception("Test error"),
            component="TestComponent",
            operation="test_operation"
        )
        
        manager.record_error(error_context)
        
        assert len(manager.error_history) == 1
        assert manager.error_history[0].component == "TestComponent"
    
    def test_get_recovery_action(self):
        """Test getting recovery action for error category."""
        manager = ErrorRecoveryManager()
        
        action = manager.get_recovery_action(ErrorCategory.FILE_PROCESSING)
        assert action.action_type == "fallback"
        
        action = manager.get_recovery_action(ErrorCategory.AGENT_EXECUTION)
        assert action.action_type == "retry"
    
    def test_circuit_breaker_creation(self):
        """Test circuit breaker creation and retrieval."""
        manager = ErrorRecoveryManager()
        
        cb1 = manager.get_circuit_breaker("test_component")
        cb2 = manager.get_circuit_breaker("test_component")
        
        assert cb1 is cb2  # Same instance
    
    def test_handle_error(self):
        """Test comprehensive error handling."""
        manager = ErrorRecoveryManager()
        
        action = manager.handle_error(
            error=Exception("Test error"),
            category=ErrorCategory.AGENT_EXECUTION,
            severity=ErrorSeverity.HIGH,
            component="AgentExecutor",
            operation="execute_task"
        )
        
        assert action.action_type == "retry"
        assert len(manager.error_history) == 1


class TestFileProcessingErrorHandler:
    """Test file processing error handlers."""
    
    def test_handle_unsupported_format(self):
        """Test handling unsupported file format."""
        result = FileProcessingErrorHandler.handle_unsupported_format(
            "test.xyz",
            Exception("Unsupported format")
        )
        
        assert result is None
    
    def test_handle_ocr_failure(self):
        """Test handling OCR failure."""
        result = FileProcessingErrorHandler.handle_ocr_failure(
            "test.pdf",
            Exception("OCR failed")
        )
        
        assert result is None
    
    def test_handle_corrupted_file(self):
        """Test handling corrupted file."""
        result = FileProcessingErrorHandler.handle_corrupted_file(
            "test.docx",
            Exception("File corrupted")
        )
        
        assert result is None


class TestAgentExecutionErrorHandler:
    """Test agent execution error handlers."""
    
    def test_handle_resource_limit_exceeded(self):
        """Test handling resource limit exceeded."""
        AgentExecutionErrorHandler.handle_resource_limit_exceeded(
            "agent_123",
            "max_steps",
            Exception("Limit exceeded")
        )
        
        # Should not raise, just log
        assert True
    
    def test_handle_tool_failure(self):
        """Test handling tool failure."""
        should_retry = AgentExecutionErrorHandler.handle_tool_failure(
            "agent_123",
            "search_tool",
            Exception("Tool failed")
        )
        
        assert should_retry is True
    
    def test_handle_infinite_loop(self):
        """Test handling infinite loop detection."""
        AgentExecutionErrorHandler.handle_infinite_loop(
            "agent_123",
            100,
            Exception("Infinite loop")
        )
        
        # Should not raise, just log
        assert True


class TestMemoryStorageErrorHandler:
    """Test memory and storage error handlers."""
    
    def test_handle_vector_db_connection_error(self):
        """Test handling vector DB connection error."""
        has_fallback = MemoryStorageErrorHandler.handle_vector_db_connection_error(
            Exception("Connection failed")
        )
        
        assert has_fallback is True
    
    def test_handle_memory_capacity_exceeded(self):
        """Test handling memory capacity exceeded."""
        MemoryStorageErrorHandler.handle_memory_capacity_exceeded(
            Exception("Capacity exceeded")
        )
        
        # Should not raise, just log
        assert True
    
    def test_handle_embedding_api_failure(self):
        """Test handling embedding API failure."""
        should_retry = MemoryStorageErrorHandler.handle_embedding_api_failure(
            Exception("API failed"),
            retry_count=1
        )
        
        assert should_retry is True
        
        should_retry = MemoryStorageErrorHandler.handle_embedding_api_failure(
            Exception("API failed"),
            retry_count=3
        )
        
        assert should_retry is False


class TestSystemErrorHandler:
    """Test system-level error handlers."""
    
    def test_handle_configuration_error(self):
        """Test handling configuration error."""
        defaults_loaded = SystemErrorHandler.handle_configuration_error(
            "config.yaml",
            Exception("Invalid config")
        )
        
        assert defaults_loaded is True
    
    def test_handle_network_connectivity_error(self):
        """Test handling network connectivity error."""
        SystemErrorHandler.handle_network_connectivity_error(
            Exception("Network error")
        )
        
        # Should not raise, just log
        assert True
    
    def test_handle_resource_exhaustion(self):
        """Test handling resource exhaustion."""
        SystemErrorHandler.handle_resource_exhaustion(
            "memory",
            Exception("Out of memory")
        )
        
        # Should not raise, just log
        assert True


class TestErrorScenarios:
    """Test complete error scenarios and recovery."""
    
    @pytest.mark.asyncio
    async def test_file_processing_with_retry(self):
        """Test file processing with retry on failure."""
        attempt_count = 0
        
        @with_retry(max_attempts=3, retry_on=(IOError,))
        async def process_file(file_path: str):
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 2:
                raise IOError("Temporary failure")
            return f"Processed {file_path}"
        
        result = await process_file("test.pdf")
        assert result == "Processed test.pdf"
        assert attempt_count == 2
    
    def test_agent_execution_with_circuit_breaker(self):
        """Test agent execution with circuit breaker."""
        cb = CircuitBreaker(failure_threshold=2)
        execution_count = 0
        
        def execute_agent():
            nonlocal execution_count
            execution_count += 1
            raise Exception("Agent failed")
        
        # First two failures
        for _ in range(2):
            with pytest.raises(Exception):
                cb.call(execute_agent)
        
        # Circuit should be open now
        assert cb.state == "OPEN"
        
        # Should block further attempts
        with pytest.raises(Exception, match="Circuit breaker is OPEN"):
            cb.call(execute_agent)
        
        # Execution count should be 2, not 3
        assert execution_count == 2
    
    def test_graceful_degradation_vector_db(self):
        """Test graceful degradation when vector DB fails."""
        has_fallback = MemoryStorageErrorHandler.handle_vector_db_connection_error(
            Exception("DB connection failed")
        )
        
        # System should indicate fallback is available
        assert has_fallback is True
    
    def test_error_context_creation(self):
        """Test error context captures all necessary information."""
        error = ValueError("Test error")
        
        context = ErrorContext(
            category=ErrorCategory.AGENT_EXECUTION,
            severity=ErrorSeverity.HIGH,
            error=error,
            component="AgentExecutor",
            operation="execute_task",
            metadata={"agent_id": "123", "task_id": "456"}
        )
        
        assert context.category == ErrorCategory.AGENT_EXECUTION
        assert context.severity == ErrorSeverity.HIGH
        assert context.component == "AgentExecutor"
        assert context.operation == "execute_task"
        assert context.metadata["agent_id"] == "123"
        assert context.stack_trace != ""


class TestResourceConstraints:
    """Test system behavior under resource constraints."""
    
    def test_memory_limit_enforcement(self):
        """Test memory limit enforcement."""
        manager = ErrorRecoveryManager()
        
        # Simulate memory capacity exceeded
        for i in range(1500):
            error_context = ErrorContext(
                category=ErrorCategory.MEMORY_STORAGE,
                severity=ErrorSeverity.LOW,
                error=Exception(f"Error {i}"),
                component="Test"
            )
            manager.record_error(error_context)
        
        # Should keep only last 1000
        assert len(manager.error_history) == 1000
    
    def test_concurrent_error_handling(self):
        """Test concurrent error handling."""
        manager = ErrorRecoveryManager()
        
        def record_errors():
            for i in range(100):
                manager.handle_error(
                    error=Exception(f"Error {i}"),
                    category=ErrorCategory.SYSTEM,
                    severity=ErrorSeverity.LOW,
                    component="Test"
                )
        
        import threading
        threads = [threading.Thread(target=record_errors) for _ in range(5)]
        
        for t in threads:
            t.start()
        
        for t in threads:
            t.join()
        
        # Should have recorded errors from all threads
        assert len(manager.error_history) >= 500


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
