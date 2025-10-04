"""
Comprehensive error handling system for Local Agent Studio.

This module provides error recovery mechanisms, circuit breakers, retry logic,
and graceful degradation for all system components.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, TypeVar, Generic
from dataclasses import dataclass, field
from functools import wraps
import traceback

logger = logging.getLogger(__name__)

T = TypeVar('T')


class ErrorSeverity(Enum):
    """Error severity levels for categorization and handling."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Categories of errors for specialized handling."""
    FILE_PROCESSING = "file_processing"
    AGENT_EXECUTION = "agent_execution"
    MEMORY_STORAGE = "memory_storage"
    VECTOR_DB = "vector_db"
    API_CALL = "api_call"
    CONFIGURATION = "configuration"
    SYSTEM = "system"


@dataclass
class ErrorContext:
    """Context information for error handling and recovery."""
    category: ErrorCategory
    severity: ErrorSeverity
    error: Exception
    timestamp: datetime = field(default_factory=datetime.now)
    component: str = ""
    operation: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    stack_trace: str = ""
    
    def __post_init__(self):
        if not self.stack_trace:
            self.stack_trace = traceback.format_exc()


@dataclass
class RecoveryAction:
    """Defines a recovery action to take after an error."""
    action_type: str  # retry, fallback, skip, abort
    max_attempts: int = 3
    delay_seconds: float = 1.0
    fallback_value: Any = None
    should_notify: bool = True


class CircuitBreaker:
    """
    Circuit breaker pattern implementation to prevent cascading failures.
    
    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Too many failures, requests fail immediately
    - HALF_OPEN: Testing if service recovered
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: type = Exception
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.state = "CLOSED"
    
    def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute function with circuit breaker protection."""
        if self.state == "OPEN":
            if self._should_attempt_reset():
                self.state = "HALF_OPEN"
            else:
                raise Exception(f"Circuit breaker is OPEN for {func.__name__}")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise e
    
    async def call_async(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute async function with circuit breaker protection."""
        if self.state == "OPEN":
            if self._should_attempt_reset():
                self.state = "HALF_OPEN"
            else:
                raise Exception(f"Circuit breaker is OPEN for {func.__name__}")
        
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt recovery."""
        if self.last_failure_time is None:
            return True
        return (datetime.now() - self.last_failure_time).total_seconds() >= self.recovery_timeout
    
    def _on_success(self):
        """Handle successful execution."""
        self.failure_count = 0
        self.state = "CLOSED"
    
    def _on_failure(self):
        """Handle failed execution."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            logger.warning(f"Circuit breaker opened after {self.failure_count} failures")


class RetryStrategy:
    """Configurable retry strategy with exponential backoff."""
    
    def __init__(
        self,
        max_attempts: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True
    ):
        self.max_attempts = max_attempts
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
    
    def get_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt number."""
        delay = min(
            self.initial_delay * (self.exponential_base ** attempt),
            self.max_delay
        )
        
        if self.jitter:
            import random
            delay *= (0.5 + random.random())
        
        return delay


def with_retry(
    max_attempts: int = 3,
    retry_on: tuple = (Exception,),
    strategy: Optional[RetryStrategy] = None
):
    """
    Decorator to add retry logic to functions.
    
    Args:
        max_attempts: Maximum number of retry attempts
        retry_on: Tuple of exception types to retry on
        strategy: RetryStrategy instance for backoff configuration
    """
    if strategy is None:
        strategy = RetryStrategy(max_attempts=max_attempts)
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except retry_on as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        delay = strategy.get_delay(attempt)
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_attempts} failed for {func.__name__}: {e}. "
                            f"Retrying in {delay:.2f}s..."
                        )
                        await asyncio.sleep(delay)
                    else:
                        logger.error(f"All {max_attempts} attempts failed for {func.__name__}")
            
            raise last_exception
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except retry_on as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        delay = strategy.get_delay(attempt)
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_attempts} failed for {func.__name__}: {e}. "
                            f"Retrying in {delay:.2f}s..."
                        )
                        import time
                        time.sleep(delay)
                    else:
                        logger.error(f"All {max_attempts} attempts failed for {func.__name__}")
            
            raise last_exception
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


class ErrorRecoveryManager:
    """
    Central error recovery manager that coordinates error handling strategies
    across all system components.
    """
    
    def __init__(self):
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.error_history: List[ErrorContext] = []
        self.recovery_strategies: Dict[ErrorCategory, RecoveryAction] = self._init_strategies()
    
    def _init_strategies(self) -> Dict[ErrorCategory, RecoveryAction]:
        """Initialize default recovery strategies for each error category."""
        return {
            ErrorCategory.FILE_PROCESSING: RecoveryAction(
                action_type="fallback",
                max_attempts=2,
                delay_seconds=0.5,
                should_notify=True
            ),
            ErrorCategory.AGENT_EXECUTION: RecoveryAction(
                action_type="retry",
                max_attempts=3,
                delay_seconds=2.0,
                should_notify=True
            ),
            ErrorCategory.MEMORY_STORAGE: RecoveryAction(
                action_type="fallback",
                max_attempts=2,
                delay_seconds=1.0,
                should_notify=False
            ),
            ErrorCategory.VECTOR_DB: RecoveryAction(
                action_type="fallback",
                max_attempts=2,
                delay_seconds=1.0,
                should_notify=True
            ),
            ErrorCategory.API_CALL: RecoveryAction(
                action_type="retry",
                max_attempts=3,
                delay_seconds=1.0,
                should_notify=True
            ),
            ErrorCategory.CONFIGURATION: RecoveryAction(
                action_type="fallback",
                max_attempts=1,
                delay_seconds=0.0,
                should_notify=True
            ),
            ErrorCategory.SYSTEM: RecoveryAction(
                action_type="abort",
                max_attempts=1,
                delay_seconds=0.0,
                should_notify=True
            ),
        }
    
    def get_circuit_breaker(self, name: str, **kwargs) -> CircuitBreaker:
        """Get or create a circuit breaker for a component."""
        if name not in self.circuit_breakers:
            self.circuit_breakers[name] = CircuitBreaker(**kwargs)
        return self.circuit_breakers[name]
    
    def record_error(self, error_context: ErrorContext):
        """Record an error for tracking and analysis."""
        self.error_history.append(error_context)
        
        # Keep only recent errors (last 1000)
        if len(self.error_history) > 1000:
            self.error_history = self.error_history[-1000:]
        
        logger.error(
            f"Error recorded - Category: {error_context.category.value}, "
            f"Severity: {error_context.severity.value}, "
            f"Component: {error_context.component}, "
            f"Error: {error_context.error}"
        )
    
    def get_recovery_action(self, category: ErrorCategory) -> RecoveryAction:
        """Get the recovery action for an error category."""
        return self.recovery_strategies.get(category, RecoveryAction(action_type="abort"))
    
    def handle_error(
        self,
        error: Exception,
        category: ErrorCategory,
        severity: ErrorSeverity,
        component: str = "",
        operation: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ) -> RecoveryAction:
        """
        Handle an error and return the appropriate recovery action.
        
        Args:
            error: The exception that occurred
            category: Category of the error
            severity: Severity level
            component: Component where error occurred
            operation: Operation being performed
            metadata: Additional context information
        
        Returns:
            RecoveryAction to take
        """
        error_context = ErrorContext(
            category=category,
            severity=severity,
            error=error,
            component=component,
            operation=operation,
            metadata=metadata or {}
        )
        
        self.record_error(error_context)
        
        return self.get_recovery_action(category)


# Global error recovery manager instance
error_recovery_manager = ErrorRecoveryManager()


# Specialized error handlers for different components

class FileProcessingErrorHandler:
    """Error handler for file processing operations."""
    
    @staticmethod
    def handle_unsupported_format(file_path: str, error: Exception) -> Optional[str]:
        """Handle unsupported file format errors."""
        logger.warning(f"Unsupported file format for {file_path}: {error}")
        error_recovery_manager.record_error(ErrorContext(
            category=ErrorCategory.FILE_PROCESSING,
            severity=ErrorSeverity.MEDIUM,
            error=error,
            component="FileProcessor",
            operation="format_detection",
            metadata={"file_path": file_path}
        ))
        return None
    
    @staticmethod
    def handle_ocr_failure(file_path: str, error: Exception) -> Optional[str]:
        """Handle OCR processing failures."""
        logger.error(f"OCR failed for {file_path}: {error}")
        error_recovery_manager.record_error(ErrorContext(
            category=ErrorCategory.FILE_PROCESSING,
            severity=ErrorSeverity.HIGH,
            error=error,
            component="OCREngine",
            operation="text_extraction",
            metadata={"file_path": file_path}
        ))
        return None
    
    @staticmethod
    def handle_corrupted_file(file_path: str, error: Exception) -> Optional[str]:
        """Handle corrupted file errors."""
        logger.error(f"Corrupted file detected: {file_path}: {error}")
        error_recovery_manager.record_error(ErrorContext(
            category=ErrorCategory.FILE_PROCESSING,
            severity=ErrorSeverity.HIGH,
            error=error,
            component="FileProcessor",
            operation="file_reading",
            metadata={"file_path": file_path}
        ))
        return None


class AgentExecutionErrorHandler:
    """Error handler for agent execution operations."""
    
    @staticmethod
    def handle_resource_limit_exceeded(agent_id: str, limit_type: str, error: Exception):
        """Handle resource limit exceeded errors."""
        logger.warning(f"Agent {agent_id} exceeded {limit_type} limit: {error}")
        error_recovery_manager.record_error(ErrorContext(
            category=ErrorCategory.AGENT_EXECUTION,
            severity=ErrorSeverity.MEDIUM,
            error=error,
            component="AgentExecutor",
            operation="resource_check",
            metadata={"agent_id": agent_id, "limit_type": limit_type}
        ))
    
    @staticmethod
    def handle_tool_failure(agent_id: str, tool_name: str, error: Exception) -> bool:
        """Handle tool execution failures. Returns True if should retry."""
        logger.error(f"Tool {tool_name} failed for agent {agent_id}: {error}")
        error_recovery_manager.record_error(ErrorContext(
            category=ErrorCategory.AGENT_EXECUTION,
            severity=ErrorSeverity.HIGH,
            error=error,
            component="AgentExecutor",
            operation="tool_execution",
            metadata={"agent_id": agent_id, "tool_name": tool_name}
        ))
        return True  # Retry tool execution
    
    @staticmethod
    def handle_infinite_loop(agent_id: str, step_count: int, error: Exception):
        """Handle potential infinite loop detection."""
        logger.critical(f"Infinite loop detected for agent {agent_id} at step {step_count}")
        error_recovery_manager.record_error(ErrorContext(
            category=ErrorCategory.AGENT_EXECUTION,
            severity=ErrorSeverity.CRITICAL,
            error=error,
            component="AgentExecutor",
            operation="step_execution",
            metadata={"agent_id": agent_id, "step_count": step_count}
        ))


class MemoryStorageErrorHandler:
    """Error handler for memory and storage operations."""
    
    @staticmethod
    def handle_vector_db_connection_error(error: Exception) -> bool:
        """Handle vector database connection errors. Returns True if fallback available."""
        logger.error(f"Vector DB connection failed: {error}")
        error_recovery_manager.record_error(ErrorContext(
            category=ErrorCategory.VECTOR_DB,
            severity=ErrorSeverity.HIGH,
            error=error,
            component="VectorStore",
            operation="connection"
        ))
        return True  # Fallback to in-memory search
    
    @staticmethod
    def handle_memory_capacity_exceeded(error: Exception):
        """Handle memory capacity exceeded errors."""
        logger.warning(f"Memory capacity exceeded: {error}")
        error_recovery_manager.record_error(ErrorContext(
            category=ErrorCategory.MEMORY_STORAGE,
            severity=ErrorSeverity.MEDIUM,
            error=error,
            component="MemoryManager",
            operation="storage"
        ))
        # Trigger retention policies
    
    @staticmethod
    def handle_embedding_api_failure(error: Exception, retry_count: int) -> bool:
        """Handle embedding API failures. Returns True if should retry."""
        logger.error(f"Embedding API failed (attempt {retry_count}): {error}")
        error_recovery_manager.record_error(ErrorContext(
            category=ErrorCategory.API_CALL,
            severity=ErrorSeverity.HIGH,
            error=error,
            component="EmbeddingService",
            operation="generate_embeddings"
        ))
        return retry_count < 3


class SystemErrorHandler:
    """Error handler for system-level operations."""
    
    @staticmethod
    def handle_configuration_error(config_file: str, error: Exception) -> bool:
        """Handle configuration errors. Returns True if defaults loaded."""
        logger.error(f"Configuration error in {config_file}: {error}")
        error_recovery_manager.record_error(ErrorContext(
            category=ErrorCategory.CONFIGURATION,
            severity=ErrorSeverity.HIGH,
            error=error,
            component="ConfigLoader",
            operation="load_config",
            metadata={"config_file": config_file}
        ))
        return True  # Load defaults
    
    @staticmethod
    def handle_network_connectivity_error(error: Exception):
        """Handle network connectivity errors."""
        logger.warning(f"Network connectivity issue: {error}")
        error_recovery_manager.record_error(ErrorContext(
            category=ErrorCategory.SYSTEM,
            severity=ErrorSeverity.MEDIUM,
            error=error,
            component="NetworkLayer",
            operation="connection"
        ))
    
    @staticmethod
    def handle_resource_exhaustion(resource_type: str, error: Exception):
        """Handle resource exhaustion errors."""
        logger.critical(f"Resource exhaustion - {resource_type}: {error}")
        error_recovery_manager.record_error(ErrorContext(
            category=ErrorCategory.SYSTEM,
            severity=ErrorSeverity.CRITICAL,
            error=error,
            component="ResourceManager",
            operation="allocation",
            metadata={"resource_type": resource_type}
        ))
