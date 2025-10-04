# Error Handling and Monitoring System

## Overview

The Local Agent Studio includes a comprehensive error handling and monitoring system that provides:

- **Error Recovery**: Circuit breakers, retry logic, and graceful degradation
- **Structured Logging**: JSON-formatted logs with metadata and trace IDs
- **Performance Monitoring**: Metrics collection and resource usage tracking
- **Execution Tracing**: Distributed tracing across components
- **Alerting**: Error reporting and notification system

## Error Handling

### Circuit Breaker Pattern

Prevents cascading failures by opening the circuit after a threshold of failures:

```python
from app.core.error_handling import CircuitBreaker

# Create circuit breaker
cb = CircuitBreaker(
    failure_threshold=5,
    recovery_timeout=60.0,
    expected_exception=Exception
)

# Use circuit breaker
try:
    result = cb.call(risky_function, arg1, arg2)
except Exception as e:
    # Handle failure
    pass
```

**States:**
- `CLOSED`: Normal operation, requests pass through
- `OPEN`: Too many failures, requests fail immediately
- `HALF_OPEN`: Testing if service recovered

### Retry Strategy

Automatic retry with exponential backoff:

```python
from app.core.error_handling import with_retry

@with_retry(max_attempts=3, retry_on=(IOError, ConnectionError))
async def process_file(file_path: str):
    # Your code here
    pass
```

**Features:**
- Configurable max attempts
- Exponential backoff with jitter
- Selective exception handling
- Works with both sync and async functions

### Error Recovery Manager

Central coordinator for error handling strategies:

```python
from app.core.error_handling import error_recovery_manager, ErrorCategory, ErrorSeverity

# Handle an error
action = error_recovery_manager.handle_error(
    error=exception,
    category=ErrorCategory.FILE_PROCESSING,
    severity=ErrorSeverity.HIGH,
    component="FileProcessor",
    operation="read_file",
    metadata={"file_path": "/path/to/file"}
)

# Get recovery action
if action.action_type == "retry":
    # Retry the operation
    pass
elif action.action_type == "fallback":
    # Use fallback mechanism
    pass
```

**Error Categories:**
- `FILE_PROCESSING`: File ingestion and processing errors
- `AGENT_EXECUTION`: Agent execution and tool failures
- `MEMORY_STORAGE`: Memory and storage issues
- `VECTOR_DB`: Vector database connection errors
- `API_CALL`: External API call failures
- `CONFIGURATION`: Configuration loading errors
- `SYSTEM`: System-level errors

**Recovery Actions:**
- `retry`: Retry the operation with backoff
- `fallback`: Use alternative mechanism
- `skip`: Skip and continue
- `abort`: Stop execution

### Specialized Error Handlers

#### File Processing Errors

```python
from app.core.error_handling import FileProcessingErrorHandler

# Handle unsupported format
result = FileProcessingErrorHandler.handle_unsupported_format(
    file_path="document.xyz",
    error=exception
)

# Handle OCR failure
result = FileProcessingErrorHandler.handle_ocr_failure(
    file_path="scan.pdf",
    error=exception
)

# Handle corrupted file
result = FileProcessingErrorHandler.handle_corrupted_file(
    file_path="broken.docx",
    error=exception
)
```

#### Agent Execution Errors

```python
from app.core.error_handling import AgentExecutionErrorHandler

# Handle resource limit exceeded
AgentExecutionErrorHandler.handle_resource_limit_exceeded(
    agent_id="agent_123",
    limit_type="max_steps",
    error=exception
)

# Handle tool failure
should_retry = AgentExecutionErrorHandler.handle_tool_failure(
    agent_id="agent_123",
    tool_name="search_tool",
    error=exception
)

# Handle infinite loop
AgentExecutionErrorHandler.handle_infinite_loop(
    agent_id="agent_123",
    step_count=100,
    error=exception
)
```

#### Memory Storage Errors

```python
from app.core.error_handling import MemoryStorageErrorHandler

# Handle vector DB connection error
has_fallback = MemoryStorageErrorHandler.handle_vector_db_connection_error(
    error=exception
)

# Handle memory capacity exceeded
MemoryStorageErrorHandler.handle_memory_capacity_exceeded(
    error=exception
)

# Handle embedding API failure
should_retry = MemoryStorageErrorHandler.handle_embedding_api_failure(
    error=exception,
    retry_count=1
)
```

## Monitoring System

### Structured Logging

JSON-formatted logs with consistent structure:

```python
from app.core.monitoring import StructuredLogger

logger = StructuredLogger("MyComponent")

# Log with metadata
logger.info(
    "Processing request",
    metadata={"user_id": "123", "request_id": "abc"},
    trace_id="trace_xyz"
)

# Different log levels
logger.debug("Debug information")
logger.warning("Warning message")
logger.error("Error occurred")
logger.critical("Critical failure")

# Get recent logs
recent_logs = logger.get_recent_logs(count=100)
```

### Performance Monitoring

Track metrics and resource usage:

```python
from app.core.monitoring import PerformanceMonitor, performance_monitor

monitor = PerformanceMonitor()

# Record metrics
monitor.record_gauge("cpu_usage", 45.5)
monitor.increment_counter("api_calls", tags={"endpoint": "/chat"})
monitor.record_histogram("response_time_ms", 123.4)

# Get metric statistics
stats = monitor.get_metric_stats("response_time_ms")
# Returns: {"count": N, "sum": X, "min": Y, "max": Z, "avg": A}

# Capture resource usage
usage = monitor.capture_resource_usage()
print(f"CPU: {usage.cpu_percent}%, Memory: {usage.memory_mb}MB")

# Start continuous monitoring
monitor.start_monitoring(interval=5.0)
```

**Using Decorators:**

```python
from app.core.monitoring import monitor_performance

@monitor_performance("my_function")
async def my_function():
    # Your code here
    pass

# Automatically records:
# - my_function.duration_ms (histogram)
# - my_function.calls (counter)
# - my_function.errors (counter on exception)
```

### Execution Tracing

Distributed tracing across components:

```python
from app.core.monitoring import ExecutionTracer, execution_tracer, trace_execution

tracer = ExecutionTracer()

# Manual tracing
trace_id = tracer.start_trace(
    component="Orchestrator",
    operation="process_request",
    metadata={"user_id": "123"}
)

# Do work...

tracer.end_trace(trace_id, status="success")

# Using decorator
@trace_execution("MyComponent", "my_operation")
async def my_operation():
    # Your code here
    pass

# Using context manager
from app.core.monitoring import trace_context

with trace_context("MyComponent", "operation") as trace_id:
    # Your code here
    pass
```

**Trace Hierarchy:**

```python
# Create parent-child traces
parent_id = tracer.start_trace("Parent", "parent_op")
child_id = tracer.start_trace("Child", "child_op", parent_trace_id=parent_id)

# Get trace tree
tree = tracer.get_trace_tree(parent_id)
# Returns hierarchical structure with all child traces
```

### Alert Management

Error reporting and notifications:

```python
from app.core.monitoring import alert_manager

# Send alert
alert_manager.send_alert(
    severity="high",
    component="FileProcessor",
    message="File processing failed",
    metadata={"file_path": "/path/to/file"}
)

# Register alert handler
def my_alert_handler(alert):
    print(f"Alert: {alert['message']}")
    # Send email, Slack notification, etc.

alert_manager.register_handler(my_alert_handler)

# Get recent alerts
alerts = alert_manager.get_recent_alerts(count=50)
```

## Integration Examples

### Complete Error Handling Flow

```python
from app.core.error_handling import with_retry, CircuitBreaker, error_recovery_manager
from app.core.monitoring import StructuredLogger, trace_execution

logger = StructuredLogger("FileProcessor")
cb = CircuitBreaker(failure_threshold=3)

@trace_execution("FileProcessor", "process_file")
@with_retry(max_attempts=3, retry_on=(IOError,))
async def process_file(file_path: str):
    try:
        # Use circuit breaker for external service
        result = await cb.call_async(external_service_call, file_path)
        logger.info("File processed successfully", metadata={"file": file_path})
        return result
    except Exception as e:
        # Handle error with recovery manager
        action = error_recovery_manager.handle_error(
            error=e,
            category=ErrorCategory.FILE_PROCESSING,
            severity=ErrorSeverity.HIGH,
            component="FileProcessor",
            operation="process_file"
        )
        
        if action.action_type == "fallback":
            # Use fallback mechanism
            return await fallback_processor(file_path)
        raise
```

### Monitoring with Metrics

```python
from app.core.monitoring import (
    StructuredLogger,
    performance_monitor,
    trace_context
)

logger = StructuredLogger("AgentExecutor")

async def execute_agent(agent_id: str):
    with trace_context("AgentExecutor", "execute", {"agent_id": agent_id}) as trace_id:
        logger.info("Starting agent execution", trace_id=trace_id)
        
        # Record metrics
        performance_monitor.increment_counter("agent_executions")
        
        start_time = time.time()
        try:
            result = await agent.run()
            
            # Record success metrics
            duration = (time.time() - start_time) * 1000
            performance_monitor.record_histogram("agent_duration_ms", duration)
            performance_monitor.increment_counter("agent_success")
            
            logger.info("Agent completed successfully", trace_id=trace_id)
            return result
            
        except Exception as e:
            performance_monitor.increment_counter("agent_errors")
            logger.error(f"Agent failed: {e}", trace_id=trace_id)
            raise
```

## Best Practices

1. **Use Circuit Breakers for External Services**: Protect against cascading failures
2. **Add Retry Logic for Transient Failures**: Network issues, temporary unavailability
3. **Log with Context**: Include trace IDs and metadata for debugging
4. **Monitor Critical Paths**: Add metrics to important operations
5. **Set Appropriate Thresholds**: Configure failure thresholds based on SLAs
6. **Handle Errors Gracefully**: Provide fallback mechanisms where possible
7. **Use Structured Logging**: Makes log analysis and searching easier
8. **Trace Distributed Operations**: Track execution flow across components
9. **Monitor Resource Usage**: Detect memory leaks and performance issues
10. **Set Up Alerts**: Get notified of critical errors immediately

## Configuration

Error handling and monitoring can be configured through environment variables or configuration files:

```yaml
# config/monitoring.yaml
error_handling:
  circuit_breaker:
    default_failure_threshold: 5
    default_recovery_timeout: 60.0
  retry:
    default_max_attempts: 3
    default_initial_delay: 1.0
    default_max_delay: 60.0

monitoring:
  logging:
    level: INFO
    buffer_size: 1000
  metrics:
    retention_count: 1000
  tracing:
    enabled: true
    trace_retention: 1000
  resource_monitoring:
    enabled: true
    interval: 5.0
```

## Testing

Comprehensive tests are available in:
- `tests/test_error_handling.py`: Error handling tests
- `tests/test_monitoring.py`: Monitoring system tests
- `tests/test_system_integration.py`: Integration and load tests

Run tests:
```bash
pytest tests/test_error_handling.py -v
pytest tests/test_monitoring.py -v
pytest tests/test_system_integration.py -v
```

## Performance Considerations

- **Log Buffer**: Limited to 1000 entries per logger
- **Metric History**: Limited to 1000 entries per metric
- **Trace History**: Limited to 1000 completed traces
- **Resource Monitoring**: Runs in background thread with configurable interval
- **Thread Safety**: All components are thread-safe for concurrent access

## Troubleshooting

### High Memory Usage

Check metric and log buffer sizes:
```python
# Get all metrics
all_metrics = performance_monitor.get_all_metrics()
print(f"Total metrics: {sum(len(m) for m in all_metrics.values())}")

# Get log count
logs = logger.get_recent_logs(count=2000)
print(f"Log entries: {len(logs)}")
```

### Circuit Breaker Stuck Open

Check circuit breaker state and reset if needed:
```python
cb = error_recovery_manager.get_circuit_breaker("my_service")
print(f"State: {cb.state}, Failures: {cb.failure_count}")

# Manual reset (use with caution)
cb.failure_count = 0
cb.state = "CLOSED"
```

### Missing Traces

Ensure traces are properly ended:
```python
# Check active traces
active = execution_tracer.get_active_traces()
print(f"Active traces: {len(active)}")

# Always end traces in finally block
trace_id = tracer.start_trace("Component", "operation")
try:
    # Your code
    pass
finally:
    tracer.end_trace(trace_id)
```
