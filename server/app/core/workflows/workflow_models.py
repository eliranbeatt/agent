"""Data models for workflow execution."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any
import uuid


class WorkflowStepStatus(Enum):
    """Status of a workflow step."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class WorkflowStepResult:
    """Result of executing a workflow step."""
    step_name: str
    status: WorkflowStepStatus
    output: Optional[Any] = None
    error: Optional[str] = None
    execution_time_ms: float = 0.0
    tokens_used: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    success_criteria_met: List[str] = field(default_factory=list)
    success_criteria_failed: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class WorkflowExecutionContext:
    """Context for workflow execution."""
    workflow_name: str
    workflow_description: str
    user_request: str
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # Step tracking
    current_step_index: int = 0
    total_steps: int = 0
    step_results: List[WorkflowStepResult] = field(default_factory=list)
    
    # Resource tracking
    total_tokens_used: int = 0
    total_execution_time_ms: float = 0.0
    
    # Context data
    input_data: Dict[str, Any] = field(default_factory=dict)
    intermediate_results: Dict[str, Any] = field(default_factory=dict)
    final_output: Optional[Any] = None
    
    # Status
    is_complete: bool = False
    is_failed: bool = False
    error_message: Optional[str] = None
    
    # Timestamps
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    def start_execution(self) -> None:
        """Mark workflow execution as started."""
        self.started_at = datetime.now()
        
    def complete_execution(self, output: Any = None) -> None:
        """Mark workflow execution as completed."""
        self.is_complete = True
        self.completed_at = datetime.now()
        self.final_output = output
        
    def fail_execution(self, error: str) -> None:
        """Mark workflow execution as failed."""
        self.is_failed = True
        self.completed_at = datetime.now()
        self.error_message = error
        
    def add_step_result(self, result: WorkflowStepResult) -> None:
        """Add a step result to the execution context."""
        self.step_results.append(result)
        self.total_tokens_used += result.tokens_used
        self.total_execution_time_ms += result.execution_time_ms
        
        # Store intermediate results
        if result.output is not None:
            self.intermediate_results[result.step_name] = result.output
            
    def get_step_result(self, step_name: str) -> Optional[WorkflowStepResult]:
        """Get result for a specific step."""
        for result in self.step_results:
            if result.step_name == step_name:
                return result
        return None
        
    def get_intermediate_result(self, step_name: str) -> Optional[Any]:
        """Get intermediate result from a specific step."""
        return self.intermediate_results.get(step_name)
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary."""
        return {
            "workflow_name": self.workflow_name,
            "workflow_description": self.workflow_description,
            "session_id": self.session_id,
            "current_step": self.current_step_index,
            "total_steps": self.total_steps,
            "total_tokens_used": self.total_tokens_used,
            "total_execution_time_ms": self.total_execution_time_ms,
            "is_complete": self.is_complete,
            "is_failed": self.is_failed,
            "error_message": self.error_message,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "step_results": [
                {
                    "step_name": r.step_name,
                    "status": r.status.value,
                    "execution_time_ms": r.execution_time_ms,
                    "tokens_used": r.tokens_used,
                    "success_criteria_met": r.success_criteria_met,
                    "success_criteria_failed": r.success_criteria_failed
                }
                for r in self.step_results
            ]
        }
