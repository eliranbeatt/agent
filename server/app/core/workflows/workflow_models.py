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


class WorkflowStatus(Enum):
    """Status of the overall workflow."""
    NOT_STARTED = "not_started"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


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
    status: WorkflowStatus = WorkflowStatus.NOT_STARTED
    is_complete: bool = False
    is_failed: bool = False
    is_paused: bool = False
    error_message: Optional[str] = None
    
    # Timestamps
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    paused_at: Optional[datetime] = None
    resumed_at: Optional[datetime] = None
    
    # Checkpointing
    checkpoints: List[Dict[str, Any]] = field(default_factory=list)
    last_checkpoint_step: int = -1
    
    def start_execution(self) -> None:
        """Mark workflow execution as started."""
        self.started_at = datetime.now()
        self.status = WorkflowStatus.RUNNING
        
    def complete_execution(self, output: Any = None) -> None:
        """Mark workflow execution as completed."""
        self.is_complete = True
        self.status = WorkflowStatus.COMPLETED
        self.completed_at = datetime.now()
        self.final_output = output
        
    def fail_execution(self, error: str) -> None:
        """Mark workflow execution as failed."""
        self.is_failed = True
        self.status = WorkflowStatus.FAILED
        self.completed_at = datetime.now()
        self.error_message = error
        
    def pause_execution(self) -> None:
        """Pause workflow execution."""
        if self.status == WorkflowStatus.RUNNING:
            self.is_paused = True
            self.status = WorkflowStatus.PAUSED
            self.paused_at = datetime.now()
            
    def resume_execution(self) -> None:
        """Resume workflow execution."""
        if self.status == WorkflowStatus.PAUSED:
            self.is_paused = False
            self.status = WorkflowStatus.RUNNING
            self.resumed_at = datetime.now()
            
    def can_resume(self) -> bool:
        """Check if workflow can be resumed."""
        return self.status == WorkflowStatus.PAUSED and not self.is_complete and not self.is_failed
        
    def create_checkpoint(self) -> Dict[str, Any]:
        """
        Create a checkpoint of the current workflow state.
        
        Returns:
            Checkpoint data dictionary
        """
        checkpoint = {
            "checkpoint_id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            "step_index": self.current_step_index,
            "tokens_used": self.total_tokens_used,
            "execution_time_ms": self.total_execution_time_ms,
            "intermediate_results": self.intermediate_results.copy(),
            "step_results": [
                {
                    "step_name": r.step_name,
                    "status": r.status.value,
                    "output": r.output
                }
                for r in self.step_results
            ]
        }
        self.checkpoints.append(checkpoint)
        self.last_checkpoint_step = self.current_step_index
        return checkpoint
        
    def restore_from_checkpoint(self, checkpoint_id: str) -> bool:
        """
        Restore workflow state from a checkpoint.
        
        Args:
            checkpoint_id: ID of the checkpoint to restore
            
        Returns:
            True if restore successful, False otherwise
        """
        for checkpoint in self.checkpoints:
            if checkpoint["checkpoint_id"] == checkpoint_id:
                self.current_step_index = checkpoint["step_index"]
                self.total_tokens_used = checkpoint["tokens_used"]
                self.total_execution_time_ms = checkpoint["execution_time_ms"]
                self.intermediate_results = checkpoint["intermediate_results"].copy()
                
                # Restore step results
                self.step_results = []
                for step_data in checkpoint["step_results"]:
                    result = WorkflowStepResult(
                        step_name=step_data["step_name"],
                        status=WorkflowStepStatus(step_data["status"]),
                        output=step_data["output"]
                    )
                    self.step_results.append(result)
                    
                return True
        return False
        
    def get_latest_checkpoint(self) -> Optional[Dict[str, Any]]:
        """Get the most recent checkpoint."""
        return self.checkpoints[-1] if self.checkpoints else None
        
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
            "status": self.status.value,
            "is_complete": self.is_complete,
            "is_failed": self.is_failed,
            "is_paused": self.is_paused,
            "error_message": self.error_message,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "paused_at": self.paused_at.isoformat() if self.paused_at else None,
            "resumed_at": self.resumed_at.isoformat() if self.resumed_at else None,
            "last_checkpoint_step": self.last_checkpoint_step,
            "num_checkpoints": len(self.checkpoints),
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
