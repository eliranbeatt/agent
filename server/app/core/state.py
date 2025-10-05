"""State management for LangGraph execution."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any
import uuid


class TaskStatus(Enum):
    """Status of individual tasks in the execution."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class ExecutionPath(Enum):
    """Execution path chosen by the orchestrator."""
    PREDEFINED_WORKFLOW = "predefined_workflow"
    PLANNER_DRIVEN = "planner_driven"
    HYBRID = "hybrid"


@dataclass
class Task:
    """Individual task in the execution plan."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    inputs: List[str] = field(default_factory=list)
    expected_output: str = ""
    success_criteria: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    assigned_agent: Optional[str] = None
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Any] = None
    error: Optional[str] = None


@dataclass
class AgentLimits:
    """Resource limits for individual agents."""
    max_steps: int = 4
    max_tokens: int = 2000
    timeout_seconds: int = 120
    allowed_tools: List[str] = field(default_factory=list)


@dataclass
class AgentSpec:
    """Specification for a dynamically created agent."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    system_prompt: str = ""
    tools: List[str] = field(default_factory=list)
    limits: AgentLimits = field(default_factory=AgentLimits)
    output_contract: Dict[str, Any] = field(default_factory=dict)
    created_for_task: str = ""
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class ExecutionState:
    """
    Central state management for LangGraph execution.
    
    Tracks session information, step counting, resource limits,
    and coordinates between all system components.
    """
    
    # Session tracking
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_request: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    
    # Step counting and limits
    current_step: int = 0
    max_steps: int = 6
    
    # Resource tracking
    tokens_used: int = 0
    token_budget: int = 50000
    
    # Execution path and workflow
    execution_path: Optional[ExecutionPath] = None
    selected_workflow: Optional[str] = None
    workflow_confidence: float = 0.0
    
    # Task management
    tasks: List[Task] = field(default_factory=list)
    completed_tasks: List[str] = field(default_factory=list)
    failed_tasks: List[str] = field(default_factory=list)
    
    # Agent management
    active_agents: List[str] = field(default_factory=list)
    agent_specs: Dict[str, AgentSpec] = field(default_factory=dict)
    
    # Context and memory
    context: Dict[str, Any] = field(default_factory=dict)
    retrieved_chunks: List[Dict[str, Any]] = field(default_factory=list)
    memory_hits: List[Dict[str, Any]] = field(default_factory=list)
    
    # Execution tracking
    execution_log: List[Dict[str, Any]] = field(default_factory=list)
    current_node: Optional[str] = None
    next_node: Optional[str] = None
    
    # Status and results
    is_complete: bool = False
    is_failed: bool = False
    final_result: Optional[Any] = None
    error_message: Optional[str] = None
    
    # Timestamps
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    def start_execution(self) -> None:
        """Mark execution as started."""
        self.started_at = datetime.now()
        self.current_step = 0
        
    def increment_step(self) -> None:
        """Increment the current step counter."""
        self.current_step += 1
        
    def add_tokens_used(self, tokens: int) -> None:
        """Add to the token usage counter."""
        self.tokens_used += tokens
        
    def is_within_limits(self) -> bool:
        """Check if execution is within resource limits."""
        return (
            self.current_step < self.max_steps and
            self.tokens_used < self.token_budget and
            not self.is_failed
        )
        
    def add_task(self, task: Task) -> None:
        """Add a task to the execution plan."""
        self.tasks.append(task)
        
    def get_task_by_id(self, task_id: str) -> Optional[Task]:
        """Get a task by its ID."""
        for task in self.tasks:
            if task.id == task_id:
                return task
        return None
        
    def mark_task_completed(self, task_id: str, result: Any = None) -> None:
        """Mark a task as completed."""
        task = self.get_task_by_id(task_id)
        if task:
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()
            task.result = result
            if task_id not in self.completed_tasks:
                self.completed_tasks.append(task_id)
                
    def mark_task_failed(self, task_id: str, error: str) -> None:
        """Mark a task as failed."""
        task = self.get_task_by_id(task_id)
        if task:
            task.status = TaskStatus.FAILED
            task.error = error
            if task_id not in self.failed_tasks:
                self.failed_tasks.append(task_id)
                
    def add_agent_spec(self, agent_spec: AgentSpec) -> None:
        """Add an agent specification."""
        self.agent_specs[agent_spec.id] = agent_spec
        
    def activate_agent(self, agent_id: str) -> None:
        """Mark an agent as active."""
        if agent_id not in self.active_agents:
            self.active_agents.append(agent_id)
            
    def deactivate_agent(self, agent_id: str) -> None:
        """Mark an agent as inactive."""
        if agent_id in self.active_agents:
            self.active_agents.remove(agent_id)
            
    def log_execution_step(self, node_name: str, action: str, details: Dict[str, Any] = None) -> None:
        """Log an execution step."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "step": self.current_step,
            "node": node_name,
            "action": action,
            "details": details or {}
        }
        self.execution_log.append(log_entry)
        
    def complete_execution(self, result: Any = None) -> None:
        """Mark execution as completed."""
        self.is_complete = True
        self.completed_at = datetime.now()
        self.final_result = result
        
    def fail_execution(self, error: str) -> None:
        """Mark execution as failed."""
        self.is_failed = True
        self.completed_at = datetime.now()
        self.error_message = error
        
    def get_pending_tasks(self) -> List[Task]:
        """Get all pending tasks."""
        return [task for task in self.tasks if task.status == TaskStatus.PENDING]
        
    def get_ready_tasks(self) -> List[Task]:
        """Get tasks that are ready to execute (dependencies satisfied)."""
        ready_tasks = []
        for task in self.get_pending_tasks():
            # Check if all dependencies are completed
            dependencies_met = all(
                dep_id in self.completed_tasks 
                for dep_id in task.dependencies
            )
            if dependencies_met:
                ready_tasks.append(task)
        return ready_tasks
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for serialization."""
        return {
            "session_id": self.session_id,
            "user_request": self.user_request,
            "current_step": self.current_step,
            "max_steps": self.max_steps,
            "tokens_used": self.tokens_used,
            "token_budget": self.token_budget,
            "execution_path": self.execution_path.value if self.execution_path else None,
            "selected_workflow": self.selected_workflow,
            "workflow_confidence": self.workflow_confidence,
            "tasks": [
                {
                    "id": task.id,
                    "name": task.name,
                    "status": task.status.value,
                    "dependencies": task.dependencies,
                    "assigned_agent": task.assigned_agent
                }
                for task in self.tasks
            ],
            "active_agents": self.active_agents,
            "is_complete": self.is_complete,
            "is_failed": self.is_failed,
            "current_node": self.current_node,
            "next_node": self.next_node,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "context": self.context,
            "retrieved_chunks": self.retrieved_chunks,
            "memory_hits": self.memory_hits,
            "final_result": self.final_result,
            "error_message": self.error_message,
        }
