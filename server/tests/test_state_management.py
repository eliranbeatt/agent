"""Unit tests for state management components."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock
from typing import Dict, Any

from app.core.state import (
    ExecutionState, Task, TaskStatus, ExecutionPath, 
    AgentSpec, AgentLimits
)


class TestExecutionState:
    """Test cases for ExecutionState class."""
    
    def test_initialization(self):
        """Test ExecutionState initialization."""
        state = ExecutionState(user_request="Test request")
        
        assert state.user_request == "Test request"
        assert state.session_id is not None
        assert state.current_step == 0
        assert state.max_steps == 6
        assert state.tokens_used == 0
        assert state.token_budget == 50000
        assert state.execution_path is None
        assert state.tasks == []
        assert state.active_agents == []
        assert state.is_complete is False
        assert state.is_failed is False
    
    def test_start_execution(self):
        """Test execution start functionality."""
        state = ExecutionState()
        
        assert state.started_at is None
        
        state.start_execution()
        
        assert state.started_at is not None
        assert state.current_step == 0
        assert isinstance(state.started_at, datetime)
    
    def test_step_increment(self):
        """Test step counter increment."""
        state = ExecutionState()
        
        assert state.current_step == 0
        
        state.increment_step()
        assert state.current_step == 1
        
        state.increment_step()
        assert state.current_step == 2
    
    def test_token_usage_tracking(self):
        """Test token usage tracking."""
        state = ExecutionState()
        
        assert state.tokens_used == 0
        
        state.add_tokens_used(100)
        assert state.tokens_used == 100
        
        state.add_tokens_used(250)
        assert state.tokens_used == 350
    
    def test_resource_limits_check(self):
        """Test resource limits checking."""
        state = ExecutionState(max_steps=5, token_budget=1000)
        
        # Within limits
        state.current_step = 3
        state.tokens_used = 500
        assert state.is_within_limits() is True
        
        # Exceed step limit
        state.current_step = 6
        assert state.is_within_limits() is False
        
        # Reset and exceed token limit
        state.current_step = 3
        state.tokens_used = 1500
        assert state.is_within_limits() is False
        
        # Failed state
        state.tokens_used = 500
        state.is_failed = True
        assert state.is_within_limits() is False
    
    def test_task_management(self):
        """Test task management functionality."""
        state = ExecutionState()
        task1 = Task(name="task1", description="First task")
        task2 = Task(name="task2", description="Second task")
        
        # Add tasks
        state.add_task(task1)
        state.add_task(task2)
        
        assert len(state.tasks) == 2
        assert state.get_task_by_id(task1.id) == task1
        assert state.get_task_by_id(task2.id) == task2
        assert state.get_task_by_id("nonexistent") is None
    
    def test_task_completion(self):
        """Test task completion tracking."""
        state = ExecutionState()
        task = Task(name="test_task")
        state.add_task(task)
        
        # Mark as completed
        result = {"output": "test result"}
        state.mark_task_completed(task.id, result)
        
        assert task.status == TaskStatus.COMPLETED
        assert task.result == result
        assert task.completed_at is not None
        assert task.id in state.completed_tasks
    
    def test_task_failure(self):
        """Test task failure tracking."""
        state = ExecutionState()
        task = Task(name="test_task")
        state.add_task(task)
        
        # Mark as failed
        error_msg = "Task failed due to error"
        state.mark_task_failed(task.id, error_msg)
        
        assert task.status == TaskStatus.FAILED
        assert task.error == error_msg
        assert task.id in state.failed_tasks
    
    def test_agent_management(self):
        """Test agent management functionality."""
        state = ExecutionState()
        
        # Create agent spec
        agent_spec = AgentSpec(
            system_prompt="Test prompt",
            tools=["tool1", "tool2"],
            created_for_task="task1"
        )
        
        # Add agent spec
        state.add_agent_spec(agent_spec)
        assert agent_spec.id in state.agent_specs
        
        # Activate agent
        state.activate_agent(agent_spec.id)
        assert agent_spec.id in state.active_agents
        
        # Deactivate agent
        state.deactivate_agent(agent_spec.id)
        assert agent_spec.id not in state.active_agents
    
    def test_execution_logging(self):
        """Test execution logging functionality."""
        state = ExecutionState()
        
        # Log execution step
        state.log_execution_step(
            node_name="TestNode",
            action="test_action",
            details={"key": "value"}
        )
        
        assert len(state.execution_log) == 1
        log_entry = state.execution_log[0]
        
        assert log_entry["node"] == "TestNode"
        assert log_entry["action"] == "test_action"
        assert log_entry["details"]["key"] == "value"
        assert "timestamp" in log_entry
        assert log_entry["step"] == state.current_step
    
    def test_execution_completion(self):
        """Test execution completion."""
        state = ExecutionState()
        result = {"final": "result"}
        
        state.complete_execution(result)
        
        assert state.is_complete is True
        assert state.completed_at is not None
        assert state.final_result == result
    
    def test_execution_failure(self):
        """Test execution failure."""
        state = ExecutionState()
        error_msg = "Execution failed"
        
        state.fail_execution(error_msg)
        
        assert state.is_failed is True
        assert state.completed_at is not None
        assert state.error_message == error_msg
    
    def test_pending_tasks_retrieval(self):
        """Test pending tasks retrieval."""
        state = ExecutionState()
        
        task1 = Task(name="pending1", status=TaskStatus.PENDING)
        task2 = Task(name="completed", status=TaskStatus.COMPLETED)
        task3 = Task(name="pending2", status=TaskStatus.PENDING)
        
        state.add_task(task1)
        state.add_task(task2)
        state.add_task(task3)
        
        pending_tasks = state.get_pending_tasks()
        
        assert len(pending_tasks) == 2
        assert task1 in pending_tasks
        assert task3 in pending_tasks
        assert task2 not in pending_tasks
    
    def test_ready_tasks_with_dependencies(self):
        """Test ready tasks retrieval with dependencies."""
        state = ExecutionState()
        
        # Create tasks with dependencies
        task1 = Task(name="independent", status=TaskStatus.PENDING)
        task2 = Task(name="dependent", status=TaskStatus.PENDING, dependencies=[task1.id])
        task3 = Task(name="multi_dependent", status=TaskStatus.PENDING, dependencies=[task1.id, task2.id])
        
        state.add_task(task1)
        state.add_task(task2)
        state.add_task(task3)
        
        # Initially, only independent task should be ready
        ready_tasks = state.get_ready_tasks()
        assert len(ready_tasks) == 1
        assert task1 in ready_tasks
        
        # Complete task1
        state.mark_task_completed(task1.id)
        
        # Now task2 should be ready
        ready_tasks = state.get_ready_tasks()
        assert len(ready_tasks) == 1
        assert task2 in ready_tasks
        
        # Complete task2
        state.mark_task_completed(task2.id)
        
        # Now task3 should be ready
        ready_tasks = state.get_ready_tasks()
        assert len(ready_tasks) == 1
        assert task3 in ready_tasks
    
    def test_state_serialization(self):
        """Test state serialization to dictionary."""
        state = ExecutionState(user_request="Test request")
        state.execution_path = ExecutionPath.PREDEFINED_WORKFLOW
        state.selected_workflow = "test_workflow"
        state.start_execution()
        
        # Add a task
        task = Task(name="test_task")
        state.add_task(task)
        
        # Serialize to dict
        state_dict = state.to_dict()
        
        # Verify serialization
        assert state_dict["session_id"] == state.session_id
        assert state_dict["user_request"] == "Test request"
        assert state_dict["execution_path"] == "predefined_workflow"
        assert state_dict["selected_workflow"] == "test_workflow"
        assert len(state_dict["tasks"]) == 1
        assert state_dict["tasks"][0]["name"] == "test_task"
        assert state_dict["started_at"] is not None


class TestTask:
    """Test cases for Task class."""
    
    def test_task_initialization(self):
        """Test Task initialization."""
        task = Task(
            name="test_task",
            description="Test description",
            inputs=["input1", "input2"],
            expected_output="output",
            success_criteria=["criteria1", "criteria2"],
            dependencies=["dep1"]
        )
        
        assert task.name == "test_task"
        assert task.description == "Test description"
        assert task.inputs == ["input1", "input2"]
        assert task.expected_output == "output"
        assert task.success_criteria == ["criteria1", "criteria2"]
        assert task.dependencies == ["dep1"]
        assert task.status == TaskStatus.PENDING
        assert task.id is not None
        assert task.created_at is not None
    
    def test_task_status_transitions(self):
        """Test task status transitions."""
        task = Task(name="test_task")
        
        # Initial status
        assert task.status == TaskStatus.PENDING
        assert task.started_at is None
        assert task.completed_at is None
        
        # Start task
        task.status = TaskStatus.IN_PROGRESS
        task.started_at = datetime.now()
        
        assert task.status == TaskStatus.IN_PROGRESS
        assert task.started_at is not None
        
        # Complete task
        task.status = TaskStatus.COMPLETED
        task.completed_at = datetime.now()
        task.result = "Task completed successfully"
        
        assert task.status == TaskStatus.COMPLETED
        assert task.completed_at is not None
        assert task.result == "Task completed successfully"


class TestAgentSpec:
    """Test cases for AgentSpec class."""
    
    def test_agent_spec_initialization(self):
        """Test AgentSpec initialization."""
        limits = AgentLimits(
            max_steps=5,
            max_tokens=2000,
            timeout_seconds=120,
            allowed_tools=["tool1", "tool2"]
        )
        
        agent_spec = AgentSpec(
            system_prompt="Test prompt",
            tools=["tool1", "tool2"],
            limits=limits,
            output_contract={"format": "json"},
            created_for_task="task123"
        )
        
        assert agent_spec.system_prompt == "Test prompt"
        assert agent_spec.tools == ["tool1", "tool2"]
        assert agent_spec.limits == limits
        assert agent_spec.output_contract == {"format": "json"}
        assert agent_spec.created_for_task == "task123"
        assert agent_spec.id is not None
        assert agent_spec.created_at is not None
    
    def test_agent_limits(self):
        """Test AgentLimits functionality."""
        limits = AgentLimits(
            max_steps=10,
            max_tokens=5000,
            timeout_seconds=300,
            allowed_tools=["search", "calculator"]
        )
        
        assert limits.max_steps == 10
        assert limits.max_tokens == 5000
        assert limits.timeout_seconds == 300
        assert limits.allowed_tools == ["search", "calculator"]


class TestTaskStatus:
    """Test cases for TaskStatus enum."""
    
    def test_task_status_values(self):
        """Test TaskStatus enum values."""
        assert TaskStatus.PENDING.value == "pending"
        assert TaskStatus.IN_PROGRESS.value == "in_progress"
        assert TaskStatus.COMPLETED.value == "completed"
        assert TaskStatus.FAILED.value == "failed"
        assert TaskStatus.SKIPPED.value == "skipped"


class TestExecutionPath:
    """Test cases for ExecutionPath enum."""
    
    def test_execution_path_values(self):
        """Test ExecutionPath enum values."""
        assert ExecutionPath.PREDEFINED_WORKFLOW.value == "predefined_workflow"
        assert ExecutionPath.PLANNER_DRIVEN.value == "planner_driven"
        assert ExecutionPath.HYBRID.value == "hybrid"


if __name__ == "__main__":
    pytest.main([__file__])