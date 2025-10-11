"""Tests for Agent Executor functionality."""

import os
import pytest
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from app.core.agent_executor import (
    AgentExecutor, AgentLifecycleManager, SubAgentRunner, ResultCollector,
    AgentExecutionStatus, AgentExecutionResult
)
from app.core.state import ExecutionState, Task, AgentSpec, AgentLimits, TaskStatus
from app.config.models import SystemConfig, AgentGeneratorConfig


OPENAI_REQUIRED = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OpenAI API key not set",
)



@pytest.fixture
def system_config():
    """Create a test system configuration."""
    return SystemConfig(
        agent_generator=AgentGeneratorConfig(
            max_concurrent_agents=3,
            default_agent_max_steps=4,
            default_agent_max_tokens=2000,
            agent_timeout_seconds=30
        )
    )


@pytest.fixture
def sample_task():
    """Create a sample task for testing."""
    return Task(
        id="test_task_1",
        name="Test Task",
        description="A test task for agent execution",
        expected_output="Test result",
        success_criteria=["Task completed successfully"]
    )


@pytest.fixture
def sample_agent_spec():
    """Create a sample agent specification."""
    return AgentSpec(
        id="test_agent_1",
        system_prompt="You are a test agent.",
        tools=["text_generator", "memory_retrieval"],
        limits=AgentLimits(max_steps=4, max_tokens=2000, timeout_seconds=30),
        output_contract={"format": "structured", "required_fields": ["result", "status"]},
        created_for_task="test_task_1"
    )


@pytest.fixture
def execution_state(sample_task, sample_agent_spec):
    """Create an execution state with sample data."""
    state = ExecutionState(
        session_id="test_session",
        user_request="Test request"
    )
    
    # Add task and agent spec
    state.add_task(sample_task)
    state.add_agent_spec(sample_agent_spec)
    
    # Add generated agents context
    state.context["generated_agents"] = [{
        "agent_id": sample_agent_spec.id,
        "task_id": sample_task.id,
        "agent_type": "general",
        "complexity": "moderate",
        "tools": sample_agent_spec.tools,
        "limits": {
            "max_steps": sample_agent_spec.limits.max_steps,
            "max_tokens": sample_agent_spec.limits.max_tokens,
            "timeout_seconds": sample_agent_spec.limits.timeout_seconds
        }
    }]
    
    return state


class TestSubAgentRunner:
    """Test the SubAgentRunner class."""
    
    @OPENAI_REQUIRED
    def test_execute_agent_success(self, system_config, sample_agent_spec, sample_task):
        """Test successful agent execution."""
        runner = SubAgentRunner(system_config)
        
        result = runner.execute_agent(sample_agent_spec, sample_task, {})
        
        assert result.agent_id == sample_agent_spec.id
        assert result.task_id == sample_task.id
        assert result.status == AgentExecutionStatus.COMPLETED
        assert result.result is not None
        assert result.execution_time > 0
        assert "status" in result.result
        assert "execution_summary" in result.result
    
    @OPENAI_REQUIRED
    def test_execute_agent_with_output_contract(self, system_config, sample_task):
        """Test agent execution with specific output contract."""
        agent_spec = AgentSpec(
            id="qa_agent",
            system_prompt="You are a QA agent.",
            tools=["search_engine"],
            limits=AgentLimits(),
            output_contract={
                "format": "structured",
                "required_fields": ["answer", "citations", "confidence_score"]
            },
            created_for_task=sample_task.id
        )
        
        runner = SubAgentRunner(system_config)
        result = runner.execute_agent(agent_spec, sample_task, {})
        
        assert result.status == AgentExecutionStatus.COMPLETED
        assert "answer" in result.result
        assert "citations" in result.result
        assert "confidence_score" in result.result


class TestAgentLifecycleManager:
    """Test the AgentLifecycleManager class."""
    
    @OPENAI_REQUIRED
    def test_instantiate_agent(self, system_config, sample_agent_spec, sample_task):
        """Test agent instantiation."""
        manager = AgentLifecycleManager(system_config)
        manager.start_monitoring()
        
        try:
            agent_id = manager.instantiate_agent(sample_agent_spec, sample_task, {})
            
            assert agent_id == sample_agent_spec.id
            assert agent_id in manager.get_running_agents()
            
            # Wait for completion
            time.sleep(0.5)
            
            # Check if agent completed
            status = manager.get_agent_status(agent_id)
            assert status in [AgentExecutionStatus.COMPLETED, AgentExecutionStatus.RUNNING]
            
        finally:
            manager.stop_monitoring()
    
    @OPENAI_REQUIRED
    def test_cancel_agent(self, system_config, sample_agent_spec, sample_task):
        """Test agent cancellation."""
        manager = AgentLifecycleManager(system_config)
        manager.start_monitoring()
        
        try:
            agent_id = manager.instantiate_agent(sample_agent_spec, sample_task, {})
            
            # Cancel immediately
            cancelled = manager.cancel_agent(agent_id)
            
            assert cancelled is True
            assert agent_id not in manager.get_running_agents()
            
            result = manager.get_agent_result(agent_id)
            assert result is not None
            assert result.status == AgentExecutionStatus.CANCELLED
            
        finally:
            manager.stop_monitoring()
    
    @OPENAI_REQUIRED
    def test_max_concurrent_agents(self, system_config, sample_agent_spec, sample_task):
        """Test maximum concurrent agents limit."""
        # Set low limit for testing
        system_config.agent_generator.max_concurrent_agents = 1
        
        manager = AgentLifecycleManager(system_config)
        manager.start_monitoring()
        
        try:
            # Start first agent
            agent_id_1 = manager.instantiate_agent(sample_agent_spec, sample_task, {})
            assert agent_id_1 == sample_agent_spec.id
            
            # Try to start second agent (should fail)
            agent_spec_2 = AgentSpec(
                id="test_agent_2",
                system_prompt="Second agent",
                tools=[],
                limits=AgentLimits(),
                output_contract={},
                created_for_task=sample_task.id
            )
            
            with pytest.raises(RuntimeError, match="Maximum concurrent agents limit reached"):
                manager.instantiate_agent(agent_spec_2, sample_task, {})
                
        finally:
            manager.stop_monitoring()


class TestResultCollector:
    """Test the ResultCollector class."""
    
    def test_collect_results_success(self, system_config, sample_task):
        """Test result collection with successful agents."""
        collector = ResultCollector(system_config)
        
        # Create mock results
        results = [
            AgentExecutionResult(
                agent_id="agent_1",
                task_id=sample_task.id,
                status=AgentExecutionStatus.COMPLETED,
                result={"result": "Success 1", "status": "completed"},
                execution_time=1.5,
                tokens_used=100
            ),
            AgentExecutionResult(
                agent_id="agent_2", 
                task_id="task_2",
                status=AgentExecutionStatus.COMPLETED,
                result={"result": "Success 2", "status": "completed"},
                execution_time=2.0,
                tokens_used=150
            )
        ]
        
        integrated = collector.collect_results(results, [sample_task])
        
        assert integrated["execution_summary"]["total_agents"] == 2
        assert integrated["execution_summary"]["successful_agents"] == 2
        assert integrated["execution_summary"]["failed_agents"] == 0
        assert integrated["execution_summary"]["total_execution_time"] == 3.5
        assert integrated["execution_summary"]["total_tokens_used"] == 250
        
        assert integrated["synthesis"]["overall_success"] is True
        assert integrated["synthesis"]["completion_rate"] == 1.0
    
    def test_collect_results_with_failures(self, system_config, sample_task):
        """Test result collection with some failures."""
        collector = ResultCollector(system_config)
        
        results = [
            AgentExecutionResult(
                agent_id="agent_1",
                task_id=sample_task.id,
                status=AgentExecutionStatus.COMPLETED,
                result={"result": "Success", "status": "completed"},
                execution_time=1.0,
                tokens_used=100
            ),
            AgentExecutionResult(
                agent_id="agent_2",
                task_id="task_2", 
                status=AgentExecutionStatus.FAILED,
                error="Test error",
                execution_time=0.5,
                tokens_used=50
            )
        ]
        
        integrated = collector.collect_results(results, [sample_task])
        
        assert integrated["execution_summary"]["successful_agents"] == 1
        assert integrated["execution_summary"]["failed_agents"] == 1
        assert integrated["synthesis"]["completion_rate"] == 0.5
        assert integrated["synthesis"]["overall_success"] is False
        assert len(integrated["errors"]) == 1
        assert integrated["errors"][0]["error"] == "Test error"


class TestAgentExecutor:
    """Test the AgentExecutor class."""
    
    @OPENAI_REQUIRED
    def test_execute_with_agents(self, system_config, execution_state):
        """Test agent executor with valid agents."""
        executor = AgentExecutor(system_config)
        
        try:
            result_state = executor.execute(execution_state)
            
            # Check that execution completed
            assert "agent_execution_results" in result_state.context
            
            results = result_state.context["agent_execution_results"]
            assert "execution_summary" in results
            assert "task_results" in results
            assert "synthesis" in results
            
            # Check that task status was updated
            task = result_state.get_task_by_id("test_task_1")
            assert task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED]
            
        finally:
            executor.lifecycle_manager.stop_monitoring()
    
    def test_execute_no_agents(self, system_config):
        """Test agent executor with no agents."""
        executor = AgentExecutor(system_config)
        
        state = ExecutionState(session_id="test", user_request="test")
        # No generated_agents in context
        
        try:
            result_state = executor.execute(state)
            
            assert result_state.is_complete
            
        finally:
            executor.lifecycle_manager.stop_monitoring()
    
    def test_validate_inputs(self, system_config, execution_state):
        """Test input validation."""
        executor = AgentExecutor(system_config)
        
        try:
            # Valid state
            assert executor.validate_inputs(execution_state) is True
            
            # Invalid state - no generated agents
            empty_state = ExecutionState(session_id="test", user_request="test")
            assert executor.validate_inputs(empty_state) is False
            
            # Invalid state - missing agent spec
            invalid_state = ExecutionState(session_id="test", user_request="test")
            invalid_state.context["generated_agents"] = [{
                "agent_id": "missing_agent",
                "task_id": "missing_task"
            }]
            assert executor.validate_inputs(invalid_state) is False
            
        finally:
            executor.lifecycle_manager.stop_monitoring()


if __name__ == "__main__":
    pytest.main([__file__])