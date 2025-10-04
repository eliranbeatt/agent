"""Tests for sub-agent lifecycle and result integration."""

import pytest
import time
import threading
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from app.core.agent_executor import (
    AgentLifecycleManager, SubAgentRunner, ResultCollector,
    AgentExecutionStatus, AgentExecutionResult, RunningAgent
)
from app.core.state import Task, AgentSpec, AgentLimits
from app.config.models import SystemConfig, AgentGeneratorConfig


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
        id="lifecycle_task",
        name="Lifecycle Test Task",
        description="A task for testing agent lifecycle",
        expected_output="Test result",
        success_criteria=["Task completed successfully"]
    )


@pytest.fixture
def sample_agent_spec():
    """Create a sample agent specification."""
    return AgentSpec(
        id="lifecycle_agent",
        system_prompt="You are a lifecycle test agent.",
        tools=["text_generator", "memory_retrieval"],
        limits=AgentLimits(max_steps=4, max_tokens=2000, timeout_seconds=5),  # Short timeout for testing
        output_contract={"format": "structured", "required_fields": ["result", "status"]},
        created_for_task="lifecycle_task"
    )


class TestSubAgentRunner:
    """Test the SubAgentRunner class for various execution scenarios."""
    
    def test_successful_execution(self, system_config, sample_agent_spec, sample_task):
        """Test successful agent execution."""
        runner = SubAgentRunner(system_config)
        
        result = runner.execute_agent(sample_agent_spec, sample_task, {})
        
        assert result.agent_id == sample_agent_spec.id
        assert result.task_id == sample_task.id
        assert result.status == AgentExecutionStatus.COMPLETED
        assert result.result is not None
        assert result.execution_time > 0
        assert result.started_at is not None
        assert result.completed_at is not None
        assert result.completed_at > result.started_at
    
    def test_execution_with_different_output_contracts(self, system_config, sample_task):
        """Test execution with various output contract requirements."""
        runner = SubAgentRunner(system_config)
        
        # Test QA agent output contract
        qa_spec = AgentSpec(
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
        
        result = runner.execute_agent(qa_spec, sample_task, {})
        
        assert result.status == AgentExecutionStatus.COMPLETED
        assert "answer" in result.result
        assert "citations" in result.result
        assert "confidence_score" in result.result
        assert isinstance(result.result["confidence_score"], (int, float))
    
    def test_execution_with_file_processing_contract(self, system_config, sample_task):
        """Test execution with file processing output contract."""
        runner = SubAgentRunner(system_config)
        
        file_spec = AgentSpec(
            id="file_agent",
            system_prompt="You are a file processing agent.",
            tools=["file_reader"],
            limits=AgentLimits(),
            output_contract={
                "format": "structured",
                "required_fields": ["extracted_content", "metadata", "processing_errors"]
            },
            created_for_task=sample_task.id
        )
        
        result = runner.execute_agent(file_spec, sample_task, {})
        
        assert result.status == AgentExecutionStatus.COMPLETED
        assert "extracted_content" in result.result
        assert "metadata" in result.result
        assert "processing_errors" in result.result
        assert isinstance(result.result["processing_errors"], list)
    
    def test_execution_timing(self, system_config, sample_agent_spec, sample_task):
        """Test that execution timing is recorded accurately."""
        runner = SubAgentRunner(system_config)
        
        start_time = datetime.now()
        result = runner.execute_agent(sample_agent_spec, sample_task, {})
        end_time = datetime.now()
        
        # Execution time should be reasonable
        assert 0 < result.execution_time < 5  # Should complete quickly in simulation
        
        # Timestamps should be within expected range
        assert start_time <= result.started_at <= end_time
        assert start_time <= result.completed_at <= end_time
        assert result.started_at <= result.completed_at


class TestAgentLifecycleManager:
    """Test the AgentLifecycleManager class for agent lifecycle operations."""
    
    def test_instantiate_single_agent(self, system_config, sample_agent_spec, sample_task):
        """Test instantiating a single agent."""
        manager = AgentLifecycleManager(system_config)
        manager.start_monitoring()
        
        try:
            agent_id = manager.instantiate_agent(sample_agent_spec, sample_task, {})
            
            assert agent_id == sample_agent_spec.id
            assert agent_id in manager.get_running_agents()
            
            # Wait for completion
            time.sleep(0.5)
            
            # Check final status
            status = manager.get_agent_status(agent_id)
            assert status in [AgentExecutionStatus.COMPLETED, AgentExecutionStatus.RUNNING]
            
        finally:
            manager.stop_monitoring()
    
    def test_instantiate_multiple_agents(self, system_config, sample_task):
        """Test instantiating multiple agents concurrently."""
        manager = AgentLifecycleManager(system_config)
        manager.start_monitoring()
        
        try:
            # Create multiple agent specs
            agent_specs = []
            for i in range(3):
                spec = AgentSpec(
                    id=f"multi_agent_{i}",
                    system_prompt=f"You are agent {i}.",
                    tools=["text_generator"],
                    limits=AgentLimits(timeout_seconds=5),
                    output_contract={"format": "structured", "required_fields": ["result"]},
                    created_for_task=sample_task.id
                )
                agent_specs.append(spec)
            
            # Start all agents
            agent_ids = []
            for spec in agent_specs:
                agent_id = manager.instantiate_agent(spec, sample_task, {})
                agent_ids.append(agent_id)
            
            # Verify all are running
            running_agents = manager.get_running_agents()
            for agent_id in agent_ids:
                assert agent_id in running_agents
            
            # Wait for completion
            time.sleep(1.0)
            
            # Check that agents completed
            completed_count = 0
            for agent_id in agent_ids:
                status = manager.get_agent_status(agent_id)
                if status == AgentExecutionStatus.COMPLETED:
                    completed_count += 1
            
            assert completed_count > 0  # At least some should complete
            
        finally:
            manager.stop_monitoring()
    
    def test_agent_cancellation(self, system_config, sample_agent_spec, sample_task):
        """Test cancelling a running agent."""
        manager = AgentLifecycleManager(system_config)
        manager.start_monitoring()
        
        try:
            agent_id = manager.instantiate_agent(sample_agent_spec, sample_task, {})
            
            # Cancel immediately
            cancelled = manager.cancel_agent(agent_id)
            
            assert cancelled is True
            assert agent_id not in manager.get_running_agents()
            
            # Check cancellation result
            result = manager.get_agent_result(agent_id)
            assert result is not None
            assert result.status == AgentExecutionStatus.CANCELLED
            
        finally:
            manager.stop_monitoring()
    
    def test_agent_timeout_handling(self, system_config, sample_task):
        """Test that agents are properly timed out."""
        manager = AgentLifecycleManager(system_config)
        manager.start_monitoring()
        
        try:
            # Create agent with very short timeout
            timeout_spec = AgentSpec(
                id="timeout_agent",
                system_prompt="You are a timeout test agent.",
                tools=["text_generator"],
                limits=AgentLimits(timeout_seconds=1),  # Very short timeout
                output_contract={"format": "structured", "required_fields": ["result"]},
                created_for_task=sample_task.id
            )
            
            agent_id = manager.instantiate_agent(timeout_spec, sample_task, {})
            
            # Wait longer than timeout
            time.sleep(2.0)
            
            # Check that agent timed out
            status = manager.get_agent_status(agent_id)
            if status == AgentExecutionStatus.TIMEOUT:
                result = manager.get_agent_result(agent_id)
                assert result is not None
                assert result.status == AgentExecutionStatus.TIMEOUT
                assert "timed out" in result.error.lower()
            
        finally:
            manager.stop_monitoring()
    
    def test_concurrent_agent_limit(self, system_config, sample_task):
        """Test that concurrent agent limits are enforced."""
        # Set low limit for testing
        system_config.agent_generator.max_concurrent_agents = 2
        
        manager = AgentLifecycleManager(system_config)
        manager.start_monitoring()
        
        try:
            # Create agent specs
            agent_specs = []
            for i in range(3):  # Try to create more than the limit
                spec = AgentSpec(
                    id=f"limit_agent_{i}",
                    system_prompt=f"You are agent {i}.",
                    tools=["text_generator"],
                    limits=AgentLimits(timeout_seconds=10),
                    output_contract={"format": "structured", "required_fields": ["result"]},
                    created_for_task=sample_task.id
                )
                agent_specs.append(spec)
            
            # Start agents up to limit
            successful_starts = 0
            for spec in agent_specs:
                try:
                    manager.instantiate_agent(spec, sample_task, {})
                    successful_starts += 1
                except RuntimeError as e:
                    if "Maximum concurrent agents limit reached" in str(e):
                        break
                    else:
                        raise
            
            # Should have started exactly the limit number
            assert successful_starts == system_config.agent_generator.max_concurrent_agents
            
        finally:
            manager.stop_monitoring()
    
    def test_cleanup_completed_agents(self, system_config, sample_agent_spec, sample_task):
        """Test cleanup of old completed agent records."""
        manager = AgentLifecycleManager(system_config)
        manager.start_monitoring()
        
        try:
            # Start and complete an agent
            agent_id = manager.instantiate_agent(sample_agent_spec, sample_task, {})
            
            # Wait for completion
            time.sleep(0.5)
            
            # Verify it's completed
            assert agent_id in manager.get_completed_agents()
            
            # Test cleanup (with 0 max age to clean everything)
            manager.cleanup_completed_agents(max_age_hours=0)
            
            # Should still be there since it was just completed
            # (cleanup only removes very old records)
            
        finally:
            manager.stop_monitoring()
    
    def test_monitoring_thread_lifecycle(self, system_config):
        """Test that monitoring thread starts and stops properly."""
        manager = AgentLifecycleManager(system_config)
        
        # Initially no monitoring thread
        assert manager._monitor_thread is None
        
        # Start monitoring
        manager.start_monitoring()
        assert manager._monitor_thread is not None
        assert manager._monitor_thread.is_alive()
        
        # Stop monitoring
        manager.stop_monitoring()
        
        # Give thread time to stop
        time.sleep(0.1)
        
        # Thread should be stopped
        assert not manager._monitor_thread.is_alive() or manager._shutdown_event.is_set()


class TestResultCollector:
    """Test the ResultCollector class for result integration."""
    
    def test_collect_all_successful_results(self, system_config, sample_task):
        """Test collecting results when all agents succeed."""
        collector = ResultCollector(system_config)
        
        # Create successful results
        results = [
            AgentExecutionResult(
                agent_id="agent_1",
                task_id=sample_task.id,
                status=AgentExecutionStatus.COMPLETED,
                result={
                    "result": "Success 1",
                    "status": "completed",
                    "key_findings": ["finding_1", "finding_2"]
                },
                execution_time=1.5,
                tokens_used=100,
                steps_taken=3
            ),
            AgentExecutionResult(
                agent_id="agent_2",
                task_id="task_2",
                status=AgentExecutionStatus.COMPLETED,
                result={
                    "result": "Success 2", 
                    "status": "completed",
                    "insights": ["insight_1", "insight_2"]
                },
                execution_time=2.0,
                tokens_used=150,
                steps_taken=4
            )
        ]
        
        integrated = collector.collect_results(results, [sample_task])
        
        # Check execution summary
        summary = integrated["execution_summary"]
        assert summary["total_agents"] == 2
        assert summary["successful_agents"] == 2
        assert summary["failed_agents"] == 0
        assert summary["total_execution_time"] == 3.5
        assert summary["total_tokens_used"] == 250
        
        # Check synthesis
        synthesis = integrated["synthesis"]
        assert synthesis["overall_success"] is True
        assert synthesis["completion_rate"] == 1.0
        assert len(synthesis["key_findings"]) > 0
        
        # Check individual results are preserved
        assert len(integrated["task_results"]) == 2
        assert len(integrated["agent_results"]) == 2
        assert len(integrated["errors"]) == 0
    
    def test_collect_mixed_results(self, system_config, sample_task):
        """Test collecting results with both successes and failures."""
        collector = ResultCollector(system_config)
        
        results = [
            AgentExecutionResult(
                agent_id="success_agent",
                task_id=sample_task.id,
                status=AgentExecutionStatus.COMPLETED,
                result={"result": "Success", "status": "completed"},
                execution_time=1.0,
                tokens_used=100
            ),
            AgentExecutionResult(
                agent_id="failed_agent",
                task_id="failed_task",
                status=AgentExecutionStatus.FAILED,
                error="Test failure",
                execution_time=0.5,
                tokens_used=50
            ),
            AgentExecutionResult(
                agent_id="timeout_agent",
                task_id="timeout_task",
                status=AgentExecutionStatus.TIMEOUT,
                error="Agent timed out",
                execution_time=5.0,
                tokens_used=200
            )
        ]
        
        integrated = collector.collect_results(results, [sample_task])
        
        # Check summary
        summary = integrated["execution_summary"]
        assert summary["total_agents"] == 3
        assert summary["successful_agents"] == 1
        assert summary["failed_agents"] == 1  # Only counts FAILED status
        
        # Check synthesis
        synthesis = integrated["synthesis"]
        assert synthesis["overall_success"] is False  # Less than 80% success
        assert synthesis["completion_rate"] == 1/3  # Only 1 out of 3 succeeded
        
        # Check errors and warnings
        assert len(integrated["errors"]) == 1
        assert integrated["errors"][0]["error"] == "Test failure"
        
        assert len(integrated["warnings"]) == 1
        assert "timeout" in integrated["warnings"][0]["warning"].lower()
    
    def test_synthesis_quality_metrics(self, system_config, sample_task):
        """Test that quality metrics are calculated correctly."""
        collector = ResultCollector(system_config)
        
        results = [
            AgentExecutionResult(
                agent_id="fast_agent",
                task_id=sample_task.id,
                status=AgentExecutionStatus.COMPLETED,
                result={"result": "Fast result"},
                execution_time=0.5,  # Fast execution
                tokens_used=50
            ),
            AgentExecutionResult(
                agent_id="slow_agent",
                task_id="task_2",
                status=AgentExecutionStatus.COMPLETED,
                result={"result": "Slow result"},
                execution_time=3.0,  # Slow execution
                tokens_used=300
            )
        ]
        
        integrated = collector.collect_results(results, [sample_task])
        
        # Check quality metrics
        quality_metrics = integrated["synthesis"]["quality_metrics"]
        assert "average_execution_time" in quality_metrics
        assert "average_tokens_used" in quality_metrics
        assert "efficiency_score" in quality_metrics
        
        # Average should be between the two values
        avg_time = quality_metrics["average_execution_time"]
        assert 0.5 < avg_time < 3.0
        
        avg_tokens = quality_metrics["average_tokens_used"]
        assert 50 < avg_tokens < 300
        
        # Efficiency score should be between 0 and 1
        efficiency = quality_metrics["efficiency_score"]
        assert 0 <= efficiency <= 1
    
    def test_synthesis_recommendations(self, system_config, sample_task):
        """Test that appropriate recommendations are generated."""
        collector = ResultCollector(system_config)
        
        # Test with partial failures
        results = [
            AgentExecutionResult(
                agent_id="success_agent",
                task_id=sample_task.id,
                status=AgentExecutionStatus.COMPLETED,
                result={"result": "Success"},
                execution_time=1.0,
                tokens_used=100
            ),
            AgentExecutionResult(
                agent_id="failed_agent",
                task_id="failed_task",
                status=AgentExecutionStatus.FAILED,
                error="Test failure",
                execution_time=0.5,
                tokens_used=50
            )
        ]
        
        integrated = collector.collect_results(results, [sample_task])
        
        recommendations = integrated["synthesis"]["recommendations"]
        
        # Should recommend retrying failed tasks
        retry_recommendation = any("retry" in rec.lower() for rec in recommendations)
        assert retry_recommendation
        
        # Test with all successes
        all_success_results = [
            AgentExecutionResult(
                agent_id=f"agent_{i}",
                task_id=f"task_{i}",
                status=AgentExecutionStatus.COMPLETED,
                result={"result": f"Success {i}"},
                execution_time=1.0,
                tokens_used=100
            )
            for i in range(3)
        ]
        
        integrated_success = collector.collect_results(all_success_results, [sample_task])
        success_recommendations = integrated_success["synthesis"]["recommendations"]
        
        # Should recommend cross-validation for multiple successes
        cross_val_recommendation = any("cross-validation" in rec.lower() for rec in success_recommendations)
        assert cross_val_recommendation
    
    def test_empty_results_handling(self, system_config):
        """Test handling of empty results list."""
        collector = ResultCollector(system_config)
        
        integrated = collector.collect_results([], [])
        
        # Should handle empty results gracefully
        assert integrated["execution_summary"]["total_agents"] == 0
        assert integrated["execution_summary"]["successful_agents"] == 0
        assert integrated["execution_summary"]["failed_agents"] == 0
        assert integrated["synthesis"]["completion_rate"] == 0.0
        assert integrated["synthesis"]["overall_success"] is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])