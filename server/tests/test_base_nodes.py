"""Unit tests for base LangGraph node classes."""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime

from app.core.base_nodes import BaseLangGraphNode, ConditionalNode, ProcessingNode
from app.core.state import ExecutionState, Task, TaskStatus


class TestBaseLangGraphNode:
    """Test cases for BaseLangGraphNode class."""
    
    def test_initialization(self):
        """Test node initialization."""
        config = {"param1": "value1", "param2": 42}
        
        class TestNode(BaseLangGraphNode):
            def execute(self, state):
                return state
        
        node = TestNode("TestNode", config)
        
        assert node.name == "TestNode"
        assert node.config == config
        assert node.get_config_value("param1") == "value1"
        assert node.get_config_value("param2") == 42
        assert node.get_config_value("nonexistent", "default") == "default"
    
    def test_successful_execution(self):
        """Test successful node execution."""
        class TestNode(BaseLangGraphNode):
            def execute(self, state):
                state.context["test_result"] = "success"
                return state
        
        node = TestNode("TestNode")
        state = ExecutionState()
        initial_step = state.current_step
        
        result_state = node(state)
        
        assert result_state.current_node == "TestNode"
        assert result_state.current_step == initial_step + 1
        assert result_state.context["test_result"] == "success"
        assert len(result_state.execution_log) >= 2  # enter and complete
        assert not result_state.is_failed
    
    def test_execution_with_resource_limits_exceeded(self):
        """Test execution when resource limits are exceeded."""
        class TestNode(BaseLangGraphNode):
            def execute(self, state):
                return state
        
        node = TestNode("TestNode")
        state = ExecutionState(current_step=10, max_steps=5)  # Exceeds limits
        
        result_state = node(state)
        
        assert result_state.is_failed is True
        assert "resource limits exceeded" in result_state.error_message.lower()
    
    def test_execution_with_critical_error(self):
        """Test execution with critical error."""
        class TestNode(BaseLangGraphNode):
            def execute(self, state):
                raise ValueError("Critical error occurred")
        
        node = TestNode("TestNode")
        state = ExecutionState()
        
        result_state = node(state)
        
        assert result_state.is_failed is True
        assert "critical error occurred" in result_state.error_message.lower()
        assert len(result_state.execution_log) >= 2  # enter and error
    
    def test_execution_with_non_critical_error(self):
        """Test execution with non-critical error."""
        class TestNode(BaseLangGraphNode):
            def execute(self, state):
                raise RuntimeError("Non-critical error")
            
            def is_critical_error(self, error):
                return False  # Override to make error non-critical
        
        node = TestNode("TestNode")
        state = ExecutionState()
        
        result_state = node(state)
        
        assert result_state.is_failed is False  # Should continue despite error
        assert len(result_state.execution_log) >= 3  # enter, error, continue_after_error
    
    def test_state_validation(self):
        """Test state validation."""
        class TestNode(BaseLangGraphNode):
            def execute(self, state):
                return state
        
        node = TestNode("TestNode")
        
        # Valid state
        valid_state = ExecutionState()
        assert node.validate_state(valid_state) is True
        
        # None state
        assert node.validate_state(None) is False
        
        # Failed state
        failed_state = ExecutionState(is_failed=True)
        assert node.validate_state(failed_state) is False
        
        # State exceeding limits
        exceeded_state = ExecutionState(current_step=10, max_steps=5)
        assert node.validate_state(exceeded_state) is False
    
    def test_next_node_determination(self):
        """Test next node determination."""
        class TestNode(BaseLangGraphNode):
            def execute(self, state):
                return state
            
            def get_next_node(self, state):
                return "NextNode"
        
        node = TestNode("TestNode")
        state = ExecutionState()
        
        next_node = node.get_next_node(state)
        assert next_node == "NextNode"
    
    def test_state_preparation_for_next_node(self):
        """Test state preparation for next node."""
        class TestNode(BaseLangGraphNode):
            def execute(self, state):
                return state
        
        node = TestNode("TestNode")
        state = ExecutionState()
        
        prepared_state = node.prepare_state_for_next_node(state, "NextNode")
        
        assert prepared_state.next_node == "NextNode"
    
    def test_state_transition_logging(self):
        """Test state transition logging."""
        class TestNode(BaseLangGraphNode):
            def execute(self, state):
                return state
        
        node = TestNode("TestNode")
        state = ExecutionState()
        
        node.log_state_transition(state, "PreviousNode", "NextNode")
        
        # Check that transition was logged
        transition_logs = [
            log for log in state.execution_log 
            if log.get("action") == "state_transition"
        ]
        assert len(transition_logs) == 1
        
        log_entry = transition_logs[0]
        assert log_entry["details"]["from_node"] == "PreviousNode"
        assert log_entry["details"]["to_node"] == "NextNode"


class TestConditionalNode:
    """Test cases for ConditionalNode class."""
    
    def test_condition_evaluation(self):
        """Test condition evaluation functionality."""
        class TestConditionalNode(ConditionalNode):
            def execute(self, state):
                return state
        
        node = TestConditionalNode("TestConditional")
        
        # Test with tasks
        state_with_tasks = ExecutionState()
        state_with_tasks.add_task(Task(name="test_task"))
        assert node.evaluate_condition(state_with_tasks, "has_tasks") is True
        
        # Test without tasks
        state_without_tasks = ExecutionState()
        assert node.evaluate_condition(state_without_tasks, "has_tasks") is False
        
        # Test within limits
        state_within_limits = ExecutionState(current_step=2, max_steps=5)
        assert node.evaluate_condition(state_within_limits, "within_limits") is True
        
        # Test exceeding limits
        state_exceeding_limits = ExecutionState(current_step=6, max_steps=5)
        assert node.evaluate_condition(state_exceeding_limits, "within_limits") is False
        
        # Test with active agents
        state_with_agents = ExecutionState(active_agents=["agent1", "agent2"])
        assert node.evaluate_condition(state_with_agents, "has_active_agents") is True
        
        # Test execution complete
        state_complete = ExecutionState(is_complete=True)
        assert node.evaluate_condition(state_complete, "execution_complete") is True
        
        # Test unknown condition
        assert node.evaluate_condition(state_complete, "unknown_condition") is False
    
    def test_routing_decision(self):
        """Test routing decision functionality."""
        class TestConditionalNode(ConditionalNode):
            def execute(self, state):
                return state
        
        node = TestConditionalNode("TestConditional")
        state = ExecutionState()
        
        # Default routing decision
        decision = node.get_routing_decision(state)
        assert decision == "end"


class TestProcessingNode:
    """Test cases for ProcessingNode class."""
    
    def test_input_validation(self):
        """Test input validation functionality."""
        class TestProcessingNode(ProcessingNode):
            def execute(self, state):
                return state
            
            def validate_inputs(self, state):
                return "required_input" in state.context
        
        node = TestProcessingNode("TestProcessing")
        
        # Valid inputs
        state_with_input = ExecutionState()
        state_with_input.context["required_input"] = "value"
        assert node.validate_inputs(state_with_input) is True
        
        # Invalid inputs
        state_without_input = ExecutionState()
        assert node.validate_inputs(state_without_input) is False
    
    def test_output_formatting(self):
        """Test output formatting functionality."""
        class TestProcessingNode(ProcessingNode):
            def execute(self, state):
                return state
            
            def format_output(self, result, state):
                return {"formatted": result, "timestamp": "2023-01-01"}
        
        node = TestProcessingNode("TestProcessing")
        state = ExecutionState()
        
        raw_result = "raw output"
        formatted_result = node.format_output(raw_result, state)
        
        assert formatted_result["formatted"] == "raw output"
        assert formatted_result["timestamp"] == "2023-01-01"
    
    def test_state_update_with_result(self):
        """Test state update with processing result."""
        class TestProcessingNode(ProcessingNode):
            def execute(self, state):
                return state
        
        node = TestProcessingNode("TestProcessing")
        state = ExecutionState()
        result = {"output": "processed data"}
        
        updated_state = node.update_state_with_result(state, result)
        
        assert updated_state.context["TestProcessing_result"] == result
    
    def test_processing_node_execution_flow(self):
        """Test complete processing node execution flow."""
        class TestProcessingNode(ProcessingNode):
            def execute(self, state):
                if not self.validate_inputs(state):
                    raise ValueError("Invalid inputs")
                
                # Simulate processing
                raw_result = "processed: " + state.context.get("input_data", "")
                formatted_result = self.format_output(raw_result, state)
                
                return self.update_state_with_result(state, formatted_result)
            
            def validate_inputs(self, state):
                return "input_data" in state.context
            
            def format_output(self, result, state):
                return {"result": result, "node": self.name}
        
        node = TestProcessingNode("TestProcessing")
        
        # Test with valid inputs
        state = ExecutionState()
        state.context["input_data"] = "test data"
        
        result_state = node(state)
        
        assert not result_state.is_failed
        assert "TestProcessing_result" in result_state.context
        
        result = result_state.context["TestProcessing_result"]
        assert result["result"] == "processed: test data"
        assert result["node"] == "TestProcessing"
        
        # Test with invalid inputs
        invalid_state = ExecutionState()
        
        result_state = node(invalid_state)
        
        assert result_state.is_failed
        assert "invalid inputs" in result_state.error_message.lower()


class TestNodeIntegration:
    """Integration tests for node interactions."""
    
    def test_node_chain_execution(self):
        """Test chaining multiple nodes together."""
        class FirstNode(ProcessingNode):
            def execute(self, state):
                state.context["first_result"] = "first output"
                state.next_node = "SecondNode"
                return state
        
        class SecondNode(ProcessingNode):
            def execute(self, state):
                first_result = state.context.get("first_result", "")
                state.context["second_result"] = f"second: {first_result}"
                state.next_node = "ThirdNode"
                return state
        
        class ThirdNode(ConditionalNode):
            def execute(self, state):
                if "second_result" in state.context:
                    state.complete_execution("Chain completed successfully")
                else:
                    state.fail_execution("Missing required data")
                return state
        
        # Execute chain
        state = ExecutionState()
        
        first_node = FirstNode("FirstNode")
        second_node = SecondNode("SecondNode")
        third_node = ThirdNode("ThirdNode")
        
        # Execute first node
        state = first_node(state)
        assert state.next_node == "SecondNode"
        assert state.context["first_result"] == "first output"
        
        # Execute second node
        state = second_node(state)
        assert state.next_node == "ThirdNode"
        assert state.context["second_result"] == "second: first output"
        
        # Execute third node
        state = third_node(state)
        assert state.is_complete is True
        assert state.final_result == "Chain completed successfully"
    
    def test_error_propagation_through_chain(self):
        """Test error propagation through node chain."""
        class FailingNode(BaseLangGraphNode):
            def execute(self, state):
                raise RuntimeError("Node failure")
        
        class RecoveryNode(BaseLangGraphNode):
            def execute(self, state):
                state.context["recovery"] = "recovered"
                return state
            
            def is_critical_error(self, error):
                return False  # Allow recovery
        
        failing_node = FailingNode("FailingNode")
        recovery_node = RecoveryNode("RecoveryNode")
        
        state = ExecutionState()
        
        # Execute failing node
        state = failing_node(state)
        assert state.is_failed is True
        
        # Try recovery (should not work since state is already failed)
        state.is_failed = False  # Reset for test
        state = recovery_node(state)
        assert "recovery" in state.context


if __name__ == "__main__":
    pytest.main([__file__])