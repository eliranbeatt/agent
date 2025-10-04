"""Base classes for LangGraph nodes."""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Type
from datetime import datetime

from .state import ExecutionState


logger = logging.getLogger(__name__)


class BaseLangGraphNode(ABC):
    """
    Base class for all LangGraph nodes in the Local Agent Studio.
    
    Provides common functionality for state management, logging,
    error handling, and flow control between nodes.
    """
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the base node.
        
        Args:
            name: Name of the node for logging and identification
            config: Optional configuration dictionary
        """
        self.name = name
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.{name}")
        
    @abstractmethod
    def execute(self, state: ExecutionState) -> ExecutionState:
        """
        Execute the node's main logic.
        
        Args:
            state: Current execution state
            
        Returns:
            Updated execution state
        """
        pass
    
    def __call__(self, state: ExecutionState) -> ExecutionState:
        """
        Main entry point for node execution with error handling and logging.
        
        Args:
            state: Current execution state
            
        Returns:
            Updated execution state
        """
        # Update current node in state
        state.current_node = self.name
        
        # Log node entry
        self.logger.info(f"Entering node: {self.name}")
        state.log_execution_step(
            node_name=self.name,
            action="enter",
            details={"step": state.current_step}
        )
        
        try:
            # Check resource limits before execution
            if not state.is_within_limits():
                self.logger.warning(f"Resource limits exceeded in node {self.name}")
                state.fail_execution(f"Resource limits exceeded in node {self.name}")
                return state
            
            # Execute the node
            start_time = datetime.now()
            updated_state = self.execute(state)
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Increment step counter
            updated_state.increment_step()
            
            # Log successful execution
            self.logger.info(f"Node {self.name} completed successfully in {execution_time:.2f}s")
            updated_state.log_execution_step(
                node_name=self.name,
                action="complete",
                details={
                    "execution_time": execution_time,
                    "step": updated_state.current_step
                }
            )
            
            return updated_state
            
        except Exception as e:
            # Log error and update state
            error_msg = f"Error in node {self.name}: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            
            state.log_execution_step(
                node_name=self.name,
                action="error",
                details={"error": error_msg}
            )
            
            # Decide whether to fail execution or continue
            if self.is_critical_error(e):
                state.fail_execution(error_msg)
            else:
                # Log non-critical error but continue
                state.log_execution_step(
                    node_name=self.name,
                    action="continue_after_error",
                    details={"error": error_msg}
                )
            
            return state
    
    def is_critical_error(self, error: Exception) -> bool:
        """
        Determine if an error should stop execution.
        
        Args:
            error: The exception that occurred
            
        Returns:
            True if execution should stop, False to continue
        """
        # By default, all errors are critical
        # Subclasses can override for specific error handling
        return True
    
    def validate_state(self, state: ExecutionState) -> bool:
        """
        Validate that the state is suitable for this node.
        
        Args:
            state: Execution state to validate
            
        Returns:
            True if state is valid, False otherwise
        """
        # Basic validation - subclasses can override
        return (
            state is not None and
            not state.is_failed and
            state.is_within_limits()
        )
    
    def get_next_node(self, state: ExecutionState) -> Optional[str]:
        """
        Determine the next node to execute based on current state.
        
        Args:
            state: Current execution state
            
        Returns:
            Name of next node or None if execution should end
        """
        # Default implementation - subclasses should override
        return None
    
    def prepare_state_for_next_node(self, state: ExecutionState, next_node: str) -> ExecutionState:
        """
        Prepare state for the next node in the execution flow.
        
        Args:
            state: Current execution state
            next_node: Name of the next node
            
        Returns:
            Updated execution state
        """
        state.next_node = next_node
        return state
    
    def get_config_value(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value with optional default.
        
        Args:
            key: Configuration key
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        return self.config.get(key, default)
    
    def log_state_transition(self, state: ExecutionState, from_node: str, to_node: str) -> None:
        """
        Log a state transition between nodes.
        
        Args:
            state: Current execution state
            from_node: Source node name
            to_node: Destination node name
        """
        self.logger.debug(f"State transition: {from_node} -> {to_node}")
        state.log_execution_step(
            node_name=self.name,
            action="state_transition",
            details={
                "from_node": from_node,
                "to_node": to_node,
                "session_id": state.session_id
            }
        )


class ConditionalNode(BaseLangGraphNode):
    """
    Base class for nodes that make routing decisions.
    
    Provides utilities for condition evaluation and flow control.
    """
    
    def evaluate_condition(self, state: ExecutionState, condition: str) -> bool:
        """
        Evaluate a condition against the current state.
        
        Args:
            state: Current execution state
            condition: Condition string to evaluate
            
        Returns:
            True if condition is met, False otherwise
        """
        # Simple condition evaluation - can be extended
        if condition == "has_tasks":
            return len(state.tasks) > 0
        elif condition == "within_limits":
            return state.is_within_limits()
        elif condition == "has_active_agents":
            return len(state.active_agents) > 0
        elif condition == "execution_complete":
            return state.is_complete
        else:
            self.logger.warning(f"Unknown condition: {condition}")
            return False
    
    def get_routing_decision(self, state: ExecutionState) -> str:
        """
        Make a routing decision based on current state.
        
        Args:
            state: Current execution state
            
        Returns:
            Name of next node to route to
        """
        # Default implementation - subclasses should override
        return "end"


class ProcessingNode(BaseLangGraphNode):
    """
    Base class for nodes that perform data processing or transformation.
    
    Provides utilities for input validation and output formatting.
    """
    
    def validate_inputs(self, state: ExecutionState) -> bool:
        """
        Validate that required inputs are present in state.
        
        Args:
            state: Current execution state
            
        Returns:
            True if inputs are valid, False otherwise
        """
        # Default implementation - subclasses should override
        return True
    
    def format_output(self, result: Any, state: ExecutionState) -> Any:
        """
        Format the processing result for storage in state.
        
        Args:
            result: Raw processing result
            state: Current execution state
            
        Returns:
            Formatted result
        """
        # Default implementation - return as-is
        return result
    
    def update_state_with_result(self, state: ExecutionState, result: Any) -> ExecutionState:
        """
        Update execution state with processing result.
        
        Args:
            state: Current execution state
            result: Processing result to store
            
        Returns:
            Updated execution state
        """
        # Store result in context by default
        state.context[f"{self.name}_result"] = result
        return state