"""
Simple graph executor that provides LangGraph-like interface without the dependency.

This is a fallback implementation that manually orchestrates the execution flow
when LangGraph is not available or has compatibility issues.
"""

import logging
from typing import Dict, Any, Optional, Iterator
from datetime import datetime

from .state import ExecutionState, ExecutionPath
from .orchestrator import MainOrchestrator
from .planner import Planner
from .task_identifier import TaskIdentifier
from .agent_generator import AgentGenerator
from .agent_executor import AgentExecutor
from .context.context_manager import ContextManager
from .memory.memory_manager import MemoryManager
from .rag.evaluator import Evaluator
from .workflows.workflow_executor import WorkflowExecutor
from ..config.models import SystemConfig


logger = logging.getLogger(__name__)


class SimpleGraphExecutor:
    """
    Simple graph executor that manually orchestrates component execution.
    
    Provides a similar interface to LangGraph but with manual flow control.
    """
    
    def __init__(self, config: SystemConfig):
        """
        Initialize the executor.
        
        Args:
            config: System configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize all components
        self.orchestrator = MainOrchestrator(config)
        self.planner = Planner(config)
        self.task_identifier = TaskIdentifier(config)
        self.agent_generator = AgentGenerator(config)
        self.agent_executor = AgentExecutor(config)
        self.context_manager = ContextManager(config)
        self.memory_manager = MemoryManager(config)
        self.evaluator = Evaluator(config)
        self.workflow_executor = WorkflowExecutor(config)
        
        # State storage for sessions
        self.states: Dict[str, ExecutionState] = {}
        
    def invoke(self, state: ExecutionState, config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute the full graph and return final state.
        
        Args:
            state: Initial execution state
            config: Optional configuration (e.g., thread_id)
            
        Returns:
            Final state as dictionary
        """
        # Store state
        thread_id = config.get("configurable", {}).get("thread_id") if config else None
        if thread_id:
            self.states[thread_id] = state
        
        # Execute all steps
        for _ in self.stream(state, config):
            pass
        
        # Return final state
        return state.to_dict()
    
    def stream(self, state: ExecutionState, config: Dict[str, Any] = None) -> Iterator[Dict[str, ExecutionState]]:
        """
        Stream execution steps.
        
        Args:
            state: Initial execution state
            config: Optional configuration
            
        Yields:
            Dictionary with node name as key and state as value
        """
        current_state = state
        max_iterations = 20  # Safety limit
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            
            # Check if execution is complete
            if current_state.is_complete or current_state.is_failed:
                break
            
            # Check resource limits
            if not current_state.is_within_limits():
                current_state.fail_execution("Resource limits exceeded")
                break
            
            # Determine next node to execute
            next_node = self._get_next_node(current_state)
            
            if not next_node:
                # No more nodes to execute
                break
            
            # Execute the node
            try:
                current_state = self._execute_node(next_node, current_state)
                yield {next_node: current_state}
            except Exception as e:
                self.logger.error(f"Error executing node {next_node}: {e}", exc_info=True)
                current_state.fail_execution(f"Error in {next_node}: {str(e)}")
                break
        
        # Ensure execution is marked complete
        if not current_state.is_complete and not current_state.is_failed:
            self._finalize_execution(current_state)
            yield {"result_assembler": current_state}
    
    def get_state(self, config: Dict[str, Any]) -> Any:
        """
        Get stored state for a thread.
        
        Args:
            config: Configuration with thread_id
            
        Returns:
            State object with values attribute
        """
        thread_id = config.get("configurable", {}).get("thread_id")
        state = self.states.get(thread_id) if thread_id else None
        
        # Return object with values attribute for compatibility
        class StateWrapper:
            def __init__(self, state_dict):
                self.values = state_dict
        
        return StateWrapper(state.to_dict() if state else {})
    
    def _get_next_node(self, state: ExecutionState) -> Optional[str]:
        """Determine the next node to execute based on state."""
        # If next_node is explicitly set, use it
        if state.next_node:
            return state.next_node
        
        # If no current node, start with orchestrator
        if not state.current_node:
            return "orchestrator"
        
        # Route based on current node and state
        current = state.current_node
        
        if current == "orchestrator":
            if state.execution_path == ExecutionPath.PREDEFINED_WORKFLOW:
                return "workflow_executor"
            elif state.execution_path == ExecutionPath.PLANNER_DRIVEN:
                return "planner"
            elif state.context.get("needs_file_processing"):
                return "context_manager"
        
        elif current == "planner":
            return "task_identifier"
        
        elif current == "task_identifier":
            return "agent_generator"
        
        elif current == "agent_generator":
            return "agent_executor"
        
        elif current == "agent_executor":
            # Check if more tasks
            if state.get_ready_tasks() and state.is_within_limits():
                return "agent_executor"
            return "evaluator"
        
        elif current == "workflow_executor":
            if state.context.get("needs_context_retrieval"):
                return "context_manager"
            if state.context.get("workflow_complete"):
                return "evaluator"
        
        elif current == "context_manager":
            if state.execution_path == ExecutionPath.PREDEFINED_WORKFLOW:
                return "workflow_executor"
            if state.context.get("retrieval_complete"):
                return "evaluator"
            return "orchestrator"
        
        elif current == "evaluator":
            if state.context.get("verification_passed", True):
                return "result_assembler"
            # Check for replan
            replan_count = state.context.get("replan_count", 0)
            if replan_count < 2 and state.is_within_limits():
                state.context["replan_count"] = replan_count + 1
                return "planner"
        
        elif current == "result_assembler":
            return "memory_manager"
        
        elif current == "memory_manager":
            return None  # End of execution
        
        return None
    
    def _execute_node(self, node_name: str, state: ExecutionState) -> ExecutionState:
        """Execute a specific node."""
        self.logger.info(f"Executing node: {node_name}")
        
        # Clear next_node to avoid loops
        state.next_node = None
        
        if node_name == "orchestrator":
            return self.orchestrator(state)
        
        elif node_name == "planner":
            return self.planner(state)
        
        elif node_name == "task_identifier":
            return self.task_identifier(state)
        
        elif node_name == "agent_generator":
            return self.agent_generator(state)
        
        elif node_name == "agent_executor":
            return self.agent_executor(state)
        
        elif node_name == "workflow_executor":
            return self.workflow_executor(state)
        
        elif node_name == "context_manager":
            return self._execute_context_manager(state)
        
        elif node_name == "memory_manager":
            return self._execute_memory_manager(state)
        
        elif node_name == "evaluator":
            return self.evaluator(state)
        
        elif node_name == "result_assembler":
            return self._execute_result_assembler(state)
        
        else:
            self.logger.warning(f"Unknown node: {node_name}")
            return state
    
    def _execute_context_manager(self, state: ExecutionState) -> ExecutionState:
        """Execute context manager node."""
        # Handle file processing
        if state.context.get("needs_file_processing"):
            files = state.context.get("uploaded_files", [])
            for file_path in files:
                try:
                    result = self.context_manager.ingest_file(file_path)
                    state.context[f"processed_{file_path}"] = result
                except Exception as e:
                    self.logger.error(f"Error processing file {file_path}: {e}")
                    state.context[f"error_{file_path}"] = str(e)
        
        # Handle context retrieval
        if state.context.get("needs_context_retrieval"):
            query = state.context.get("retrieval_query", state.user_request)
            k = state.context.get("retrieval_k", 10)
            try:
                chunks = self.context_manager.retrieve_context(query, k=k)
                state.retrieved_chunks = chunks
                state.context["retrieval_complete"] = True
            except Exception as e:
                self.logger.error(f"Error retrieving context: {e}")
                state.context["retrieval_error"] = str(e)
        
        return state
    
    def _execute_memory_manager(self, state: ExecutionState) -> ExecutionState:
        """Execute memory manager node."""
        # Store execution results in memory
        if state.final_result:
            try:
                self.memory_manager.store_conversation(
                    session_id=state.session_id,
                    user_message=state.user_request,
                    assistant_message=str(state.final_result),
                    metadata={
                        "execution_path": state.execution_path.value if state.execution_path else None,
                        "workflow": state.selected_workflow,
                        "tokens_used": state.tokens_used
                    }
                )
            except Exception as e:
                self.logger.error(f"Error storing conversation: {e}")
        
        return state
    
    def _execute_result_assembler(self, state: ExecutionState) -> ExecutionState:
        """Execute result assembler node."""
        # Assemble final result
        results = []
        
        for task_id in state.completed_tasks:
            task = state.get_task_by_id(task_id)
            if task and task.result:
                results.append({
                    "task": task.name,
                    "result": task.result
                })
        
        # Create final result
        final_result = {
            "response": state.context.get("final_response", "Request processed successfully"),
            "sources": [
                {
                    "chunk_id": chunk.get("id"),
                    "content": chunk.get("content", "")[:200],
                    "source_file": chunk.get("source_file"),
                    "page_number": chunk.get("page_number")
                }
                for chunk in state.retrieved_chunks
            ],
            "execution_path": state.execution_path.value if state.execution_path else "unknown",
            "workflow": state.selected_workflow,
            "tasks_completed": len(state.completed_tasks),
            "tokens_used": state.tokens_used,
            "execution_time": (
                (datetime.now() - state.started_at).total_seconds()
                if state.started_at
                else 0
            ),
            "metadata": {
                "session_id": state.session_id,
                "steps": state.current_step,
                "active_agents": len(state.active_agents),
                "workflow_confidence": state.workflow_confidence
            }
        }
        
        state.complete_execution(final_result)
        state.final_result = final_result
        
        return state
    
    def _finalize_execution(self, state: ExecutionState) -> None:
        """Finalize execution if not already complete."""
        if not state.final_result:
            final_result = {
                "response": state.context.get("final_response", "Execution completed"),
                "sources": [],
                "execution_path": state.execution_path.value if state.execution_path else "unknown",
                "metadata": {
                    "session_id": state.session_id,
                    "steps": state.current_step
                }
            }
            state.complete_execution(final_result)
            state.final_result = final_result


def create_simple_execution_graph(config: SystemConfig) -> SimpleGraphExecutor:
    """
    Factory function to create a simple execution graph.
    
    Args:
        config: System configuration
        
    Returns:
        SimpleGraphExecutor ready for execution
    """
    return SimpleGraphExecutor(config)
