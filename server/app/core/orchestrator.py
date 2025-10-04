"""Main Orchestrator node for workflow routing and coordination."""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

from ..config.models import SystemConfig, WorkflowConfig
from .base_nodes import ConditionalNode
from .state import ExecutionState, ExecutionPath, TaskStatus


logger = logging.getLogger(__name__)


from .workflows.workflow_matcher import WorkflowMatcher


class ResourceLimitEnforcer:
    """Handles resource limit enforcement and monitoring."""
    
    def __init__(self, config: SystemConfig):
        """
        Initialize resource limit enforcer.
        
        Args:
            config: System configuration
        """
        self.config = config
        
    def check_limits(self, state: ExecutionState) -> tuple[bool, Optional[str]]:
        """
        Check if execution is within resource limits.
        
        Args:
            state: Current execution state
            
        Returns:
            Tuple of (within_limits, error_message)
        """
        # Check step limit
        if state.current_step >= state.max_steps:
            return False, f"Maximum steps exceeded ({state.current_step}/{state.max_steps})"
            
        # Check token budget
        if state.tokens_used >= state.token_budget:
            return False, f"Token budget exceeded ({state.tokens_used}/{state.token_budget})"
            
        # Check timeout
        if state.started_at:
            elapsed = (datetime.now() - state.started_at).total_seconds()
            timeout = self.config.orchestrator.timeout_seconds
            if elapsed > timeout:
                return False, f"Execution timeout exceeded ({elapsed:.1f}s/{timeout}s)"
                
        # Check agent limits
        max_agents = self.config.agent_generator.max_concurrent_agents
        if len(state.active_agents) > max_agents:
            return False, f"Too many active agents ({len(state.active_agents)}/{max_agents})"
            
        return True, None
    
    def enforce_graceful_termination(self, state: ExecutionState, reason: str) -> ExecutionState:
        """
        Gracefully terminate execution when limits are exceeded.
        
        Args:
            state: Current execution state
            reason: Reason for termination
            
        Returns:
            Updated execution state
        """
        logger.warning(f"Enforcing graceful termination: {reason}")
        
        # Deactivate all agents
        for agent_id in state.active_agents.copy():
            state.deactivate_agent(agent_id)
            
        # Mark execution as failed with reason
        state.fail_execution(f"Resource limits exceeded: {reason}")
        
        # Log termination
        state.log_execution_step(
            node_name="ResourceLimitEnforcer",
            action="graceful_termination",
            details={"reason": reason}
        )
        
        return state


class MainOrchestrator(ConditionalNode):
    """
    Main Orchestrator node that coordinates the entire execution flow.
    
    Responsibilities:
    - Route requests between predefined workflows and planner-driven execution
    - Enforce resource limits and constraints
    - Coordinate state transitions between components
    - Monitor execution progress and handle failures
    """
    
    def __init__(self, config: SystemConfig):
        """
        Initialize the Main Orchestrator.
        
        Args:
            config: System configuration
        """
        super().__init__("MainOrchestrator", config.orchestrator.__dict__)
        self.system_config = config
        self.workflow_matcher = WorkflowMatcher(config.workflows)
        self.resource_enforcer = ResourceLimitEnforcer(config)
        
    def execute(self, state: ExecutionState) -> ExecutionState:
        """
        Execute orchestrator logic.
        
        Args:
            state: Current execution state
            
        Returns:
            Updated execution state
        """
        # Initialize execution if not started
        if not state.started_at:
            state.start_execution()
            state.max_steps = self.system_config.orchestrator.max_iterations
            state.token_budget = self.system_config.orchestrator.token_budget
            
        # Check resource limits
        within_limits, limit_error = self.resource_enforcer.check_limits(state)
        if not within_limits:
            return self.resource_enforcer.enforce_graceful_termination(state, limit_error)
            
        # Determine execution path if not already set
        if state.execution_path is None:
            state = self._select_execution_path(state)
            
        # Route to appropriate execution flow
        if state.execution_path == ExecutionPath.PREDEFINED_WORKFLOW:
            state = self._handle_predefined_workflow(state)
        elif state.execution_path == ExecutionPath.PLANNER_DRIVEN:
            state = self._handle_planner_driven_execution(state)
        elif state.execution_path == ExecutionPath.HYBRID:
            state = self._handle_hybrid_execution(state)
            
        return state
    
    def _select_execution_path(self, state: ExecutionState) -> ExecutionState:
        """
        Select the appropriate execution path based on workflow matching.
        
        Args:
            state: Current execution state
            
        Returns:
            Updated execution state with selected path
        """
        # Match user request to workflows
        workflow_name, confidence = self.workflow_matcher.match_workflow(state.user_request)
        
        state.workflow_confidence = confidence
        threshold = self.system_config.orchestrator.workflow_confidence_threshold
        
        # Log matching details for debugging
        self.logger.debug(
            f"Workflow matching: request='{state.user_request[:50]}...', "
            f"best_match='{workflow_name}', confidence={confidence:.3f}, threshold={threshold}"
        )
        
        if workflow_name and confidence >= threshold:
            # Use predefined workflow
            state.execution_path = ExecutionPath.PREDEFINED_WORKFLOW
            state.selected_workflow = workflow_name
            
            self.logger.info(
                f"✓ Selected predefined workflow '{workflow_name}' "
                f"with confidence {confidence:.3f} (threshold: {threshold})"
            )
            
        else:
            # Use planner-driven execution
            state.execution_path = ExecutionPath.PLANNER_DRIVEN
            
            self.logger.info(
                f"→ Selected planner-driven execution "
                f"(best match: {workflow_name or 'none'}, confidence: {confidence:.3f} < threshold: {threshold})"
            )
            
        # Log path selection
        state.log_execution_step(
            node_name=self.name,
            action="path_selection",
            details={
                "execution_path": state.execution_path.value,
                "selected_workflow": state.selected_workflow,
                "confidence": confidence,
                "threshold": threshold
            }
        )
        
        return state
    
    def _handle_predefined_workflow(self, state: ExecutionState) -> ExecutionState:
        """
        Handle execution using a predefined workflow.
        
        Args:
            state: Current execution state
            
        Returns:
            Updated execution state
        """
        workflow_config = self.system_config.workflows.get(state.selected_workflow)
        if not workflow_config:
            error_msg = f"Workflow '{state.selected_workflow}' not found"
            self.logger.error(error_msg)
            state.fail_execution(error_msg)
            return state
            
        self.logger.info(f"Executing predefined workflow: {state.selected_workflow}")
        
        # Store workflow steps in state context
        state.context["workflow_steps"] = [
            {
                "name": step.name,
                "description": step.description,
                "node_type": step.node_type,
                "parameters": step.parameters,
                "success_criteria": step.success_criteria
            }
            for step in workflow_config.steps
        ]
        
        # Set next node based on workflow
        state.next_node = "WorkflowExecutor"
        
        return state
    
    def _handle_planner_driven_execution(self, state: ExecutionState) -> ExecutionState:
        """
        Handle execution using dynamic planning.
        
        Args:
            state: Current execution state
            
        Returns:
            Updated execution state
        """
        self.logger.info("Executing planner-driven workflow")
        
        # Route to planner for task decomposition
        state.next_node = "Planner"
        
        return state
    
    def _handle_hybrid_execution(self, state: ExecutionState) -> ExecutionState:
        """
        Handle hybrid execution combining workflows and planning.
        
        Args:
            state: Current execution state
            
        Returns:
            Updated execution state
        """
        # For now, treat hybrid as planner-driven
        # This can be extended in the future
        return self._handle_planner_driven_execution(state)
    
    def get_next_node(self, state: ExecutionState) -> Optional[str]:
        """
        Determine the next node based on execution state.
        
        Args:
            state: Current execution state
            
        Returns:
            Name of next node or None if execution should end
        """
        # Check if execution is complete or failed
        if state.is_complete or state.is_failed:
            return None
            
        # Return the next node set during execution
        return state.next_node
    
    def coordinate_components(self, state: ExecutionState) -> ExecutionState:
        """
        Coordinate between system components and manage state transitions.
        
        Args:
            state: Current execution state
            
        Returns:
            Updated execution state
        """
        # Update component status
        state.context["orchestrator_status"] = {
            "current_step": state.current_step,
            "execution_path": state.execution_path.value if state.execution_path else None,
            "active_agents": len(state.active_agents),
            "completed_tasks": len(state.completed_tasks),
            "resource_usage": {
                "tokens_used": state.tokens_used,
                "token_budget": state.token_budget,
                "steps_used": state.current_step,
                "max_steps": state.max_steps,
                "utilization_percentage": (state.tokens_used / state.token_budget) * 100
            },
            "execution_metrics": {
                "workflow_confidence": state.workflow_confidence,
                "selected_workflow": state.selected_workflow,
                "tasks_pending": len(state.get_pending_tasks()),
                "tasks_completed": len(state.completed_tasks),
                "tasks_failed": len(state.failed_tasks)
            }
        }
        
        # Coordinate agent lifecycle
        self._coordinate_agent_lifecycle(state)
        
        # Monitor execution health
        self._monitor_execution_health(state)
        
        return state
    
    def _coordinate_agent_lifecycle(self, state: ExecutionState) -> None:
        """Coordinate agent creation, activation, and cleanup."""
        # Clean up completed agents
        completed_agents = []
        for agent_id in state.active_agents:
            # Check if agent's task is completed
            agent_spec = state.agent_specs.get(agent_id)
            if agent_spec:
                task = state.get_task_by_id(agent_spec.created_for_task)
                if task and task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
                    completed_agents.append(agent_id)
                    
        # Deactivate completed agents
        for agent_id in completed_agents:
            state.deactivate_agent(agent_id)
            self.logger.info(f"Deactivated completed agent: {agent_id}")
    
    def _monitor_execution_health(self, state: ExecutionState) -> None:
        """Monitor execution health and trigger interventions if needed."""
        # Check for stuck execution
        if state.current_step > 0 and not state.completed_tasks and not state.active_agents:
            self.logger.warning("Execution appears stuck - no progress and no active agents")
            state.log_execution_step(
                node_name=self.name,
                action="health_warning",
                details={"issue": "stuck_execution", "step": state.current_step}
            )
            
        # Check resource utilization
        token_utilization = (state.tokens_used / state.token_budget) * 100
        if token_utilization > 80:
            self.logger.warning(f"High token utilization: {token_utilization:.1f}%")
            
        # Check for excessive failures
        failure_rate = len(state.failed_tasks) / max(len(state.tasks), 1)
        if failure_rate > 0.5:
            self.logger.error(f"High task failure rate: {failure_rate:.1%}")
            state.log_execution_step(
                node_name=self.name,
                action="health_alert",
                details={"issue": "high_failure_rate", "rate": failure_rate}
            )
    
    def select_execution_path(self, state: ExecutionState, user_context: Dict[str, Any] = None) -> ExecutionState:
        """
        Advanced execution path selection with context awareness.
        
        Args:
            state: Current execution state
            user_context: Optional additional context for path selection
            
        Returns:
            Updated execution state with selected path
        """
        if state.execution_path is not None:
            return state  # Path already selected
            
        # Enhanced workflow matching with context
        workflow_name, confidence = self.workflow_matcher.match_workflow(state.user_request)
        
        # Consider user context for path selection
        context_boost = 0.0
        if user_context:
            if user_context.get("has_uploaded_files") and workflow_name in ["rag_qa", "summarize"]:
                context_boost = 0.1
            if user_context.get("previous_workflow") == workflow_name:
                context_boost = 0.05  # Slight boost for workflow continuity
                
        adjusted_confidence = min(confidence + context_boost, 1.0)
        threshold = self.system_config.orchestrator.workflow_confidence_threshold
        
        # Path selection logic
        if workflow_name and adjusted_confidence >= threshold:
            # High confidence - use predefined workflow
            state.execution_path = ExecutionPath.PREDEFINED_WORKFLOW
            state.selected_workflow = workflow_name
            
        elif workflow_name and adjusted_confidence >= (threshold * 0.7):
            # Medium confidence - hybrid approach
            state.execution_path = ExecutionPath.HYBRID
            state.selected_workflow = workflow_name
            
        else:
            # Low confidence - planner-driven
            state.execution_path = ExecutionPath.PLANNER_DRIVEN
            
        # Store selection metadata
        state.workflow_confidence = adjusted_confidence
        state.context["path_selection"] = {
            "original_confidence": confidence,
            "context_boost": context_boost,
            "final_confidence": adjusted_confidence,
            "threshold": threshold,
            "selection_reason": self._get_selection_reason(adjusted_confidence, threshold)
        }
        
        self.logger.info(
            f"Selected execution path: {state.execution_path.value} "
            f"(confidence: {adjusted_confidence:.3f}, threshold: {threshold})"
        )
        
        return state
    
    def _get_selection_reason(self, confidence: float, threshold: float) -> str:
        """Get human-readable reason for path selection."""
        if confidence >= threshold:
            return "High confidence workflow match"
        elif confidence >= (threshold * 0.7):
            return "Medium confidence - hybrid approach"
        else:
            return "Low confidence - dynamic planning required"
    
    def is_critical_error(self, error: Exception) -> bool:
        """
        Determine if an error should stop execution.
        
        Args:
            error: The exception that occurred
            
        Returns:
            True if execution should stop, False to continue
        """
        # Configuration errors are critical
        if isinstance(error, (KeyError, ValueError, AttributeError)):
            return True
            
        # Resource limit errors are handled gracefully
        if "limit" in str(error).lower():
            return False
            
        # Default to critical for orchestrator
        return True
    
    def process_request(
        self, 
        user_request: str, 
        context: Dict[str, Any], 
        execution_state: Optional[ExecutionState] = None
    ) -> Dict[str, Any]:
        """
        Process a user request through the orchestration system.
        
        This is the main entry point for request processing that integrates
        with the LangGraph execution flow.
        
        Args:
            user_request: The user's input request
            context: Additional context for the request
            execution_state: Optional pre-initialized execution state
            
        Returns:
            Dictionary containing the response and metadata
        """
        # Initialize or update execution state
        if execution_state is None:
            execution_state = ExecutionState(
                user_request=user_request,
                max_steps=self.system_config.orchestrator.max_iterations,
                token_budget=self.system_config.orchestrator.token_budget
            )
        else:
            execution_state.user_request = user_request
        
        # Merge context
        execution_state.context.update(context)
        
        # Execute orchestrator logic
        result_state = self.execute(execution_state)
        
        # Extract result
        if result_state.final_result:
            return result_state.final_result
        
        # Build response from state
        return {
            "response": result_state.context.get("final_response", "Request processed"),
            "execution_path": result_state.execution_path.value if result_state.execution_path else "unknown",
            "selected_workflow": result_state.selected_workflow,
            "sources": [
                {
                    "chunk_id": chunk.get("id"),
                    "content": chunk.get("content", "")[:200],
                    "source_file": chunk.get("source_file")
                }
                for chunk in result_state.retrieved_chunks
            ],
            "metadata": {
                "session_id": result_state.session_id,
                "steps": result_state.current_step,
                "tokens_used": result_state.tokens_used,
                "workflow_confidence": result_state.workflow_confidence,
                "is_complete": result_state.is_complete,
                "is_failed": result_state.is_failed,
                "error": result_state.error_message
            }
        }