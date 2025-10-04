"""Sub-agent execution framework for Local Agent Studio."""

import logging
import asyncio
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import uuid
import threading
import time

from ..config.models import SystemConfig
from .base_nodes import ProcessingNode
from .state import ExecutionState, Task, AgentSpec, TaskStatus


logger = logging.getLogger(__name__)


class AgentExecutionStatus(Enum):
    """Status of agent execution."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


@dataclass
class AgentExecutionResult:
    """Result of agent execution."""
    agent_id: str
    task_id: str
    status: AgentExecutionStatus
    result: Optional[Any] = None
    error: Optional[str] = None
    execution_time: float = 0.0
    steps_taken: int = 0
    tokens_used: int = 0
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    execution_log: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class RunningAgent:
    """Information about a currently running agent."""
    agent_id: str
    task_id: str
    agent_spec: AgentSpec
    started_at: datetime
    timeout_at: datetime
    current_step: int = 0
    tokens_used: int = 0
    is_cancelled: bool = False
    execution_thread: Optional[threading.Thread] = None
    result_future: Optional[Any] = None


class SubAgentRunner:
    """Handles the actual execution of individual sub-agents."""
    
    def __init__(self, config: SystemConfig):
        """
        Initialize sub-agent runner.
        
        Args:
            config: System configuration
        """
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.SubAgentRunner")
        
    def execute_agent(self, agent_spec: AgentSpec, task: Task, context: Dict[str, Any]) -> AgentExecutionResult:
        """
        Execute a single sub-agent.
        
        Args:
            agent_spec: Agent specification
            task: Task to execute
            context: Execution context
            
        Returns:
            Agent execution result
        """
        start_time = datetime.now()
        result = AgentExecutionResult(
            agent_id=agent_spec.id,
            task_id=task.id,
            status=AgentExecutionStatus.RUNNING,
            started_at=start_time
        )
        
        try:
            self.logger.info(f"Starting execution of agent {agent_spec.id} for task {task.id}")
            
            # Simulate agent execution (in real implementation, this would interface with LangChain/LangGraph)
            execution_result = self._simulate_agent_execution(agent_spec, task, context)
            
            # Process the result
            result.result = execution_result
            result.status = AgentExecutionStatus.COMPLETED
            result.completed_at = datetime.now()
            result.execution_time = (result.completed_at - start_time).total_seconds()
            
            self.logger.info(f"Agent {agent_spec.id} completed successfully in {result.execution_time:.2f}s")
            
        except TimeoutError:
            result.status = AgentExecutionStatus.TIMEOUT
            result.error = f"Agent execution timed out after {agent_spec.limits.timeout_seconds}s"
            result.completed_at = datetime.now()
            result.execution_time = (result.completed_at - start_time).total_seconds()
            self.logger.warning(f"Agent {agent_spec.id} timed out")
            
        except Exception as e:
            result.status = AgentExecutionStatus.FAILED
            result.error = str(e)
            result.completed_at = datetime.now()
            result.execution_time = (result.completed_at - start_time).total_seconds()
            self.logger.error(f"Agent {agent_spec.id} failed: {e}", exc_info=True)
            
        return result
    
    def _simulate_agent_execution(self, agent_spec: AgentSpec, task: Task, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simulate agent execution (placeholder for actual LangChain integration).
        
        Args:
            agent_spec: Agent specification
            task: Task to execute
            context: Execution context
            
        Returns:
            Execution result
        """
        # This is a simulation - in real implementation, this would:
        # 1. Create a LangChain agent with the system prompt
        # 2. Provide the agent with the specified tools
        # 3. Execute the agent within the specified limits
        # 4. Collect and return the results
        
        # Simulate some processing time
        import time
        time.sleep(0.1)  # Simulate work
        
        # Create a mock result based on the output contract
        output_contract = agent_spec.output_contract
        required_fields = output_contract.get("required_fields", [])
        
        mock_result = {
            "status": "completed",
            "execution_summary": f"Successfully executed {task.name}",
        }
        
        # Add required fields based on output contract
        for field in required_fields:
            if field == "result":
                mock_result["result"] = f"Completed task: {task.description}"
            elif field == "answer":
                mock_result["answer"] = f"Answer to: {task.description}"
            elif field == "citations":
                mock_result["citations"] = ["source_1", "source_2"]
            elif field == "confidence_score":
                mock_result["confidence_score"] = 0.85
            elif field == "extracted_content":
                mock_result["extracted_content"] = "Sample extracted content"
            elif field == "metadata":
                mock_result["metadata"] = {"pages": 5, "format": "pdf"}
            elif field == "processing_errors":
                mock_result["processing_errors"] = []
            elif field == "analysis_results":
                mock_result["analysis_results"] = {"key_findings": ["finding_1", "finding_2"]}
            elif field == "insights":
                mock_result["insights"] = ["insight_1", "insight_2"]
            elif field == "recommendations":
                mock_result["recommendations"] = ["recommendation_1"]
            elif field == "generated_content":
                mock_result["generated_content"] = f"Generated content for: {task.name}"
            elif field == "sources_used":
                mock_result["sources_used"] = ["source_a", "source_b"]
            elif field == "quality_metrics":
                mock_result["quality_metrics"] = {"readability": 0.8, "accuracy": 0.9}
            elif field == "task_breakdown":
                mock_result["task_breakdown"] = ["subtask_1", "subtask_2"]
            elif field == "dependencies":
                mock_result["dependencies"] = task.dependencies
            elif field == "resource_estimates":
                mock_result["resource_estimates"] = {"time": "5 minutes", "complexity": "moderate"}
            elif field == "quality_score":
                mock_result["quality_score"] = 0.88
            elif field == "evaluation_details":
                mock_result["evaluation_details"] = {"criteria_met": 4, "criteria_total": 5}
            elif field == "improvement_suggestions":
                mock_result["improvement_suggestions"] = ["suggestion_1"]
        
        return mock_result


class AgentLifecycleManager:
    """Manages the lifecycle of sub-agents including instantiation, monitoring, and cleanup."""
    
    def __init__(self, config: SystemConfig):
        """
        Initialize agent lifecycle manager.
        
        Args:
            config: System configuration
        """
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.AgentLifecycleManager")
        self.running_agents: Dict[str, RunningAgent] = {}
        self.completed_agents: Dict[str, AgentExecutionResult] = {}
        self.agent_runner = SubAgentRunner(config)
        self._shutdown_event = threading.Event()
        self._monitor_thread = None
        
    def start_monitoring(self):
        """Start the agent monitoring thread."""
        if self._monitor_thread is None or not self._monitor_thread.is_alive():
            self._shutdown_event.clear()
            self._monitor_thread = threading.Thread(target=self._monitor_agents, daemon=True)
            self._monitor_thread.start()
            self.logger.info("Agent monitoring started")
    
    def stop_monitoring(self):
        """Stop the agent monitoring thread."""
        self._shutdown_event.set()
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=5)
            self.logger.info("Agent monitoring stopped")
    
    def instantiate_agent(self, agent_spec: AgentSpec, task: Task, context: Dict[str, Any]) -> str:
        """
        Instantiate and start a sub-agent.
        
        Args:
            agent_spec: Agent specification
            task: Task to execute
            context: Execution context
            
        Returns:
            Agent execution ID
        """
        if len(self.running_agents) >= self.config.agent_generator.max_concurrent_agents:
            raise RuntimeError("Maximum concurrent agents limit reached")
        
        # Create running agent record
        now = datetime.now()
        timeout_at = now + timedelta(seconds=agent_spec.limits.timeout_seconds)
        
        running_agent = RunningAgent(
            agent_id=agent_spec.id,
            task_id=task.id,
            agent_spec=agent_spec,
            started_at=now,
            timeout_at=timeout_at
        )
        
        # Start execution in a separate thread
        def execute_wrapper():
            try:
                result = self.agent_runner.execute_agent(agent_spec, task, context)
                self._handle_agent_completion(agent_spec.id, result)
            except Exception as e:
                error_result = AgentExecutionResult(
                    agent_id=agent_spec.id,
                    task_id=task.id,
                    status=AgentExecutionStatus.FAILED,
                    error=str(e),
                    completed_at=datetime.now()
                )
                self._handle_agent_completion(agent_spec.id, error_result)
        
        execution_thread = threading.Thread(target=execute_wrapper, daemon=True)
        running_agent.execution_thread = execution_thread
        
        # Register and start the agent
        self.running_agents[agent_spec.id] = running_agent
        execution_thread.start()
        
        self.logger.info(f"Instantiated agent {agent_spec.id} for task {task.id}")
        return agent_spec.id
    
    def cancel_agent(self, agent_id: str) -> bool:
        """
        Cancel a running agent.
        
        Args:
            agent_id: ID of agent to cancel
            
        Returns:
            True if agent was cancelled, False if not found or already completed
        """
        if agent_id in self.running_agents:
            running_agent = self.running_agents[agent_id]
            running_agent.is_cancelled = True
            
            # Create cancellation result
            cancel_result = AgentExecutionResult(
                agent_id=agent_id,
                task_id=running_agent.task_id,
                status=AgentExecutionStatus.CANCELLED,
                completed_at=datetime.now(),
                execution_time=(datetime.now() - running_agent.started_at).total_seconds()
            )
            
            self._handle_agent_completion(agent_id, cancel_result)
            self.logger.info(f"Cancelled agent {agent_id}")
            return True
        
        return False
    
    def get_agent_status(self, agent_id: str) -> Optional[AgentExecutionStatus]:
        """
        Get the status of an agent.
        
        Args:
            agent_id: ID of agent
            
        Returns:
            Agent status or None if not found
        """
        if agent_id in self.running_agents:
            return AgentExecutionStatus.RUNNING
        elif agent_id in self.completed_agents:
            return self.completed_agents[agent_id].status
        else:
            return None
    
    def get_agent_result(self, agent_id: str) -> Optional[AgentExecutionResult]:
        """
        Get the result of a completed agent.
        
        Args:
            agent_id: ID of agent
            
        Returns:
            Agent execution result or None if not completed
        """
        return self.completed_agents.get(agent_id)
    
    def get_running_agents(self) -> List[str]:
        """Get list of currently running agent IDs."""
        return list(self.running_agents.keys())
    
    def get_completed_agents(self) -> List[str]:
        """Get list of completed agent IDs."""
        return list(self.completed_agents.keys())
    
    def cleanup_completed_agents(self, max_age_hours: int = 24):
        """
        Clean up old completed agent records.
        
        Args:
            max_age_hours: Maximum age in hours for completed agents
        """
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        
        to_remove = []
        for agent_id, result in self.completed_agents.items():
            if result.completed_at and result.completed_at < cutoff_time:
                to_remove.append(agent_id)
        
        for agent_id in to_remove:
            del self.completed_agents[agent_id]
        
        if to_remove:
            self.logger.info(f"Cleaned up {len(to_remove)} old agent records")
    
    def _handle_agent_completion(self, agent_id: str, result: AgentExecutionResult):
        """Handle completion of an agent."""
        # Move from running to completed
        if agent_id in self.running_agents:
            del self.running_agents[agent_id]
        
        self.completed_agents[agent_id] = result
        
        self.logger.info(f"Agent {agent_id} completed with status: {result.status}")
    
    def _monitor_agents(self):
        """Monitor running agents for timeouts and cleanup."""
        while not self._shutdown_event.is_set():
            try:
                now = datetime.now()
                timed_out_agents = []
                
                # Check for timeouts
                for agent_id, running_agent in self.running_agents.items():
                    if now > running_agent.timeout_at:
                        timed_out_agents.append(agent_id)
                
                # Handle timeouts
                for agent_id in timed_out_agents:
                    running_agent = self.running_agents[agent_id]
                    timeout_result = AgentExecutionResult(
                        agent_id=agent_id,
                        task_id=running_agent.task_id,
                        status=AgentExecutionStatus.TIMEOUT,
                        error=f"Agent timed out after {running_agent.agent_spec.limits.timeout_seconds}s",
                        completed_at=now,
                        execution_time=(now - running_agent.started_at).total_seconds()
                    )
                    self._handle_agent_completion(agent_id, timeout_result)
                
                # Periodic cleanup
                if len(self.completed_agents) > 100:  # Arbitrary threshold
                    self.cleanup_completed_agents(max_age_hours=1)
                
                # Sleep before next check
                time.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                self.logger.error(f"Error in agent monitoring: {e}", exc_info=True)
                time.sleep(10)  # Wait longer on error


class ResultCollector:
    """Collects and integrates results from multiple sub-agents."""
    
    def __init__(self, config: SystemConfig):
        """
        Initialize result collector.
        
        Args:
            config: System configuration
        """
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.ResultCollector")
    
    def collect_results(self, agent_results: List[AgentExecutionResult], tasks: List[Task]) -> Dict[str, Any]:
        """
        Collect and integrate results from multiple agents.
        
        Args:
            agent_results: List of agent execution results
            tasks: List of tasks that were executed
            
        Returns:
            Integrated results
        """
        integrated_results = {
            "execution_summary": {
                "total_agents": len(agent_results),
                "successful_agents": len([r for r in agent_results if r.status == AgentExecutionStatus.COMPLETED]),
                "failed_agents": len([r for r in agent_results if r.status == AgentExecutionStatus.FAILED]),
                "total_execution_time": sum(r.execution_time for r in agent_results),
                "total_tokens_used": sum(r.tokens_used for r in agent_results)
            },
            "task_results": {},
            "agent_results": {},
            "errors": [],
            "warnings": []
        }
        
        # Process each agent result
        for result in agent_results:
            # Store agent-specific result
            integrated_results["agent_results"][result.agent_id] = {
                "status": result.status.value,
                "execution_time": result.execution_time,
                "tokens_used": result.tokens_used,
                "steps_taken": result.steps_taken
            }
            
            # Store task-specific result
            if result.task_id:
                integrated_results["task_results"][result.task_id] = {
                    "status": result.status.value,
                    "result": result.result,
                    "agent_id": result.agent_id
                }
            
            # Collect errors and warnings
            if result.error:
                integrated_results["errors"].append({
                    "agent_id": result.agent_id,
                    "task_id": result.task_id,
                    "error": result.error
                })
            
            if result.status in [AgentExecutionStatus.TIMEOUT, AgentExecutionStatus.CANCELLED]:
                integrated_results["warnings"].append({
                    "agent_id": result.agent_id,
                    "task_id": result.task_id,
                    "warning": f"Agent {result.status.value}"
                })
        
        # Synthesize cross-agent insights
        integrated_results["synthesis"] = self._synthesize_results(agent_results, tasks)
        
        return integrated_results
    
    def _synthesize_results(self, agent_results: List[AgentExecutionResult], tasks: List[Task]) -> Dict[str, Any]:
        """
        Synthesize insights across multiple agent results.
        
        Args:
            agent_results: List of agent execution results
            tasks: List of tasks
            
        Returns:
            Synthesized insights
        """
        synthesis = {
            "overall_success": True,
            "completion_rate": 0.0,
            "key_findings": [],
            "recommendations": [],
            "quality_metrics": {}
        }
        
        successful_results = [r for r in agent_results if r.status == AgentExecutionStatus.COMPLETED]
        synthesis["completion_rate"] = len(successful_results) / len(agent_results) if agent_results else 0.0
        synthesis["overall_success"] = synthesis["completion_rate"] >= 0.8  # 80% success threshold
        
        # Extract key findings from successful results
        for result in successful_results:
            if result.result and isinstance(result.result, dict):
                # Extract various types of findings
                if "key_findings" in result.result:
                    synthesis["key_findings"].extend(result.result["key_findings"])
                if "insights" in result.result:
                    synthesis["key_findings"].extend(result.result["insights"])
                if "analysis_results" in result.result:
                    if isinstance(result.result["analysis_results"], dict):
                        synthesis["key_findings"].extend(result.result["analysis_results"].get("key_findings", []))
        
        # Generate recommendations based on results
        if synthesis["completion_rate"] < 1.0:
            synthesis["recommendations"].append("Some agents failed - consider retrying failed tasks")
        
        if len(successful_results) > 1:
            synthesis["recommendations"].append("Multiple agents completed successfully - results may benefit from cross-validation")
        
        # Calculate quality metrics
        if successful_results:
            avg_execution_time = sum(r.execution_time for r in successful_results) / len(successful_results)
            avg_tokens_used = sum(r.tokens_used for r in successful_results) / len(successful_results)
            
            synthesis["quality_metrics"] = {
                "average_execution_time": avg_execution_time,
                "average_tokens_used": avg_tokens_used,
                "efficiency_score": min(1.0, 60.0 / avg_execution_time) if avg_execution_time > 0 else 1.0  # Higher score for faster execution
            }
        
        return synthesis


class AgentExecutor(ProcessingNode):
    """
    Agent Executor node that manages sub-agent instantiation, lifecycle, and result collection.
    
    Responsibilities:
    - Instantiate sub-agents based on agent specifications
    - Monitor agent execution and enforce limits
    - Collect and integrate results from multiple agents
    - Handle agent failures and timeouts gracefully
    """
    
    def __init__(self, config: SystemConfig):
        """
        Initialize the Agent Executor.
        
        Args:
            config: System configuration
        """
        super().__init__("AgentExecutor", config.agent_generator.__dict__)
        self.system_config = config
        self.lifecycle_manager = AgentLifecycleManager(config)
        self.result_collector = ResultCollector(config)
        
        # Start monitoring
        self.lifecycle_manager.start_monitoring()
    
    def execute(self, state: ExecutionState) -> ExecutionState:
        """
        Execute sub-agents and collect results.
        
        Args:
            state: Current execution state
            
        Returns:
            Updated execution state with agent results
        """
        generated_agents = state.context.get("generated_agents", [])
        
        if not generated_agents:
            self.logger.warning("No generated agents found in state")
            state.complete_execution("No agents to execute")
            return state
        
        self.logger.info(f"Starting execution of {len(generated_agents)} agents")
        
        # Start all agents
        running_agent_ids = []
        for agent_info in generated_agents:
            agent_id = agent_info["agent_id"]
            task_id = agent_info["task_id"]
            
            # Get agent spec and task from state
            agent_spec = state.agent_specs.get(agent_id)
            task = state.get_task_by_id(task_id)
            
            if not agent_spec or not task:
                self.logger.error(f"Missing agent spec or task for agent {agent_id}")
                continue
            
            try:
                # Start the agent
                execution_id = self.lifecycle_manager.instantiate_agent(
                    agent_spec, task, state.context
                )
                running_agent_ids.append(agent_id)
                
                # Mark task as in progress
                task.status = TaskStatus.IN_PROGRESS
                task.started_at = datetime.now()
                
                # Activate agent in state
                state.activate_agent(agent_id)
                
            except Exception as e:
                self.logger.error(f"Failed to start agent {agent_id}: {e}")
                state.mark_task_failed(task_id, str(e))
        
        # Wait for all agents to complete
        agent_results = self._wait_for_agents(running_agent_ids, state)
        
        # Process results
        self._process_agent_results(agent_results, state)
        
        # Collect and integrate results
        tasks = [state.get_task_by_id(info["task_id"]) for info in generated_agents]
        tasks = [t for t in tasks if t is not None]
        
        integrated_results = self.result_collector.collect_results(agent_results, tasks)
        
        # Store results in state
        state.context["agent_execution_results"] = integrated_results
        
        # Log execution summary
        state.log_execution_step(
            node_name=self.name,
            action="agent_execution_complete",
            details={
                "total_agents": len(generated_agents),
                "successful_agents": integrated_results["execution_summary"]["successful_agents"],
                "failed_agents": integrated_results["execution_summary"]["failed_agents"],
                "total_execution_time": integrated_results["execution_summary"]["total_execution_time"],
                "completion_rate": integrated_results["synthesis"]["completion_rate"]
            }
        )
        
        # Determine next node
        if integrated_results["synthesis"]["overall_success"]:
            state.next_node = "Evaluator"
        else:
            # Some agents failed - might need replanning
            state.next_node = "Planner"  # Could replan failed tasks
        
        return state
    
    def _wait_for_agents(self, agent_ids: List[str], state: ExecutionState) -> List[AgentExecutionResult]:
        """
        Wait for all agents to complete execution.
        
        Args:
            agent_ids: List of agent IDs to wait for
            state: Current execution state
            
        Returns:
            List of agent execution results
        """
        results = []
        max_wait_time = 300  # 5 minutes maximum wait
        check_interval = 1  # Check every second
        start_time = datetime.now()
        
        remaining_agents = set(agent_ids)
        
        while remaining_agents and (datetime.now() - start_time).total_seconds() < max_wait_time:
            completed_this_cycle = []
            
            for agent_id in remaining_agents:
                status = self.lifecycle_manager.get_agent_status(agent_id)
                
                if status in [AgentExecutionStatus.COMPLETED, AgentExecutionStatus.FAILED, 
                             AgentExecutionStatus.TIMEOUT, AgentExecutionStatus.CANCELLED]:
                    result = self.lifecycle_manager.get_agent_result(agent_id)
                    if result:
                        results.append(result)
                        completed_this_cycle.append(agent_id)
                        
                        # Deactivate agent in state
                        state.deactivate_agent(agent_id)
            
            # Remove completed agents from remaining set
            for agent_id in completed_this_cycle:
                remaining_agents.remove(agent_id)
            
            if remaining_agents:
                time.sleep(check_interval)
        
        # Handle any remaining agents (timeout)
        for agent_id in remaining_agents:
            self.logger.warning(f"Agent {agent_id} did not complete within timeout, cancelling")
            self.lifecycle_manager.cancel_agent(agent_id)
            
            # Get the cancellation result
            result = self.lifecycle_manager.get_agent_result(agent_id)
            if result:
                results.append(result)
            
            state.deactivate_agent(agent_id)
        
        return results
    
    def _process_agent_results(self, agent_results: List[AgentExecutionResult], state: ExecutionState):
        """
        Process agent results and update task statuses.
        
        Args:
            agent_results: List of agent execution results
            state: Current execution state
        """
        for result in agent_results:
            task = state.get_task_by_id(result.task_id)
            if not task:
                continue
            
            if result.status == AgentExecutionStatus.COMPLETED:
                state.mark_task_completed(result.task_id, result.result)
                self.logger.info(f"Task {result.task_id} completed successfully by agent {result.agent_id}")
            else:
                error_msg = result.error or f"Agent execution {result.status.value}"
                state.mark_task_failed(result.task_id, error_msg)
                self.logger.warning(f"Task {result.task_id} failed: {error_msg}")
    
    def validate_inputs(self, state: ExecutionState) -> bool:
        """
        Validate that required inputs are present.
        
        Args:
            state: Current execution state
            
        Returns:
            True if inputs are valid, False otherwise
        """
        generated_agents = state.context.get("generated_agents", [])
        
        if not generated_agents:
            return False
        
        # Check that all referenced agents and tasks exist
        for agent_info in generated_agents:
            agent_id = agent_info.get("agent_id")
            task_id = agent_info.get("task_id")
            
            if not agent_id or not task_id:
                return False
            
            if agent_id not in state.agent_specs:
                return False
            
            if not state.get_task_by_id(task_id):
                return False
        
        return True
    
    def get_next_node(self, state: ExecutionState) -> Optional[str]:
        """
        Determine the next node based on execution results.
        
        Args:
            state: Current execution state
            
        Returns:
            Name of next node
        """
        results = state.context.get("agent_execution_results", {})
        
        if not results:
            return None
        
        synthesis = results.get("synthesis", {})
        overall_success = synthesis.get("overall_success", False)
        
        if overall_success:
            return "Evaluator"
        else:
            # Could implement retry logic or replanning
            return "Planner"
    
    def is_critical_error(self, error: Exception) -> bool:
        """
        Determine if an error should stop execution.
        
        Args:
            error: The exception that occurred
            
        Returns:
            True if execution should stop, False to continue
        """
        # Agent execution errors are generally not critical for the overall system
        # Individual agent failures are handled gracefully
        return False
    
    def __del__(self):
        """Cleanup when executor is destroyed."""
        if hasattr(self, 'lifecycle_manager'):
            self.lifecycle_manager.stop_monitoring()