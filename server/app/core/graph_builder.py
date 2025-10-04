"""LangGraph execution graph builder that wires all components together."""

import logging
from typing import Dict, Any, Optional, Callable

try:
    from langgraph.graph import StateGraph, END
    from langgraph.checkpoint.memory import MemorySaver
    LANGGRAPH_AVAILABLE = True
except (ImportError, TypeError) as e:
    # LangGraph may not be available or have compatibility issues
    logging.warning(f"LangGraph not available: {e}")
    LANGGRAPH_AVAILABLE = False
    StateGraph = None
    END = None
    MemorySaver = None

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


class GraphBuilder:
    """
    Builds the complete LangGraph execution graph with all nodes and edges.
    
    This class wires together all system components into a cohesive execution flow,
    handling both predefined workflows and dynamic agent generation paths.
    """
    
    def __init__(self, config: SystemConfig):
        """
        Initialize the graph builder.
        
        Args:
            config: System configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize all nodes
        self.orchestrator = MainOrchestrator(config)
        self.planner = Planner(config)
        self.task_identifier = TaskIdentifier(config)
        self.agent_generator = AgentGenerator(config)
        self.agent_executor = AgentExecutor(config)
        self.context_manager = ContextManager(config)
        self.memory_manager = MemoryManager(config)
        self.evaluator = Evaluator(config)
        self.workflow_executor = WorkflowExecutor(config)
        
        # Create checkpoint saver for state persistence
        self.checkpointer = MemorySaver()
        
    def build_graph(self) -> StateGraph:
        """
        Build the complete execution graph with all nodes and edges.
        
        Returns:
            Compiled StateGraph ready for execution
        """
        # Create the graph
        graph = StateGraph(ExecutionState)
        
        # Add all nodes
        self._add_nodes(graph)
        
        # Add edges and conditional routing
        self._add_edges(graph)
        
        # Set entry point
        graph.set_entry_point("orchestrator")
        
        # Compile the graph with checkpointing
        compiled_graph = graph.compile(checkpointer=self.checkpointer)
        
        self.logger.info("LangGraph execution graph built successfully")
        
        return compiled_graph
    
    def _add_nodes(self, graph: StateGraph) -> None:
        """Add all nodes to the graph."""
        # Core orchestration nodes
        graph.add_node("orchestrator", self._orchestrator_node)
        graph.add_node("planner", self._planner_node)
        graph.add_node("task_identifier", self._task_identifier_node)
        graph.add_node("agent_generator", self._agent_generator_node)
        graph.add_node("agent_executor", self._agent_executor_node)
        
        # Workflow execution node
        graph.add_node("workflow_executor", self._workflow_executor_node)
        
        # Context and memory nodes
        graph.add_node("context_manager", self._context_manager_node)
        graph.add_node("memory_manager", self._memory_manager_node)
        
        # Evaluation and result assembly
        graph.add_node("evaluator", self._evaluator_node)
        graph.add_node("result_assembler", self._result_assembler_node)
        
        self.logger.debug("Added all nodes to graph")
    
    def _add_edges(self, graph: StateGraph) -> None:
        """Add edges and conditional routing to the graph."""
        # Orchestrator routing based on execution path
        graph.add_conditional_edges(
            "orchestrator",
            self._route_from_orchestrator,
            {
                "workflow_executor": "workflow_executor",
                "planner": "planner",
                "context_manager": "context_manager",
                "end": END
            }
        )
        
        # Planner routes to task identifier
        graph.add_edge("planner", "task_identifier")
        
        # Task identifier routes to agent generator
        graph.add_edge("task_identifier", "agent_generator")
        
        # Agent generator routes to agent executor
        graph.add_edge("agent_generator", "agent_executor")
        
        # Agent executor routes to evaluator
        graph.add_conditional_edges(
            "agent_executor",
            self._route_from_agent_executor,
            {
                "evaluator": "evaluator",
                "agent_executor": "agent_executor",  # Loop for multiple agents
                "end": END
            }
        )
        
        # Workflow executor routes to context manager or evaluator
        graph.add_conditional_edges(
            "workflow_executor",
            self._route_from_workflow_executor,
            {
                "context_manager": "context_manager",
                "evaluator": "evaluator",
                "end": END
            }
        )
        
        # Context manager routes back to workflow executor or orchestrator
        graph.add_conditional_edges(
            "context_manager",
            self._route_from_context_manager,
            {
                "workflow_executor": "workflow_executor",
                "orchestrator": "orchestrator",
                "evaluator": "evaluator"
            }
        )
        
        # Evaluator routes to result assembler or back to planner for replan
        graph.add_conditional_edges(
            "evaluator",
            self._route_from_evaluator,
            {
                "result_assembler": "result_assembler",
                "planner": "planner",  # Replan if verification fails
                "end": END
            }
        )
        
        # Result assembler routes to memory manager
        graph.add_edge("result_assembler", "memory_manager")
        
        # Memory manager is the final node before END
        graph.add_edge("memory_manager", END)
        
        self.logger.debug("Added all edges to graph")
    
    # Node wrapper functions
    
    def _orchestrator_node(self, state: ExecutionState) -> ExecutionState:
        """Orchestrator node wrapper."""
        return self.orchestrator(state)
    
    def _planner_node(self, state: ExecutionState) -> ExecutionState:
        """Planner node wrapper."""
        return self.planner(state)
    
    def _task_identifier_node(self, state: ExecutionState) -> ExecutionState:
        """Task identifier node wrapper."""
        return self.task_identifier(state)
    
    def _agent_generator_node(self, state: ExecutionState) -> ExecutionState:
        """Agent generator node wrapper."""
        return self.agent_generator(state)
    
    def _agent_executor_node(self, state: ExecutionState) -> ExecutionState:
        """Agent executor node wrapper."""
        return self.agent_executor(state)
    
    def _workflow_executor_node(self, state: ExecutionState) -> ExecutionState:
        """Workflow executor node wrapper."""
        return self.workflow_executor(state)
    
    def _context_manager_node(self, state: ExecutionState) -> ExecutionState:
        """Context manager node wrapper."""
        # Context manager handles file processing and retrieval
        if state.context.get("needs_file_processing"):
            files = state.context.get("uploaded_files", [])
            for file_path in files:
                try:
                    result = self.context_manager.ingest_file(file_path)
                    state.context[f"processed_{file_path}"] = result
                except Exception as e:
                    self.logger.error(f"Error processing file {file_path}: {e}")
                    state.context[f"error_{file_path}"] = str(e)
        
        # Handle context retrieval for RAG
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
    
    def _memory_manager_node(self, state: ExecutionState) -> ExecutionState:
        """Memory manager node wrapper."""
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
    
    def _evaluator_node(self, state: ExecutionState) -> ExecutionState:
        """Evaluator node wrapper."""
        return self.evaluator(state)
    
    def _result_assembler_node(self, state: ExecutionState) -> ExecutionState:
        """Result assembler node that combines outputs into final result."""
        # Assemble final result from completed tasks
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
            "response": state.context.get("final_response", ""),
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
                (state.completed_at - state.started_at).total_seconds()
                if state.completed_at and state.started_at
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
    
    # Routing functions
    
    def _route_from_orchestrator(self, state: ExecutionState) -> str:
        """Route from orchestrator based on execution path."""
        if state.is_failed or state.is_complete:
            return "end"
        
        if state.execution_path == ExecutionPath.PREDEFINED_WORKFLOW:
            return "workflow_executor"
        elif state.execution_path == ExecutionPath.PLANNER_DRIVEN:
            return "planner"
        elif state.execution_path == ExecutionPath.HYBRID:
            # For hybrid, start with workflow but allow fallback to planner
            return "workflow_executor"
        else:
            # Default to context manager if files need processing
            if state.context.get("needs_file_processing"):
                return "context_manager"
            return "end"
    
    def _route_from_agent_executor(self, state: ExecutionState) -> str:
        """Route from agent executor based on task completion."""
        if state.is_failed:
            return "end"
        
        # Check if there are more ready tasks
        ready_tasks = state.get_ready_tasks()
        if ready_tasks and state.is_within_limits():
            return "agent_executor"  # Loop back for next task
        
        # All tasks complete, go to evaluator
        if state.completed_tasks:
            return "evaluator"
        
        return "end"
    
    def _route_from_workflow_executor(self, state: ExecutionState) -> str:
        """Route from workflow executor based on workflow needs."""
        if state.is_failed:
            return "end"
        
        # Check if workflow needs context
        if state.context.get("needs_context_retrieval"):
            return "context_manager"
        
        # Check if workflow is complete
        if state.context.get("workflow_complete"):
            return "evaluator"
        
        return "end"
    
    def _route_from_context_manager(self, state: ExecutionState) -> str:
        """Route from context manager back to appropriate node."""
        # If we came from workflow executor, go back
        if state.execution_path == ExecutionPath.PREDEFINED_WORKFLOW:
            return "workflow_executor"
        
        # If we have retrieved context, go to evaluator
        if state.context.get("retrieval_complete"):
            return "evaluator"
        
        # Otherwise go back to orchestrator
        return "orchestrator"
    
    def _route_from_evaluator(self, state: ExecutionState) -> str:
        """Route from evaluator based on verification results."""
        if state.is_failed:
            return "end"
        
        # Check if verification passed
        verification_passed = state.context.get("verification_passed", True)
        
        if verification_passed:
            return "result_assembler"
        else:
            # Check if we should replan
            replan_count = state.context.get("replan_count", 0)
            max_replans = self.config.orchestrator.max_replans if hasattr(self.config.orchestrator, 'max_replans') else 2
            
            if replan_count < max_replans and state.is_within_limits():
                state.context["replan_count"] = replan_count + 1
                return "planner"
            else:
                # Max replans reached, end execution
                return "end"


def create_execution_graph(config: SystemConfig):
    """
    Factory function to create a complete execution graph.
    
    Args:
        config: System configuration
        
    Returns:
        Compiled StateGraph or SimpleGraphExecutor ready for execution
    """
    if LANGGRAPH_AVAILABLE:
        try:
            builder = GraphBuilder(config)
            return builder.build_graph()
        except Exception as e:
            logger.warning(f"Failed to create LangGraph, falling back to simple executor: {e}")
    
    # Fallback to simple executor
    from .simple_graph_executor import create_simple_execution_graph
    return create_simple_execution_graph(config)
