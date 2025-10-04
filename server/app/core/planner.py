"""Planner node for request decomposition and task management."""

import logging
import re
from typing import Dict, Any, Optional, List, Set
from datetime import datetime

from ..config.models import SystemConfig
from .base_nodes import ProcessingNode
from .state import ExecutionState, Task, TaskStatus


logger = logging.getLogger(__name__)


class RequestAnalyzer:
    """Analyzes user requests to understand complexity and requirements."""
    
    def __init__(self):
        """Initialize request analyzer."""
        # Keywords that indicate different types of operations
        self.file_keywords = ["upload", "file", "document", "pdf", "word", "excel", "image"]
        self.question_keywords = ["what", "how", "why", "when", "where", "who", "explain", "describe"]
        self.action_keywords = ["create", "generate", "build", "make", "write", "analyze", "compare"]
        self.complex_keywords = ["and", "then", "after", "before", "also", "additionally", "furthermore"]
        
    def analyze_request(self, request: str) -> Dict[str, Any]:
        """
        Analyze a user request to understand its complexity and requirements.
        
        Args:
            request: User's input request
            
        Returns:
            Analysis results dictionary
        """
        request_lower = request.lower()
        
        # Detect request type
        request_type = self._detect_request_type(request_lower)
        
        # Estimate complexity
        complexity = self._estimate_complexity(request_lower)
        
        # Extract entities and actions
        entities = self._extract_entities(request)
        actions = self._extract_actions(request_lower)
        
        # Detect dependencies
        has_dependencies = self._detect_dependencies(request_lower)
        
        return {
            "request_type": request_type,
            "complexity": complexity,
            "entities": entities,
            "actions": actions,
            "has_dependencies": has_dependencies,
            "requires_file_processing": any(kw in request_lower for kw in self.file_keywords),
            "is_question": any(kw in request_lower for kw in self.question_keywords),
            "requires_generation": any(kw in request_lower for kw in self.action_keywords)
        }
    
    def _detect_request_type(self, request: str) -> str:
        """Detect the primary type of request."""
        if any(kw in request for kw in self.question_keywords):
            return "question"
        elif any(kw in request for kw in self.action_keywords):
            return "action"
        elif any(kw in request for kw in self.file_keywords):
            return "file_processing"
        else:
            return "general"
    
    def _estimate_complexity(self, request: str) -> int:
        """Estimate request complexity on a scale of 1-5."""
        complexity = 1
        
        # Length-based complexity
        if len(request.split()) > 20:
            complexity += 1
        if len(request.split()) > 50:
            complexity += 1
            
        # Keyword-based complexity
        if any(kw in request for kw in self.complex_keywords):
            complexity += 1
            
        # Multiple actions increase complexity
        action_count = sum(1 for kw in self.action_keywords if kw in request)
        if action_count > 1:
            complexity += 1
            
        return min(complexity, 5)
    
    def _extract_entities(self, request: str) -> List[str]:
        """Extract potential entities from the request."""
        # Simple entity extraction - can be enhanced with NLP
        entities = []
        
        # Extract quoted strings
        quoted = re.findall(r'"([^"]*)"', request)
        entities.extend(quoted)
        
        # Extract file extensions
        file_exts = re.findall(r'\.\w{2,4}\b', request)
        entities.extend(file_exts)
        
        return entities
    
    def _extract_actions(self, request: str) -> List[str]:
        """Extract action verbs from the request."""
        actions = []
        for keyword in self.action_keywords:
            if keyword in request:
                actions.append(keyword)
        return actions
    
    def _detect_dependencies(self, request: str) -> bool:
        """Detect if the request has sequential dependencies."""
        return any(kw in request for kw in self.complex_keywords)


class TaskDecomposer:
    """Decomposes complex requests into manageable tasks."""
    
    def __init__(self, config: SystemConfig):
        """
        Initialize task decomposer.
        
        Args:
            config: System configuration
        """
        self.config = config
        self.max_tasks = config.planner.max_tasks_per_request
        
        # Enhanced decomposition patterns for better task identification
        self.decomposition_patterns = {
            "sequential": ["then", "after", "next", "following", "subsequently"],
            "parallel": ["and", "also", "additionally", "simultaneously", "concurrently"],
            "conditional": ["if", "when", "unless", "provided", "assuming"],
            "iterative": ["for each", "iterate", "loop", "repeat", "multiple"],
            "comparative": ["compare", "contrast", "versus", "against", "difference"]
        }
        
    def decompose_request(self, request: str, analysis: Dict[str, Any]) -> List[Task]:
        """
        Decompose a request into individual tasks using enhanced algorithm.
        
        Args:
            request: Original user request
            analysis: Request analysis results
            
        Returns:
            List of decomposed tasks with proper dependencies
        """
        tasks = []
        
        # Enhanced decomposition based on request complexity and patterns
        if analysis["complexity"] >= 4:
            tasks.extend(self._decompose_complex_request(request, analysis))
        elif analysis["has_dependencies"]:
            tasks.extend(self._decompose_sequential_request(request, analysis))
        else:
            # Use existing simple decomposition for basic requests
            if analysis["request_type"] == "file_processing":
                tasks.extend(self._create_file_processing_tasks(request, analysis))
            elif analysis["request_type"] == "question":
                tasks.extend(self._create_question_answering_tasks(request, analysis))
            elif analysis["request_type"] == "action":
                tasks.extend(self._create_action_tasks(request, analysis))
            else:
                tasks.extend(self._create_general_tasks(request, analysis))
        
        # Enhance tasks with better success criteria
        tasks = self._enhance_task_success_criteria(tasks, analysis)
        
        # Limit number of tasks
        if len(tasks) > self.max_tasks:
            logger.warning(f"Too many tasks generated ({len(tasks)}), limiting to {self.max_tasks}")
            tasks = tasks[:self.max_tasks]
            
        return tasks
    
    def _decompose_complex_request(self, request: str, analysis: Dict[str, Any]) -> List[Task]:
        """Decompose complex requests into minimal, focused tasks."""
        tasks = []
        request_lower = request.lower()
        
        # Identify task boundaries using linguistic patterns
        task_segments = self._identify_task_segments(request)
        
        for i, segment in enumerate(task_segments):
            # Create focused task for each segment
            task_name = f"complex_task_{i+1}"
            task_description = f"Execute: {segment.strip()}"
            
            # Determine task type and inputs
            task_type = self._classify_task_segment(segment)
            inputs = self._extract_task_inputs(segment, analysis)
            
            task = Task(
                name=task_name,
                description=task_description,
                inputs=inputs,
                expected_output=self._determine_expected_output(segment, task_type),
                success_criteria=self._generate_success_criteria(segment, task_type)
            )
            
            # Add dependencies for sequential tasks
            if i > 0 and self._requires_previous_output(segment):
                task.dependencies.append(tasks[i-1].id)
            
            tasks.append(task)
        
        return tasks
    
    def _decompose_sequential_request(self, request: str, analysis: Dict[str, Any]) -> List[Task]:
        """Decompose requests with clear sequential dependencies."""
        tasks = []
        
        # Split on sequential indicators
        segments = self._split_on_sequential_patterns(request)
        
        for i, segment in enumerate(segments):
            task = Task(
                name=f"sequential_task_{i+1}",
                description=f"Execute step {i+1}: {segment.strip()}",
                inputs=self._extract_task_inputs(segment, analysis),
                expected_output=self._determine_expected_output(segment, "sequential"),
                success_criteria=self._generate_success_criteria(segment, "sequential")
            )
            
            # Each task depends on the previous one
            if i > 0:
                task.dependencies.append(tasks[i-1].id)
            
            tasks.append(task)
        
        return tasks
    
    def _identify_task_segments(self, request: str) -> List[str]:
        """Identify distinct task segments in a complex request."""
        # Simple segmentation based on conjunctions and punctuation
        segments = []
        
        # Split on major separators
        for separator in ['. ', '; ', ' and then ', ' after that ', ' next ']:
            if separator in request:
                segments = request.split(separator)
                break
        
        if not segments:
            # Fallback to sentence splitting
            segments = [s.strip() for s in request.split('.') if s.strip()]
        
        return segments if segments else [request]
    
    def _split_on_sequential_patterns(self, request: str) -> List[str]:
        """Split request on sequential pattern indicators."""
        import re
        
        # Pattern for sequential indicators
        pattern = r'\b(?:then|after|next|following|subsequently)\b'
        segments = re.split(pattern, request, flags=re.IGNORECASE)
        
        return [s.strip() for s in segments if s.strip()]
    
    def _classify_task_segment(self, segment: str) -> str:
        """Classify a task segment by type."""
        segment_lower = segment.lower()
        
        if any(kw in segment_lower for kw in ["analyze", "examine", "study"]):
            return "analysis"
        elif any(kw in segment_lower for kw in ["create", "generate", "build"]):
            return "generation"
        elif any(kw in segment_lower for kw in ["find", "search", "retrieve"]):
            return "retrieval"
        elif any(kw in segment_lower for kw in ["compare", "contrast", "evaluate"]):
            return "comparison"
        else:
            return "general"
    
    def _extract_task_inputs(self, segment: str, analysis: Dict[str, Any]) -> List[str]:
        """Extract required inputs for a task segment."""
        inputs = []
        segment_lower = segment.lower()
        
        # Add context-based inputs
        if analysis.get("requires_file_processing"):
            inputs.append("uploaded_files")
        
        if "document" in segment_lower or "file" in segment_lower:
            inputs.append("document_content")
        
        if "previous" in segment_lower or "result" in segment_lower:
            inputs.append("previous_results")
        
        if "question" in segment_lower:
            inputs.append("user_question")
        
        # Default input if none identified
        if not inputs:
            inputs.append("user_request")
        
        return inputs
    
    def _determine_expected_output(self, segment: str, task_type: str) -> str:
        """Determine expected output for a task segment."""
        segment_lower = segment.lower()
        
        if task_type == "analysis":
            return "analysis_results"
        elif task_type == "generation":
            return "generated_content"
        elif task_type == "retrieval":
            return "retrieved_information"
        elif task_type == "comparison":
            return "comparison_results"
        elif "answer" in segment_lower:
            return "answer_with_citations"
        elif "summary" in segment_lower:
            return "summary_content"
        else:
            return "task_output"
    
    def _generate_success_criteria(self, segment: str, task_type: str) -> List[str]:
        """Generate success criteria for a task segment."""
        criteria = []
        segment_lower = segment.lower()
        
        # Base criteria for all tasks
        criteria.append("Task completes without errors")
        criteria.append("Output is relevant to the request")
        
        # Type-specific criteria
        if task_type == "analysis":
            criteria.extend([
                "Analysis is thorough and accurate",
                "Key insights are identified"
            ])
        elif task_type == "generation":
            criteria.extend([
                "Generated content meets requirements",
                "Content is coherent and well-structured"
            ])
        elif task_type == "retrieval":
            criteria.extend([
                "Relevant information is retrieved",
                "Sources are properly identified"
            ])
        elif "citation" in segment_lower or "source" in segment_lower:
            criteria.append("Proper citations are included")
        
        return criteria
    
    def _requires_previous_output(self, segment: str) -> bool:
        """Check if a task segment requires output from previous tasks."""
        segment_lower = segment.lower()
        dependency_indicators = [
            "based on", "using the", "from the previous", "with the results",
            "taking into account", "considering the", "building on"
        ]
        
        return any(indicator in segment_lower for indicator in dependency_indicators)
    
    def _enhance_task_success_criteria(self, tasks: List[Task], analysis: Dict[str, Any]) -> List[Task]:
        """Enhance tasks with more detailed success criteria."""
        for task in tasks:
            # Add quality criteria
            if not any("quality" in criterion.lower() for criterion in task.success_criteria):
                task.success_criteria.append("Output meets quality standards")
            
            # Add verification criteria for complex tasks
            if analysis["complexity"] >= 3:
                task.success_criteria.append("Results can be verified against requirements")
            
            # Add citation criteria for information tasks
            if any(keyword in task.description.lower() for keyword in ["question", "information", "document"]):
                if not any("citation" in criterion.lower() for criterion in task.success_criteria):
                    task.success_criteria.append("Sources are properly cited when applicable")
        
        return tasks
    
    def _create_file_processing_tasks(self, request: str, analysis: Dict[str, Any]) -> List[Task]:
        """Create tasks for file processing requests."""
        tasks = []
        
        # File ingestion task
        ingest_task = Task(
            name="file_ingestion",
            description="Process and ingest uploaded files",
            inputs=["uploaded_files"],
            expected_output="processed_file_chunks",
            success_criteria=[
                "Files are successfully processed",
                "Content is extracted and chunked",
                "Embeddings are generated and stored"
            ]
        )
        tasks.append(ingest_task)
        
        # If it's also a question, add QA task
        if analysis["is_question"]:
            qa_task = Task(
                name="document_qa",
                description="Answer questions about processed documents",
                inputs=["user_question", "processed_file_chunks"],
                expected_output="answer_with_citations",
                success_criteria=[
                    "Question is answered accurately",
                    "Answer includes proper citations",
                    "Sources are verified"
                ],
                dependencies=[ingest_task.id]
            )
            tasks.append(qa_task)
            
        return tasks
    
    def _create_question_answering_tasks(self, request: str, analysis: Dict[str, Any]) -> List[Task]:
        """Create tasks for question answering requests."""
        tasks = []
        
        # Context retrieval task
        retrieval_task = Task(
            name="context_retrieval",
            description="Retrieve relevant context for the question",
            inputs=["user_question"],
            expected_output="relevant_context",
            success_criteria=[
                "Relevant information is retrieved",
                "Context is sufficient to answer question"
            ]
        )
        tasks.append(retrieval_task)
        
        # Answer generation task
        answer_task = Task(
            name="answer_generation",
            description="Generate answer based on retrieved context",
            inputs=["user_question", "relevant_context"],
            expected_output="final_answer",
            success_criteria=[
                "Answer is accurate and complete",
                "Answer is grounded in provided context",
                "Citations are included where appropriate"
            ],
            dependencies=[retrieval_task.id]
        )
        tasks.append(answer_task)
        
        return tasks
    
    def _create_action_tasks(self, request: str, analysis: Dict[str, Any]) -> List[Task]:
        """Create tasks for action-oriented requests."""
        tasks = []
        
        for action in analysis["actions"]:
            task = Task(
                name=f"{action}_task",
                description=f"Execute {action} operation",
                inputs=["user_requirements"],
                expected_output=f"{action}_result",
                success_criteria=[
                    f"{action.capitalize()} operation completed successfully",
                    "Output meets user requirements"
                ]
            )
            tasks.append(task)
            
        return tasks
    
    def _create_general_tasks(self, request: str, analysis: Dict[str, Any]) -> List[Task]:
        """Create tasks for general requests."""
        tasks = []
        
        # Single general processing task
        general_task = Task(
            name="general_processing",
            description="Process general user request",
            inputs=["user_request"],
            expected_output="response",
            success_criteria=[
                "Request is understood and processed",
                "Appropriate response is generated"
            ]
        )
        tasks.append(general_task)
        
        return tasks


class DependencyResolver:
    """Resolves task dependencies and creates execution order with enhanced topological sorting."""
    
    def resolve_dependencies(self, tasks: List[Task]) -> List[str]:
        """
        Resolve task dependencies and return execution order using enhanced topological sorting.
        
        Args:
            tasks: List of tasks with dependencies
            
        Returns:
            List of task IDs in execution order
        """
        if not tasks:
            return []
        
        # Create task lookup
        task_lookup = {task.id: task for task in tasks}
        
        # Validate dependencies exist
        self._validate_dependencies(tasks, task_lookup)
        
        # Build dependency graph with reverse edges for efficient processing
        graph, reverse_graph, in_degree = self._build_dependency_graph(tasks)
        
        # Enhanced topological sort with priority handling
        execution_order = self._topological_sort_with_priority(
            tasks, graph, reverse_graph, in_degree, task_lookup
        )
        
        # Verify and handle circular dependencies
        if len(execution_order) != len(tasks):
            execution_order = self._handle_circular_dependencies(
                tasks, execution_order, task_lookup
            )
        
        return execution_order
    
    def _validate_dependencies(self, tasks: List[Task], task_lookup: Dict[str, Task]) -> None:
        """Validate that all dependencies reference existing tasks."""
        for task in tasks:
            for dep_id in task.dependencies:
                if dep_id not in task_lookup:
                    logger.warning(f"Task {task.name} has invalid dependency: {dep_id}")
                    # Remove invalid dependency
                    task.dependencies.remove(dep_id)
    
    def _build_dependency_graph(self, tasks: List[Task]) -> tuple:
        """Build forward and reverse dependency graphs with in-degree tracking."""
        graph = {}  # task_id -> list of tasks that depend on it
        reverse_graph = {}  # task_id -> list of dependencies
        in_degree = {}  # task_id -> number of dependencies
        
        # Initialize structures
        for task in tasks:
            graph[task.id] = []
            reverse_graph[task.id] = task.dependencies.copy()
            in_degree[task.id] = len(task.dependencies)
        
        # Build forward graph
        for task in tasks:
            for dep_id in task.dependencies:
                if dep_id in graph:
                    graph[dep_id].append(task.id)
        
        return graph, reverse_graph, in_degree
    
    def _topological_sort_with_priority(
        self, 
        tasks: List[Task], 
        graph: Dict[str, List[str]], 
        reverse_graph: Dict[str, List[str]], 
        in_degree: Dict[str, int],
        task_lookup: Dict[str, Task]
    ) -> List[str]:
        """Enhanced topological sort with task priority consideration."""
        execution_order = []
        
        # Use priority queue to handle tasks with same dependency level
        from heapq import heappush, heappop
        
        # Initialize with tasks that have no dependencies
        ready_queue = []
        for task_id, degree in in_degree.items():
            if degree == 0:
                priority = self._calculate_task_priority(task_lookup[task_id])
                heappush(ready_queue, (priority, task_id))
        
        while ready_queue:
            # Get highest priority ready task
            _, current_task_id = heappop(ready_queue)
            execution_order.append(current_task_id)
            
            # Update dependent tasks
            for dependent_task_id in graph[current_task_id]:
                in_degree[dependent_task_id] -= 1
                
                # If all dependencies satisfied, add to ready queue
                if in_degree[dependent_task_id] == 0:
                    priority = self._calculate_task_priority(task_lookup[dependent_task_id])
                    heappush(ready_queue, (priority, dependent_task_id))
        
        return execution_order
    
    def _calculate_task_priority(self, task: Task) -> int:
        """Calculate task priority for execution ordering (lower number = higher priority)."""
        priority = 0
        
        # Prioritize by task type
        task_name_lower = task.name.lower()
        
        # File processing tasks should go first
        if "file" in task_name_lower or "ingest" in task_name_lower:
            priority -= 10
        
        # Context retrieval should be early
        elif "context" in task_name_lower or "retrieval" in task_name_lower:
            priority -= 5
        
        # Generation tasks should be later
        elif "generate" in task_name_lower or "create" in task_name_lower:
            priority += 5
        
        # Final tasks (like verification) should be last
        elif "verify" in task_name_lower or "final" in task_name_lower:
            priority += 10
        
        # Consider number of success criteria (more criteria = more important)
        priority -= len(task.success_criteria)
        
        return priority
    
    def _handle_circular_dependencies(
        self, 
        tasks: List[Task], 
        execution_order: List[str], 
        task_lookup: Dict[str, Task]
    ) -> List[str]:
        """Handle circular dependencies by breaking cycles and adding remaining tasks."""
        remaining_task_ids = [
            task.id for task in tasks 
            if task.id not in execution_order
        ]
        
        if remaining_task_ids:
            logger.error(f"Circular dependency detected in tasks: {remaining_task_ids}")
            
            # Try to break cycles by removing problematic dependencies
            cycle_broken = self._break_dependency_cycles(remaining_task_ids, task_lookup)
            
            if cycle_broken:
                # Retry resolution with broken cycles
                remaining_tasks = [task_lookup[task_id] for task_id in remaining_task_ids]
                _, _, in_degree = self._build_dependency_graph(remaining_tasks)
                
                # Add tasks with no remaining dependencies
                for task_id in remaining_task_ids:
                    if in_degree.get(task_id, 0) == 0:
                        execution_order.append(task_id)
                        remaining_task_ids.remove(task_id)
            
            # Add any still-remaining tasks at the end
            execution_order.extend(remaining_task_ids)
        
        return execution_order
    
    def _break_dependency_cycles(self, task_ids: List[str], task_lookup: Dict[str, Task]) -> bool:
        """Attempt to break dependency cycles by removing the least critical dependencies."""
        # Simple cycle breaking: remove dependencies between tasks in the cycle
        cycle_broken = False
        
        for task_id in task_ids:
            task = task_lookup[task_id]
            # Remove dependencies that point to other tasks in the cycle
            original_deps = task.dependencies.copy()
            task.dependencies = [
                dep_id for dep_id in task.dependencies 
                if dep_id not in task_ids
            ]
            
            if len(task.dependencies) < len(original_deps):
                cycle_broken = True
                logger.warning(
                    f"Broke circular dependency for task {task.name}: "
                    f"removed {set(original_deps) - set(task.dependencies)}"
                )
        
        return cycle_broken
    
    def get_dependency_graph_info(self, tasks: List[Task]) -> Dict[str, Any]:
        """Get information about the dependency graph for debugging/monitoring."""
        if not tasks:
            return {"total_tasks": 0, "has_cycles": False, "max_depth": 0}
        
        task_lookup = {task.id: task for task in tasks}
        graph, reverse_graph, in_degree = self._build_dependency_graph(tasks)
        
        # Calculate graph metrics
        total_tasks = len(tasks)
        independent_tasks = sum(1 for degree in in_degree.values() if degree == 0)
        max_dependencies = max(in_degree.values()) if in_degree else 0
        
        # Check for cycles by attempting resolution
        execution_order = self._topological_sort_with_priority(
            tasks, graph, reverse_graph, in_degree.copy(), task_lookup
        )
        has_cycles = len(execution_order) != total_tasks
        
        # Calculate maximum depth
        max_depth = self._calculate_max_depth(tasks, graph, reverse_graph)
        
        return {
            "total_tasks": total_tasks,
            "independent_tasks": independent_tasks,
            "max_dependencies": max_dependencies,
            "has_cycles": has_cycles,
            "max_depth": max_depth,
            "execution_order": execution_order
        }
    
    def _calculate_max_depth(
        self, 
        tasks: List[Task], 
        graph: Dict[str, List[str]], 
        reverse_graph: Dict[str, List[str]]
    ) -> int:
        """Calculate the maximum depth of the dependency graph."""
        depths = {}
        
        def calculate_depth(task_id: str) -> int:
            if task_id in depths:
                return depths[task_id]
            
            if not reverse_graph[task_id]:  # No dependencies
                depths[task_id] = 0
                return 0
            
            max_dep_depth = max(
                calculate_depth(dep_id) 
                for dep_id in reverse_graph[task_id]
            )
            depths[task_id] = max_dep_depth + 1
            return depths[task_id]
        
        if not tasks:
            return 0
        
        return max(calculate_depth(task.id) for task in tasks)


class Planner(ProcessingNode):
    """
    Planner node that decomposes complex requests into structured task graphs.
    
    Responsibilities:
    - Analyze user requests for complexity and requirements
    - Generate minimal task sets with clear dependencies
    - Define success criteria for each task
    - Optimize task ordering for efficiency
    """
    
    def __init__(self, config: SystemConfig):
        """
        Initialize the Planner.
        
        Args:
            config: System configuration
        """
        super().__init__("Planner", config.planner.__dict__)
        self.system_config = config
        self.request_analyzer = RequestAnalyzer()
        self.task_decomposer = TaskDecomposer(config)
        self.dependency_resolver = DependencyResolver()
        
    def execute(self, state: ExecutionState) -> ExecutionState:
        """
        Execute planner logic to decompose request into tasks.
        
        Args:
            state: Current execution state
            
        Returns:
            Updated execution state with task plan
        """
        # Analyze the user request
        analysis = self.request_analyzer.analyze_request(state.user_request)
        
        self.logger.info(f"Request analysis: {analysis}")
        state.context["request_analysis"] = analysis
        
        # Decompose request into tasks
        tasks = self.task_decomposer.decompose_request(state.user_request, analysis)
        
        self.logger.info(f"Generated {len(tasks)} tasks")
        
        # Add tasks to state
        for task in tasks:
            state.add_task(task)
            
        # Resolve dependencies and create execution order
        execution_order = self.dependency_resolver.resolve_dependencies(tasks)
        state.context["task_execution_order"] = execution_order
        
        self.logger.info(f"Task execution order: {execution_order}")
        
        # Log planning results
        state.log_execution_step(
            node_name=self.name,
            action="planning_complete",
            details={
                "num_tasks": len(tasks),
                "execution_order": execution_order,
                "request_analysis": analysis
            }
        )
        
        # Set next node to Task Identifier
        state.next_node = "TaskIdentifier"
        
        return state
    
    def validate_inputs(self, state: ExecutionState) -> bool:
        """
        Validate that required inputs are present.
        
        Args:
            state: Current execution state
            
        Returns:
            True if inputs are valid, False otherwise
        """
        return bool(state.user_request and state.user_request.strip())
    
    def get_next_node(self, state: ExecutionState) -> Optional[str]:
        """
        Determine the next node based on planning results.
        
        Args:
            state: Current execution state
            
        Returns:
            Name of next node
        """
        if state.tasks:
            return "TaskIdentifier"
        else:
            # No tasks generated, complete execution
            state.complete_execution("No tasks required for this request")
            return None