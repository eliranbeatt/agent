"""Task Identifier node for mapping tasks to tools and context."""

import logging
from typing import Dict, Any, Optional, List, Set

from ..config.models import SystemConfig
from .base_nodes import ProcessingNode
from .state import ExecutionState, Task, TaskStatus


logger = logging.getLogger(__name__)


class ToolMapper:
    """Maps tasks to appropriate tools based on task requirements with enhanced algorithms."""
    
    def __init__(self, available_tools: List[str]):
        """
        Initialize tool mapper.
        
        Args:
            available_tools: List of available tool names
        """
        self.available_tools = set(available_tools)
        
        # Enhanced tool categories with more granular capabilities
        self.tool_categories = {
            "file_processing": {
                "tools": ["file_reader", "pdf_processor", "ocr_engine", "document_parser", "unstructured_loader"],
                "capabilities": ["read_files", "extract_text", "parse_structure", "handle_images"]
            },
            "web_search": {
                "tools": ["web_search", "search_engine", "web_scraper"],
                "capabilities": ["search_web", "retrieve_urls", "extract_content"]
            },
            "computation": {
                "tools": ["calculator", "code_executor", "data_analyzer", "statistical_analyzer"],
                "capabilities": ["calculate", "execute_code", "analyze_data", "statistical_analysis"]
            },
            "text_generation": {
                "tools": ["text_generator", "summarizer", "translator", "content_creator"],
                "capabilities": ["generate_text", "summarize", "translate", "create_content"]
            },
            "database": {
                "tools": ["database_query", "vector_search", "memory_retrieval", "semantic_search"],
                "capabilities": ["query_db", "vector_similarity", "retrieve_memory", "semantic_matching"]
            },
            "api_calls": {
                "tools": ["api_client", "http_client", "rest_client", "openai_client"],
                "capabilities": ["make_requests", "handle_apis", "external_services"]
            },
            "validation": {
                "tools": ["fact_checker", "citation_validator", "quality_assessor"],
                "capabilities": ["verify_facts", "validate_sources", "assess_quality"]
            }
        }
        
        # Enhanced task type to tool mapping with complexity consideration
        self.task_tool_mapping = {
            "file_ingestion": {
                "primary": ["file_reader", "pdf_processor", "ocr_engine", "unstructured_loader"],
                "secondary": ["document_parser"],
                "complexity_scaling": True
            },
            "document_qa": {
                "primary": ["vector_search", "semantic_search", "text_generator"],
                "secondary": ["memory_retrieval", "citation_validator"],
                "complexity_scaling": True
            },
            "context_retrieval": {
                "primary": ["vector_search", "memory_retrieval"],
                "secondary": ["web_search", "semantic_search"],
                "complexity_scaling": False
            },
            "answer_generation": {
                "primary": ["text_generator", "content_creator"],
                "secondary": ["memory_retrieval", "fact_checker"],
                "complexity_scaling": True
            },
            "analysis_task": {
                "primary": ["data_analyzer", "statistical_analyzer"],
                "secondary": ["text_generator", "quality_assessor"],
                "complexity_scaling": True
            },
            "comparison_task": {
                "primary": ["data_analyzer", "text_generator"],
                "secondary": ["statistical_analyzer", "quality_assessor"],
                "complexity_scaling": True
            },
            "generation_task": {
                "primary": ["text_generator", "content_creator"],
                "secondary": ["code_executor", "quality_assessor"],
                "complexity_scaling": True
            }
        }
    
    def map_task_to_tools(self, task: Task, complexity: int = 1) -> List[str]:
        """
        Map a task to appropriate tools using enhanced algorithm.
        
        Args:
            task: Task to map
            complexity: Task complexity score (1-5)
            
        Returns:
            List of recommended tool names ordered by relevance
        """
        recommended_tools = []
        
        # Enhanced direct mapping based on task classification
        task_type = self._classify_task_type(task)
        
        if task_type in self.task_tool_mapping:
            mapping = self.task_tool_mapping[task_type]
            
            # Always include primary tools
            recommended_tools.extend(mapping["primary"])
            
            # Include secondary tools based on complexity
            if complexity >= 3 or len(task.success_criteria) > 2:
                recommended_tools.extend(mapping["secondary"])
        
        # Pattern-based enhancement
        additional_tools = self._get_pattern_based_tools(task)
        recommended_tools.extend(additional_tools)
        
        # Input/output based tool selection
        io_tools = self._get_io_based_tools(task)
        recommended_tools.extend(io_tools)
        
        # Success criteria based tool selection
        criteria_tools = self._get_criteria_based_tools(task)
        recommended_tools.extend(criteria_tools)
        
        # Filter to available tools and prioritize
        available_tools = [tool for tool in recommended_tools if tool in self.available_tools]
        
        # Remove duplicates while preserving order and add scoring
        prioritized_tools = self._prioritize_tools(available_tools, task, complexity)
        
        # Fallback to general tools if none found
        if not prioritized_tools:
            general_tools = ["text_generator", "web_search", "memory_retrieval"]
            prioritized_tools = [tool for tool in general_tools if tool in self.available_tools]
        
        return prioritized_tools
    
    def _classify_task_type(self, task: Task) -> str:
        """Classify task type based on name, description, and characteristics."""
        task_text = f"{task.name} {task.description}".lower()
        
        # Classification patterns
        if any(pattern in task_text for pattern in ["file", "document", "upload", "ingest"]):
            return "file_ingestion"
        elif any(pattern in task_text for pattern in ["question", "answer", "qa", "ask"]):
            return "document_qa"
        elif any(pattern in task_text for pattern in ["retrieve", "find", "search", "context"]):
            return "context_retrieval"
        elif any(pattern in task_text for pattern in ["generate", "create", "write", "produce"]):
            return "generation_task"
        elif any(pattern in task_text for pattern in ["analyze", "examine", "study", "investigate"]):
            return "analysis_task"
        elif any(pattern in task_text for pattern in ["compare", "contrast", "evaluate", "assess"]):
            return "comparison_task"
        else:
            return "answer_generation"  # Default fallback
    
    def _get_pattern_based_tools(self, task: Task) -> List[str]:
        """Get tools based on text patterns in task."""
        tools = []
        task_text = f"{task.name} {task.description}".lower()
        
        # File processing patterns
        if any(pattern in task_text for pattern in ["pdf", "word", "excel", "image", "ocr"]):
            tools.extend(self.tool_categories["file_processing"]["tools"])
        
        # Web search patterns
        if any(pattern in task_text for pattern in ["web", "online", "internet", "url"]):
            tools.extend(self.tool_categories["web_search"]["tools"])
        
        # Computation patterns
        if any(pattern in task_text for pattern in ["calculate", "compute", "math", "statistics"]):
            tools.extend(self.tool_categories["computation"]["tools"])
        
        # Validation patterns
        if any(pattern in task_text for pattern in ["verify", "validate", "check", "confirm"]):
            tools.extend(self.tool_categories["validation"]["tools"])
        
        return tools
    
    def _get_io_based_tools(self, task: Task) -> List[str]:
        """Get tools based on task inputs and expected outputs."""
        tools = []
        
        # Analyze inputs
        for input_item in task.inputs:
            input_lower = input_item.lower()
            
            if "file" in input_lower or "document" in input_lower:
                tools.extend(["file_reader", "pdf_processor", "document_parser"])
            elif "question" in input_lower or "query" in input_lower:
                tools.extend(["vector_search", "semantic_search"])
            elif "data" in input_lower:
                tools.extend(["data_analyzer", "statistical_analyzer"])
        
        # Analyze expected output
        output_lower = task.expected_output.lower()
        
        if "answer" in output_lower or "response" in output_lower:
            tools.extend(["text_generator", "content_creator"])
        elif "analysis" in output_lower:
            tools.extend(["data_analyzer", "statistical_analyzer"])
        elif "summary" in output_lower:
            tools.extend(["summarizer", "text_generator"])
        elif "citation" in output_lower:
            tools.extend(["citation_validator", "fact_checker"])
        
        return tools
    
    def _get_criteria_based_tools(self, task: Task) -> List[str]:
        """Get tools based on success criteria."""
        tools = []
        
        for criterion in task.success_criteria:
            criterion_lower = criterion.lower()
            
            if "citation" in criterion_lower or "source" in criterion_lower:
                tools.extend(["citation_validator", "fact_checker"])
            elif "quality" in criterion_lower or "accurate" in criterion_lower:
                tools.extend(["quality_assessor", "fact_checker"])
            elif "verify" in criterion_lower or "validate" in criterion_lower:
                tools.extend(["fact_checker", "citation_validator"])
        
        return tools
    
    def _prioritize_tools(self, tools: List[str], task: Task, complexity: int) -> List[str]:
        """Prioritize tools based on relevance and complexity."""
        if not tools:
            return []
        
        # Score each tool
        tool_scores = {}
        
        for tool in tools:
            score = 0
            
            # Base score from frequency (more mentions = higher relevance)
            score += tools.count(tool) * 10
            
            # Complexity-based scoring
            if complexity >= 4 and tool in ["quality_assessor", "fact_checker", "citation_validator"]:
                score += 20  # High complexity tasks need validation tools
            
            # Task type specific bonuses
            task_text = f"{task.name} {task.description}".lower()
            
            if "document" in task_text and tool in ["pdf_processor", "document_parser"]:
                score += 15
            elif "question" in task_text and tool in ["vector_search", "semantic_search"]:
                score += 15
            elif "generate" in task_text and tool in ["text_generator", "content_creator"]:
                score += 15
            
            tool_scores[tool] = score
        
        # Sort by score (descending) and remove duplicates
        unique_tools = list(dict.fromkeys(tools))  # Preserve order while removing duplicates
        prioritized = sorted(unique_tools, key=lambda t: tool_scores.get(t, 0), reverse=True)
        
        # Limit to reasonable number of tools
        return prioritized[:8]  # Max 8 tools per task


class ContextAnalyzer:
    """Analyzes tasks to identify required context and dependencies with enhanced algorithms."""
    
    def analyze_context_requirements(self, task: Task, state: ExecutionState) -> Dict[str, Any]:
        """
        Analyze what context a task needs to execute successfully using enhanced analysis.
        
        Args:
            task: Task to analyze
            state: Current execution state
            
        Returns:
            Dictionary of comprehensive context requirements
        """
        requirements = {
            "requires_file_context": False,
            "requires_memory_context": False,
            "requires_web_context": False,
            "requires_previous_results": False,
            "requires_user_profile": False,
            "context_sources": [],
            "estimated_context_size": "small",  # small, medium, large
            "context_priority": "normal",  # low, normal, high, critical
            "retrieval_strategy": "standard",  # standard, comprehensive, targeted
            "context_filters": [],
            "temporal_requirements": None,
            "quality_requirements": []
        }
        
        # Enhanced input analysis
        self._analyze_task_inputs(task, requirements)
        
        # Dependency analysis
        self._analyze_task_dependencies(task, state, requirements)
        
        # Content analysis from task text
        self._analyze_task_content(task, requirements)
        
        # Success criteria analysis
        self._analyze_success_criteria(task, requirements)
        
        # State-based context analysis
        self._analyze_state_context(state, requirements)
        
        # Determine retrieval strategy
        self._determine_retrieval_strategy(task, requirements)
        
        # Estimate context size and priority
        self._estimate_context_metrics(task, requirements)
        
        return requirements
    
    def _analyze_task_inputs(self, task: Task, requirements: Dict[str, Any]) -> None:
        """Analyze task inputs for context requirements."""
        for input_item in task.inputs:
            input_lower = input_item.lower()
            
            # File and document context
            if any(keyword in input_lower for keyword in ["file", "document", "pdf", "upload"]):
                requirements["requires_file_context"] = True
                requirements["context_sources"].append("file_system")
                requirements["context_filters"].append("file_type")
            
            # Memory and history context
            if any(keyword in input_lower for keyword in ["memory", "history", "previous", "past"]):
                requirements["requires_memory_context"] = True
                requirements["context_sources"].append("memory")
                requirements["temporal_requirements"] = "historical"
            
            # Web and external context
            if any(keyword in input_lower for keyword in ["web", "search", "online", "external"]):
                requirements["requires_web_context"] = True
                requirements["context_sources"].append("web")
                requirements["context_filters"].append("relevance_score")
            
            # User profile context
            if any(keyword in input_lower for keyword in ["user", "profile", "preference", "personal"]):
                requirements["requires_user_profile"] = True
                requirements["context_sources"].append("user_profile")
            
            # Previous results context
            if any(keyword in input_lower for keyword in ["result", "output", "previous_task"]):
                requirements["requires_previous_results"] = True
                requirements["context_sources"].append("previous_tasks")
    
    def _analyze_task_dependencies(self, task: Task, state: ExecutionState, requirements: Dict[str, Any]) -> None:
        """Analyze task dependencies for context requirements."""
        if task.dependencies:
            requirements["requires_previous_results"] = True
            if "previous_tasks" not in requirements["context_sources"]:
                requirements["context_sources"].append("previous_tasks")
            
            # Analyze dependency types
            for dep_id in task.dependencies:
                dep_task = state.get_task_by_id(dep_id)
                if dep_task:
                    # If dependency involves file processing, we need file context
                    if "file" in dep_task.name.lower() or "document" in dep_task.name.lower():
                        requirements["requires_file_context"] = True
                        if "file_system" not in requirements["context_sources"]:
                            requirements["context_sources"].append("file_system")
                    
                    # If dependency involves retrieval, we need comprehensive context
                    if "retrieval" in dep_task.name.lower() or "search" in dep_task.name.lower():
                        requirements["retrieval_strategy"] = "comprehensive"
    
    def _analyze_task_content(self, task: Task, requirements: Dict[str, Any]) -> None:
        """Analyze task name and description for context clues."""
        task_text = f"{task.name} {task.description}".lower()
        
        # Document processing tasks
        if any(keyword in task_text for keyword in ["document", "pdf", "file", "upload", "ingest"]):
            requirements["requires_file_context"] = True
            if "file_system" not in requirements["context_sources"]:
                requirements["context_sources"].append("file_system")
        
        # Question answering tasks
        if any(keyword in task_text for keyword in ["question", "answer", "qa", "ask"]):
            requirements["requires_file_context"] = True
            requirements["requires_memory_context"] = True
            requirements["context_sources"].extend(["vector_db", "memory"])
            requirements["retrieval_strategy"] = "comprehensive"
            requirements["context_priority"] = "high"
        
        # Analysis tasks
        if any(keyword in task_text for keyword in ["analyze", "analysis", "examine", "study"]):
            requirements["estimated_context_size"] = "large"
            requirements["retrieval_strategy"] = "comprehensive"
            requirements["quality_requirements"].append("comprehensive_coverage")
        
        # Comparison tasks
        if any(keyword in task_text for keyword in ["compare", "contrast", "versus", "difference"]):
            requirements["estimated_context_size"] = "large"
            requirements["context_filters"].append("comparative_relevance")
            requirements["quality_requirements"].append("balanced_representation")
        
        # Summarization tasks
        if any(keyword in task_text for keyword in ["summarize", "summary", "brief", "overview"]):
            requirements["estimated_context_size"] = "large"
            requirements["quality_requirements"].append("completeness")
        
        # Generation tasks
        if any(keyword in task_text for keyword in ["generate", "create", "write", "produce"]):
            requirements["requires_memory_context"] = True
            requirements["context_sources"].append("memory")
            requirements["quality_requirements"].append("creativity_support")
    
    def _analyze_success_criteria(self, task: Task, requirements: Dict[str, Any]) -> None:
        """Analyze success criteria for additional context requirements."""
        for criterion in task.success_criteria:
            criterion_lower = criterion.lower()
            
            # Citation requirements
            if any(keyword in criterion_lower for keyword in ["citation", "source", "reference"]):
                requirements["quality_requirements"].append("source_tracking")
                requirements["context_filters"].append("source_metadata")
            
            # Accuracy requirements
            if any(keyword in criterion_lower for keyword in ["accurate", "correct", "precise"]):
                requirements["quality_requirements"].append("high_accuracy")
                requirements["context_priority"] = "high"
            
            # Completeness requirements
            if any(keyword in criterion_lower for keyword in ["complete", "comprehensive", "thorough"]):
                requirements["retrieval_strategy"] = "comprehensive"
                requirements["quality_requirements"].append("completeness")
            
            # Verification requirements
            if any(keyword in criterion_lower for keyword in ["verify", "validate", "check"]):
                requirements["quality_requirements"].append("verification_support")
                requirements["context_filters"].append("reliability_score")
    
    def _analyze_state_context(self, state: ExecutionState, requirements: Dict[str, Any]) -> None:
        """Analyze execution state for additional context insights."""
        # Check if files have been uploaded
        if state.context.get("uploaded_files"):
            requirements["requires_file_context"] = True
            if "file_system" not in requirements["context_sources"]:
                requirements["context_sources"].append("file_system")
        
        # Check conversation history
        if state.context.get("conversation_history"):
            requirements["requires_memory_context"] = True
            if "memory" not in requirements["context_sources"]:
                requirements["context_sources"].append("memory")
        
        # Check for user profile information
        if state.context.get("user_profile"):
            requirements["requires_user_profile"] = True
            if "user_profile" not in requirements["context_sources"]:
                requirements["context_sources"].append("user_profile")
        
        # Adjust based on execution path
        if state.execution_path and "predefined" in str(state.execution_path).lower():
            requirements["retrieval_strategy"] = "targeted"
        elif state.execution_path and "planner" in str(state.execution_path).lower():
            requirements["retrieval_strategy"] = "comprehensive"
    
    def _determine_retrieval_strategy(self, task: Task, requirements: Dict[str, Any]) -> None:
        """Determine the optimal retrieval strategy based on requirements."""
        # Start with current strategy
        current_strategy = requirements["retrieval_strategy"]
        
        # Upgrade strategy based on complexity
        complexity_indicators = [
            len(task.inputs) > 2,
            len(task.success_criteria) > 3,
            len(task.dependencies) > 1,
            requirements["context_priority"] == "high",
            "comprehensive" in requirements["quality_requirements"]
        ]
        
        if sum(complexity_indicators) >= 3:
            requirements["retrieval_strategy"] = "comprehensive"
        elif sum(complexity_indicators) >= 1 and current_strategy == "standard":
            requirements["retrieval_strategy"] = "targeted"
    
    def _estimate_context_metrics(self, task: Task, requirements: Dict[str, Any]) -> None:
        """Estimate context size and priority based on all analyzed factors."""
        # Calculate context size score
        size_score = 0
        
        # Base score from inputs and dependencies
        size_score += len(task.inputs) * 2
        size_score += len(task.dependencies) * 3
        size_score += len(task.success_criteria)
        
        # Bonus for context sources
        size_score += len(requirements["context_sources"]) * 2
        
        # Bonus for quality requirements
        size_score += len(requirements["quality_requirements"]) * 1.5
        
        # Determine size category
        if size_score >= 15:
            requirements["estimated_context_size"] = "large"
        elif size_score >= 8:
            requirements["estimated_context_size"] = "medium"
        else:
            requirements["estimated_context_size"] = "small"
        
        # Determine priority
        priority_indicators = [
            "high_accuracy" in requirements["quality_requirements"],
            "verification_support" in requirements["quality_requirements"],
            requirements["retrieval_strategy"] == "comprehensive",
            len(requirements["context_sources"]) > 3,
            "question" in task.name.lower() or "answer" in task.name.lower()
        ]
        
        if sum(priority_indicators) >= 3:
            requirements["context_priority"] = "critical"
        elif sum(priority_indicators) >= 2:
            requirements["context_priority"] = "high"
        elif sum(priority_indicators) >= 1:
            requirements["context_priority"] = "normal"
        else:
            requirements["context_priority"] = "low"


class ComplexityEstimator:
    """Estimates task complexity for resource allocation with enhanced algorithms."""
    
    def estimate_task_complexity(self, task: Task, context_requirements: Dict[str, Any]) -> int:
        """
        Estimate task complexity on a scale of 1-5 using enhanced analysis.
        
        Args:
            task: Task to analyze
            context_requirements: Context requirements analysis
            
        Returns:
            Complexity score (1-5)
        """
        complexity_score = 0
        
        # Base complexity from task structure
        complexity_score += self._calculate_structural_complexity(task)
        
        # Context-based complexity
        complexity_score += self._calculate_context_complexity(context_requirements)
        
        # Task type complexity
        complexity_score += self._calculate_type_complexity(task)
        
        # Quality requirements complexity
        complexity_score += self._calculate_quality_complexity(context_requirements)
        
        # Dependency complexity
        complexity_score += self._calculate_dependency_complexity(task)
        
        # Normalize to 1-5 scale
        normalized_complexity = max(1, min(5, int(complexity_score / 2) + 1))
        
        return normalized_complexity
    
    def _calculate_structural_complexity(self, task: Task) -> float:
        """Calculate complexity based on task structure."""
        score = 0
        
        # Input complexity
        num_inputs = len(task.inputs)
        if num_inputs > 4:
            score += 3
        elif num_inputs > 2:
            score += 2
        elif num_inputs > 1:
            score += 1
        
        # Success criteria complexity
        num_criteria = len(task.success_criteria)
        if num_criteria > 5:
            score += 2
        elif num_criteria > 3:
            score += 1.5
        elif num_criteria > 2:
            score += 1
        
        # Description complexity (length and detail)
        description_words = len(task.description.split())
        if description_words > 50:
            score += 2
        elif description_words > 20:
            score += 1
        
        return score
    
    def _calculate_context_complexity(self, context_requirements: Dict[str, Any]) -> float:
        """Calculate complexity based on context requirements."""
        score = 0
        
        # Context size impact
        context_size = context_requirements.get("estimated_context_size", "small")
        if context_size == "large":
            score += 3
        elif context_size == "medium":
            score += 1.5
        
        # Number of context sources
        num_sources = len(context_requirements.get("context_sources", []))
        if num_sources > 4:
            score += 2
        elif num_sources > 2:
            score += 1
        
        # Context priority impact
        priority = context_requirements.get("context_priority", "normal")
        if priority == "critical":
            score += 2
        elif priority == "high":
            score += 1
        
        # Retrieval strategy impact
        strategy = context_requirements.get("retrieval_strategy", "standard")
        if strategy == "comprehensive":
            score += 2
        elif strategy == "targeted":
            score += 1
        
        # Quality requirements impact
        quality_reqs = context_requirements.get("quality_requirements", [])
        score += len(quality_reqs) * 0.5
        
        return score
    
    def _calculate_type_complexity(self, task: Task) -> float:
        """Calculate complexity based on task type."""
        score = 0
        task_text = f"{task.name} {task.description}".lower()
        
        # High complexity operations
        high_complexity_ops = [
            "analyze", "analysis", "synthesize", "synthesis", 
            "compare", "comparison", "evaluate", "evaluation"
        ]
        
        # Medium complexity operations
        medium_complexity_ops = [
            "generate", "create", "build", "construct",
            "summarize", "extract", "transform"
        ]
        
        # Low complexity operations
        low_complexity_ops = [
            "retrieve", "find", "search", "read", "load"
        ]
        
        if any(op in task_text for op in high_complexity_ops):
            score += 3
        elif any(op in task_text for op in medium_complexity_ops):
            score += 2
        elif any(op in task_text for op in low_complexity_ops):
            score += 1
        else:
            score += 1.5  # Default for unknown operations
        
        # Multi-step operations
        multi_step_indicators = ["then", "after", "following", "subsequently", "and"]
        if any(indicator in task_text for indicator in multi_step_indicators):
            score += 1
        
        return score
    
    def _calculate_quality_complexity(self, context_requirements: Dict[str, Any]) -> float:
        """Calculate complexity based on quality requirements."""
        score = 0
        quality_reqs = context_requirements.get("quality_requirements", [])
        
        # High-impact quality requirements
        high_impact_reqs = ["high_accuracy", "verification_support", "comprehensive_coverage"]
        medium_impact_reqs = ["source_tracking", "completeness", "balanced_representation"]
        
        for req in quality_reqs:
            if req in high_impact_reqs:
                score += 1.5
            elif req in medium_impact_reqs:
                score += 1
            else:
                score += 0.5
        
        return score
    
    def _calculate_dependency_complexity(self, task: Task) -> float:
        """Calculate complexity based on task dependencies."""
        score = 0
        num_deps = len(task.dependencies)
        
        if num_deps > 3:
            score += 3
        elif num_deps > 1:
            score += 2
        elif num_deps > 0:
            score += 1
        
        return score
    
    def get_complexity_breakdown(self, task: Task, context_requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Get detailed breakdown of complexity calculation for debugging/monitoring."""
        breakdown = {
            "structural": self._calculate_structural_complexity(task),
            "context": self._calculate_context_complexity(context_requirements),
            "type": self._calculate_type_complexity(task),
            "quality": self._calculate_quality_complexity(context_requirements),
            "dependency": self._calculate_dependency_complexity(task)
        }
        
        breakdown["total_raw"] = sum(breakdown.values())
        breakdown["normalized"] = self.estimate_task_complexity(task, context_requirements)
        
        return breakdown


class TaskIdentifier(ProcessingNode):
    """
    Task Identifier node that maps tasks to tools and identifies context requirements.
    
    Responsibilities:
    - Map tasks to appropriate tools based on task types and requirements
    - Identify context requirements for each task
    - Estimate task complexity for resource allocation
    - Prepare tasks for agent assignment
    """
    
    def __init__(self, config: SystemConfig):
        """
        Initialize the Task Identifier.
        
        Args:
            config: System configuration
        """
        super().__init__("TaskIdentifier", {})
        self.system_config = config
        self.tool_mapper = ToolMapper(config.agent_generator.available_tools)
        self.context_analyzer = ContextAnalyzer()
        self.complexity_estimator = ComplexityEstimator()
        
    def execute(self, state: ExecutionState) -> ExecutionState:
        """
        Execute task identification logic.
        
        Args:
            state: Current execution state
            
        Returns:
            Updated execution state with task mappings
        """
        if not state.tasks:
            self.logger.warning("No tasks found in state")
            state.next_node = "AgentGenerator"
            return state
            
        task_mappings = []
        
        # Process each task
        for task in state.tasks:
            # Analyze context requirements first
            context_requirements = self.context_analyzer.analyze_context_requirements(task, state)
            
            # Estimate complexity
            complexity = self.complexity_estimator.estimate_task_complexity(task, context_requirements)
            
            # Map task to tools using complexity information
            recommended_tools = self.tool_mapper.map_task_to_tools(task, complexity)
            
            # Create task mapping
            task_mapping = {
                "task_id": task.id,
                "task_name": task.name,
                "recommended_tools": recommended_tools,
                "context_requirements": context_requirements,
                "complexity": complexity,
                "estimated_resources": self._estimate_resources(complexity, context_requirements)
            }
            
            task_mappings.append(task_mapping)
            
            self.logger.info(
                f"Task '{task.name}' mapped to tools: {recommended_tools}, "
                f"complexity: {complexity}"
            )
            
        # Store mappings in state
        state.context["task_mappings"] = task_mappings
        
        # Log identification results
        state.log_execution_step(
            node_name=self.name,
            action="task_identification_complete",
            details={
                "num_tasks_processed": len(task_mappings),
                "total_complexity": sum(mapping["complexity"] for mapping in task_mappings),
                "unique_tools_needed": list(set(
                    tool for mapping in task_mappings 
                    for tool in mapping["recommended_tools"]
                ))
            }
        )
        
        # Set next node to Agent Generator
        state.next_node = "AgentGenerator"
        
        return state
    
    def _estimate_resources(self, complexity: int, context_requirements: Dict[str, Any]) -> Dict[str, Any]:
        """
        Estimate resource requirements for a task using enhanced algorithms.
        
        Args:
            complexity: Task complexity score
            context_requirements: Context requirements analysis
            
        Returns:
            Comprehensive resource estimation dictionary
        """
        # Base resource requirements
        base_tokens = 1000
        base_steps = 2
        base_timeout = 60
        
        # Complexity-based scaling
        complexity_multiplier = {
            1: 1.0,
            2: 1.5,
            3: 2.0,
            4: 3.0,
            5: 4.0
        }.get(complexity, 2.0)
        
        estimated_tokens = int(base_tokens * complexity_multiplier)
        estimated_steps = base_steps + (complexity - 1)
        estimated_timeout = int(base_timeout * complexity_multiplier)
        
        # Context-based adjustments
        context_adjustments = self._calculate_context_adjustments(context_requirements)
        
        estimated_tokens = int(estimated_tokens * context_adjustments["token_multiplier"])
        estimated_steps += context_adjustments["additional_steps"]
        estimated_timeout += context_adjustments["additional_timeout"]
        
        # Quality requirements adjustments
        quality_adjustments = self._calculate_quality_adjustments(context_requirements)
        
        estimated_tokens = int(estimated_tokens * quality_adjustments["token_multiplier"])
        estimated_steps += quality_adjustments["additional_steps"]
        
        # Apply system limits
        max_tokens = self.system_config.agent_generator.default_agent_max_tokens
        max_steps = self.system_config.agent_generator.default_agent_max_steps
        max_timeout = self.system_config.agent_generator.agent_timeout_seconds
        
        # Calculate confidence and risk factors
        confidence_score = self._calculate_resource_confidence(complexity, context_requirements)
        risk_factors = self._identify_resource_risks(complexity, context_requirements)
        
        return {
            "estimated_tokens": min(estimated_tokens, max_tokens),
            "estimated_steps": min(estimated_steps, max_steps),
            "timeout_seconds": min(estimated_timeout, max_timeout),
            "confidence_score": confidence_score,
            "risk_factors": risk_factors,
            "resource_breakdown": {
                "base_tokens": base_tokens,
                "complexity_multiplier": complexity_multiplier,
                "context_adjustments": context_adjustments,
                "quality_adjustments": quality_adjustments,
                "final_tokens": min(estimated_tokens, max_tokens)
            },
            "scaling_factors": {
                "complexity": complexity,
                "context_size": context_requirements.get("estimated_context_size", "small"),
                "context_priority": context_requirements.get("context_priority", "normal"),
                "retrieval_strategy": context_requirements.get("retrieval_strategy", "standard")
            }
        }
    
    def _calculate_context_adjustments(self, context_requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate resource adjustments based on context requirements."""
        adjustments = {
            "token_multiplier": 1.0,
            "additional_steps": 0,
            "additional_timeout": 0
        }
        
        # Context size adjustments
        context_size = context_requirements.get("estimated_context_size", "small")
        if context_size == "large":
            adjustments["token_multiplier"] *= 2.5
            adjustments["additional_steps"] += 2
            adjustments["additional_timeout"] += 60
        elif context_size == "medium":
            adjustments["token_multiplier"] *= 1.8
            adjustments["additional_steps"] += 1
            adjustments["additional_timeout"] += 30
        
        # Context priority adjustments
        priority = context_requirements.get("context_priority", "normal")
        if priority == "critical":
            adjustments["token_multiplier"] *= 1.5
            adjustments["additional_steps"] += 1
        elif priority == "high":
            adjustments["token_multiplier"] *= 1.3
        
        # Retrieval strategy adjustments
        strategy = context_requirements.get("retrieval_strategy", "standard")
        if strategy == "comprehensive":
            adjustments["token_multiplier"] *= 1.4
            adjustments["additional_timeout"] += 30
        elif strategy == "targeted":
            adjustments["token_multiplier"] *= 1.2
        
        # Multiple context sources
        num_sources = len(context_requirements.get("context_sources", []))
        if num_sources > 3:
            adjustments["token_multiplier"] *= 1.3
            adjustments["additional_steps"] += 1
        
        return adjustments
    
    def _calculate_quality_adjustments(self, context_requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate resource adjustments based on quality requirements."""
        adjustments = {
            "token_multiplier": 1.0,
            "additional_steps": 0
        }
        
        quality_reqs = context_requirements.get("quality_requirements", [])
        
        # High-impact quality requirements
        high_impact_count = sum(1 for req in quality_reqs if req in [
            "high_accuracy", "verification_support", "comprehensive_coverage"
        ])
        
        if high_impact_count > 0:
            adjustments["token_multiplier"] *= (1 + high_impact_count * 0.3)
            adjustments["additional_steps"] += high_impact_count
        
        # Source tracking and citation requirements
        if "source_tracking" in quality_reqs:
            adjustments["token_multiplier"] *= 1.2
        
        return adjustments
    
    def _calculate_resource_confidence(self, complexity: int, context_requirements: Dict[str, Any]) -> float:
        """Calculate confidence score for resource estimates (0.0 to 1.0)."""
        confidence = 0.8  # Base confidence
        
        # Reduce confidence for high complexity
        if complexity >= 4:
            confidence -= 0.2
        elif complexity >= 3:
            confidence -= 0.1
        
        # Reduce confidence for large context requirements
        if context_requirements.get("estimated_context_size") == "large":
            confidence -= 0.15
        
        # Reduce confidence for comprehensive retrieval
        if context_requirements.get("retrieval_strategy") == "comprehensive":
            confidence -= 0.1
        
        # Reduce confidence for many quality requirements
        quality_count = len(context_requirements.get("quality_requirements", []))
        if quality_count > 3:
            confidence -= 0.1
        
        return max(0.3, confidence)  # Minimum 30% confidence
    
    def _identify_resource_risks(self, complexity: int, context_requirements: Dict[str, Any]) -> List[str]:
        """Identify potential resource risks for the task."""
        risks = []
        
        # Complexity risks
        if complexity >= 4:
            risks.append("high_complexity_may_exceed_token_limits")
        
        # Context risks
        if context_requirements.get("estimated_context_size") == "large":
            risks.append("large_context_may_cause_memory_issues")
        
        if len(context_requirements.get("context_sources", [])) > 3:
            risks.append("multiple_context_sources_may_slow_retrieval")
        
        # Quality risks
        quality_reqs = context_requirements.get("quality_requirements", [])
        if "verification_support" in quality_reqs:
            risks.append("verification_requirements_may_increase_processing_time")
        
        if "comprehensive_coverage" in quality_reqs:
            risks.append("comprehensive_analysis_may_exceed_step_limits")
        
        # Strategy risks
        if context_requirements.get("retrieval_strategy") == "comprehensive":
            risks.append("comprehensive_retrieval_may_be_slow")
        
        return risks
    
    def validate_inputs(self, state: ExecutionState) -> bool:
        """
        Validate that required inputs are present.
        
        Args:
            state: Current execution state
            
        Returns:
            True if inputs are valid, False otherwise
        """
        return len(state.tasks) > 0
    
    def get_next_node(self, state: ExecutionState) -> Optional[str]:
        """
        Determine the next node based on identification results.
        
        Args:
            state: Current execution state
            
        Returns:
            Name of next node
        """
        if "task_mappings" in state.context and state.context["task_mappings"]:
            return "AgentGenerator"
        else:
            # No valid task mappings, complete execution
            state.complete_execution("No valid task mappings generated")
            return None