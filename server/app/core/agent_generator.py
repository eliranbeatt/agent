"""Agent Generator node for creating specialized sub-agents."""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path

from ..config.models import SystemConfig
from .base_nodes import ProcessingNode
from .state import ExecutionState, Task, AgentSpec, AgentLimits


logger = logging.getLogger(__name__)


class PromptTemplateManager:
    """Manages system prompt templates for different agent types."""
    
    def __init__(self, template_path: str):
        """
        Initialize prompt template manager.
        
        Args:
            template_path: Path to agent prompt templates
        """
        self.template_path = Path(template_path)
        self.templates = self._load_templates()
        self.agent_limits = self._load_agent_limits()
        
    def _load_templates(self) -> Dict[str, str]:
        """Load prompt templates from configuration."""
        try:
            import yaml
            if self.template_path.exists():
                with open(self.template_path, 'r') as f:
                    config = yaml.safe_load(f)
                    return config.get('templates', {})
        except Exception as e:
            logger.warning(f"Could not load templates from {self.template_path}: {e}")
            
        # Fallback to default templates
        return {
            "file_processing": """You are a specialized file processing agent.

Your role: Process and analyze files to extract meaningful information.
Goal: {goal}
Available tools: {tools}
Hard limits: {limits}
Required outputs: {outputs}

Instructions:
1. Use appropriate file processing tools for the given file types
2. Extract content systematically and thoroughly
3. Handle errors gracefully and report issues
4. Provide structured output with metadata

Stop conditions: {stop_conditions}""",

            "question_answering": """You are a specialized question answering agent.

Your role: Answer questions accurately using available context and sources.
Goal: {goal}
Available tools: {tools}
Hard limits: {limits}
Required outputs: {outputs}

Instructions:
1. Retrieve relevant context using search tools
2. Analyze the question to understand what's being asked
3. Generate accurate answers based on retrieved information
4. Include proper citations and source references
5. Verify answer quality before responding

Stop conditions: {stop_conditions}""",

            "content_generation": """You are a specialized content generation agent.

Your role: Create high-quality content based on user requirements.
Goal: {goal}
Available tools: {tools}
Hard limits: {limits}
Required outputs: {outputs}

Instructions:
1. Understand the content requirements clearly
2. Use available tools to gather necessary information
3. Generate content that meets specified criteria
4. Ensure quality, accuracy, and relevance
5. Format output appropriately

Stop conditions: {stop_conditions}""",

            "data_analysis": """You are a specialized data analysis agent.

Your role: Analyze data and provide insights and conclusions.
Goal: {goal}
Available tools: {tools}
Hard limits: {limits}
Required outputs: {outputs}

Instructions:
1. Load and examine the provided data
2. Apply appropriate analysis techniques
3. Identify patterns, trends, and insights
4. Present findings clearly with supporting evidence
5. Provide actionable recommendations where appropriate

Stop conditions: {stop_conditions}""",

            "general": """You are a general-purpose agent.

Your role: Handle various tasks using available tools and capabilities.
Goal: {goal}
Available tools: {tools}
Hard limits: {limits}
Required outputs: {outputs}

Instructions:
1. Understand the task requirements
2. Use appropriate tools for the task
3. Work systematically toward the goal
4. Provide clear and complete outputs
5. Handle errors and edge cases gracefully

Stop conditions: {stop_conditions}"""
        }
    
    def _load_agent_limits(self) -> Dict[str, Dict[str, Any]]:
        """Load agent limit configurations by complexity level."""
        try:
            import yaml
            if self.template_path.exists():
                with open(self.template_path, 'r') as f:
                    config = yaml.safe_load(f)
                    return config.get('agent_limits', {})
        except Exception as e:
            logger.warning(f"Could not load agent limits from {self.template_path}: {e}")
            
        # Fallback to default limits
        return {
            "simple": {"max_steps": 3, "max_tokens": 1000, "timeout_seconds": 60},
            "moderate": {"max_steps": 5, "max_tokens": 2000, "timeout_seconds": 120},
            "complex": {"max_steps": 8, "max_tokens": 4000, "timeout_seconds": 300},
            "expert": {"max_steps": 12, "max_tokens": 6000, "timeout_seconds": 600}
        }
    
    def get_template(self, agent_type: str) -> str:
        """
        Get prompt template for agent type.
        
        Args:
            agent_type: Type of agent (file_processing, question_answering, etc.)
            
        Returns:
            Prompt template string
        """
        # Map internal types to template keys
        template_mapping = {
            "file_processing": "document_analyzer",
            "question_answering": "qa_agent", 
            "content_generation": "research_agent",
            "data_analysis": "comparison_agent",
            "planning": "planning_agent",
            "evaluation": "evaluation_agent"
        }
        
        template_key = template_mapping.get(agent_type, "base_agent")
        
        # Try to get specific template, fallback to base_agent, then general
        if template_key in self.templates:
            return self.templates[template_key]
        elif "base_agent" in self.templates:
            return self.templates["base_agent"]
        else:
            return self.templates.get("general", "")
    
    def format_prompt(self, template: str, **kwargs) -> str:
        """
        Format prompt template with provided parameters.
        
        Args:
            template: Template string
            **kwargs: Template parameters
            
        Returns:
            Formatted prompt
        """
        try:
            return template.format(**kwargs)
        except KeyError as e:
            logger.warning(f"Missing template parameter: {e}")
            # Return template with placeholder for missing parameters
            return template
    
    def get_limits_for_complexity(self, complexity: str) -> Dict[str, Any]:
        """
        Get agent limits for a given complexity level.
        
        Args:
            complexity: Complexity level (simple, moderate, complex, expert)
            
        Returns:
            Dictionary with limit values
        """
        return self.agent_limits.get(complexity, self.agent_limits.get("moderate", {}))


class AgentTypeClassifier:
    """Classifies tasks to determine appropriate agent types and complexity."""
    
    def __init__(self):
        """Initialize agent type classifier."""
        self.type_keywords = {
            "file_processing": ["file", "document", "pdf", "upload", "ingest", "process", "extract", "ocr"],
            "question_answering": ["question", "answer", "qa", "ask", "explain", "what", "how", "why", "find"],
            "content_generation": ["generate", "create", "write", "compose", "build", "make", "draft", "produce"],
            "data_analysis": ["analyze", "analysis", "compare", "summarize", "extract", "insights", "pattern", "trend"],
            "planning": ["plan", "organize", "structure", "breakdown", "sequence", "schedule"],
            "evaluation": ["evaluate", "verify", "check", "validate", "assess", "review", "quality"]
        }
        
        self.complexity_indicators = {
            "simple": ["single", "one", "basic", "simple", "quick"],
            "moderate": ["multiple", "several", "detailed", "thorough"],
            "complex": ["comprehensive", "complex", "advanced", "multi-step", "integrated"],
            "expert": ["expert", "sophisticated", "deep", "extensive", "cross-domain"]
        }
    
    def classify_task(self, task: Task, task_mapping: Dict[str, Any]) -> tuple[str, str]:
        """
        Classify a task to determine the appropriate agent type and complexity.
        
        Args:
            task: Task to classify
            task_mapping: Task mapping information
            
        Returns:
            Tuple of (agent_type, complexity_level)
        """
        task_name_lower = task.name.lower()
        task_desc_lower = task.description.lower()
        combined_text = f"{task_name_lower} {task_desc_lower}"
        
        # Score each agent type
        type_scores = {}
        
        for agent_type, keywords in self.type_keywords.items():
            score = 0
            
            # Check task name (higher weight)
            for keyword in keywords:
                if keyword in task_name_lower:
                    score += 3
                if keyword in task_desc_lower:
                    score += 2
                    
            # Check recommended tools
            for tool in task_mapping.get("recommended_tools", []):
                tool_lower = tool.lower()
                if agent_type == "file_processing" and any(kw in tool_lower for kw in ["file", "pdf", "ocr", "document"]):
                    score += 2
                elif agent_type == "question_answering" and any(kw in tool_lower for kw in ["search", "retrieval", "query"]):
                    score += 2
                elif agent_type == "content_generation" and any(kw in tool_lower for kw in ["generator", "writer", "creator"]):
                    score += 2
                elif agent_type == "data_analysis" and any(kw in tool_lower for kw in ["analyzer", "processor", "extractor"]):
                    score += 2
                elif agent_type == "planning" and any(kw in tool_lower for kw in ["planner", "organizer"]):
                    score += 2
                elif agent_type == "evaluation" and any(kw in tool_lower for kw in ["evaluator", "validator", "checker"]):
                    score += 2
                    
            # Check task inputs and dependencies for complexity indicators
            num_inputs = len(task.inputs)
            num_dependencies = len(task.dependencies)
            
            if num_inputs > 3 or num_dependencies > 2:
                score += 1  # More complex tasks get slight boost
                    
            type_scores[agent_type] = score
            
        # Determine best agent type
        best_type = "general"
        if type_scores:
            best_type = max(type_scores, key=type_scores.get)
            if type_scores[best_type] == 0:
                best_type = "general"
        
        # Determine complexity level
        complexity = self._determine_complexity(task, task_mapping, combined_text)
        
        return best_type, complexity
    
    def _determine_complexity(self, task: Task, task_mapping: Dict[str, Any], combined_text: str) -> str:
        """
        Determine the complexity level of a task.
        
        Args:
            task: Task to analyze
            task_mapping: Task mapping information
            combined_text: Combined task name and description
            
        Returns:
            Complexity level string
        """
        complexity_score = 0
        
        # Check for complexity keywords
        for complexity, keywords in self.complexity_indicators.items():
            for keyword in keywords:
                if keyword in combined_text:
                    if complexity == "simple":
                        complexity_score -= 1
                    elif complexity == "moderate":
                        complexity_score += 0
                    elif complexity == "complex":
                        complexity_score += 2
                    elif complexity == "expert":
                        complexity_score += 3
        
        # Factor in task structure
        num_inputs = len(task.inputs)
        num_dependencies = len(task.dependencies)
        num_success_criteria = len(task.success_criteria)
        num_tools = len(task_mapping.get("recommended_tools", []))
        
        # Structural complexity indicators
        if num_inputs > 5:
            complexity_score += 2
        elif num_inputs > 2:
            complexity_score += 1
            
        if num_dependencies > 3:
            complexity_score += 2
        elif num_dependencies > 1:
            complexity_score += 1
            
        if num_success_criteria > 5:
            complexity_score += 1
            
        if num_tools > 4:
            complexity_score += 1
        
        # Estimated resources
        resources = task_mapping.get("estimated_resources", {})
        estimated_steps = resources.get("estimated_steps", 0)
        estimated_tokens = resources.get("estimated_tokens", 0)
        
        if estimated_steps > 8:
            complexity_score += 2
        elif estimated_steps > 5:
            complexity_score += 1
            
        if estimated_tokens > 4000:
            complexity_score += 2
        elif estimated_tokens > 2000:
            complexity_score += 1
        
        # Map score to complexity level
        if complexity_score <= -1:
            return "simple"
        elif complexity_score <= 2:
            return "moderate"
        elif complexity_score <= 5:
            return "complex"
        else:
            return "expert"


class AgentSpecBuilder:
    """Builds complete agent specifications."""
    
    def __init__(self, config: SystemConfig, prompt_manager: PromptTemplateManager):
        """
        Initialize agent spec builder.
        
        Args:
            config: System configuration
            prompt_manager: Prompt template manager
        """
        self.config = config
        self.prompt_manager = prompt_manager
        
    def build_agent_spec(self, task: Task, task_mapping: Dict[str, Any], agent_type: str, complexity: str = "moderate") -> AgentSpec:
        """
        Build a complete agent specification for a task.
        
        Args:
            task: Task to create agent for
            task_mapping: Task mapping information
            agent_type: Type of agent to create
            complexity: Complexity level of the task
            
        Returns:
            Complete agent specification
        """
        # Create agent limits based on complexity
        limits = self._create_agent_limits(task_mapping, complexity)
        
        # Select tools based on task requirements and agent capabilities
        tools = self._select_tools(task_mapping, limits, agent_type)
        
        # Generate system prompt with task-specific context
        system_prompt = self._generate_system_prompt(task, agent_type, tools, limits, complexity)
        
        # Create output contract
        output_contract = self._create_output_contract(task, agent_type)
        
        # Build agent spec
        agent_spec = AgentSpec(
            system_prompt=system_prompt,
            tools=tools,
            limits=limits,
            output_contract=output_contract,
            created_for_task=task.id
        )
        
        return agent_spec
    
    def _create_agent_limits(self, task_mapping: Dict[str, Any], complexity: str = "moderate") -> AgentLimits:
        """Create agent limits based on task requirements and complexity."""
        resources = task_mapping.get("estimated_resources", {})
        
        # Get complexity-based limits
        complexity_limits = self.prompt_manager.get_limits_for_complexity(complexity)
        
        # Use task-specific resources if available, otherwise use complexity defaults, then system defaults
        max_steps = resources.get("estimated_steps") or complexity_limits.get("max_steps") or self.config.agent_generator.default_agent_max_steps
        max_tokens = resources.get("estimated_tokens") or complexity_limits.get("max_tokens") or self.config.agent_generator.default_agent_max_tokens
        timeout_seconds = resources.get("timeout_seconds") or complexity_limits.get("timeout_seconds") or self.config.agent_generator.agent_timeout_seconds
        
        return AgentLimits(
            max_steps=max_steps,
            max_tokens=max_tokens,
            timeout_seconds=timeout_seconds,
            allowed_tools=task_mapping.get("recommended_tools", [])
        )
    
    def _select_tools(self, task_mapping: Dict[str, Any], limits: AgentLimits, agent_type: str) -> List[str]:
        """Select appropriate tools for the agent based on task requirements and agent capabilities."""
        recommended_tools = task_mapping.get("recommended_tools", [])
        available_tools = self.config.agent_generator.available_tools
        
        # Filter to available tools
        selected_tools = [tool for tool in recommended_tools if tool in available_tools]
        
        # Add agent-type specific essential tools
        essential_tools_by_type = {
            "file_processing": ["file_reader", "text_extractor", "ocr_processor"],
            "question_answering": ["search_engine", "retrieval_system", "memory_retrieval"],
            "content_generation": ["text_generator", "template_engine"],
            "data_analysis": ["data_processor", "analyzer", "chart_generator"],
            "planning": ["task_planner", "dependency_resolver"],
            "evaluation": ["quality_checker", "validator", "grader"],
            "general": ["text_generator", "memory_retrieval"]
        }
        
        type_tools = essential_tools_by_type.get(agent_type, essential_tools_by_type["general"])
        
        # Add essential tools for this agent type
        for tool in type_tools:
            if tool in available_tools and tool not in selected_tools:
                selected_tools.append(tool)
        
        # Always ensure basic tools are available
        basic_tools = ["text_generator", "memory_retrieval"]
        for tool in basic_tools:
            if tool in available_tools and tool not in selected_tools:
                selected_tools.append(tool)
                
        return selected_tools
    
    def _generate_system_prompt(self, task: Task, agent_type: str, tools: List[str], limits: AgentLimits, complexity: str) -> str:
        """Generate system prompt for the agent with task-specific context."""
        template = self.prompt_manager.get_template(agent_type)
        
        # Prepare comprehensive template parameters
        params = {
            "role": f"{agent_type.replace('_', ' ').title()} Agent",
            "goal": f"{task.name}: {task.description}",
            "task_description": task.description,
            "tools": ", ".join(tools) if tools else "No specific tools assigned",
            "limits": f"Max steps: {limits.max_steps}, Max tokens: {limits.max_tokens}, Timeout: {limits.timeout_seconds}s",
            "max_steps": limits.max_steps,
            "max_tokens": limits.max_tokens,
            "timeout_seconds": limits.timeout_seconds,
            "outputs": task.expected_output or "Structured result with status and findings",
            "required_outputs": task.expected_output or "Structured result with status and findings",
            "stop_conditions": "; ".join(task.success_criteria) if task.success_criteria else "Task completion with satisfactory results",
            "success_criteria": task.success_criteria,
            "complexity": complexity,
            "task_inputs": task.inputs,
            "dependencies": task.dependencies
        }
        
        # Add context-specific parameters based on agent type
        if agent_type == "question_answering":
            params["question"] = task.description
            params["context"] = "Retrieved from available sources and memory"
        elif agent_type == "file_processing":
            params["document_context"] = "Files and documents provided by the user"
            params["analysis_requirements"] = task.success_criteria
        elif agent_type == "data_analysis":
            params["comparison_task"] = task.description
            params["comparison_items"] = task.inputs
            params["comparison_dimensions"] = task.success_criteria
        elif agent_type == "content_generation":
            params["research_topic"] = task.name
            params["research_scope"] = task.description
            params["available_sources"] = "System knowledge and retrieved documents"
        elif agent_type == "planning":
            params["planning_request"] = task.description
            params["available_resources"] = tools
            params["constraints"] = f"Steps: {limits.max_steps}, Tokens: {limits.max_tokens}"
        elif agent_type == "evaluation":
            params["evaluation_task"] = task.description
            params["content"] = "Output from previous agents or user input"
            
        return self.prompt_manager.format_prompt(template, **params)
    
    def _create_output_contract(self, task: Task, agent_type: str) -> Dict[str, Any]:
        """Create output contract specification based on task and agent type."""
        base_contract = {
            "format": "structured",
            "required_fields": ["result", "status", "execution_summary"],
            "expected_output": task.expected_output or "Structured result with findings",
            "success_criteria": task.success_criteria
        }
        
        # Add agent-type specific output requirements
        if agent_type == "question_answering":
            base_contract["required_fields"].extend(["answer", "citations", "confidence_score"])
        elif agent_type == "file_processing":
            base_contract["required_fields"].extend(["extracted_content", "metadata", "processing_errors"])
        elif agent_type == "data_analysis":
            base_contract["required_fields"].extend(["analysis_results", "insights", "recommendations"])
        elif agent_type == "content_generation":
            base_contract["required_fields"].extend(["generated_content", "sources_used", "quality_metrics"])
        elif agent_type == "planning":
            base_contract["required_fields"].extend(["task_breakdown", "dependencies", "resource_estimates"])
        elif agent_type == "evaluation":
            base_contract["required_fields"].extend(["quality_score", "evaluation_details", "improvement_suggestions"])
        
        return base_contract


class AgentGenerator(ProcessingNode):
    """
    Agent Generator node that creates specialized sub-agents for tasks.
    
    Responsibilities:
    - Generate context-specific system prompts using templates
    - Select appropriate tools based on task requirements
    - Set execution limits based on task complexity
    - Create complete agent specifications for instantiation
    """
    
    def __init__(self, config: SystemConfig):
        """
        Initialize the Agent Generator.
        
        Args:
            config: System configuration
        """
        super().__init__("AgentGenerator", config.agent_generator.__dict__)
        self.system_config = config
        self.prompt_manager = PromptTemplateManager(config.agent_generator.prompt_template_path)
        self.type_classifier = AgentTypeClassifier()
        self.spec_builder = AgentSpecBuilder(config, self.prompt_manager)
        
    def execute(self, state: ExecutionState) -> ExecutionState:
        """
        Execute agent generation logic.
        
        Args:
            state: Current execution state
            
        Returns:
            Updated execution state with agent specifications
        """
        task_mappings = state.context.get("task_mappings", [])
        
        if not task_mappings:
            self.logger.warning("No task mappings found in state")
            state.complete_execution("No tasks to generate agents for")
            return state
            
        generated_agents = []
        
        # Generate agent for each task
        for task_mapping in task_mappings:
            task_id = task_mapping["task_id"]
            task = state.get_task_by_id(task_id)
            
            if not task:
                self.logger.error(f"Task {task_id} not found in state")
                continue
                
            # Check agent limits
            if len(state.active_agents) >= self.system_config.agent_generator.max_concurrent_agents:
                self.logger.warning("Maximum concurrent agents reached, queuing remaining tasks")
                break
                
            # Classify task to determine agent type and complexity
            agent_type, complexity = self.type_classifier.classify_task(task, task_mapping)
            
            # Build agent specification
            agent_spec = self.spec_builder.build_agent_spec(task, task_mapping, agent_type, complexity)
            
            # Add agent spec to state
            state.add_agent_spec(agent_spec)
            
            # Assign agent to task
            task.assigned_agent = agent_spec.id
            
            generated_agents.append({
                "agent_id": agent_spec.id,
                "task_id": task_id,
                "agent_type": agent_type,
                "complexity": complexity,
                "tools": agent_spec.tools,
                "limits": {
                    "max_steps": agent_spec.limits.max_steps,
                    "max_tokens": agent_spec.limits.max_tokens,
                    "timeout_seconds": agent_spec.limits.timeout_seconds
                },
                "output_contract": agent_spec.output_contract
            })
            
            self.logger.info(
                f"Generated {agent_type} agent {agent_spec.id} (complexity: {complexity}) for task '{task.name}'"
            )
            
        # Store generation results in state
        state.context["generated_agents"] = generated_agents
        
        # Log generation results
        state.log_execution_step(
            node_name=self.name,
            action="agent_generation_complete",
            details={
                "num_agents_generated": len(generated_agents),
                "agent_types": [agent["agent_type"] for agent in generated_agents],
                "total_tasks": len(task_mappings)
            }
        )
        
        # Set next node based on execution path
        if generated_agents:
            state.next_node = "AgentExecutor"
        else:
            state.complete_execution("No agents generated")
            
        return state
    
    def validate_inputs(self, state: ExecutionState) -> bool:
        """
        Validate that required inputs are present.
        
        Args:
            state: Current execution state
            
        Returns:
            True if inputs are valid, False otherwise
        """
        return "task_mappings" in state.context and len(state.context["task_mappings"]) > 0
    
    def get_next_node(self, state: ExecutionState) -> Optional[str]:
        """
        Determine the next node based on generation results.
        
        Args:
            state: Current execution state
            
        Returns:
            Name of next node
        """
        generated_agents = state.context.get("generated_agents", [])
        
        if generated_agents:
            return "AgentExecutor"
        else:
            # No agents generated, complete execution
            state.complete_execution("No agents were generated")
            return None
    
    def is_critical_error(self, error: Exception) -> bool:
        """
        Determine if an error should stop execution.
        
        Args:
            error: The exception that occurred
            
        Returns:
            True if execution should stop, False to continue
        """
        # Template errors are not critical - can use default templates
        if "template" in str(error).lower():
            return False
            
        # Configuration errors are critical
        if isinstance(error, (KeyError, ValueError, AttributeError)):
            return True
            
        # Default to non-critical for agent generation
        return False