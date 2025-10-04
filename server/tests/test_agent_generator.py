"""Comprehensive unit tests for Agent Generator functionality."""

import pytest
import tempfile
import yaml
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from app.core.agent_generator import (
    AgentGenerator, PromptTemplateManager, AgentTypeClassifier, 
    AgentSpecBuilder
)
from app.core.state import ExecutionState, Task, AgentSpec, AgentLimits, TaskStatus
from app.config.models import SystemConfig, AgentGeneratorConfig


@pytest.fixture
def system_config():
    """Create a test system configuration."""
    return SystemConfig(
        agent_generator=AgentGeneratorConfig(
            max_concurrent_agents=5,
            default_agent_max_steps=4,
            default_agent_max_tokens=2000,
            agent_timeout_seconds=120,
            prompt_template_path="config/templates/agent_prompts.yaml",
            available_tools=["file_reader", "web_search", "calculator", "text_generator", "memory_retrieval"]
        )
    )


@pytest.fixture
def sample_template_config():
    """Create sample template configuration for testing."""
    return {
        "templates": {
            "base_agent": """You are a specialized agent.
Role: {role}
Goal: {goal}
Tools: {tools}
Limits: {limits}
Outputs: {required_outputs}""",
            "document_analyzer": """You are a Document Analysis Agent.
Task: {task_description}
Context: {document_context}
Requirements: {analysis_requirements}""",
            "qa_agent": """You are a Question Answering Agent.
Question: {question}
Context: {context}
Requirements: Answer with citations and confidence."""
        },
        "agent_limits": {
            "simple": {"max_steps": 3, "max_tokens": 1000, "timeout_seconds": 60},
            "moderate": {"max_steps": 5, "max_tokens": 2000, "timeout_seconds": 120},
            "complex": {"max_steps": 8, "max_tokens": 4000, "timeout_seconds": 300},
            "expert": {"max_steps": 12, "max_tokens": 6000, "timeout_seconds": 600}
        }
    }


@pytest.fixture
def temp_template_file(sample_template_config):
    """Create a temporary template file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(sample_template_config, f)
        return f.name


@pytest.fixture
def sample_tasks():
    """Create various sample tasks for testing."""
    return {
        "file_processing": Task(
            id="file_task",
            name="Process PDF Document",
            description="Extract text and analyze content from uploaded PDF file",
            inputs=["document.pdf"],
            expected_output="Structured analysis with key findings",
            success_criteria=["Text extracted successfully", "Key topics identified"],
            dependencies=[]
        ),
        "question_answering": Task(
            id="qa_task", 
            name="Answer User Question",
            description="What are the main benefits of renewable energy?",
            inputs=["user_question"],
            expected_output="Comprehensive answer with sources",
            success_criteria=["Question answered accurately", "Sources cited"],
            dependencies=[]
        ),
        "content_generation": Task(
            id="content_task",
            name="Generate Report",
            description="Create a comprehensive report on market trends",
            inputs=["market_data", "research_notes"],
            expected_output="Professional report document",
            success_criteria=["Report is well-structured", "Data is accurate"],
            dependencies=["data_analysis_task"]
        ),
        "data_analysis": Task(
            id="analysis_task",
            name="Analyze Sales Data",
            description="Compare quarterly sales performance and identify trends",
            inputs=["q1_sales.csv", "q2_sales.csv", "q3_sales.csv"],
            expected_output="Analysis report with insights",
            success_criteria=["Trends identified", "Recommendations provided"],
            dependencies=[]
        ),
        "complex_multi_step": Task(
            id="complex_task",
            name="Comprehensive Market Analysis",
            description="Conduct extensive multi-dimensional analysis of market conditions across multiple sectors",
            inputs=["sector_data_1", "sector_data_2", "sector_data_3", "economic_indicators", "competitor_analysis"],
            expected_output="Executive summary with detailed findings and strategic recommendations",
            success_criteria=[
                "All sectors analyzed thoroughly",
                "Cross-sector correlations identified", 
                "Strategic recommendations provided",
                "Risk assessment completed",
                "Implementation timeline suggested"
            ],
            dependencies=["data_collection", "preliminary_analysis"]
        )
    }


@pytest.fixture
def sample_task_mappings():
    """Create sample task mappings for testing."""
    return {
        "file_processing": {
            "task_id": "file_task",
            "recommended_tools": ["file_reader", "text_extractor", "ocr_processor"],
            "estimated_resources": {
                "estimated_steps": 4,
                "estimated_tokens": 1500,
                "timeout_seconds": 90
            }
        },
        "question_answering": {
            "task_id": "qa_task",
            "recommended_tools": ["web_search", "memory_retrieval", "search_engine"],
            "estimated_resources": {
                "estimated_steps": 3,
                "estimated_tokens": 1200
            }
        },
        "content_generation": {
            "task_id": "content_task", 
            "recommended_tools": ["text_generator", "template_engine", "data_processor"],
            "estimated_resources": {
                "estimated_steps": 6,
                "estimated_tokens": 3000,
                "timeout_seconds": 180
            }
        },
        "complex_multi_step": {
            "task_id": "complex_task",
            "recommended_tools": ["data_processor", "analyzer", "chart_generator", "report_generator"],
            "estimated_resources": {
                "estimated_steps": 12,
                "estimated_tokens": 8000,
                "timeout_seconds": 600
            }
        }
    }


class TestPromptTemplateManager:
    """Test the PromptTemplateManager class."""
    
    def test_init_with_valid_template_file(self, temp_template_file):
        """Test initialization with a valid template file."""
        manager = PromptTemplateManager(temp_template_file)
        
        assert "base_agent" in manager.templates
        assert "document_analyzer" in manager.templates
        assert "qa_agent" in manager.templates
        assert "simple" in manager.agent_limits
        assert "moderate" in manager.agent_limits
        
        # Cleanup
        Path(temp_template_file).unlink()
    
    def test_init_with_missing_template_file(self):
        """Test initialization with missing template file falls back to defaults."""
        manager = PromptTemplateManager("nonexistent_file.yaml")
        
        # Should have fallback templates
        assert "file_processing" in manager.templates
        assert "question_answering" in manager.templates
        assert "general" in manager.templates
        assert "simple" in manager.agent_limits
    
    def test_get_template_existing(self, temp_template_file):
        """Test getting an existing template."""
        manager = PromptTemplateManager(temp_template_file)
        
        template = manager.get_template("file_processing")
        assert "document_analyzer" in template or "base_agent" in template
        
        Path(temp_template_file).unlink()
    
    def test_get_template_fallback(self, temp_template_file):
        """Test template fallback behavior."""
        manager = PromptTemplateManager(temp_template_file)
        
        # Non-existent type should fall back to base_agent
        template = manager.get_template("nonexistent_type")
        assert "specialized agent" in template.lower()
        
        Path(temp_template_file).unlink()
    
    def test_format_prompt_success(self, temp_template_file):
        """Test successful prompt formatting."""
        manager = PromptTemplateManager(temp_template_file)
        
        template = "Hello {name}, your role is {role}"
        formatted = manager.format_prompt(template, name="Agent", role="Assistant")
        
        assert formatted == "Hello Agent, your role is Assistant"
        
        Path(temp_template_file).unlink()
    
    def test_format_prompt_missing_params(self, temp_template_file):
        """Test prompt formatting with missing parameters."""
        manager = PromptTemplateManager(temp_template_file)
        
        template = "Hello {name}, your role is {role}"
        formatted = manager.format_prompt(template, name="Agent")  # Missing 'role'
        
        # Should return original template when parameters are missing
        assert "{role}" in formatted
        
        Path(temp_template_file).unlink()
    
    def test_get_limits_for_complexity(self, temp_template_file):
        """Test getting limits for different complexity levels."""
        manager = PromptTemplateManager(temp_template_file)
        
        simple_limits = manager.get_limits_for_complexity("simple")
        assert simple_limits["max_steps"] == 3
        assert simple_limits["max_tokens"] == 1000
        
        expert_limits = manager.get_limits_for_complexity("expert")
        assert expert_limits["max_steps"] == 12
        assert expert_limits["max_tokens"] == 6000
        
        # Non-existent complexity should return moderate
        default_limits = manager.get_limits_for_complexity("nonexistent")
        assert default_limits["max_steps"] == 5  # moderate default
        
        Path(temp_template_file).unlink()


class TestAgentTypeClassifier:
    """Test the AgentTypeClassifier class."""
    
    def test_classify_file_processing_task(self, sample_tasks, sample_task_mappings):
        """Test classification of file processing tasks."""
        classifier = AgentTypeClassifier()
        task = sample_tasks["file_processing"]
        mapping = sample_task_mappings["file_processing"]
        
        agent_type, complexity = classifier.classify_task(task, mapping)
        
        assert agent_type == "file_processing"
        assert complexity in ["simple", "moderate", "complex", "expert"]
    
    def test_classify_question_answering_task(self, sample_tasks, sample_task_mappings):
        """Test classification of question answering tasks."""
        classifier = AgentTypeClassifier()
        task = sample_tasks["question_answering"]
        mapping = sample_task_mappings["question_answering"]
        
        agent_type, complexity = classifier.classify_task(task, mapping)
        
        assert agent_type == "question_answering"
        assert complexity in ["simple", "moderate", "complex", "expert"]
    
    def test_classify_content_generation_task(self, sample_tasks, sample_task_mappings):
        """Test classification of content generation tasks."""
        classifier = AgentTypeClassifier()
        task = sample_tasks["content_generation"]
        mapping = sample_task_mappings["content_generation"]
        
        agent_type, complexity = classifier.classify_task(task, mapping)
        
        assert agent_type == "content_generation"
        assert complexity in ["simple", "moderate", "complex", "expert"]
    
    def test_classify_data_analysis_task(self, sample_tasks, sample_task_mappings):
        """Test classification of data analysis tasks."""
        classifier = AgentTypeClassifier()
        task = sample_tasks["data_analysis"]
        mapping = sample_task_mappings["data_analysis"]
        
        agent_type, complexity = classifier.classify_task(task, mapping)
        
        assert agent_type == "data_analysis"
        assert complexity in ["simple", "moderate", "complex", "expert"]
    
    def test_classify_complex_task_complexity(self, sample_tasks, sample_task_mappings):
        """Test that complex tasks are classified with appropriate complexity."""
        classifier = AgentTypeClassifier()
        task = sample_tasks["complex_multi_step"]
        mapping = sample_task_mappings["complex_multi_step"]
        
        agent_type, complexity = classifier.classify_task(task, mapping)
        
        # Complex task should get higher complexity rating
        assert complexity in ["complex", "expert"]
        assert len(task.inputs) > 3  # Verify it's actually complex
        assert len(task.success_criteria) > 3
    
    def test_classify_unknown_task_type(self, sample_tasks):
        """Test classification of tasks that don't match known patterns."""
        classifier = AgentTypeClassifier()
        
        # Create a task with no clear type indicators
        unknown_task = Task(
            id="unknown_task",
            name="Do Something",
            description="Perform some generic operation",
            inputs=["input"],
            expected_output="output",
            success_criteria=["completed"]
        )
        
        mapping = {"task_id": "unknown_task", "recommended_tools": []}
        
        agent_type, complexity = classifier.classify_task(unknown_task, mapping)
        
        assert agent_type == "general"
        assert complexity in ["simple", "moderate", "complex", "expert"]
    
    def test_complexity_determination_factors(self, sample_tasks):
        """Test that complexity is determined by multiple factors."""
        classifier = AgentTypeClassifier()
        
        # Simple task
        simple_task = Task(
            id="simple",
            name="Simple Task",
            description="A basic simple operation",
            inputs=["one_input"],
            expected_output="simple output",
            success_criteria=["done"]
        )
        
        simple_mapping = {
            "task_id": "simple",
            "recommended_tools": ["basic_tool"],
            "estimated_resources": {"estimated_steps": 2, "estimated_tokens": 500}
        }
        
        _, simple_complexity = classifier.classify_task(simple_task, simple_mapping)
        
        # Complex task
        complex_task = sample_tasks["complex_multi_step"]
        complex_mapping = sample_task_mappings["complex_multi_step"]
        
        _, complex_complexity = classifier.classify_task(complex_task, complex_mapping)
        
        # Complex task should have higher complexity than simple task
        complexity_order = ["simple", "moderate", "complex", "expert"]
        simple_idx = complexity_order.index(simple_complexity)
        complex_idx = complexity_order.index(complex_complexity)
        
        assert complex_idx >= simple_idx


class TestAgentSpecBuilder:
    """Test the AgentSpecBuilder class."""
    
    def test_build_agent_spec_basic(self, system_config, temp_template_file, sample_tasks, sample_task_mappings):
        """Test basic agent spec building."""
        prompt_manager = PromptTemplateManager(temp_template_file)
        builder = AgentSpecBuilder(system_config, prompt_manager)
        
        task = sample_tasks["file_processing"]
        mapping = sample_task_mappings["file_processing"]
        
        agent_spec = builder.build_agent_spec(task, mapping, "file_processing", "moderate")
        
        assert isinstance(agent_spec, AgentSpec)
        assert agent_spec.created_for_task == task.id
        assert len(agent_spec.tools) > 0
        assert agent_spec.limits.max_steps > 0
        assert agent_spec.limits.max_tokens > 0
        assert agent_spec.system_prompt != ""
        assert isinstance(agent_spec.output_contract, dict)
        
        Path(temp_template_file).unlink()
    
    def test_agent_limits_creation_with_complexity(self, system_config, temp_template_file):
        """Test agent limits creation based on complexity."""
        prompt_manager = PromptTemplateManager(temp_template_file)
        builder = AgentSpecBuilder(system_config, prompt_manager)
        
        # Test different complexity levels
        simple_limits = builder._create_agent_limits({}, "simple")
        expert_limits = builder._create_agent_limits({}, "expert")
        
        assert simple_limits.max_steps < expert_limits.max_steps
        assert simple_limits.max_tokens < expert_limits.max_tokens
        assert simple_limits.timeout_seconds < expert_limits.timeout_seconds
        
        Path(temp_template_file).unlink()
    
    def test_agent_limits_with_task_resources(self, system_config, temp_template_file):
        """Test that task-specific resources override complexity defaults."""
        prompt_manager = PromptTemplateManager(temp_template_file)
        builder = AgentSpecBuilder(system_config, prompt_manager)
        
        task_mapping = {
            "estimated_resources": {
                "estimated_steps": 10,
                "estimated_tokens": 5000,
                "timeout_seconds": 300
            }
        }
        
        limits = builder._create_agent_limits(task_mapping, "simple")
        
        # Task-specific resources should override complexity defaults
        assert limits.max_steps == 10
        assert limits.max_tokens == 5000
        assert limits.timeout_seconds == 300
        
        Path(temp_template_file).unlink()
    
    def test_tool_selection_by_agent_type(self, system_config, temp_template_file):
        """Test that tools are selected appropriately for different agent types."""
        prompt_manager = PromptTemplateManager(temp_template_file)
        builder = AgentSpecBuilder(system_config, prompt_manager)
        
        # Mock available tools
        system_config.agent_generator.available_tools = [
            "file_reader", "text_extractor", "search_engine", "text_generator", 
            "memory_retrieval", "data_processor", "analyzer"
        ]
        
        mapping = {"recommended_tools": ["file_reader", "text_extractor"]}
        limits = AgentLimits()
        
        # File processing agent should get file-related tools
        file_tools = builder._select_tools(mapping, limits, "file_processing")
        assert "file_reader" in file_tools
        assert "text_extractor" in file_tools
        
        # QA agent should get search-related tools
        qa_mapping = {"recommended_tools": ["search_engine"]}
        qa_tools = builder._select_tools(qa_mapping, limits, "question_answering")
        assert "search_engine" in qa_tools
        assert "memory_retrieval" in qa_tools  # Essential tool
        
        Path(temp_template_file).unlink()
    
    def test_system_prompt_generation_with_context(self, system_config, temp_template_file, sample_tasks):
        """Test system prompt generation with task-specific context."""
        prompt_manager = PromptTemplateManager(temp_template_file)
        builder = AgentSpecBuilder(system_config, prompt_manager)
        
        task = sample_tasks["question_answering"]
        tools = ["search_engine", "memory_retrieval"]
        limits = AgentLimits(max_steps=5, max_tokens=2000, timeout_seconds=120)
        
        prompt = builder._generate_system_prompt(task, "question_answering", tools, limits, "moderate")
        
        assert task.description in prompt
        assert "search_engine" in prompt
        assert "memory_retrieval" in prompt
        assert str(limits.max_steps) in prompt
        assert str(limits.max_tokens) in prompt
        
        Path(temp_template_file).unlink()
    
    def test_output_contract_by_agent_type(self, system_config, temp_template_file, sample_tasks):
        """Test that output contracts are customized by agent type."""
        prompt_manager = PromptTemplateManager(temp_template_file)
        builder = AgentSpecBuilder(system_config, prompt_manager)
        
        task = sample_tasks["question_answering"]
        
        # QA agent should have answer-specific fields
        qa_contract = builder._create_output_contract(task, "question_answering")
        assert "answer" in qa_contract["required_fields"]
        assert "citations" in qa_contract["required_fields"]
        assert "confidence_score" in qa_contract["required_fields"]
        
        # File processing agent should have extraction-specific fields
        file_contract = builder._create_output_contract(task, "file_processing")
        assert "extracted_content" in file_contract["required_fields"]
        assert "metadata" in file_contract["required_fields"]
        
        Path(temp_template_file).unlink()


class TestAgentGenerator:
    """Test the main AgentGenerator class."""
    
    def test_init(self, system_config):
        """Test AgentGenerator initialization."""
        generator = AgentGenerator(system_config)
        
        assert generator.name == "AgentGenerator"
        assert generator.system_config == system_config
        assert isinstance(generator.prompt_manager, PromptTemplateManager)
        assert isinstance(generator.type_classifier, AgentTypeClassifier)
        assert isinstance(generator.spec_builder, AgentSpecBuilder)
    
    def test_execute_with_valid_task_mappings(self, system_config, sample_tasks, sample_task_mappings):
        """Test execution with valid task mappings."""
        generator = AgentGenerator(system_config)
        
        # Create execution state
        state = ExecutionState(session_id="test", user_request="Test request")
        task = sample_tasks["file_processing"]
        state.add_task(task)
        
        # Add task mappings
        state.context["task_mappings"] = [sample_task_mappings["file_processing"]]
        
        result_state = generator.execute(state)
        
        assert "generated_agents" in result_state.context
        generated_agents = result_state.context["generated_agents"]
        assert len(generated_agents) == 1
        
        agent_info = generated_agents[0]
        assert agent_info["task_id"] == task.id
        assert "agent_id" in agent_info
        assert "agent_type" in agent_info
        assert "complexity" in agent_info
        assert "tools" in agent_info
        assert "limits" in agent_info
        
        # Check that agent spec was added to state
        agent_id = agent_info["agent_id"]
        assert agent_id in result_state.agent_specs
        
        # Check that task was assigned to agent
        assert task.assigned_agent == agent_id
    
    def test_execute_multiple_tasks(self, system_config, sample_tasks, sample_task_mappings):
        """Test execution with multiple tasks."""
        generator = AgentGenerator(system_config)
        
        state = ExecutionState(session_id="test", user_request="Multiple tasks")
        
        # Add multiple tasks
        file_task = sample_tasks["file_processing"]
        qa_task = sample_tasks["question_answering"]
        state.add_task(file_task)
        state.add_task(qa_task)
        
        # Add task mappings
        state.context["task_mappings"] = [
            sample_task_mappings["file_processing"],
            sample_task_mappings["question_answering"]
        ]
        
        result_state = generator.execute(state)
        
        generated_agents = result_state.context["generated_agents"]
        assert len(generated_agents) == 2
        
        # Check that different agent types were created
        agent_types = [agent["agent_type"] for agent in generated_agents]
        assert "file_processing" in agent_types
        assert "question_answering" in agent_types
    
    def test_execute_with_agent_limit(self, system_config, sample_tasks, sample_task_mappings):
        """Test that agent generation respects concurrent agent limits."""
        # Set low limit for testing
        system_config.agent_generator.max_concurrent_agents = 1
        generator = AgentGenerator(system_config)
        
        state = ExecutionState(session_id="test", user_request="Limited agents")
        
        # Add multiple tasks
        for i, (task_key, task) in enumerate(sample_tasks.items()):
            if i >= 2:  # Only add 2 tasks
                break
            state.add_task(task)
        
        # Add task mappings
        mappings = list(sample_task_mappings.values())[:2]
        state.context["task_mappings"] = mappings
        
        result_state = generator.execute(state)
        
        # Should only generate 1 agent due to limit
        generated_agents = result_state.context["generated_agents"]
        assert len(generated_agents) == 1
    
    def test_execute_no_task_mappings(self, system_config):
        """Test execution with no task mappings."""
        generator = AgentGenerator(system_config)
        
        state = ExecutionState(session_id="test", user_request="No tasks")
        # No task_mappings in context
        
        result_state = generator.execute(state)
        
        assert result_state.is_complete
        assert "generated_agents" not in result_state.context or len(result_state.context.get("generated_agents", [])) == 0
    
    def test_execute_missing_task(self, system_config):
        """Test execution with task mapping referencing non-existent task."""
        generator = AgentGenerator(system_config)
        
        state = ExecutionState(session_id="test", user_request="Missing task")
        
        # Add task mapping for non-existent task
        state.context["task_mappings"] = [{
            "task_id": "nonexistent_task",
            "recommended_tools": ["text_generator"]
        }]
        
        result_state = generator.execute(state)
        
        # Should handle gracefully
        generated_agents = result_state.context.get("generated_agents", [])
        assert len(generated_agents) == 0
    
    def test_validate_inputs_valid(self, system_config):
        """Test input validation with valid inputs."""
        generator = AgentGenerator(system_config)
        
        state = ExecutionState(session_id="test", user_request="Valid")
        state.context["task_mappings"] = [{"task_id": "test", "recommended_tools": []}]
        
        assert generator.validate_inputs(state) is True
    
    def test_validate_inputs_invalid(self, system_config):
        """Test input validation with invalid inputs."""
        generator = AgentGenerator(system_config)
        
        # No task mappings
        state = ExecutionState(session_id="test", user_request="Invalid")
        assert generator.validate_inputs(state) is False
        
        # Empty task mappings
        state.context["task_mappings"] = []
        assert generator.validate_inputs(state) is False
    
    def test_get_next_node_with_agents(self, system_config):
        """Test next node determination when agents are generated."""
        generator = AgentGenerator(system_config)
        
        state = ExecutionState(session_id="test", user_request="Test")
        state.context["generated_agents"] = [{"agent_id": "test_agent"}]
        
        next_node = generator.get_next_node(state)
        assert next_node == "AgentExecutor"
    
    def test_get_next_node_no_agents(self, system_config):
        """Test next node determination when no agents are generated."""
        generator = AgentGenerator(system_config)
        
        state = ExecutionState(session_id="test", user_request="Test")
        state.context["generated_agents"] = []
        
        next_node = generator.get_next_node(state)
        assert next_node is None
        assert state.is_complete
    
    def test_is_critical_error(self, system_config):
        """Test critical error determination."""
        generator = AgentGenerator(system_config)
        
        # Template errors should not be critical
        template_error = Exception("template error")
        assert generator.is_critical_error(template_error) is False
        
        # Configuration errors should be critical
        config_error = KeyError("missing config")
        assert generator.is_critical_error(config_error) is True
        
        value_error = ValueError("invalid value")
        assert generator.is_critical_error(value_error) is True


class TestAgentGenerationIntegration:
    """Integration tests for the complete agent generation process."""
    
    def test_end_to_end_agent_generation(self, system_config, sample_tasks, sample_task_mappings, temp_template_file):
        """Test complete end-to-end agent generation process."""
        # Use real template file
        system_config.agent_generator.prompt_template_path = temp_template_file
        
        generator = AgentGenerator(system_config)
        
        # Create comprehensive test scenario
        state = ExecutionState(session_id="integration_test", user_request="Comprehensive test")
        
        # Add various types of tasks
        tasks_to_add = ["file_processing", "question_answering", "content_generation"]
        for task_key in tasks_to_add:
            state.add_task(sample_tasks[task_key])
        
        # Add corresponding task mappings
        mappings = [sample_task_mappings[key] for key in tasks_to_add]
        state.context["task_mappings"] = mappings
        
        # Execute agent generation
        result_state = generator.execute(state)
        
        # Verify results
        assert "generated_agents" in result_state.context
        generated_agents = result_state.context["generated_agents"]
        assert len(generated_agents) == 3
        
        # Verify each agent has required properties
        for agent_info in generated_agents:
            assert "agent_id" in agent_info
            assert "task_id" in agent_info
            assert "agent_type" in agent_info
            assert "complexity" in agent_info
            assert "tools" in agent_info
            assert "limits" in agent_info
            assert "output_contract" in agent_info
            
            # Verify agent spec exists in state
            agent_id = agent_info["agent_id"]
            assert agent_id in result_state.agent_specs
            
            agent_spec = result_state.agent_specs[agent_id]
            assert agent_spec.system_prompt != ""
            assert len(agent_spec.tools) > 0
            assert agent_spec.limits.max_steps > 0
            assert agent_spec.output_contract != {}
        
        # Verify tasks were assigned to agents
        for task in result_state.tasks:
            assert task.assigned_agent is not None
            assert task.assigned_agent in result_state.agent_specs
        
        # Verify different agent types were created
        agent_types = [agent["agent_type"] for agent in generated_agents]
        assert len(set(agent_types)) > 1  # Should have multiple different types
        
        # Verify execution log
        assert len(result_state.execution_log) > 0
        generation_logs = [log for log in result_state.execution_log if log["action"] == "agent_generation_complete"]
        assert len(generation_logs) == 1
        
        log_details = generation_logs[0]["details"]
        assert log_details["num_agents_generated"] == 3
        assert len(log_details["agent_types"]) == 3
        
        Path(temp_template_file).unlink()
    
    def test_agent_generation_with_various_complexities(self, system_config, sample_tasks, sample_task_mappings):
        """Test that different task complexities result in appropriate agent configurations."""
        generator = AgentGenerator(system_config)
        
        state = ExecutionState(session_id="complexity_test", user_request="Test complexities")
        
        # Add tasks of different complexities
        simple_task = sample_tasks["question_answering"]  # Simple QA
        complex_task = sample_tasks["complex_multi_step"]  # Complex analysis
        
        state.add_task(simple_task)
        state.add_task(complex_task)
        
        state.context["task_mappings"] = [
            sample_task_mappings["question_answering"],
            sample_task_mappings["complex_multi_step"]
        ]
        
        result_state = generator.execute(state)
        
        generated_agents = result_state.context["generated_agents"]
        assert len(generated_agents) == 2
        
        # Find agents by task
        simple_agent = next(a for a in generated_agents if a["task_id"] == simple_task.id)
        complex_agent = next(a for a in generated_agents if a["task_id"] == complex_task.id)
        
        # Complex agent should have higher limits
        simple_limits = simple_agent["limits"]
        complex_limits = complex_agent["limits"]
        
        assert complex_limits["max_steps"] >= simple_limits["max_steps"]
        assert complex_limits["max_tokens"] >= simple_limits["max_tokens"]
        assert complex_limits["timeout_seconds"] >= simple_limits["timeout_seconds"]
        
        # Verify complexity classifications
        complexity_order = ["simple", "moderate", "complex", "expert"]
        simple_idx = complexity_order.index(simple_agent["complexity"])
        complex_idx = complexity_order.index(complex_agent["complexity"])
        
        assert complex_idx >= simple_idx


if __name__ == "__main__":
    pytest.main([__file__, "-v"])