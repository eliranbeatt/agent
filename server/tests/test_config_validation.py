"""Tests for configuration validation and loading."""

import pytest
import tempfile
import shutil
from pathlib import Path
import yaml

from app.config.loader import ConfigLoader
from app.config.models import (
    SystemConfig,
    OrchestratorConfig,
    PlannerConfig,
    AgentGeneratorConfig,
    ContextConfig,
    MemoryConfig,
    WorkflowConfig,
)


class TestConfigDataclassValidation:
    """Test validation in configuration dataclasses."""
    
    def test_orchestrator_config_valid(self):
        """Test valid OrchestratorConfig."""
        config = OrchestratorConfig(
            max_iterations=6,
            token_budget=50000,
            workflow_confidence_threshold=0.7,
            timeout_seconds=300,
            fallback_behavior="graceful_degradation"
        )
        assert config.max_iterations == 6
        assert config.token_budget == 50000
    
    def test_orchestrator_config_invalid_iterations(self):
        """Test OrchestratorConfig with invalid max_iterations."""
        with pytest.raises(ValueError, match="max_iterations must be positive"):
            OrchestratorConfig(max_iterations=0)
    
    def test_orchestrator_config_invalid_token_budget(self):
        """Test OrchestratorConfig with invalid token_budget."""
        with pytest.raises(ValueError, match="token_budget must be positive"):
            OrchestratorConfig(token_budget=-1000)
    
    def test_orchestrator_config_invalid_confidence_threshold(self):
        """Test OrchestratorConfig with invalid confidence threshold."""
        with pytest.raises(ValueError, match="workflow_confidence_threshold must be between 0 and 1"):
            OrchestratorConfig(workflow_confidence_threshold=1.5)
    
    def test_orchestrator_config_invalid_fallback_behavior(self):
        """Test OrchestratorConfig with invalid fallback behavior."""
        with pytest.raises(ValueError, match="fallback_behavior must be one of"):
            OrchestratorConfig(fallback_behavior="invalid_behavior")
    
    def test_planner_config_valid(self):
        """Test valid PlannerConfig."""
        config = PlannerConfig(
            max_tasks=10,
            min_task_complexity=1,
            max_task_complexity=5
        )
        assert config.max_tasks == 10
    
    def test_planner_config_invalid_max_tasks(self):
        """Test PlannerConfig with invalid max_tasks."""
        with pytest.raises(ValueError, match="max_tasks must be positive"):
            PlannerConfig(max_tasks=0)
    
    def test_planner_config_invalid_complexity_range(self):
        """Test PlannerConfig with invalid complexity range."""
        with pytest.raises(ValueError, match="max_task_complexity.*must be >= min_task_complexity"):
            PlannerConfig(min_task_complexity=5, max_task_complexity=2)
    
    def test_agent_generator_config_valid(self):
        """Test valid AgentGeneratorConfig."""
        config = AgentGeneratorConfig(
            max_concurrent_agents=5,
            default_agent_max_steps=4,
            available_tools=["file_reader", "calculator"]
        )
        assert config.max_concurrent_agents == 5
    
    def test_agent_generator_config_invalid_concurrent_agents(self):
        """Test AgentGeneratorConfig with invalid max_concurrent_agents."""
        with pytest.raises(ValueError, match="max_concurrent_agents must be positive"):
            AgentGeneratorConfig(max_concurrent_agents=0)
    
    def test_agent_generator_config_empty_tools(self):
        """Test AgentGeneratorConfig with empty tools list."""
        with pytest.raises(ValueError, match="available_tools cannot be empty"):
            AgentGeneratorConfig(available_tools=[])
    
    def test_context_config_valid(self):
        """Test valid ContextConfig."""
        config = ContextConfig(
            chunk_size=1000,
            chunk_overlap=100,
            supported_file_types=[".pdf", ".docx"]
        )
        assert config.chunk_size == 1000
    
    def test_context_config_invalid_chunk_overlap(self):
        """Test ContextConfig with chunk_overlap >= chunk_size."""
        with pytest.raises(ValueError, match="chunk_overlap.*must be less than chunk_size"):
            ContextConfig(chunk_size=1000, chunk_overlap=1000)
    
    def test_context_config_empty_file_types(self):
        """Test ContextConfig with empty supported_file_types."""
        with pytest.raises(ValueError, match="supported_file_types cannot be empty"):
            ContextConfig(supported_file_types=[])
    
    def test_memory_config_valid(self):
        """Test valid MemoryConfig."""
        config = MemoryConfig(
            profile_ttl_days=365,
            facts_ttl_days=180,
            retention_policy="lru"
        )
        assert config.profile_ttl_days == 365
    
    def test_memory_config_invalid_ttl(self):
        """Test MemoryConfig with invalid TTL."""
        with pytest.raises(ValueError, match="profile_ttl_days must be positive"):
            MemoryConfig(profile_ttl_days=0)
    
    def test_memory_config_invalid_retention_policy(self):
        """Test MemoryConfig with invalid retention policy."""
        with pytest.raises(ValueError, match="retention_policy must be one of"):
            MemoryConfig(retention_policy="invalid_policy")
    
    def test_workflow_config_valid(self):
        """Test valid WorkflowConfig."""
        from app.config.models import WorkflowStep
        
        steps = [
            WorkflowStep(
                name="step1",
                description="Test step",
                node_type="test_node"
            )
        ]
        config = WorkflowConfig(
            name="test_workflow",
            description="Test workflow",
            steps=steps
        )
        assert config.name == "test_workflow"
    
    def test_workflow_config_empty_name(self):
        """Test WorkflowConfig with empty name."""
        from app.config.models import WorkflowStep
        
        steps = [WorkflowStep(name="step1", description="Test", node_type="test")]
        with pytest.raises(ValueError, match="workflow name cannot be empty"):
            WorkflowConfig(name="", description="Test", steps=steps)
    
    def test_workflow_config_no_steps(self):
        """Test WorkflowConfig with no steps."""
        with pytest.raises(ValueError, match="must have at least one step"):
            WorkflowConfig(name="test", description="Test", steps=[])


class TestConfigLoader:
    """Test configuration loading from files."""
    
    @pytest.fixture
    def temp_config_dir(self):
        """Create a temporary config directory."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    def test_load_valid_config(self, temp_config_dir):
        """Test loading valid configuration files."""
        # Create valid config files
        agents_config = {
            'orchestrator': {
                'max_iterations': 6,
                'token_budget': 50000,
                'workflow_confidence_threshold': 0.7,
                'timeout_seconds': 300,
                'fallback_behavior': 'graceful_degradation'
            },
            'planner': {
                'max_tasks_per_request': 10,
                'min_task_complexity': 1,
                'max_task_complexity': 5,
                'dependency_resolution_timeout': 30,
                'enable_task_optimization': True
            },
            'agent_generator': {
                'max_concurrent_agents': 5,
                'default_agent_max_steps': 4,
                'default_agent_max_tokens': 2000,
                'agent_timeout_seconds': 120,
                'prompt_template_path': 'config/templates/agent_prompts.yaml',
                'available_tools': ['file_reader', 'calculator']
            },
            'context': {
                'chunk_size': 1000,
                'chunk_overlap': 100,
                'max_file_size_mb': 100,
                'supported_file_types': ['.pdf', '.docx'],
                'ocr_enabled': True,
                'ocr_language': 'eng',
                'embedding_model': 'text-embedding-3-small',
                'vector_db_path': 'data/vector_db',
                'retrieval_k': 10,
                'use_mmr': True
            }
        }
        
        memory_config = {
            'memory': {
                'mem0_enabled': True,
                'memory_db_path': 'data/memory',
                'profile_ttl_days': 365,
                'facts_ttl_days': 180,
                'conversation_ttl_days': 30,
                'max_memory_size_mb': 500,
                'enable_memory_compression': True,
                'retention_policy': 'lru',
                'max_relevant_memories': 10,
                'similarity_threshold': 0.7,
                'context_window_size': 5,
                'enable_temporal_weighting': True
            }
        }
        
        # Write config files
        with open(temp_config_dir / 'agents.yaml', 'w') as f:
            yaml.dump(agents_config, f)
        
        with open(temp_config_dir / 'memory.yaml', 'w') as f:
            yaml.dump(memory_config, f)
        
        # Load configuration
        loader = ConfigLoader(temp_config_dir)
        config = loader.load_config()
        
        assert config is not None
        assert config.orchestrator.max_iterations == 6
        assert config.planner.max_tasks == 10
        assert config.memory.profile_ttl_days == 365
    
    def test_load_config_with_invalid_values(self, temp_config_dir):
        """Test loading configuration with invalid values."""
        # Create config with invalid values
        invalid_config = {
            'orchestrator': {
                'max_iterations': -5,  # Invalid: must be positive
                'token_budget': 50000,
                'workflow_confidence_threshold': 0.7,
                'timeout_seconds': 300,
                'fallback_behavior': 'graceful_degradation'
            }
        }
        
        with open(temp_config_dir / 'agents.yaml', 'w') as f:
            yaml.dump(invalid_config, f)
        
        # Loading should fail with validation error
        loader = ConfigLoader(temp_config_dir)
        config = loader.load_config()
        
        # Should return default config on failure
        assert config is not None
        assert config.orchestrator.max_iterations == 6  # Default value
    
    def test_load_config_with_missing_fields(self, temp_config_dir):
        """Test loading configuration with missing fields (should use defaults)."""
        # Create config with only some fields
        partial_config = {
            'orchestrator': {
                'max_iterations': 10
                # Other fields missing, should use defaults
            }
        }
        
        with open(temp_config_dir / 'agents.yaml', 'w') as f:
            yaml.dump(partial_config, f)
        
        loader = ConfigLoader(temp_config_dir)
        config = loader.load_config()
        
        assert config is not None
        assert config.orchestrator.max_iterations == 10
        assert config.orchestrator.token_budget == 50000  # Default
    
    def test_load_config_with_invalid_yaml(self, temp_config_dir):
        """Test loading configuration with invalid YAML syntax."""
        # Create invalid YAML file
        with open(temp_config_dir / 'agents.yaml', 'w') as f:
            f.write("invalid: yaml: syntax:\n  - broken")
        
        loader = ConfigLoader(temp_config_dir)
        config = loader.load_config()
        
        # Should return default config on failure
        assert config is not None
        assert isinstance(config, SystemConfig)
    
    def test_load_config_missing_files(self, temp_config_dir):
        """Test loading configuration when files don't exist."""
        # Don't create any config files
        loader = ConfigLoader(temp_config_dir)
        config = loader.load_config()
        
        # Should return default config
        assert config is not None
        assert config.orchestrator.max_iterations == 6  # Default
    
    def test_fallback_to_defaults(self, temp_config_dir):
        """Test that system falls back to defaults on error."""
        # Create config that will cause validation error
        bad_config = {
            'orchestrator': {
                'max_iterations': 5,
                'token_budget': 50000,
                'workflow_confidence_threshold': 2.5,  # Invalid: > 1
                'timeout_seconds': 300,
                'fallback_behavior': 'graceful_degradation'
            }
        }
        
        with open(temp_config_dir / 'agents.yaml', 'w') as f:
            yaml.dump(bad_config, f)
        
        loader = ConfigLoader(temp_config_dir)
        config = loader.load_config()
        
        # Should have default config
        assert config is not None
        assert config.orchestrator.workflow_confidence_threshold == 0.7  # Default


class TestSystemConfigValidation:
    """Test SystemConfig validation method."""
    
    def test_system_config_validate_success(self):
        """Test validation of valid SystemConfig."""
        config = SystemConfig()
        errors = config.validate()
        assert len(errors) == 0
    
    def test_system_config_validate_orchestrator_errors(self):
        """Test validation catches orchestrator errors."""
        config = SystemConfig()
        config.orchestrator.max_iterations = -1
        errors = config.validate()
        assert len(errors) > 0
        assert any("max_iterations" in error for error in errors)
    
    def test_system_config_validate_context_errors(self):
        """Test validation catches context errors."""
        config = SystemConfig()
        config.context.chunk_overlap = config.context.chunk_size + 100
        errors = config.validate()
        assert len(errors) > 0
        assert any("chunk_overlap" in error for error in errors)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
