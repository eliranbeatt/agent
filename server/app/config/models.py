"""Configuration schema models for Local Agent Studio."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path


@dataclass
class OrchestratorConfig:
    """Configuration for the Main Orchestrator."""
    max_iterations: int = 6
    token_budget: int = 50000
    workflow_confidence_threshold: float = 0.7
    timeout_seconds: int = 300
    fallback_behavior: str = "graceful_degradation"
    
    def __post_init__(self):
        """Validate OrchestratorConfig fields after initialization."""
        if self.max_iterations <= 0:
            raise ValueError(f"max_iterations must be positive, got {self.max_iterations}")
        if self.token_budget <= 0:
            raise ValueError(f"token_budget must be positive, got {self.token_budget}")
        if not 0 <= self.workflow_confidence_threshold <= 1:
            raise ValueError(f"workflow_confidence_threshold must be between 0 and 1, got {self.workflow_confidence_threshold}")
        if self.timeout_seconds <= 0:
            raise ValueError(f"timeout_seconds must be positive, got {self.timeout_seconds}")
        valid_fallback_behaviors = ["graceful_degradation", "strict", "retry"]
        if self.fallback_behavior not in valid_fallback_behaviors:
            raise ValueError(f"fallback_behavior must be one of {valid_fallback_behaviors}, got '{self.fallback_behavior}'")


@dataclass
class PlannerConfig:
    """Configuration for the Planner component."""
    max_tasks_per_request: int = 10
    min_task_complexity: int = 1
    max_task_complexity: int = 5
    dependency_resolution_timeout: int = 30
    enable_task_optimization: bool = True
    
    def __post_init__(self):
        """Validate PlannerConfig fields after initialization."""
        if self.max_tasks_per_request <= 0:
            raise ValueError(f"max_tasks_per_request must be positive, got {self.max_tasks_per_request}")
        if self.min_task_complexity < 1:
            raise ValueError(f"min_task_complexity must be >= 1, got {self.min_task_complexity}")
        if self.max_task_complexity < self.min_task_complexity:
            raise ValueError(f"max_task_complexity ({self.max_task_complexity}) must be >= min_task_complexity ({self.min_task_complexity})")
        if self.dependency_resolution_timeout <= 0:
            raise ValueError(f"dependency_resolution_timeout must be positive, got {self.dependency_resolution_timeout}")


@dataclass
class AgentGeneratorConfig:
    """Configuration for the Agent Generator."""
    max_concurrent_agents: int = 5
    default_agent_max_steps: int = 4
    default_agent_max_tokens: int = 2000
    agent_timeout_seconds: int = 120
    prompt_template_path: str = "config/templates/agent_prompts.yaml"
    available_tools: List[str] = field(default_factory=lambda: [
        "file_reader", "web_search", "calculator", "code_executor"
    ])
    
    def __post_init__(self):
        """Validate AgentGeneratorConfig fields after initialization."""
        if self.max_concurrent_agents <= 0:
            raise ValueError(f"max_concurrent_agents must be positive, got {self.max_concurrent_agents}")
        if self.default_agent_max_steps <= 0:
            raise ValueError(f"default_agent_max_steps must be positive, got {self.default_agent_max_steps}")
        if self.default_agent_max_tokens <= 0:
            raise ValueError(f"default_agent_max_tokens must be positive, got {self.default_agent_max_tokens}")
        if self.agent_timeout_seconds <= 0:
            raise ValueError(f"agent_timeout_seconds must be positive, got {self.agent_timeout_seconds}")
        if not self.available_tools:
            raise ValueError("available_tools cannot be empty")


@dataclass
class ContextConfig:
    """Configuration for Context Manager and file processing."""
    chunk_size: int = 1000
    chunk_overlap: int = 100
    max_file_size_mb: int = 100
    supported_file_types: List[str] = field(default_factory=lambda: [
        ".pdf", ".docx", ".xlsx", ".pptx", ".txt", ".md", ".png", ".jpg", ".jpeg"
    ])
    ocr_enabled: bool = True
    ocr_language: str = "eng"
    embedding_model: str = "text-embedding-3-small"
    vector_db_path: str = "data/vector_db"
    retrieval_k: int = 10
    use_mmr: bool = True
    
    def __post_init__(self):
        """Validate ContextConfig fields after initialization."""
        if self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {self.chunk_size}")
        if self.chunk_overlap < 0:
            raise ValueError(f"chunk_overlap must be non-negative, got {self.chunk_overlap}")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(f"chunk_overlap ({self.chunk_overlap}) must be less than chunk_size ({self.chunk_size})")
        if self.max_file_size_mb <= 0:
            raise ValueError(f"max_file_size_mb must be positive, got {self.max_file_size_mb}")
        if not self.supported_file_types:
            raise ValueError("supported_file_types cannot be empty")
        if self.retrieval_k <= 0:
            raise ValueError(f"retrieval_k must be positive, got {self.retrieval_k}")
        if not self.embedding_model:
            raise ValueError("embedding_model cannot be empty")
        if not self.vector_db_path:
            raise ValueError("vector_db_path cannot be empty")


@dataclass
class MemoryConfig:
    """Configuration for Memory Manager and Mem0 integration."""
    mem0_enabled: bool = True
    memory_db_path: str = "data/memory"
    profile_ttl_days: int = 365
    facts_ttl_days: int = 180
    conversation_ttl_days: int = 30
    max_memory_size_mb: int = 500
    enable_memory_compression: bool = True
    retention_policy: str = "lru"  # lru, fifo, or custom
    max_relevant_memories: int = 10
    similarity_threshold: float = 0.7
    context_window_size: int = 5
    enable_temporal_weighting: bool = True
    
    def __post_init__(self):
        """Validate MemoryConfig fields after initialization."""
        if self.profile_ttl_days <= 0:
            raise ValueError(f"profile_ttl_days must be positive, got {self.profile_ttl_days}")
        if self.facts_ttl_days <= 0:
            raise ValueError(f"facts_ttl_days must be positive, got {self.facts_ttl_days}")
        if self.conversation_ttl_days <= 0:
            raise ValueError(f"conversation_ttl_days must be positive, got {self.conversation_ttl_days}")
        if self.max_memory_size_mb <= 0:
            raise ValueError(f"max_memory_size_mb must be positive, got {self.max_memory_size_mb}")
        valid_retention_policies = ["lru", "fifo", "custom"]
        if self.retention_policy not in valid_retention_policies:
            raise ValueError(f"retention_policy must be one of {valid_retention_policies}, got '{self.retention_policy}'")
        if self.max_relevant_memories <= 0:
            raise ValueError(f"max_relevant_memories must be positive, got {self.max_relevant_memories}")
        if not 0 <= self.similarity_threshold <= 1:
            raise ValueError(f"similarity_threshold must be between 0 and 1, got {self.similarity_threshold}")
        if self.context_window_size < 0:
            raise ValueError(f"context_window_size must be non-negative, got {self.context_window_size}")
        if not self.memory_db_path:
            raise ValueError("memory_db_path cannot be empty")


@dataclass
class WorkflowStep:
    """Configuration for a single workflow step."""
    name: str
    description: str
    node_type: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    success_criteria: List[str] = field(default_factory=list)


@dataclass
class WorkflowConfig:
    """Configuration for predefined workflows."""
    name: str
    description: str
    triggers: List[str] = field(default_factory=list)
    steps: List[WorkflowStep] = field(default_factory=list)
    success_criteria: List[str] = field(default_factory=list)
    confidence_threshold: float = 0.7
    enabled: bool = True
    
    def __post_init__(self):
        """Validate WorkflowConfig fields after initialization."""
        if not self.name:
            raise ValueError("workflow name cannot be empty")
        if not self.description:
            raise ValueError(f"workflow '{self.name}' description cannot be empty")
        if not 0 <= self.confidence_threshold <= 1:
            raise ValueError(f"workflow '{self.name}' confidence_threshold must be between 0 and 1, got {self.confidence_threshold}")
        if not self.steps:
            raise ValueError(f"workflow '{self.name}' must have at least one step")


@dataclass
class SystemConfig:
    """Root configuration for the entire Local Agent Studio system."""
    orchestrator: OrchestratorConfig = field(default_factory=OrchestratorConfig)
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    agent_generator: AgentGeneratorConfig = field(default_factory=AgentGeneratorConfig)
    context: ContextConfig = field(default_factory=ContextConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    workflows: Dict[str, WorkflowConfig] = field(default_factory=dict)
    
    # Global settings
    openai_api_key: Optional[str] = None
    log_level: str = "INFO"
    debug_mode: bool = False
    config_reload_enabled: bool = True
    config_watch_interval: int = 5  # seconds
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of errors."""
        errors = []
        
        # Validate orchestrator settings
        if self.orchestrator.max_iterations <= 0:
            errors.append("orchestrator.max_iterations must be positive")
        if self.orchestrator.token_budget <= 0:
            errors.append("orchestrator.token_budget must be positive")
        if not 0 <= self.orchestrator.workflow_confidence_threshold <= 1:
            errors.append("orchestrator.workflow_confidence_threshold must be between 0 and 1")
            
        # Validate planner settings
        if self.planner.max_tasks_per_request <= 0:
            errors.append("planner.max_tasks_per_request must be positive")
            
        # Validate agent generator settings
        if self.agent_generator.max_concurrent_agents <= 0:
            errors.append("agent_generator.max_concurrent_agents must be positive")
        if self.agent_generator.default_agent_max_steps <= 0:
            errors.append("agent_generator.default_agent_max_steps must be positive")
            
        # Validate context settings
        if self.context.chunk_size <= 0:
            errors.append("context.chunk_size must be positive")
        if self.context.chunk_overlap < 0:
            errors.append("context.chunk_overlap must be non-negative")
        if self.context.chunk_overlap >= self.context.chunk_size:
            errors.append("context.chunk_overlap must be less than chunk_size")
            
        # Validate memory settings
        if self.memory.max_memory_size_mb <= 0:
            errors.append("memory.max_memory_size_mb must be positive")
            
        # Validate workflows
        for workflow_name, workflow in self.workflows.items():
            if not workflow.steps:
                errors.append(f"workflow '{workflow_name}' must have at least one step")
            if not 0 <= workflow.confidence_threshold <= 1:
                errors.append(f"workflow '{workflow_name}' confidence_threshold must be between 0 and 1")
                
        return errors