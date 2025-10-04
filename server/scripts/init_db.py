"""
Database initialization script.

Creates necessary directories and initializes vector database and memory storage.
"""

import os
import sys
import logging
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config.loader import ConfigLoader
from app.core.context.vector_store import VectorStore
from app.core.memory.memory_manager import MemoryManager


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_directories():
    """Create necessary data directories."""
    directories = [
        "data",
        "data/vector_db",
        "data/memory",
        "data/uploads",
        "data/processed",
    ]
    
    for directory in directories:
        path = Path(directory)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {directory}")
        else:
            logger.info(f"Directory already exists: {directory}")


def initialize_vector_db(config):
    """Initialize vector database."""
    try:
        logger.info("Initializing vector database...")
        vector_store = VectorStore(config)
        
        # Test connection
        test_result = vector_store.search("test query", k=1)
        logger.info(f"Vector database initialized successfully (found {len(test_result)} results)")
        
    except Exception as e:
        logger.error(f"Error initializing vector database: {e}")
        raise


def initialize_memory(config):
    """Initialize memory system."""
    try:
        logger.info("Initializing memory system...")
        memory_manager = MemoryManager.from_config(config)
        
        # Test memory operations
        test_session = "init_test_session"
        memory_manager.store_conversation(
            session_id=test_session,
            user_message="Test initialization",
            assistant_message="Memory system initialized",
            metadata={"test": True}
        )
        
        logger.info("Memory system initialized successfully")
        
    except Exception as e:
        logger.error(f"Error initializing memory system: {e}")
        raise


def create_default_config():
    """Create default configuration files if they don't exist."""
    config_dir = Path("config")
    config_dir.mkdir(exist_ok=True)
    
    # Default agents.yaml
    agents_config = """# Agent Configuration
orchestrator:
  max_iterations: 6
  token_budget: 50000
  workflow_confidence_threshold: 0.7
  timeout_seconds: 300

planner:
  max_tasks: 10
  decomposition_strategy: "minimal"

agent_generator:
  max_concurrent_agents: 5
  default_max_steps: 4
  default_token_limit: 2000
"""
    
    agents_path = config_dir / "agents.yaml"
    if not agents_path.exists():
        agents_path.write_text(agents_config)
        logger.info("Created default agents.yaml")
    
    # Default workflows.yaml
    workflows_config = """# Workflow Configuration
workflows:
  rag_qa:
    name: "RAG Question Answering"
    description: "Answer questions using retrieved context"
    enabled: true
    triggers:
      - "what"
      - "how"
      - "why"
      - "explain"
      - "tell me about"
    steps:
      - name: "retrieve_context"
        description: "Retrieve relevant context"
        node_type: "context_manager"
      - name: "generate_answer"
        description: "Generate answer with citations"
        node_type: "rag_workflow"
      - name: "verify_answer"
        description: "Verify answer quality"
        node_type: "evaluator"

  summarize_extract:
    name: "Summarize and Extract"
    description: "Summarize documents and extract key information"
    enabled: true
    triggers:
      - "summarize"
      - "summary"
      - "extract"
      - "key points"
    steps:
      - name: "process_document"
        description: "Process and chunk document"
        node_type: "context_manager"
      - name: "generate_summary"
        description: "Generate summary"
        node_type: "rag_workflow"
"""
    
    workflows_path = config_dir / "workflows.yaml"
    if not workflows_path.exists():
        workflows_path.write_text(workflows_config)
        logger.info("Created default workflows.yaml")
    
    # Default memory.yaml
    memory_config = """# Memory Configuration
memory:
  provider: "mem0"
  ttl_days: 30
  max_entries: 10000
  
  collections:
    profile:
      enabled: true
      max_size: 100
    facts:
      enabled: true
      max_size: 1000
    conversations:
      enabled: true
      max_size: 5000
      retention_days: 30
"""
    
    memory_path = config_dir / "memory.yaml"
    if not memory_path.exists():
        memory_path.write_text(memory_config)
        logger.info("Created default memory.yaml")


def main():
    """Main initialization function."""
    logger.info("=" * 50)
    logger.info("Local Agent Studio - Database Initialization")
    logger.info("=" * 50)
    
    try:
        # Create directories
        logger.info("\n[1/4] Creating directories...")
        create_directories()
        
        # Create default config
        logger.info("\n[2/4] Creating default configuration...")
        create_default_config()
        
        # Load configuration
        logger.info("\n[3/4] Loading configuration...")
        config_loader = ConfigLoader()
        config = config_loader.load_config()
        logger.info("Configuration loaded successfully")
        
        # Initialize vector database
        logger.info("\n[4/4] Initializing databases...")
        try:
            initialize_vector_db(config)
        except Exception as e:
            logger.warning(f"Vector database initialization skipped: {e}")
        
        try:
            initialize_memory(config)
        except Exception as e:
            logger.warning(f"Memory initialization skipped: {e}")
        
        logger.info("\n" + "=" * 50)
        logger.info("Initialization completed successfully!")
        logger.info("=" * 50)
        logger.info("\nYou can now start the application with:")
        logger.info("  Windows: start-dev.bat")
        logger.info("  Linux/Mac: ./start-dev.sh")
        
    except Exception as e:
        logger.error(f"\nInitialization failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
