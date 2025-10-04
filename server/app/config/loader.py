"""Configuration loader with YAML/JSON support, validation, and hot-reload."""

import json
import logging
import re
import threading
import time
from pathlib import Path
from typing import Dict, Any, Optional, Callable
from dataclasses import asdict
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

import yaml

from .models import (
    SystemConfig,
    OrchestratorConfig,
    PlannerConfig,
    AgentGeneratorConfig,
    ContextConfig,
    MemoryConfig,
    WorkflowConfig,
    WorkflowStep,
)


logger = logging.getLogger(__name__)


class ConfigFileHandler(FileSystemEventHandler):
    """File system event handler for configuration hot-reload."""
    
    def __init__(self, config_loader: 'ConfigLoader'):
        self.config_loader = config_loader
        
    def on_modified(self, event):
        """Handle file modification events."""
        if event.is_directory:
            return
            
        file_path = Path(event.src_path)
        if file_path.suffix in ['.yaml', '.yml', '.json'] and file_path.name in self.config_loader.watched_files:
            logger.info(f"Configuration file {file_path} modified, reloading...")
            try:
                self.config_loader.reload_config()
            except Exception as e:
                logger.error(f"Failed to reload configuration: {e}")


class ConfigLoader:
    """Configuration loader with validation and hot-reload capabilities."""
    
    def __init__(self, config_dir: Path = Path("C:/Users/elira/Dev/agent/config")):
        self.config_dir = Path(config_dir)
        self.config: Optional[SystemConfig] = None
        self.watched_files: set = set()
        self.observer: Optional[Observer] = None
        self.reload_callbacks: list[Callable[[SystemConfig], None]] = []
        self._lock = threading.Lock()
        
    def load_config(self, config_files: Optional[Dict[str, str]] = None) -> SystemConfig:
        """
        Load configuration from files.
        
        Args:
            config_files: Optional dict mapping config types to file paths
                         Default: {"agents": "agents.yaml", "workflows": "workflows.yaml", "memory": "memory.yaml"}
        """
        if config_files is None:
            config_files = {
                "agents": "agents.yaml",
                "workflows": "workflows.yaml", 
                "memory": "memory.yaml"
            }
            
        with self._lock:
            loaded_files = []
            try:
                # Start with default configuration
                config_data = {}
                
                # Load each configuration file
                for config_type, filename in config_files.items():
                    file_path = self.config_dir / filename
                    if file_path.exists():
                        try:
                            data = self._load_file(file_path)
                            config_data.update(data)
                            self.watched_files.add(filename)
                            loaded_files.append(str(file_path))
                        except Exception as e:
                            raise ValueError(f"Error loading {file_path}: {e}")
                    else:
                        logger.warning(f"Configuration file {file_path} not found, using defaults")
                
                # Create SystemConfig from loaded data
                try:
                    self.config = self._create_system_config(config_data)
                except ValueError as e:
                    # Add file context to error message
                    error_msg = f"Configuration error in files {loaded_files}:\n  {e}"
                    logger.error(error_msg)
                    raise ValueError(error_msg)
                
                # Validate configuration
                errors = self.config.validate()
                if errors:
                    error_msg = f"Configuration validation failed for files {loaded_files}:\n" + "\n".join(f"  - {error}" for error in errors)
                    logger.error(error_msg)
                    raise ValueError(error_msg)
                
                logger.info(f"Configuration loaded successfully from: {', '.join(loaded_files)}")
                return self.config
                
            except Exception as e:
                logger.error(f"Failed to load configuration: {e}")
                # Return default configuration on failure
                logger.warning("Using default configuration due to load failure")
                self.config = SystemConfig()
                return self.config
    
    def _load_file(self, file_path: Path) -> Dict[str, Any]:
        """Load a single configuration file (YAML or JSON)."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                if file_path.suffix.lower() in ['.yaml', '.yml']:
                    try:
                        data = yaml.safe_load(f)
                        return data or {}
                    except yaml.YAMLError as e:
                        # Provide line number information for YAML errors
                        if hasattr(e, 'problem_mark'):
                            mark = e.problem_mark
                            raise ValueError(f"YAML syntax error at line {mark.line + 1}, column {mark.column + 1}: {e.problem}")
                        else:
                            raise ValueError(f"YAML parsing error: {e}")
                elif file_path.suffix.lower() == '.json':
                    try:
                        return json.load(f)
                    except json.JSONDecodeError as e:
                        raise ValueError(f"JSON syntax error at line {e.lineno}, column {e.colno}: {e.msg}")
                else:
                    raise ValueError(f"Unsupported file format: {file_path.suffix}")
        except FileNotFoundError:
            raise ValueError(f"Configuration file not found: {file_path}")
        except PermissionError:
            raise ValueError(f"Permission denied reading configuration file: {file_path}")
        except Exception as e:
            if isinstance(e, ValueError):
                raise
            logger.error(f"Unexpected error loading {file_path}: {e}")
            raise ValueError(f"Failed to load configuration file: {e}")
    
    def _create_system_config(self, config_data: Dict[str, Any]) -> SystemConfig:
        """Create SystemConfig from loaded configuration data."""
        
        try:
            # Extract orchestrator config
            orchestrator_data = config_data.get('orchestrator', {})
            try:
                orchestrator = OrchestratorConfig(**orchestrator_data)
            except (TypeError, ValueError) as e:
                # Extract field name from error message if possible
                field_match = re.search(r"'(\w+)'", str(e))
                field = f" on field '{field_match.group(1)}'" if field_match else ""
                raise ValueError(f"Invalid orchestrator configuration{field}: {e}")
            
            # Extract planner config
            planner_data = config_data.get('planner', {})
            try:
                planner = PlannerConfig(**planner_data)
            except (TypeError, ValueError) as e:
                # Extract field name from error message if possible
                field_match = re.search(r"'(\w+)'", str(e))
                field = f" on field '{field_match.group(1)}'" if field_match else ""
                raise ValueError(f"Invalid planner configuration{field}: {e}")
            
            # Extract agent generator config
            agent_generator_data = config_data.get('agent_generator', {})
            try:
                agent_generator = AgentGeneratorConfig(**agent_generator_data)
            except (TypeError, ValueError) as e:
                # Extract field name from error message if possible
                field_match = re.search(r"'(\w+)'", str(e))
                field = f" on field '{field_match.group(1)}'" if field_match else ""
                raise ValueError(f"Invalid agent_generator configuration{field}: {e}")
            
            # Extract context config
            context_data = config_data.get('context', {})
            try:
                context = ContextConfig(**context_data)
            except (TypeError, ValueError) as e:
                # Extract field name from error message if possible
                field_match = re.search(r"'(\w+)'", str(e))
                field = f" on field '{field_match.group(1)}'" if field_match else ""
                raise ValueError(f"Invalid context configuration{field}: {e}")
            
            # Extract memory config
            memory_data = config_data.get('memory', {})
            try:
                memory = MemoryConfig(**memory_data)
            except (TypeError, ValueError) as e:
                # Extract field name from error message if possible
                field_match = re.search(r"'(\w+)'", str(e))
                field = f" on field '{field_match.group(1)}'" if field_match else ""
                raise ValueError(f"Invalid memory configuration{field}: {e}")
            
            # Extract workflows
            workflows = {}
            workflows_data = config_data.get('workflows', {})
            for workflow_name, workflow_data in workflows_data.items():
                try:
                    # Convert steps data to WorkflowStep objects
                    steps = []
                    for step_idx, step_data in enumerate(workflow_data.get('steps', [])):
                        try:
                            step = WorkflowStep(**step_data)
                            steps.append(step)
                        except (TypeError, ValueError) as e:
                            raise ValueError(f"Invalid step {step_idx} in workflow '{workflow_name}': {e}")
                    
                    workflow_data['steps'] = steps
                    workflows[workflow_name] = WorkflowConfig(**workflow_data)
                except (TypeError, ValueError) as e:
                    raise ValueError(f"Invalid workflow '{workflow_name}': {e}")
            
            # Extract global settings
            global_settings = {
                'openai_api_key': config_data.get('openai_api_key'),
                'log_level': config_data.get('log_level', 'INFO'),
                'debug_mode': config_data.get('debug_mode', False),
                'config_reload_enabled': config_data.get('config_reload_enabled', True),
                'config_watch_interval': config_data.get('config_watch_interval', 5),
            }
            
            # Validate global settings
            valid_log_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
            if global_settings['log_level'] not in valid_log_levels:
                raise ValueError(f"log_level must be one of {valid_log_levels}, got '{global_settings['log_level']}'")
            
            if global_settings['config_watch_interval'] <= 0:
                raise ValueError(f"config_watch_interval must be positive, got {global_settings['config_watch_interval']}")
            
            return SystemConfig(
                orchestrator=orchestrator,
                planner=planner,
                agent_generator=agent_generator,
                context=context,
                memory=memory,
                workflows=workflows,
                **global_settings
            )
        except Exception as e:
            logger.error(f"Failed to create system configuration: {e}")
            raise
    
    def reload_config(self) -> SystemConfig:
        """Reload configuration from files."""
        logger.info("Reloading configuration...")
        old_config = self.config
        new_config = self.load_config()
        
        # Notify callbacks of configuration change
        for callback in self.reload_callbacks:
            try:
                callback(new_config)
            except Exception as e:
                logger.error(f"Configuration reload callback failed: {e}")
        
        return new_config
    
    def start_hot_reload(self) -> None:
        """Start watching configuration files for changes."""
        if not self.config or not self.config.config_reload_enabled:
            logger.info("Configuration hot-reload is disabled")
            return
            
        if self.observer is not None:
            logger.warning("Hot-reload is already running")
            return
            
        try:
            self.observer = Observer()
            event_handler = ConfigFileHandler(self)
            self.observer.schedule(event_handler, str(self.config_dir), recursive=False)
            self.observer.start()
            logger.info(f"Started configuration hot-reload watching {self.config_dir}")
        except Exception as e:
            logger.error(f"Failed to start configuration hot-reload: {e}")
    
    def stop_hot_reload(self) -> None:
        """Stop watching configuration files."""
        if self.observer is not None:
            self.observer.stop()
            self.observer.join()
            self.observer = None
            logger.info("Stopped configuration hot-reload")
    
    def add_reload_callback(self, callback: Callable[[SystemConfig], None]) -> None:
        """Add a callback to be called when configuration is reloaded."""
        self.reload_callbacks.append(callback)
    
    def remove_reload_callback(self, callback: Callable[[SystemConfig], None]) -> None:
        """Remove a reload callback."""
        if callback in self.reload_callbacks:
            self.reload_callbacks.remove(callback)
    
    def get_config(self) -> SystemConfig:
        """Get the current configuration."""
        if self.config is None:
            return self.load_config()
        return self.config
    
    def save_config(self, config: SystemConfig, filename: str = "system_config.yaml") -> None:
        """Save configuration to a file."""
        file_path = self.config_dir / filename
        
        # Ensure config directory exists
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert config to dict
        config_dict = asdict(config)
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            logger.info(f"Configuration saved to {file_path}")
        except Exception as e:
            logger.error(f"Failed to save configuration to {file_path}: {e}")
            raise


# Global configuration loader instance
_config_loader: Optional[ConfigLoader] = None


def get_config_loader(config_dir: Path = Path("config")) -> ConfigLoader:
    """Get or create the global configuration loader instance."""
    global _config_loader
    if _config_loader is None:
        _config_loader = ConfigLoader(config_dir)
    return _config_loader


def get_config() -> SystemConfig:
    """Get the current system configuration."""
    return get_config_loader().get_config()
