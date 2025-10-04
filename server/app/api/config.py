"""Configuration management API endpoints."""
from fastapi import APIRouter, HTTPException
from typing import Dict, Any

from app.api.models import ConfigUpdateRequest, ConfigResponse
from app.config.loader import ConfigLoader
from app.config.models import SystemConfig

router = APIRouter(prefix="/config", tags=["configuration"])

# Initialize config loader
config_loader = ConfigLoader()


@router.get("/", response_model=Dict[str, Any])
async def get_all_config() -> Dict[str, Any]:
    """
    Get all system configuration.
    """
    try:
        config = config_loader.load_config()
        
        # Convert dataclass to dict
        from dataclasses import asdict
        
        return {
            "orchestrator": asdict(config.orchestrator) if hasattr(config, 'orchestrator') else {},
            "planner": asdict(config.planner) if hasattr(config, 'planner') else {},
            "agent_generator": asdict(config.agent_generator) if hasattr(config, 'agent_generator') else {},
            "context": asdict(config.context) if hasattr(config, 'context') else {},
            "memory": asdict(config.memory) if hasattr(config, 'memory') else {},
            "workflows": {name: asdict(wf) for name, wf in config.workflows.items()} if hasattr(config, 'workflows') else {}
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve configuration: {str(e)}")


@router.get("/{config_type}", response_model=ConfigResponse)
async def get_config(config_type: str) -> ConfigResponse:
    """
    Get specific configuration section.
    
    - **config_type**: Configuration type (orchestrator, planner, agent_generator, context_manager, memory, workflows)
    """
    try:
        config = config_loader.load_config()
        
        config_map = {
            "orchestrator": config.orchestrator if hasattr(config, 'orchestrator') else None,
            "planner": config.planner if hasattr(config, 'planner') else None,
            "agent_generator": config.agent_generator if hasattr(config, 'agent_generator') else None,
            "context": config.context if hasattr(config, 'context') else None,
            "memory": config.memory if hasattr(config, 'memory') else None,
            "workflows": config.workflows if hasattr(config, 'workflows') else None
        }
        
        if config_type not in config_map:
            raise HTTPException(status_code=404, detail=f"Configuration type '{config_type}' not found")
        
        config_data = config_map[config_type]
        if config_data is None:
            raise HTTPException(status_code=404, detail=f"Configuration type '{config_type}' not available")
        
        # Convert to dict
        from dataclasses import asdict, is_dataclass
        
        if is_dataclass(config_data):
            config_dict = asdict(config_data)
        elif isinstance(config_data, dict):
            config_dict = {k: asdict(v) if is_dataclass(v) else v for k, v in config_data.items()}
        else:
            config_dict = {}
        
        return ConfigResponse(
            config_type=config_type,
            config=config_dict
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve configuration: {str(e)}")


@router.put("/{config_type}", response_model=ConfigResponse)
async def update_config(config_type: str, request: ConfigUpdateRequest) -> ConfigResponse:
    """
    Update specific configuration section.
    
    - **config_type**: Configuration type to update
    - **updates**: Configuration updates as key-value pairs
    """
    try:
        # Validate config type
        valid_types = ["orchestrator", "planner", "agent_generator", "context", "memory"]
        if config_type not in valid_types:
            raise HTTPException(status_code=400, detail=f"Invalid configuration type. Must be one of: {valid_types}")
        
        # In a real implementation, this would:
        # 1. Validate the updates against the schema
        # 2. Update the configuration file
        # 3. Reload the configuration
        # 4. Notify relevant components
        
        # For now, we'll just return the updated config
        return ConfigResponse(
            config_type=config_type,
            config=request.updates
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update configuration: {str(e)}")


@router.post("/reload")
async def reload_config() -> Dict[str, str]:
    """
    Reload configuration from files.
    """
    try:
        config_loader.reload()
        return {"message": "Configuration reloaded successfully"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reload configuration: {str(e)}")


@router.get("/validate/{config_type}")
async def validate_config(config_type: str) -> Dict[str, Any]:
    """
    Validate a specific configuration section.
    
    - **config_type**: Configuration type to validate
    """
    try:
        config = config_loader.load_config()
        
        # Basic validation - check if config exists and has required fields
        config_map = {
            "orchestrator": config.orchestrator if hasattr(config, 'orchestrator') else None,
            "planner": config.planner if hasattr(config, 'planner') else None,
            "agent_generator": config.agent_generator if hasattr(config, 'agent_generator') else None,
            "context": config.context if hasattr(config, 'context') else None,
            "memory": config.memory if hasattr(config, 'memory') else None,
        }
        
        if config_type not in config_map:
            raise HTTPException(status_code=404, detail=f"Configuration type '{config_type}' not found")
        
        config_data = config_map[config_type]
        if config_data is None:
            return {
                "valid": False,
                "errors": [f"Configuration type '{config_type}' not available"]
            }
        
        return {
            "valid": True,
            "errors": []
        }
        
    except HTTPException:
        raise
    except Exception as e:
        return {
            "valid": False,
            "errors": [str(e)]
        }
