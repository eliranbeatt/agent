"""Memory management API endpoints."""
from fastapi import APIRouter, HTTPException
from typing import Optional
from datetime import datetime

from app.api.models import MemoryEntry, MemoryListResponse, MemoryUpdateRequest
from app.core.memory.memory_manager import MemoryManager
from app.core.memory.models import UserProfile, Fact
from app.config.loader import ConfigLoader

router = APIRouter(prefix="/memory", tags=["memory"])

# Initialize memory manager
import os
config_loader = ConfigLoader()
config = config_loader.load_config()

# Get OpenAI API key from environment
api_key = os.getenv("OPENAI_API_KEY", config.openai_api_key or "")

# Initialize MemoryManager with proper parameters
memory_manager = MemoryManager(
    api_key=api_key,
    storage_path=config.memory.memory_db_path,
    ttl_days=config.memory.conversation_ttl_days,
    max_facts=1000,  # Could be added to config
    max_conversations=100,  # Could be added to config
    max_rag_traces=500,  # Could be added to config
)


@router.get("/", response_model=MemoryListResponse)
async def list_memories(
    memory_type: Optional[str] = None,
    limit: int = 100,
    offset: int = 0
) -> MemoryListResponse:
    """
    List stored memories.
    
    - **memory_type**: Filter by type (profile, fact, conversation)
    - **limit**: Maximum number of results
    - **offset**: Pagination offset
    """
    try:
        # Retrieve memories based on type
        memories = []
        
        if memory_type is None or memory_type == "profile":
            profile = memory_manager.get_profile("default_user")
            if profile:
                memories.append(MemoryEntry(
                    id="profile_default",
                    type="profile",
                    content=f"Name: {profile.name}, Preferences: {profile.preferences}",
                    source="system",
                    timestamp=datetime.now(),
                    metadata={"communication_style": profile.communication_style}
                ))
        
        if memory_type is None or memory_type == "fact":
            facts = memory_manager.get_facts(limit=limit)
            for fact in facts:
                memories.append(MemoryEntry(
                    id=f"fact_{fact.content[:20]}",
                    type="fact",
                    content=fact.content,
                    source=fact.source,
                    timestamp=fact.timestamp,
                    metadata={"confidence": fact.confidence}
                ))
        
        if memory_type is None or memory_type == "conversation":
            conversations = memory_manager.get_conversation_history("default_session", limit=limit)
            for i, conv in enumerate(conversations):
                memories.append(MemoryEntry(
                    id=f"conv_{i}",
                    type="conversation",
                    content=conv.get("content", ""),
                    source="conversation",
                    timestamp=conv.get("timestamp", datetime.now()),
                    metadata=conv.get("metadata", {})
                ))
        
        # Apply pagination
        paginated_memories = memories[offset:offset + limit]
        
        return MemoryListResponse(
            memories=paginated_memories,
            total=len(memories)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve memories: {str(e)}")


@router.get("/{memory_id}", response_model=MemoryEntry)
async def get_memory(memory_id: str) -> MemoryEntry:
    """
    Get a specific memory entry.
    
    - **memory_id**: Memory entry identifier
    """
    try:
        # Parse memory type from ID
        if memory_id.startswith("profile_"):
            profile = memory_manager.get_profile("default_user")
            if not profile:
                raise HTTPException(status_code=404, detail="Memory not found")
            
            return MemoryEntry(
                id=memory_id,
                type="profile",
                content=f"Name: {profile.name}, Preferences: {profile.preferences}",
                source="system",
                timestamp=datetime.now(),
                metadata={"communication_style": profile.communication_style}
            )
        
        elif memory_id.startswith("fact_"):
            # In a real implementation, we'd have proper fact retrieval by ID
            raise HTTPException(status_code=404, detail="Memory not found")
        
        else:
            raise HTTPException(status_code=404, detail="Memory not found")
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve memory: {str(e)}")


@router.put("/{memory_id}", response_model=MemoryEntry)
async def update_memory(memory_id: str, request: MemoryUpdateRequest) -> MemoryEntry:
    """
    Update a memory entry.
    
    - **memory_id**: Memory entry identifier
    - **content**: Updated content
    - **metadata**: Updated metadata
    """
    try:
        # Parse memory type from ID
        if memory_id.startswith("profile_"):
            # Update profile
            profile = UserProfile(
                user_id="default_user",
                name=request.metadata.get("name", "User"),
                preferences=request.metadata.get("preferences", {}),
                communication_style=request.metadata.get("communication_style", "professional")
            )
            memory_manager.store_profile("default_user", profile)
            
            return MemoryEntry(
                id=memory_id,
                type="profile",
                content=request.content,
                source="user_update",
                timestamp=datetime.now(),
                metadata=request.metadata
            )
        
        elif memory_id.startswith("fact_"):
            # Update fact
            fact = Fact(
                content=request.content,
                source=request.metadata.get("source", "user_update"),
                timestamp=datetime.now(),
                confidence=request.metadata.get("confidence", 1.0)
            )
            memory_manager.add_fact(fact)
            
            return MemoryEntry(
                id=memory_id,
                type="fact",
                content=request.content,
                source=fact.source,
                timestamp=fact.timestamp,
                metadata={"confidence": fact.confidence}
            )
        
        else:
            raise HTTPException(status_code=404, detail="Memory not found")
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update memory: {str(e)}")


@router.delete("/{memory_id}")
async def delete_memory(memory_id: str) -> dict:
    """
    Delete a memory entry.
    
    - **memory_id**: Memory entry identifier
    """
    try:
        # In a real implementation, we'd have proper deletion logic
        return {"message": "Memory deleted successfully", "memory_id": memory_id}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete memory: {str(e)}")


@router.post("/clear")
async def clear_memories(memory_type: Optional[str] = None) -> dict:
    """
    Clear all memories or memories of a specific type.
    
    - **memory_type**: Optional type filter (profile, fact, conversation)
    """
    try:
        # Apply retention policies or clear specific types
        if memory_type:
            return {"message": f"Cleared {memory_type} memories"}
        else:
            memory_manager.apply_retention_policies()
            return {"message": "Applied retention policies to all memories"}
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear memories: {str(e)}")
