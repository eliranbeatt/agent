"""API request and response models."""
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    message: str = Field(..., description="User message")
    session_id: Optional[str] = Field(None, description="Session ID for conversation continuity")
    context: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional context")


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    response: str = Field(..., description="Agent response")
    session_id: str = Field(..., description="Session ID")
    sources: Optional[List[Dict[str, Any]]] = Field(default_factory=list, description="Retrieved sources")
    execution_path: Optional[str] = Field(None, description="Execution path taken")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")


class FileUploadResponse(BaseModel):
    """Response model for file upload."""
    file_id: str = Field(..., description="Unique file identifier")
    filename: str = Field(..., description="Original filename")
    status: str = Field(..., description="Processing status")
    message: str = Field(..., description="Status message")


class FileProcessingStatus(BaseModel):
    """File processing status model."""
    file_id: str = Field(..., description="File identifier")
    filename: str = Field(..., description="Original filename")
    status: str = Field(..., description="Processing status: pending, processing, completed, failed")
    progress: float = Field(..., description="Processing progress (0-100)")
    chunks_created: Optional[int] = Field(None, description="Number of chunks created")
    error: Optional[str] = Field(None, description="Error message if failed")
    completed_at: Optional[datetime] = Field(None, description="Completion timestamp")


class MemoryEntry(BaseModel):
    """Memory entry model."""
    id: str = Field(..., description="Memory entry ID")
    type: str = Field(..., description="Memory type: profile, fact, conversation")
    content: str = Field(..., description="Memory content")
    source: Optional[str] = Field(None, description="Source of the memory")
    timestamp: datetime = Field(..., description="Creation timestamp")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")


class MemoryListResponse(BaseModel):
    """Response model for memory list."""
    memories: List[MemoryEntry] = Field(..., description="List of memory entries")
    total: int = Field(..., description="Total number of memories")


class MemoryUpdateRequest(BaseModel):
    """Request model for updating memory."""
    content: str = Field(..., description="Updated content")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Updated metadata")


class ConfigUpdateRequest(BaseModel):
    """Request model for configuration updates."""
    config_type: str = Field(..., description="Configuration type: orchestrator, planner, agent_generator, etc.")
    updates: Dict[str, Any] = Field(..., description="Configuration updates")


class ConfigResponse(BaseModel):
    """Response model for configuration."""
    config_type: str = Field(..., description="Configuration type")
    config: Dict[str, Any] = Field(..., description="Configuration data")


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp")
