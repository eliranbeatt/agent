"""File upload and processing API endpoints."""
from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import Dict
import uuid
import os
from datetime import datetime
from pathlib import Path

from app.api.models import FileUploadResponse, FileProcessingStatus
from app.core.context.file_processor import FileProcessor
from app.core.context.context_manager import ContextManager
from app.config.loader import ConfigLoader

router = APIRouter(prefix="/files", tags=["files"])

# Initialize components
config_loader = ConfigLoader()
config = config_loader.load_config()

# Initialize context manager with proper configuration
context_manager = ContextManager(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    vector_db_path=config.context.vector_db_path,
    chunk_size=config.context.chunk_size,
    chunk_overlap=config.context.chunk_overlap,
    embedding_model=config.context.embedding_model,
    context_config=config.context
)

# In-memory storage for file processing status (should be replaced with proper storage)
file_status_store: Dict[str, FileProcessingStatus] = {}


@router.post("/upload", response_model=FileUploadResponse)
async def upload_file(file: UploadFile = File(...)) -> FileUploadResponse:
    """
    Upload a file for processing.
    
    - **file**: File to upload (Office documents, PDFs, images)
    """
    try:
        # Generate unique file ID
        file_id = str(uuid.uuid4())
        
        # Create upload directory if it doesn't exist
        upload_dir = Path("server/data/uploads")
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        # Save file
        file_path = upload_dir / f"{file_id}_{file.filename}"
        content = await file.read()
        
        with open(file_path, "wb") as f:
            f.write(content)
        
        # Initialize processing status
        file_status_store[file_id] = FileProcessingStatus(
            file_id=file_id,
            filename=file.filename or "unknown",
            status="pending",
            progress=0.0,
            chunks_created=None,
            error=None,
            completed_at=None
        )
        
        # Start async processing (in production, this would be a background task)
        try:
            file_status_store[file_id].status = "processing"
            file_status_store[file_id].progress = 25.0
            
            # Process file through complete RAG pipeline
            result = context_manager.ingest_file(str(file_path))
            
            # Update status based on ingestion result
            if result.success:
                file_status_store[file_id].status = "completed"
                file_status_store[file_id].progress = 100.0
                file_status_store[file_id].chunks_created = result.chunks_created
                file_status_store[file_id].completed_at = datetime.now()
            else:
                file_status_store[file_id].status = "failed"
                file_status_store[file_id].progress = 0.0
                file_status_store[file_id].error = result.error
            
        except Exception as e:
            file_status_store[file_id].status = "failed"
            file_status_store[file_id].error = str(e)
        
        return FileUploadResponse(
            file_id=file_id,
            filename=file.filename or "unknown",
            status="processing",
            message="File uploaded successfully and processing started"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File upload failed: {str(e)}")


@router.get("/status/{file_id}", response_model=FileProcessingStatus)
async def get_file_status(file_id: str) -> FileProcessingStatus:
    """
    Get the processing status of an uploaded file.
    
    - **file_id**: Unique file identifier
    """
    if file_id not in file_status_store:
        raise HTTPException(status_code=404, detail="File not found")
    
    return file_status_store[file_id]


@router.get("/list")
async def list_files() -> Dict[str, list]:
    """
    List all uploaded files and their processing status.
    """
    return {
        "files": [
            status.model_dump() for status in file_status_store.values()
        ]
    }


@router.delete("/{file_id}")
async def delete_file(file_id: str) -> Dict[str, str]:
    """
    Delete an uploaded file and its processing data.
    
    - **file_id**: Unique file identifier
    """
    if file_id not in file_status_store:
        raise HTTPException(status_code=404, detail="File not found")
    
    try:
        # Get file info
        file_info = file_status_store[file_id]
        
        # Delete physical file
        upload_dir = Path("server/data/uploads")
        file_path = upload_dir / f"{file_id}_{file_info.filename}"
        
        if file_path.exists():
            os.remove(file_path)
        
        # Remove from status store
        del file_status_store[file_id]
        
        return {"message": "File deleted successfully"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File deletion failed: {str(e)}")
