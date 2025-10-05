"""Chat API endpoints."""
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from typing import AsyncGenerator
import json
import uuid
import os
from datetime import datetime

from app.api.models import ChatRequest, ChatResponse, ErrorResponse
from app.core.graph_builder import create_execution_graph
from app.core.state import ExecutionState
from app.config.loader import ConfigLoader
from app.core.rag.rag_workflow import RAGWorkflow
from app.core.context.context_manager import ContextManager

router = APIRouter(prefix="/chat", tags=["chat"])

# Initialize execution graph
config_loader = ConfigLoader()
system_config = config_loader.load_config()
execution_graph = create_execution_graph(system_config)

# Initialize RAG workflow for direct question answering
context_manager = ContextManager(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    vector_db_path=system_config.context.vector_db_path,
    chunk_size=system_config.context.chunk_size,
    chunk_overlap=system_config.context.chunk_overlap,
    embedding_model=system_config.context.embedding_model,
    context_config=system_config.context
)

rag_workflow = RAGWorkflow(
    context_manager=context_manager,
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    answer_model="gpt-4o-mini",
    retrieval_k=10,
    use_mmr=True,
    enable_verification=True
)


async def generate_streaming_response(
    message: str, 
    session_id: str, 
    context: dict
) -> AsyncGenerator[str, None]:
    """Generate streaming response from execution graph."""
    try:
        # Create execution state
        execution_state = ExecutionState(
            session_id=session_id,
            user_request=message,
            current_step=0,
            max_steps=system_config.orchestrator.max_iterations,
            tokens_used=0,
            token_budget=system_config.orchestrator.token_budget
        )
        
        # Merge context
        execution_state.context.update(context)
        
        # Execute through graph
        config = {"configurable": {"thread_id": session_id}}
        
        # Stream execution steps
        for step_output in execution_graph.stream(execution_state, config):
            # Extract node name and state
            node_name = list(step_output.keys())[0]
            state_dict = step_output[node_name]
            
            # Stream progress update
            progress_data = {
                "type": "progress",
                "node": node_name,
                "step": state_dict.get("current_step", 0),
                "session_id": session_id
            }
            yield f"data: {json.dumps(progress_data)}\n\n"
        
        # Get final state
        final_state = execution_graph.get_state(config)
        
        # Stream final response
        response_data = {
            "type": "response",
            "response": final_state.values.get("final_result", {}).get("response", ""),
            "session_id": session_id,
            "sources": final_state.values.get("final_result", {}).get("sources", []),
            "execution_path": final_state.values.get("execution_path", "unknown"),
            "metadata": final_state.values.get("final_result", {}).get("metadata", {})
        }
        
        yield f"data: {json.dumps(response_data)}\n\n"
        
    except Exception as e:
        error_data = {
            "type": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }
        yield f"data: {json.dumps(error_data)}\n\n"


@router.post("/", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Process a chat message and return a response.
    
    - **message**: User message to process
    - **session_id**: Optional session ID for conversation continuity
    - **context**: Optional additional context
    """
    try:
        # Generate or use provided session ID
        session_id = request.session_id or str(uuid.uuid4())
        
        # Check if this is a RAG question (has uploaded files or context indicates RAG)
        use_rag = request.context and (
            request.context.get("use_rag", False) or 
            request.context.get("has_uploaded_files", False)
        )
        
        if use_rag:
            # Use RAG workflow for direct question answering
            source_filter = request.context.get("source_filter")
            page_filter = request.context.get("page_filter")
            
            rag_result = rag_workflow.process_question(
                question=request.message,
                source_filter=source_filter,
                page_filter=page_filter
            )
            
            # Convert RAG result to chat response
            sources = [
                {
                    "chunk_id": citation.chunk_id,
                    "content": citation.content,
                    "source_file": citation.source_file,
                    "page_number": citation.page_number,
                    "score": citation.score
                }
                for citation in rag_result.citations
            ]
            
            return ChatResponse(
                response=rag_result.answer,
                session_id=session_id,
                sources=sources,
                execution_path="rag_qa",
                metadata={
                    "confidence": rag_result.confidence,
                    "tokens_used": rag_result.tokens_used,
                    "processing_time_ms": rag_result.processing_time_ms,
                    "chunks_retrieved": rag_result.metadata.get("chunks_retrieved", 0),
                    "verification_passed": rag_result.verification.passed if rag_result.verification else None,
                    **rag_result.metadata
                }
            )
        
        # Otherwise, use full orchestrator workflow
        # Create execution state
        execution_state = ExecutionState(
            session_id=session_id,
            user_request=request.message,
            current_step=0,
            max_steps=system_config.orchestrator.max_iterations,
            tokens_used=0,
            token_budget=system_config.orchestrator.token_budget
        )
        
        # Merge context
        execution_state.context.update(request.context or {})
        
        # Execute through graph
        config = {"configurable": {"thread_id": session_id}}
        final_state = execution_graph.invoke(execution_state, config)
        
        # Extract result
        final_result = final_state.get("final_result", {})
        
        return ChatResponse(
            response=final_result.get("response", ""),
            session_id=session_id,
            sources=final_result.get("sources", []),
            execution_path=final_result.get("execution_path", "unknown"),
            metadata=final_result.get("metadata", {})
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stream")
async def chat_stream(request: ChatRequest):
    """
    Process a chat message and stream the response.
    
    - **message**: User message to process
    - **session_id**: Optional session ID for conversation continuity
    - **context**: Optional additional context
    """
    session_id = request.session_id or str(uuid.uuid4())
    
    return StreamingResponse(
        generate_streaming_response(request.message, session_id, request.context or {}),
        media_type="text/event-stream"
    )


@router.post("/rag", response_model=ChatResponse)
async def rag_question(request: ChatRequest) -> ChatResponse:
    """
    Process a question using RAG (Retrieval-Augmented Generation).
    
    This endpoint is optimized for document question answering with:
    - Semantic search over uploaded documents
    - Answer generation with citations
    - Source verification
    
    - **message**: Question to ask about documents
    - **session_id**: Optional session ID
    - **context**: Optional context with source_filter, page_filter, etc.
    """
    try:
        session_id = request.session_id or str(uuid.uuid4())
        
        # Extract RAG-specific parameters from context
        source_filter = request.context.get("source_filter") if request.context else None
        page_filter = request.context.get("page_filter") if request.context else None
        custom_k = request.context.get("retrieval_k") if request.context else None
        
        # Process question through RAG workflow
        rag_result = rag_workflow.process_question(
            question=request.message,
            source_filter=source_filter,
            page_filter=page_filter,
            custom_k=custom_k
        )
        
        if not rag_result.success:
            raise HTTPException(status_code=500, detail=rag_result.error)
        
        # Convert citations to sources format
        sources = [
            {
                "chunk_id": citation.chunk_id,
                "content": citation.content,
                "source_file": citation.source_file,
                "page_number": citation.page_number,
                "score": citation.score
            }
            for citation in rag_result.citations
        ]
        
        return ChatResponse(
            response=rag_result.answer,
            session_id=session_id,
            sources=sources,
            execution_path="rag_qa",
            metadata={
                "confidence": rag_result.confidence,
                "tokens_used": rag_result.tokens_used,
                "processing_time_ms": rag_result.processing_time_ms,
                "chunks_retrieved": rag_result.metadata.get("chunks_retrieved", 0),
                "question_type": rag_result.metadata.get("question_type"),
                "verification_passed": rag_result.verification.passed if rag_result.verification else None,
                "verification_confidence": rag_result.verification.confidence if rag_result.verification else None,
                **rag_result.metadata
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
