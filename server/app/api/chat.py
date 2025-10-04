"""Chat API endpoints."""
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from typing import AsyncGenerator
import json
import uuid
from datetime import datetime

from app.api.models import ChatRequest, ChatResponse, ErrorResponse
from app.core.graph_builder import create_execution_graph
from app.core.state import ExecutionState
from app.config.loader import ConfigLoader

router = APIRouter(prefix="/chat", tags=["chat"])

# Initialize execution graph
config_loader = ConfigLoader()
system_config = config_loader.load_config()
execution_graph = create_execution_graph(system_config)


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
            state = step_output[node_name]
            
            # Stream progress update
            progress_data = {
                "type": "progress",
                "node": node_name,
                "step": state.current_step,
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
