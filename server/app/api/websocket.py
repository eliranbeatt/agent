"""WebSocket endpoints for real-time updates."""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import Dict, Set
import json
import asyncio
from datetime import datetime
import uuid

router = APIRouter(tags=["websocket"])

# Connection manager for WebSocket connections
class ConnectionManager:
    """Manages WebSocket connections and broadcasts."""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.session_connections: Dict[str, Set[str]] = {}
    
    async def connect(self, websocket: WebSocket, client_id: str, session_id: str = None):
        """Accept and register a new WebSocket connection."""
        await websocket.accept()
        self.active_connections[client_id] = websocket
        
        if session_id:
            if session_id not in self.session_connections:
                self.session_connections[session_id] = set()
            self.session_connections[session_id].add(client_id)
    
    def disconnect(self, client_id: str, session_id: str = None):
        """Remove a WebSocket connection."""
        if client_id in self.active_connections:
            del self.active_connections[client_id]
        
        if session_id and session_id in self.session_connections:
            self.session_connections[session_id].discard(client_id)
            if not self.session_connections[session_id]:
                del self.session_connections[session_id]
    
    async def send_personal_message(self, message: dict, client_id: str):
        """Send a message to a specific client."""
        if client_id in self.active_connections:
            websocket = self.active_connections[client_id]
            await websocket.send_json(message)
    
    async def broadcast_to_session(self, message: dict, session_id: str):
        """Broadcast a message to all clients in a session."""
        if session_id in self.session_connections:
            for client_id in self.session_connections[session_id]:
                await self.send_personal_message(message, client_id)
    
    async def broadcast(self, message: dict):
        """Broadcast a message to all connected clients."""
        for client_id in list(self.active_connections.keys()):
            await self.send_personal_message(message, client_id)


# Global connection manager
manager = ConnectionManager()


@router.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """
    WebSocket endpoint for real-time updates.
    
    - **client_id**: Unique client identifier
    """
    await manager.connect(websocket, client_id)
    
    try:
        # Send connection confirmation
        await manager.send_personal_message({
            "type": "connection",
            "status": "connected",
            "client_id": client_id,
            "timestamp": datetime.now().isoformat()
        }, client_id)
        
        # Listen for messages
        while True:
            data = await websocket.receive_json()
            
            # Handle different message types
            message_type = data.get("type")
            
            if message_type == "ping":
                await manager.send_personal_message({
                    "type": "pong",
                    "timestamp": datetime.now().isoformat()
                }, client_id)
            
            elif message_type == "subscribe":
                session_id = data.get("session_id")
                if session_id:
                    if session_id not in manager.session_connections:
                        manager.session_connections[session_id] = set()
                    manager.session_connections[session_id].add(client_id)
                    
                    await manager.send_personal_message({
                        "type": "subscribed",
                        "session_id": session_id,
                        "timestamp": datetime.now().isoformat()
                    }, client_id)
            
            elif message_type == "unsubscribe":
                session_id = data.get("session_id")
                if session_id and session_id in manager.session_connections:
                    manager.session_connections[session_id].discard(client_id)
                    
                    await manager.send_personal_message({
                        "type": "unsubscribed",
                        "session_id": session_id,
                        "timestamp": datetime.now().isoformat()
                    }, client_id)
            
    except WebSocketDisconnect:
        manager.disconnect(client_id)
    except Exception as e:
        print(f"WebSocket error for client {client_id}: {str(e)}")
        manager.disconnect(client_id)


async def send_chat_update(session_id: str, message: str, is_complete: bool = False):
    """
    Send a chat response update via WebSocket.
    
    - **session_id**: Session identifier
    - **message**: Message content
    - **is_complete**: Whether this is the final message
    """
    await manager.broadcast_to_session({
        "type": "chat_update",
        "session_id": session_id,
        "message": message,
        "is_complete": is_complete,
        "timestamp": datetime.now().isoformat()
    }, session_id)


async def send_file_processing_update(
    file_id: str,
    status: str,
    progress: float,
    message: str = None,
    error: str = None
):
    """
    Send a file processing status update via WebSocket.
    
    - **file_id**: File identifier
    - **status**: Processing status
    - **progress**: Progress percentage (0-100)
    - **message**: Optional status message
    - **error**: Optional error message
    """
    await manager.broadcast({
        "type": "file_processing",
        "file_id": file_id,
        "status": status,
        "progress": progress,
        "message": message,
        "error": error,
        "timestamp": datetime.now().isoformat()
    })


async def send_agent_execution_update(
    session_id: str,
    agent_id: str,
    status: str,
    step: int,
    max_steps: int,
    message: str = None
):
    """
    Send an agent execution status update via WebSocket.
    
    - **session_id**: Session identifier
    - **agent_id**: Agent identifier
    - **status**: Execution status
    - **step**: Current step number
    - **max_steps**: Maximum steps
    - **message**: Optional status message
    """
    await manager.broadcast_to_session({
        "type": "agent_execution",
        "session_id": session_id,
        "agent_id": agent_id,
        "status": status,
        "step": step,
        "max_steps": max_steps,
        "progress": (step / max_steps * 100) if max_steps > 0 else 0,
        "message": message,
        "timestamp": datetime.now().isoformat()
    }, session_id)


async def send_execution_monitoring_update(
    session_id: str,
    execution_path: str,
    active_agents: list,
    completed_tasks: list,
    tokens_used: int,
    token_budget: int
):
    """
    Send execution monitoring data via WebSocket.
    
    - **session_id**: Session identifier
    - **execution_path**: Execution path taken
    - **active_agents**: List of active agent IDs
    - **completed_tasks**: List of completed task IDs
    - **tokens_used**: Tokens used so far
    - **token_budget**: Total token budget
    """
    await manager.broadcast_to_session({
        "type": "execution_monitoring",
        "session_id": session_id,
        "execution_path": execution_path,
        "active_agents": active_agents,
        "completed_tasks": completed_tasks,
        "tokens_used": tokens_used,
        "token_budget": token_budget,
        "token_usage_percent": (tokens_used / token_budget * 100) if token_budget > 0 else 0,
        "timestamp": datetime.now().isoformat()
    }, session_id)


async def send_plan_graph_update(
    session_id: str,
    tasks: list,
    dependencies: dict
):
    """
    Send plan graph update via WebSocket.
    
    - **session_id**: Session identifier
    - **tasks**: List of tasks in the plan
    - **dependencies**: Task dependency graph
    """
    await manager.broadcast_to_session({
        "type": "plan_graph",
        "session_id": session_id,
        "tasks": tasks,
        "dependencies": dependencies,
        "timestamp": datetime.now().isoformat()
    }, session_id)


# Export helper functions for use in other modules
__all__ = [
    "router",
    "manager",
    "send_chat_update",
    "send_file_processing_update",
    "send_agent_execution_update",
    "send_execution_monitoring_update",
    "send_plan_graph_update"
]
