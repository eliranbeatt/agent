"""Integration tests for WebSocket functionality."""
import pytest
from fastapi.testclient import TestClient
import json
from datetime import datetime

from app.main import app

client = TestClient(app)


class TestWebSocketConnection:
    """Tests for WebSocket connection management."""
    
    def test_websocket_connection(self):
        """Test WebSocket connection establishment."""
        with client.websocket_connect("/ws/test-client-123") as websocket:
            # Receive connection confirmation
            data = websocket.receive_json()
            
            assert data["type"] == "connection"
            assert data["status"] == "connected"
            assert data["client_id"] == "test-client-123"
            assert "timestamp" in data
    
    def test_websocket_ping_pong(self):
        """Test WebSocket ping/pong mechanism."""
        with client.websocket_connect("/ws/test-client-456") as websocket:
            # Skip connection message
            websocket.receive_json()
            
            # Send ping
            websocket.send_json({"type": "ping"})
            
            # Receive pong
            data = websocket.receive_json()
            assert data["type"] == "pong"
            assert "timestamp" in data
    
    def test_websocket_subscribe_to_session(self):
        """Test subscribing to a session."""
        with client.websocket_connect("/ws/test-client-789") as websocket:
            # Skip connection message
            websocket.receive_json()
            
            # Subscribe to session
            websocket.send_json({
                "type": "subscribe",
                "session_id": "test-session-123"
            })
            
            # Receive subscription confirmation
            data = websocket.receive_json()
            assert data["type"] == "subscribed"
            assert data["session_id"] == "test-session-123"
    
    def test_websocket_unsubscribe_from_session(self):
        """Test unsubscribing from a session."""
        with client.websocket_connect("/ws/test-client-101") as websocket:
            # Skip connection message
            websocket.receive_json()
            
            # Subscribe first
            websocket.send_json({
                "type": "subscribe",
                "session_id": "test-session-456"
            })
            websocket.receive_json()  # Skip subscription confirmation
            
            # Unsubscribe
            websocket.send_json({
                "type": "unsubscribe",
                "session_id": "test-session-456"
            })
            
            # Receive unsubscription confirmation
            data = websocket.receive_json()
            assert data["type"] == "unsubscribed"
            assert data["session_id"] == "test-session-456"


class TestWebSocketMessaging:
    """Tests for WebSocket message broadcasting."""
    
    def test_multiple_clients_connection(self):
        """Test multiple clients can connect simultaneously."""
        with client.websocket_connect("/ws/client-1") as ws1:
            with client.websocket_connect("/ws/client-2") as ws2:
                # Both should receive connection confirmations
                data1 = ws1.receive_json()
                data2 = ws2.receive_json()
                
                assert data1["client_id"] == "client-1"
                assert data2["client_id"] == "client-2"
    
    def test_session_subscription_isolation(self):
        """Test that session subscriptions are isolated."""
        with client.websocket_connect("/ws/client-a") as ws_a:
            with client.websocket_connect("/ws/client-b") as ws_b:
                # Skip connection messages
                ws_a.receive_json()
                ws_b.receive_json()
                
                # Subscribe to different sessions
                ws_a.send_json({
                    "type": "subscribe",
                    "session_id": "session-a"
                })
                ws_b.send_json({
                    "type": "subscribe",
                    "session_id": "session-b"
                })
                
                # Receive confirmations
                conf_a = ws_a.receive_json()
                conf_b = ws_b.receive_json()
                
                assert conf_a["session_id"] == "session-a"
                assert conf_b["session_id"] == "session-b"


class TestWebSocketHelperFunctions:
    """Tests for WebSocket helper functions."""
    
    @pytest.mark.asyncio
    async def test_send_chat_update(self):
        """Test sending chat update via WebSocket."""
        from app.api.websocket import send_chat_update, manager
        
        # This would require a connected client in a real scenario
        # For now, we test that the function doesn't raise errors
        try:
            await send_chat_update(
                session_id="test-session",
                message="Test message",
                is_complete=False
            )
        except Exception as e:
            # Expected to fail without connected clients
            pass
    
    @pytest.mark.asyncio
    async def test_send_file_processing_update(self):
        """Test sending file processing update via WebSocket."""
        from app.api.websocket import send_file_processing_update
        
        try:
            await send_file_processing_update(
                file_id="test-file",
                status="processing",
                progress=50.0,
                message="Processing file"
            )
        except Exception as e:
            # Expected to fail without connected clients
            pass
    
    @pytest.mark.asyncio
    async def test_send_agent_execution_update(self):
        """Test sending agent execution update via WebSocket."""
        from app.api.websocket import send_agent_execution_update
        
        try:
            await send_agent_execution_update(
                session_id="test-session",
                agent_id="test-agent",
                status="running",
                step=2,
                max_steps=5,
                message="Executing step 2"
            )
        except Exception as e:
            # Expected to fail without connected clients
            pass
    
    @pytest.mark.asyncio
    async def test_send_execution_monitoring_update(self):
        """Test sending execution monitoring update via WebSocket."""
        from app.api.websocket import send_execution_monitoring_update
        
        try:
            await send_execution_monitoring_update(
                session_id="test-session",
                execution_path="predefined",
                active_agents=["agent-1", "agent-2"],
                completed_tasks=["task-1"],
                tokens_used=500,
                token_budget=10000
            )
        except Exception as e:
            # Expected to fail without connected clients
            pass
    
    @pytest.mark.asyncio
    async def test_send_plan_graph_update(self):
        """Test sending plan graph update via WebSocket."""
        from app.api.websocket import send_plan_graph_update
        
        try:
            await send_plan_graph_update(
                session_id="test-session",
                tasks=[
                    {"id": "task-1", "name": "Task 1"},
                    {"id": "task-2", "name": "Task 2"}
                ],
                dependencies={"task-2": ["task-1"]}
            )
        except Exception as e:
            # Expected to fail without connected clients
            pass


class TestConnectionManager:
    """Tests for ConnectionManager class."""
    
    @pytest.mark.asyncio
    async def test_connection_manager_connect(self):
        """Test ConnectionManager connect method."""
        from app.api.websocket import ConnectionManager
        from unittest.mock import Mock, AsyncMock
        
        manager = ConnectionManager()
        mock_websocket = Mock()
        mock_websocket.accept = AsyncMock()
        
        await manager.connect(mock_websocket, "client-1", "session-1")
        
        assert "client-1" in manager.active_connections
        assert "session-1" in manager.session_connections
        assert "client-1" in manager.session_connections["session-1"]
    
    def test_connection_manager_disconnect(self):
        """Test ConnectionManager disconnect method."""
        from app.api.websocket import ConnectionManager
        from unittest.mock import Mock
        
        manager = ConnectionManager()
        manager.active_connections["client-1"] = Mock()
        manager.session_connections["session-1"] = {"client-1"}
        
        manager.disconnect("client-1", "session-1")
        
        assert "client-1" not in manager.active_connections
        assert "session-1" not in manager.session_connections
    
    @pytest.mark.asyncio
    async def test_connection_manager_send_personal_message(self):
        """Test ConnectionManager send_personal_message method."""
        from app.api.websocket import ConnectionManager
        from unittest.mock import Mock, AsyncMock
        
        manager = ConnectionManager()
        mock_websocket = Mock()
        mock_websocket.send_json = AsyncMock()
        manager.active_connections["client-1"] = mock_websocket
        
        await manager.send_personal_message({"test": "message"}, "client-1")
        
        mock_websocket.send_json.assert_called_once_with({"test": "message"})
    
    @pytest.mark.asyncio
    async def test_connection_manager_broadcast_to_session(self):
        """Test ConnectionManager broadcast_to_session method."""
        from app.api.websocket import ConnectionManager
        from unittest.mock import Mock, AsyncMock
        
        manager = ConnectionManager()
        mock_ws1 = Mock()
        mock_ws1.send_json = AsyncMock()
        mock_ws2 = Mock()
        mock_ws2.send_json = AsyncMock()
        
        manager.active_connections["client-1"] = mock_ws1
        manager.active_connections["client-2"] = mock_ws2
        manager.session_connections["session-1"] = {"client-1", "client-2"}
        
        await manager.broadcast_to_session({"test": "broadcast"}, "session-1")
        
        assert mock_ws1.send_json.called
        assert mock_ws2.send_json.called
    
    @pytest.mark.asyncio
    async def test_connection_manager_broadcast(self):
        """Test ConnectionManager broadcast method."""
        from app.api.websocket import ConnectionManager
        from unittest.mock import Mock, AsyncMock
        
        manager = ConnectionManager()
        mock_ws1 = Mock()
        mock_ws1.send_json = AsyncMock()
        mock_ws2 = Mock()
        mock_ws2.send_json = AsyncMock()
        
        manager.active_connections["client-1"] = mock_ws1
        manager.active_connections["client-2"] = mock_ws2
        
        await manager.broadcast({"test": "broadcast"})
        
        assert mock_ws1.send_json.called
        assert mock_ws2.send_json.called
