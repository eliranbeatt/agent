"""Integration tests for API endpoints."""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, MagicMock
import json
from datetime import datetime

from app.main import app

client = TestClient(app)


class TestHealthEndpoint:
    """Tests for health check endpoint."""
    
    def test_health_check(self):
        """Test health check endpoint returns ok status."""
        response = client.get("/healthz")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestChatEndpoints:
    """Tests for chat API endpoints."""
    
    @patch("app.api.chat.orchestrator")
    def test_chat_endpoint_success(self, mock_orchestrator):
        """Test successful chat request."""
        # Mock orchestrator response
        mock_orchestrator.process_request.return_value = {
            "response": "Test response",
            "sources": [],
            "execution_path": "predefined",
            "metadata": {}
        }
        
        response = client.post(
            "/chat/",
            json={
                "message": "Hello, how are you?",
                "session_id": "test-session-123"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["response"] == "Test response"
        assert data["session_id"] == "test-session-123"
        assert "sources" in data
        assert "execution_path" in data
    
    @patch("app.api.chat.orchestrator")
    def test_chat_endpoint_generates_session_id(self, mock_orchestrator):
        """Test chat endpoint generates session ID if not provided."""
        mock_orchestrator.process_request.return_value = {
            "response": "Test response",
            "sources": [],
            "execution_path": "predefined",
            "metadata": {}
        }
        
        response = client.post(
            "/chat/",
            json={"message": "Hello"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "session_id" in data
        assert len(data["session_id"]) > 0
    
    @patch("app.api.chat.orchestrator")
    def test_chat_endpoint_with_context(self, mock_orchestrator):
        """Test chat endpoint with additional context."""
        mock_orchestrator.process_request.return_value = {
            "response": "Test response",
            "sources": [],
            "execution_path": "predefined",
            "metadata": {}
        }
        
        response = client.post(
            "/chat/",
            json={
                "message": "What is in the document?",
                "context": {"file_id": "test-file-123"}
            }
        )
        
        assert response.status_code == 200
        assert mock_orchestrator.process_request.called
    
    @patch("app.api.chat.orchestrator")
    def test_chat_stream_endpoint(self, mock_orchestrator):
        """Test streaming chat endpoint."""
        mock_orchestrator.process_request.return_value = {
            "response": "Streaming response",
            "sources": [],
            "execution_path": "predefined",
            "metadata": {}
        }
        
        response = client.post(
            "/chat/stream",
            json={"message": "Hello"}
        )
        
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/event-stream; charset=utf-8"


class TestFileEndpoints:
    """Tests for file API endpoints."""
    
    @patch("app.api.files.context_manager")
    def test_file_upload_success(self, mock_context_manager):
        """Test successful file upload."""
        mock_context_manager.ingest_file.return_value = {
            "chunks_created": 10,
            "status": "completed"
        }
        
        # Create a test file
        files = {"file": ("test.txt", b"Test file content", "text/plain")}
        response = client.post("/files/upload", files=files)
        
        assert response.status_code == 200
        data = response.json()
        assert "file_id" in data
        assert data["filename"] == "test.txt"
        assert data["status"] == "processing"
    
    def test_file_upload_no_file(self):
        """Test file upload without file."""
        response = client.post("/files/upload")
        assert response.status_code == 422  # Validation error
    
    @patch("app.api.files.file_status_store")
    def test_get_file_status_success(self, mock_store):
        """Test getting file processing status."""
        from app.api.models import FileProcessingStatus
        
        mock_store.__getitem__.return_value = FileProcessingStatus(
            file_id="test-file-123",
            filename="test.txt",
            status="completed",
            progress=100.0,
            chunks_created=10,
            error=None,
            completed_at=datetime.now()
        )
        mock_store.__contains__.return_value = True
        
        response = client.get("/files/status/test-file-123")
        
        assert response.status_code == 200
        data = response.json()
        assert data["file_id"] == "test-file-123"
        assert data["status"] == "completed"
    
    def test_get_file_status_not_found(self):
        """Test getting status for non-existent file."""
        response = client.get("/files/status/non-existent-file")
        assert response.status_code == 404
    
    @patch("app.api.files.file_status_store")
    def test_list_files(self, mock_store):
        """Test listing all files."""
        from app.api.models import FileProcessingStatus
        
        mock_store.values.return_value = [
            FileProcessingStatus(
                file_id="file-1",
                filename="test1.txt",
                status="completed",
                progress=100.0,
                chunks_created=5,
                error=None,
                completed_at=datetime.now()
            )
        ]
        
        response = client.get("/files/list")
        
        assert response.status_code == 200
        data = response.json()
        assert "files" in data
        assert len(data["files"]) == 1


class TestMemoryEndpoints:
    """Tests for memory API endpoints."""
    
    @patch("app.api.memory.memory_manager")
    def test_list_memories(self, mock_memory_manager):
        """Test listing memories."""
        from app.core.memory.models import UserProfile, Fact
        
        mock_memory_manager.get_profile.return_value = UserProfile(
            user_id="default_user",
            name="Test User",
            preferences={},
            communication_style="professional"
        )
        mock_memory_manager.get_facts.return_value = [
            Fact(
                content="Test fact",
                source="test",
                timestamp=datetime.now(),
                confidence=1.0
            )
        ]
        mock_memory_manager.get_conversation_history.return_value = []
        
        response = client.get("/memory/")
        
        assert response.status_code == 200
        data = response.json()
        assert "memories" in data
        assert "total" in data
    
    @patch("app.api.memory.memory_manager")
    def test_list_memories_filtered(self, mock_memory_manager):
        """Test listing memories with type filter."""
        from app.core.memory.models import Fact
        
        mock_memory_manager.get_facts.return_value = [
            Fact(
                content="Test fact",
                source="test",
                timestamp=datetime.now(),
                confidence=1.0
            )
        ]
        
        response = client.get("/memory/?memory_type=fact")
        
        assert response.status_code == 200
        data = response.json()
        assert "memories" in data
    
    @patch("app.api.memory.memory_manager")
    def test_get_memory_profile(self, mock_memory_manager):
        """Test getting a specific profile memory."""
        from app.core.memory.models import UserProfile
        
        mock_memory_manager.get_profile.return_value = UserProfile(
            user_id="default_user",
            name="Test User",
            preferences={"theme": "dark"},
            communication_style="casual"
        )
        
        response = client.get("/memory/profile_default")
        
        assert response.status_code == 200
        data = response.json()
        assert data["type"] == "profile"
        assert "Test User" in data["content"]
    
    def test_get_memory_not_found(self):
        """Test getting non-existent memory."""
        response = client.get("/memory/non_existent_memory")
        assert response.status_code == 404
    
    @patch("app.api.memory.memory_manager")
    def test_update_memory_profile(self, mock_memory_manager):
        """Test updating a profile memory."""
        response = client.put(
            "/memory/profile_default",
            json={
                "content": "Updated profile",
                "metadata": {
                    "name": "Updated User",
                    "preferences": {"theme": "light"},
                    "communication_style": "professional"
                }
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["type"] == "profile"
        assert mock_memory_manager.store_profile.called
    
    def test_delete_memory(self):
        """Test deleting a memory."""
        response = client.delete("/memory/test_memory")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
    
    @patch("app.api.memory.memory_manager")
    def test_clear_memories(self, mock_memory_manager):
        """Test clearing memories."""
        response = client.post("/memory/clear")
        
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert mock_memory_manager.apply_retention_policies.called


class TestConfigEndpoints:
    """Tests for configuration API endpoints."""
    
    @patch("app.api.config.config_loader")
    def test_get_all_config(self, mock_config_loader):
        """Test getting all configuration."""
        from app.config.models import SystemConfig, OrchestratorConfig
        
        mock_config = SystemConfig()
        mock_config_loader.load_config.return_value = mock_config
        
        response = client.get("/config/")
        
        assert response.status_code == 200
        data = response.json()
        assert "orchestrator" in data
        assert "planner" in data
    
    @patch("app.api.config.config_loader")
    def test_get_specific_config(self, mock_config_loader):
        """Test getting specific configuration section."""
        from app.config.models import SystemConfig
        
        mock_config = SystemConfig()
        mock_config_loader.load_config.return_value = mock_config
        
        response = client.get("/config/orchestrator")
        
        assert response.status_code == 200
        data = response.json()
        assert data["config_type"] == "orchestrator"
        assert "config" in data
    
    def test_get_invalid_config_type(self):
        """Test getting invalid configuration type."""
        response = client.get("/config/invalid_type")
        assert response.status_code == 404
    
    def test_update_config(self):
        """Test updating configuration."""
        response = client.put(
            "/config/orchestrator",
            json={
                "config_type": "orchestrator",
                "updates": {"max_iterations": 10}
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["config_type"] == "orchestrator"
    
    def test_update_invalid_config_type(self):
        """Test updating invalid configuration type."""
        response = client.put(
            "/config/invalid_type",
            json={
                "config_type": "invalid_type",
                "updates": {}
            }
        )
        
        assert response.status_code == 400
    
    @patch("app.api.config.config_loader")
    def test_reload_config(self, mock_config_loader):
        """Test reloading configuration."""
        response = client.post("/config/reload")
        
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert mock_config_loader.reload.called
    
    @patch("app.api.config.config_loader")
    def test_validate_config(self, mock_config_loader):
        """Test validating configuration."""
        from app.config.models import SystemConfig
        
        mock_config = SystemConfig()
        mock_config_loader.load_config.return_value = mock_config
        
        response = client.get("/config/validate/orchestrator")
        
        assert response.status_code == 200
        data = response.json()
        assert "valid" in data
        assert data["valid"] is True


class TestErrorHandling:
    """Tests for error handling across endpoints."""
    
    @patch("app.api.chat.orchestrator")
    def test_chat_endpoint_error(self, mock_orchestrator):
        """Test chat endpoint handles errors gracefully."""
        mock_orchestrator.process_request.side_effect = Exception("Test error")
        
        response = client.post(
            "/chat/",
            json={"message": "Hello"}
        )
        
        assert response.status_code == 500
    
    @patch("app.api.files.context_manager")
    def test_file_upload_processing_error(self, mock_context_manager):
        """Test file upload handles processing errors."""
        mock_context_manager.ingest_file.side_effect = Exception("Processing error")
        
        files = {"file": ("test.txt", b"Test content", "text/plain")}
        response = client.post("/files/upload", files=files)
        
        # Should still return 200 but mark as failed in status
        assert response.status_code == 200
    
    @patch("app.api.memory.memory_manager")
    def test_memory_list_error(self, mock_memory_manager):
        """Test memory list handles errors gracefully."""
        mock_memory_manager.get_profile.side_effect = Exception("Memory error")
        
        response = client.get("/memory/")
        
        assert response.status_code == 500
