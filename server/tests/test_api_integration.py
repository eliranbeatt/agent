"""Integration tests for API endpoints."""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, MagicMock
from types import SimpleNamespace
import json
from datetime import datetime

from app.main import app
from app.core.rag.rag_workflow import RAGResult

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

    @patch("app.api.chat.intent_agent.analyze")
    @patch("app.api.chat.orchestrator")
    def test_chat_endpoint_success(self, mock_orchestrator, mock_analyze):
        """Test successful chat request with orchestrator escalation."""
        mock_analyze.return_value = {
            "response": "Quick triage summary",
            "requires_followup": True,
            "metadata": {"model": "gpt-5-mini"},
        }
        mock_orchestrator.process_request.return_value = {
            "response": "Test response",
            "sources": [],
            "execution_path": "predefined",
            "metadata": {},
        }

        response = client.post(
            "/chat/",
            json={"message": "Hello, how are you?", "session_id": "test-session-123"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["response"] == "Test response"
        assert data["session_id"] == "test-session-123"
        assert data["metadata"]["analysis"]["model"] == "gpt-5-mini"
        assert data["metadata"]["initial_response"] == "Quick triage summary"

    @patch("app.api.chat.intent_agent.analyze")
    @patch("app.api.chat.orchestrator")
    def test_chat_endpoint_generates_session_id(self, mock_orchestrator, mock_analyze):
        """Test chat endpoint generates session ID if not provided."""
        mock_analyze.return_value = {
            "response": "Quick triage summary",
            "requires_followup": True,
            "metadata": {"model": "gpt-5-mini"},
        }
        mock_orchestrator.process_request.return_value = {
            "response": "Test response",
            "sources": [],
            "execution_path": "predefined",
            "metadata": {},
        }

        response = client.post("/chat/", json={"message": "Hello"})

        assert response.status_code == 200
        data = response.json()
        assert "session_id" in data and data["session_id"]
        assert data["metadata"]["analysis"]["model"] == "gpt-5-mini"

    @patch("app.api.chat.intent_agent.analyze")
    @patch("app.api.chat.orchestrator")
    def test_chat_endpoint_with_context(self, mock_orchestrator, mock_analyze):
        """Test chat endpoint with additional context triggers orchestrator."""
        mock_analyze.return_value = {
            "response": "Quick triage summary",
            "requires_followup": True,
            "metadata": {"model": "gpt-5-mini"},
        }
        mock_orchestrator.process_request.return_value = {
            "response": "Test response",
            "sources": [],
            "execution_path": "predefined",
            "metadata": {},
        }

        response = client.post(
            "/chat/",
            json={"message": "What is in the document?", "context": {"file_id": "test-file-123"}},
        )

        assert response.status_code == 200
        assert mock_orchestrator.process_request.called

    @patch("app.api.chat.intent_agent.analyze")
    def test_chat_endpoint_light_intent_only(self, mock_analyze):
        """Lightweight intent agent should respond immediately when no follow-up needed."""
        mock_analyze.return_value = {
            "response": "Quick acknowledgment",
            "requires_followup": False,
            "metadata": {"model": "gpt-5-mini"},
        }

        response = client.post("/chat/", json={"message": "Just saying hi"})

        assert response.status_code == 200
        data = response.json()
        assert data["execution_path"] == "light_intent"
        assert data["metadata"]["initial_model"] == "gpt-5-mini"
        assert data["metadata"]["fallback_used"] is False

    @patch("app.api.chat.intent_agent.analyze")
    def test_chat_stream_endpoint(self, mock_analyze):
        """Test streaming chat endpoint with synthetic stream output."""
        mock_analyze.return_value = {
            "response": "Quick triage summary",
            "requires_followup": True,
            "metadata": {"model": "gpt-5-mini"},
        }
        fake_state = SimpleNamespace(
            values={
                "final_result": {
                    "response": "Streaming response",
                    "sources": [],
                    "metadata": {},
                    "execution_path": "predefined",
                },
                "execution_path": "predefined",
            }
        )

        with patch("app.api.chat.execution_graph.stream", return_value=[]), patch(
            "app.api.chat.execution_graph.get_state", return_value=fake_state
        ):
            response = client.post("/chat/stream", json={"message": "Hello"})

        assert response.status_code == 200
        assert response.headers["content-type"] == "text/event-stream; charset=utf-8"

    def test_chat_endpoint_fallback_uses_rag(self):
        """Ensure orchestrator fallback produces a real answer via RAG."""
        fallback_result = RAGResult(
            success=True,
            question="Hello",
            answer="Fallback answer",
            confidence=0.85,
            tokens_used=42,
            processing_time_ms=120.5,
        )
        analysis_result = {
            "response": "Routing to orchestrator",
            "requires_followup": True,
            "metadata": {"model": "gpt-5-mini"},
        }

        with patch("app.api.chat.intent_agent.analyze", return_value=analysis_result), patch(
            "app.api.chat.orchestrator.process_request",
            return_value={"response": "", "sources": [], "execution_path": "predefined", "metadata": {}},
        ) as mock_process, patch(
            "app.api.chat.rag_workflow.process_question", return_value=fallback_result
        ) as mock_rag:
            response = client.post("/chat/", json={"message": "Hello there"})

        assert response.status_code == 200
        data = response.json()
        assert data["response"] == "Fallback answer"
        assert data["execution_path"] == "rag_fallback"
        assert data["metadata"]["fallback_used"] is True
        assert data["metadata"]["fallback_strategy"] == "rag_workflow"
        assert data["metadata"]["analysis"]["model"] == "gpt-5-mini"
        assert mock_process.called
        assert mock_rag.called

    def test_chat_stream_endpoint_fallback_uses_rag(self):
        """Ensure streaming endpoint delivers fallback response when needed."""
        fallback_result = RAGResult(
            success=True,
            question="Hello",
            answer="Streamed fallback answer",
            confidence=0.72,
            tokens_used=18,
            processing_time_ms=88.0,
        )
        analysis_result = {
            "response": "Routing to orchestrator",
            "requires_followup": True,
            "metadata": {"model": "gpt-5-mini"},
        }
        fake_state = SimpleNamespace(
            values={
                "final_result": {"response": "", "metadata": {}},
                "execution_path": "predefined",
            }
        )

        with patch("app.api.chat.intent_agent.analyze", return_value=analysis_result), patch(
            "app.api.chat.execution_graph.stream", return_value=[]
        ), patch(
            "app.api.chat.execution_graph.get_state", return_value=fake_state
        ), patch(
            "app.api.chat.rag_workflow.process_question", return_value=fallback_result
        ) as mock_rag:
            with client.stream("POST", "/chat/stream", json={"message": "Hello"}) as stream_response:
                assert stream_response.status_code == 200
                payloads = []
                for line in stream_response.iter_lines():
                    if not line:
                        continue
                    if isinstance(line, bytes):
                        line = line.decode()
                    if line.startswith("data: "):
                        payloads.append(json.loads(line[6:]))

        assert len(payloads) >= 1
        final_event = payloads[-1]
        assert final_event["type"] == "response"
        assert final_event["response"] == "Streamed fallback answer"
        assert final_event["execution_path"] == "rag_fallback"
        assert final_event["metadata"]["fallback_used"] is True
        assert final_event["metadata"]["fallback_strategy"] == "rag_workflow"
        assert mock_rag.called


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
    
    @patch("app.api.chat.intent_agent.analyze")
    @patch("app.api.chat.orchestrator")
    def test_chat_endpoint_error(self, mock_orchestrator, mock_analyze):
        """Test chat endpoint handles errors gracefully."""
        mock_analyze.return_value = {
            "response": "Routing to orchestrator",
            "requires_followup": True,
            "metadata": {"model": "gpt-5-mini"},
        }
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
