"""Integration tests for predefined workflows."""

import pytest
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime

from app.core.workflows.workflow_executor import WorkflowExecutor
from app.core.workflows.workflow_matcher import WorkflowMatcher
from app.core.workflows.predefined import (
    RAGQAWorkflow,
    SummarizeExtractWorkflow,
    CompareSynthesizeWorkflow,
    ImageOCRQAWorkflow
)
from app.core.state import ExecutionState, ExecutionPath
from app.core.workflows.workflow_models import WorkflowExecutionContext
from app.config.models import WorkflowConfig, WorkflowStep


class TestWorkflowMatcher:
    """Test workflow matching algorithm."""
    
    def test_match_rag_qa_workflow(self):
        """Test matching RAG QA workflow."""
        # Create workflow configs
        workflows = {
            "rag_qa": WorkflowConfig(
                name="RAG QA",
                description="Answer questions based on documents",
                triggers=["question about document", "what does the document say"],
                steps=[WorkflowStep(name="test", description="test", node_type="test")],
                confidence_threshold=0.7,
                enabled=True
            )
        }
        
        matcher = WorkflowMatcher(workflows)
        
        # Test matching
        workflow_name, confidence = matcher.match_workflow(
            "What does the document say about climate change?"
        )
        
        assert workflow_name == "rag_qa"
        assert confidence > 0.5  # Should match well with exact trigger phrase
    
    def test_match_summarize_workflow(self):
        """Test matching summarize workflow."""
        workflows = {
            "summarize": WorkflowConfig(
                name="Summarize",
                description="Summarize documents",
                triggers=["summarize", "summary", "key points"],
                steps=[WorkflowStep(name="test", description="test", node_type="test")],
                confidence_threshold=0.7,
                enabled=True
            )
        }
        
        matcher = WorkflowMatcher(workflows)
        
        workflow_name, confidence = matcher.match_workflow(
            "Please summarize this document"
        )
        
        assert workflow_name == "summarize"
        assert confidence > 0.5
    
    def test_match_compare_workflow(self):
        """Test matching compare workflow."""
        workflows = {
            "compare": WorkflowConfig(
                name="Compare",
                description="Compare multiple documents",
                triggers=["compare", "difference", "versus"],
                steps=[WorkflowStep(name="test", description="test", node_type="test")],
                confidence_threshold=0.7,
                enabled=True
            )
        }
        
        matcher = WorkflowMatcher(workflows)
        
        workflow_name, confidence = matcher.match_workflow(
            "Compare these two documents"
        )
        
        assert workflow_name == "compare"
        assert confidence > 0.5
    
    def test_no_match_low_confidence(self):
        """Test no match when confidence is low."""
        workflows = {
            "rag_qa": WorkflowConfig(
                name="RAG QA",
                description="Answer questions",
                triggers=["specific trigger phrase"],
                steps=[WorkflowStep(name="test", description="test", node_type="test")],
                confidence_threshold=0.9,
                enabled=True
            )
        }
        
        matcher = WorkflowMatcher(workflows)
        
        workflow_name, confidence = matcher.match_workflow(
            "Completely unrelated request"
        )
        
        # Should still return best match but with low confidence
        assert confidence < 0.5
    
    def test_disabled_workflow_not_matched(self):
        """Test that disabled workflows are not matched."""
        workflows = {
            "disabled_workflow": WorkflowConfig(
                name="Disabled",
                description="This is disabled",
                triggers=["disabled trigger"],
                steps=[WorkflowStep(name="test", description="test", node_type="test")],
                confidence_threshold=0.7,
                enabled=False
            ),
            "enabled_workflow": WorkflowConfig(
                name="Enabled",
                description="This is enabled",
                triggers=["enabled trigger"],
                steps=[WorkflowStep(name="test", description="test", node_type="test")],
                confidence_threshold=0.7,
                enabled=True
            )
        }
        
        matcher = WorkflowMatcher(workflows)
        
        workflow_name, confidence = matcher.match_workflow(
            "disabled trigger"
        )
        
        # Should not match disabled workflow
        assert workflow_name != "disabled_workflow"
    
    def test_context_aware_matching(self):
        """Test context-aware workflow matching."""
        workflows = {
            "rag_qa": WorkflowConfig(
                name="RAG QA",
                description="Answer questions about documents",
                triggers=["question", "what"],
                steps=[WorkflowStep(name="test", description="test", node_type="test")],
                confidence_threshold=0.7,
                enabled=True
            )
        }
        
        matcher = WorkflowMatcher(workflows)
        
        # With file context
        workflow_name, confidence_with_context = matcher.match_workflow(
            "What is in the document?",
            context={"has_uploaded_files": True}
        )
        
        # Without file context
        _, confidence_without_context = matcher.match_workflow(
            "What is in the document?",
            context={"has_uploaded_files": False}
        )
        
        # Confidence should be higher with relevant context
        assert confidence_with_context >= confidence_without_context
    
    def test_get_workflow_suggestions(self):
        """Test getting multiple workflow suggestions."""
        workflows = {
            "rag_qa": WorkflowConfig(
                name="RAG QA",
                description="Answer questions",
                triggers=["question", "what"],
                steps=[WorkflowStep(name="test", description="test", node_type="test")],
                confidence_threshold=0.7,
                enabled=True
            ),
            "summarize": WorkflowConfig(
                name="Summarize",
                description="Summarize documents",
                triggers=["summarize", "summary"],
                steps=[WorkflowStep(name="test", description="test", node_type="test")],
                confidence_threshold=0.7,
                enabled=True
            )
        }
        
        matcher = WorkflowMatcher(workflows)
        
        suggestions = matcher.get_workflow_suggestions(
            "What is the summary of this document?",
            top_k=2,
            min_confidence=0.1
        )
        
        assert len(suggestions) <= 2
        assert all(isinstance(s, tuple) and len(s) == 2 for s in suggestions)
        # Should be sorted by confidence
        if len(suggestions) > 1:
            assert suggestions[0][1] >= suggestions[1][1]


class TestRAGQAWorkflow:
    """Test RAG QA workflow."""
    
    def test_rag_qa_workflow_execution(self):
        """Test RAG QA workflow execution."""
        # Mock RAG workflow
        mock_rag = Mock()
        mock_rag.process_question.return_value = Mock(
            success=True,
            answer="This is the answer",
            citations=[],
            confidence=0.85,
            tokens_used=100,
            processing_time_ms=500,
            verification=None,
            metadata={"chunks_retrieved": 5}
        )
        
        # Create workflow
        workflow = RAGQAWorkflow()
        workflow.rag_workflow = mock_rag
        
        # Create execution state
        state = ExecutionState(user_request="What is the answer?")
        context = WorkflowExecutionContext(
            workflow_name="RAG QA",
            workflow_description="Test",
            user_request="What is the answer?"
        )
        
        # Execute
        result = workflow.execute(
            user_request="What is the answer?",
            execution_state=state,
            workflow_context=context
        )
        
        assert result["success"] is True
        assert result["answer"] == "This is the answer"
        assert result["confidence"] == 0.85
        assert result["tokens_used"] == 100
        assert state.tokens_used == 100
    
    def test_rag_qa_workflow_failure(self):
        """Test RAG QA workflow handles failures."""
        # Mock RAG workflow with failure
        mock_rag = Mock()
        mock_rag.process_question.return_value = Mock(
            success=False,
            answer="",
            citations=[],
            confidence=0.0,
            tokens_used=0,
            processing_time_ms=100,
            verification=None,
            metadata={},
            error="Retrieval failed"
        )
        
        workflow = RAGQAWorkflow()
        workflow.rag_workflow = mock_rag
        
        state = ExecutionState(user_request="Test question")
        context = WorkflowExecutionContext(
            workflow_name="RAG QA",
            workflow_description="Test",
            user_request="Test question"
        )
        
        result = workflow.execute(
            user_request="Test question",
            execution_state=state,
            workflow_context=context
        )
        
        assert result["success"] is False
        assert result["error"] == "Retrieval failed"


class TestSummarizeExtractWorkflow:
    """Test Summarize & Extract workflow."""
    
    def test_summarize_extract_execution(self):
        """Test summarize & extract workflow execution."""
        # Mock context manager
        mock_context = Mock()
        mock_context.retrieve_context.return_value = Mock(
            success=True,
            chunks=[
                {"content": "This is chunk 1", "source_file": "test.pdf"},
                {"content": "This is chunk 2", "source_file": "test.pdf"}
            ]
        )
        
        workflow = SummarizeExtractWorkflow(context_manager=mock_context)
        
        state = ExecutionState(user_request="Summarize this document")
        context = WorkflowExecutionContext(
            workflow_name="Summarize",
            workflow_description="Test",
            user_request="Summarize this document"
        )
        
        result = workflow.execute(
            user_request="Summarize this document",
            execution_state=state,
            workflow_context=context
        )
        
        assert result["success"] is True
        assert "summary" in result
        assert "facts" in result
        assert "key_points" in result
    
    def test_summarize_extract_no_chunks(self):
        """Test summarize & extract with no chunks."""
        mock_context = Mock()
        mock_context.retrieve_context.return_value = Mock(
            success=True,
            chunks=[]
        )
        
        workflow = SummarizeExtractWorkflow(context_manager=mock_context)
        
        state = ExecutionState(user_request="Summarize")
        context = WorkflowExecutionContext(
            workflow_name="Summarize",
            workflow_description="Test",
            user_request="Summarize"
        )
        
        result = workflow.execute(
            user_request="Summarize",
            execution_state=state,
            workflow_context=context
        )
        
        assert result["success"] is False
        assert "error" in result


class TestCompareSynthesizeWorkflow:
    """Test Compare & Synthesize workflow."""
    
    def test_compare_synthesize_execution(self):
        """Test compare & synthesize workflow execution."""
        # Mock context manager with multiple documents
        mock_context = Mock()
        mock_context.retrieve_context.return_value = Mock(
            success=True,
            chunks=[
                {"content": "Doc 1 content", "source_file": "doc1.pdf"},
                {"content": "Doc 2 content", "source_file": "doc2.pdf"}
            ]
        )
        
        workflow = CompareSynthesizeWorkflow(context_manager=mock_context)
        
        state = ExecutionState(user_request="Compare these documents")
        context = WorkflowExecutionContext(
            workflow_name="Compare",
            workflow_description="Test",
            user_request="Compare these documents"
        )
        
        result = workflow.execute(
            user_request="Compare these documents",
            execution_state=state,
            workflow_context=context
        )
        
        assert result["success"] is True
        assert "comparison" in result
        assert "synthesis" in result
        assert "similarities" in result
        assert "differences" in result
    
    def test_compare_synthesize_insufficient_docs(self):
        """Test compare & synthesize with insufficient documents."""
        mock_context = Mock()
        mock_context.retrieve_context.return_value = Mock(
            success=True,
            chunks=[
                {"content": "Only one doc", "source_file": "doc1.pdf"}
            ]
        )
        
        workflow = CompareSynthesizeWorkflow(context_manager=mock_context)
        
        state = ExecutionState(user_request="Compare")
        context = WorkflowExecutionContext(
            workflow_name="Compare",
            workflow_description="Test",
            user_request="Compare"
        )
        
        result = workflow.execute(
            user_request="Compare",
            execution_state=state,
            workflow_context=context
        )
        
        assert result["success"] is False
        assert "at least 2 documents" in result["error"]


class TestImageOCRQAWorkflow:
    """Test Image OCR → QA workflow."""
    
    def test_image_ocr_qa_execution(self):
        """Test image OCR → QA workflow execution."""
        workflow = ImageOCRQAWorkflow()
        
        state = ExecutionState(user_request="What does this image say?")
        context = WorkflowExecutionContext(
            workflow_name="Image OCR QA",
            workflow_description="Test",
            user_request="What does this image say?"
        )
        
        result = workflow.execute(
            user_request="What does this image say?",
            execution_state=state,
            workflow_context=context,
            image_path="test_image.png",
            question="What does this image say?"
        )
        
        assert result["success"] is True
        assert "extracted_text" in result
        assert "ocr_confidence" in result
    
    def test_image_ocr_qa_no_image(self):
        """Test image OCR → QA without image."""
        workflow = ImageOCRQAWorkflow()
        
        state = ExecutionState(user_request="Read image")
        context = WorkflowExecutionContext(
            workflow_name="Image OCR QA",
            workflow_description="Test",
            user_request="Read image"
        )
        
        result = workflow.execute(
            user_request="Read image",
            execution_state=state,
            workflow_context=context,
            image_path=None
        )
        
        assert result["success"] is False
        assert "error" in result


class TestWorkflowExecutor:
    """Test workflow executor."""
    
    def test_workflow_executor_initialization(self):
        """Test workflow executor initialization."""
        workflow_config = WorkflowConfig(
            name="Test Workflow",
            description="Test",
            steps=[
                WorkflowStep(
                    name="step1",
                    description="Test step",
                    node_type="context_manager",
                    parameters={},
                    success_criteria=[]
                )
            ],
            confidence_threshold=0.7,
            enabled=True
        )
        
        executor = WorkflowExecutor(workflow_config)
        
        assert executor.workflow_config == workflow_config
        assert executor.step_executor is not None
    
    def test_workflow_execution_success(self):
        """Test successful workflow execution."""
        workflow_config = WorkflowConfig(
            name="Test Workflow",
            description="Test",
            steps=[
                WorkflowStep(
                    name="test_step",
                    description="Test step",
                    node_type="context_manager",
                    parameters={"retrieval_k": 5},
                    success_criteria=["Retrieved at least 3 chunks"]
                )
            ],
            success_criteria=["Workflow completed"],
            confidence_threshold=0.7,
            enabled=True
        )
        
        # Mock context manager - retrieve_context returns chunks directly
        mock_context = Mock()
        mock_context.retrieve_context.return_value = [{"content": "test"}] * 5
        
        executor = WorkflowExecutor(
            workflow_config,
            context_manager=mock_context
        )
        
        state = ExecutionState(user_request="Test request")
        
        result = executor.execute(
            user_request="Test request",
            execution_state=state
        )
        
        assert result.is_complete is True
        assert result.is_failed is False
        assert len(result.step_results) == 1
    
    def test_workflow_execution_step_failure(self):
        """Test workflow execution with step failure."""
        workflow_config = WorkflowConfig(
            name="Test Workflow",
            description="Test",
            steps=[
                WorkflowStep(
                    name="failing_step",
                    description="This will fail",
                    node_type="unknown_type",
                    parameters={},
                    success_criteria=[]
                )
            ],
            confidence_threshold=0.7,
            enabled=True
        )
        
        executor = WorkflowExecutor(workflow_config)
        state = ExecutionState(user_request="Test")
        
        result = executor.execute(
            user_request="Test",
            execution_state=state
        )
        
        assert result.is_failed is True
        assert result.error_message is not None
    
    def test_workflow_success_criteria_checking(self):
        """Test workflow success criteria checking."""
        workflow_config = WorkflowConfig(
            name="Test Workflow",
            description="Test",
            steps=[
                WorkflowStep(
                    name="step1",
                    description="Test",
                    node_type="context_manager",
                    parameters={"retrieval_k": 5},
                    success_criteria=[]
                )
            ],
            success_criteria=["Answer provided", "Citations included"],
            confidence_threshold=0.7,
            enabled=True
        )
        
        executor = WorkflowExecutor(workflow_config)
        
        # Create mock context with answer in step results
        context = WorkflowExecutionContext(
            workflow_name="Test",
            workflow_description="Test",
            user_request="Test"
        )
        
        # Add step result with answer and citations
        from app.core.workflows.workflow_models import WorkflowStepResult, WorkflowStepStatus
        step_result = WorkflowStepResult(
            step_name="step1",
            status=WorkflowStepStatus.COMPLETED,
            output={
                "answer": "Test answer",
                "citations": ["source1"]
            }
        )
        context.step_results.append(step_result)
        
        # Check success criteria
        success = executor._check_workflow_success(context)
        
        # Should pass since we have answer and citations in step results
        assert success is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
