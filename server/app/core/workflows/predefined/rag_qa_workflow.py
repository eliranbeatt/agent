"""RAG Question Answering predefined workflow."""

import logging
from typing import Dict, Any, Optional

from ...rag.rag_workflow import RAGWorkflow, RAGResult
from ...context.context_manager import ContextManager
from ...state import ExecutionState
from ..workflow_models import WorkflowExecutionContext


logger = logging.getLogger(__name__)


class RAGQAWorkflow:
    """
    Predefined workflow for RAG-based question answering.
    
    Pipeline:
    1. Context retrieval - Retrieve relevant document chunks
    2. Answer generation - Generate answer with LLM
    3. Answer verification - Verify quality and grounding
    """
    
    def __init__(
        self,
        context_manager: Optional[ContextManager] = None,
        openai_api_key: Optional[str] = None,
        retrieval_k: int = 10,
        use_mmr: bool = True
    ):
        """
        Initialize RAG QA workflow.
        
        Args:
            context_manager: Context manager instance
            openai_api_key: OpenAI API key
            retrieval_k: Number of chunks to retrieve
            use_mmr: Whether to use MMR for retrieval
        """
        self.rag_workflow = RAGWorkflow(
            context_manager=context_manager,
            openai_api_key=openai_api_key,
            retrieval_k=retrieval_k,
            use_mmr=use_mmr,
            enable_verification=True
        )
        
        logger.info("RAG QA workflow initialized")
    
    def execute(
        self,
        user_request: str,
        execution_state: ExecutionState,
        workflow_context: WorkflowExecutionContext,
        source_filter: Optional[str] = None,
        page_filter: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Execute the RAG QA workflow.
        
        Args:
            user_request: User's question
            execution_state: Global execution state
            workflow_context: Workflow execution context
            source_filter: Optional filter by source file
            page_filter: Optional filter by page number
            
        Returns:
            Workflow result dictionary
        """
        logger.info(f"Executing RAG QA workflow for question: {user_request}")
        
        # Process question through RAG pipeline
        rag_result = self.rag_workflow.process_question(
            question=user_request,
            source_filter=source_filter,
            page_filter=page_filter
        )
        
        # Update execution state
        if rag_result.success:
            execution_state.add_tokens_used(rag_result.tokens_used)
            
            # Store retrieved chunks in state
            if rag_result.metadata.get("chunks_retrieved"):
                execution_state.context["chunks_retrieved"] = rag_result.metadata["chunks_retrieved"]
        
        # Convert to workflow result format
        result = {
            "success": rag_result.success,
            "answer": rag_result.answer,
            "citations": [c.to_dict() for c in rag_result.citations],
            "confidence": rag_result.confidence,
            "tokens_used": rag_result.tokens_used,
            "processing_time_ms": rag_result.processing_time_ms,
            "verification": rag_result.verification.to_dict() if rag_result.verification else None,
            "metadata": rag_result.metadata,
            "error": rag_result.error
        }
        
        logger.info(
            f"RAG QA workflow completed: success={rag_result.success}, "
            f"citations={len(rag_result.citations)}, "
            f"confidence={rag_result.confidence:.2f}"
        )
        
        return result
    
    def get_formatted_answer(
        self,
        result: Dict[str, Any],
        include_citations: bool = True
    ) -> str:
        """
        Get formatted answer from workflow result.
        
        Args:
            result: Workflow result dictionary
            include_citations: Whether to include citations
            
        Returns:
            Formatted answer string
        """
        if not result.get("success"):
            return f"Error: {result.get('error', 'Unknown error')}"
        
        answer = result.get("answer", "")
        
        if include_citations and result.get("citations"):
            citations_text = "\n\nSources:\n"
            for i, citation in enumerate(result["citations"], 1):
                source = citation.get("source_file", "Unknown")
                page = citation.get("page_number")
                page_text = f", page {page}" if page else ""
                citations_text += f"[{i}] {source}{page_text}\n"
            
            answer += citations_text
        
        return answer
