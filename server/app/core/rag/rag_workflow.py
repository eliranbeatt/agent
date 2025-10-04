"""Main RAG workflow orchestration."""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime

from ..context.context_manager import ContextManager, RetrievalResult
from .question_processor import QuestionProcessor, ProcessedQuestion
from .answer_generator import AnswerGenerator, GeneratedAnswer
from .citation_manager import CitationManager, Citation
from .evaluator import Evaluator, VerificationResult

logger = logging.getLogger(__name__)


@dataclass
class RAGResult:
    """Complete result from RAG workflow."""
    success: bool
    question: str
    answer: str
    citations: List[Citation] = field(default_factory=list)
    confidence: float = 0.0
    tokens_used: int = 0
    processing_time_ms: float = 0.0
    verification: Optional[VerificationResult] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "question": self.question,
            "answer": self.answer,
            "citations": [c.to_dict() for c in self.citations],
            "confidence": self.confidence,
            "tokens_used": self.tokens_used,
            "processing_time_ms": self.processing_time_ms,
            "verification": self.verification.to_dict() if self.verification else None,
            "metadata": self.metadata,
            "error": self.error
        }


class RAGWorkflow:
    """
    Complete RAG workflow: ingest → chunk → embed → retrieve → answer → verify.
    
    This class orchestrates the entire RAG pipeline from question to answer.
    """
    
    def __init__(
        self,
        context_manager: Optional[ContextManager] = None,
        openai_api_key: Optional[str] = None,
        answer_model: str = "gpt-4o-mini",
        retrieval_k: int = 10,
        use_mmr: bool = True,
        enable_verification: bool = True
    ):
        """
        Initialize RAG workflow.
        
        Args:
            context_manager: Context manager instance (creates new if None)
            openai_api_key: OpenAI API key
            answer_model: Model to use for answer generation
            retrieval_k: Number of chunks to retrieve
            use_mmr: Whether to use MMR for retrieval
            enable_verification: Whether to enable answer verification
        """
        # Initialize components
        self.context_manager = context_manager or ContextManager(
            openai_api_key=openai_api_key
        )
        
        self.question_processor = QuestionProcessor()
        self.answer_generator = AnswerGenerator(
            openai_api_key=openai_api_key,
            model=answer_model
        )
        self.citation_manager = CitationManager()
        self.evaluator = Evaluator(
            openai_api_key=openai_api_key,
            model=answer_model
        )
        
        self.retrieval_k = retrieval_k
        self.use_mmr = use_mmr
        self.enable_verification = enable_verification
        
        logger.info("RAG workflow initialized")
    
    def process_question(
        self,
        question: str,
        source_filter: Optional[str] = None,
        page_filter: Optional[int] = None,
        custom_k: Optional[int] = None
    ) -> RAGResult:
        """
        Process a question through the complete RAG pipeline.
        
        Pipeline steps:
        1. Process question to extract metadata
        2. Retrieve relevant context chunks
        3. Create citations
        4. Generate answer with LLM
        5. Return complete result
        
        Args:
            question: User's question
            source_filter: Optional filter by source file
            page_filter: Optional filter by page number
            custom_k: Optional custom number of chunks to retrieve
            
        Returns:
            Complete RAG result with answer and citations
        """
        start_time = datetime.now()
        
        try:
            logger.info(f"Processing question: {question}")
            
            # Step 1: Process question
            processed_question = self.question_processor.process_question(question)
            
            # Step 2: Retrieve relevant context
            k = custom_k or processed_question.context_requirements.get("max_chunks", self.retrieval_k)
            
            retrieval_result = self.context_manager.retrieve_context(
                query=processed_question.processed_question,
                k=k,
                use_mmr=self.use_mmr,
                source_filter=source_filter,
                page_filter=page_filter
            )
            
            if not retrieval_result.success:
                return RAGResult(
                    success=False,
                    question=question,
                    answer="",
                    error=f"Context retrieval failed: {retrieval_result.error}"
                )
            
            if not retrieval_result.chunks:
                return RAGResult(
                    success=True,
                    question=question,
                    answer="I don't have enough information in the available documents to answer this question.",
                    confidence=0.0,
                    metadata={"reason": "no_relevant_chunks"}
                )
            
            # Step 3: Create citations
            citations = self.citation_manager.create_citations(
                retrieval_result.chunks,
                retrieval_result.scores
            )
            
            # Step 4: Generate answer
            generated_answer = self.answer_generator.generate_answer(
                question=question,
                chunks=retrieval_result.chunks,
                citations=citations,
                question_type=processed_question.question_type
            )
            
            # Step 5: Verify answer (if enabled)
            verification = None
            if self.enable_verification:
                verification = self.evaluator.verify_answer(
                    question=question,
                    answer=generated_answer.answer,
                    chunks=retrieval_result.chunks,
                    citations=citations
                )
                
                # Update confidence based on verification
                if verification.passed:
                    final_confidence = (generated_answer.confidence + verification.confidence) / 2
                else:
                    final_confidence = min(generated_answer.confidence, verification.confidence)
            else:
                final_confidence = generated_answer.confidence
            
            # Calculate processing time
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds() * 1000
            
            # Build result
            result = RAGResult(
                success=True,
                question=question,
                answer=generated_answer.answer,
                citations=citations,
                confidence=final_confidence,
                tokens_used=generated_answer.tokens_used,
                processing_time_ms=processing_time,
                verification=verification,
                metadata={
                    "question_type": processed_question.question_type,
                    "chunks_retrieved": len(retrieval_result.chunks),
                    "model": generated_answer.model,
                    "keywords": processed_question.keywords,
                    "entities": processed_question.entities,
                    "verification_enabled": self.enable_verification
                }
            )
            
            logger.info(
                f"RAG workflow completed: {len(citations)} citations, "
                f"{generated_answer.tokens_used} tokens, "
                f"{processing_time:.0f}ms, "
                f"verified={verification.passed if verification else 'N/A'}"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"RAG workflow failed: {e}")
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds() * 1000
            
            return RAGResult(
                success=False,
                question=question,
                answer="",
                processing_time_ms=processing_time,
                error=str(e)
            )
    
    def process_batch_questions(
        self,
        questions: List[str],
        source_filter: Optional[str] = None
    ) -> List[RAGResult]:
        """
        Process multiple questions in batch.
        
        Args:
            questions: List of questions
            source_filter: Optional filter by source file
            
        Returns:
            List of RAG results
        """
        results = []
        
        for i, question in enumerate(questions, 1):
            logger.info(f"Processing question {i}/{len(questions)}")
            result = self.process_question(question, source_filter=source_filter)
            results.append(result)
        
        successful = sum(1 for r in results if r.success)
        logger.info(f"Batch processing complete: {successful}/{len(questions)} successful")
        
        return results
    
    def ingest_and_query(
        self,
        file_path: str,
        question: str
    ) -> RAGResult:
        """
        Convenience method to ingest a file and immediately query it.
        
        Args:
            file_path: Path to file to ingest
            question: Question to ask about the file
            
        Returns:
            RAG result
        """
        logger.info(f"Ingesting file and querying: {file_path}")
        
        # Ingest file
        ingestion_result = self.context_manager.ingest_file(file_path)
        
        if not ingestion_result.success:
            return RAGResult(
                success=False,
                question=question,
                answer="",
                error=f"File ingestion failed: {ingestion_result.error}"
            )
        
        # Query the file
        return self.process_question(
            question=question,
            source_filter=file_path
        )
    
    def get_formatted_answer(
        self,
        result: RAGResult,
        include_citations: bool = True,
        citation_format: str = "markdown"
    ) -> str:
        """
        Get formatted answer with optional citations.
        
        Args:
            result: RAG result
            include_citations: Whether to include citations
            citation_format: Format for citations (markdown, html, text)
            
        Returns:
            Formatted answer string
        """
        if not result.success:
            return f"Error: {result.error}"
        
        output = result.answer
        
        if include_citations and result.citations:
            citations_text = self.citation_manager.format_citations(
                result.citations,
                format_type=citation_format
            )
            output += f"\n\n{citations_text}"
        
        return output
    
    def get_workflow_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the RAG workflow.
        
        Returns:
            Dictionary with workflow statistics
        """
        context_stats = self.context_manager.get_stats()
        
        return {
            "context_manager": context_stats,
            "answer_model": self.answer_generator.model,
            "retrieval_k": self.retrieval_k,
            "use_mmr": self.use_mmr,
            "enable_verification": self.enable_verification,
            "components": {
                "question_processor": True,
                "answer_generator": self.answer_generator.client is not None,
                "citation_manager": True,
                "evaluator": self.evaluator.client is not None
            }
        }
