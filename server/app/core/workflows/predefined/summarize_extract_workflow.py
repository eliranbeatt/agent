"""Summarize & Extract predefined workflow."""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

from ...context.context_manager import ContextManager
from ...state import ExecutionState
from ..workflow_models import WorkflowExecutionContext


logger = logging.getLogger(__name__)


class SummarizeExtractWorkflow:
    """
    Predefined workflow for document summarization and fact extraction.
    
    Pipeline:
    1. Document chunking - Process and chunk the entire document
    2. Content analysis - Analyze chunks for key information
    3. Summary generation - Generate comprehensive summary
    4. Fact extraction - Extract and structure key facts
    """
    
    def __init__(
        self,
        context_manager: Optional[ContextManager] = None,
        openai_api_key: Optional[str] = None
    ):
        """
        Initialize Summarize & Extract workflow.
        
        Args:
            context_manager: Context manager instance
            openai_api_key: OpenAI API key
        """
        self.context_manager = context_manager
        self.openai_api_key = openai_api_key
        
        logger.info("Summarize & Extract workflow initialized")
    
    def execute(
        self,
        user_request: str,
        execution_state: ExecutionState,
        workflow_context: WorkflowExecutionContext,
        source_filter: Optional[str] = None,
        summary_length: str = "medium"
    ) -> Dict[str, Any]:
        """
        Execute the Summarize & Extract workflow.
        
        Args:
            user_request: User's request
            execution_state: Global execution state
            workflow_context: Workflow execution context
            source_filter: Optional filter by source file
            summary_length: Length of summary (short, medium, long)
            
        Returns:
            Workflow result dictionary
        """
        logger.info(f"Executing Summarize & Extract workflow: {user_request}")
        
        start_time = datetime.now()
        
        try:
            # Step 1: Get all chunks from the document
            chunks = self._get_document_chunks(source_filter)
            
            if not chunks:
                return {
                    "success": False,
                    "error": "No document chunks available for summarization"
                }
            
            # Step 2: Analyze content
            analysis = self._analyze_content(chunks)
            
            # Step 3: Generate summary
            summary = self._generate_summary(chunks, analysis, summary_length)
            
            # Step 4: Extract facts
            facts = self._extract_facts(chunks, analysis)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            result = {
                "success": True,
                "summary": summary,
                "key_points": analysis.get("key_points", []),
                "facts": facts,
                "entities": analysis.get("entities", []),
                "themes": analysis.get("themes", []),
                "statistics": {
                    "total_chunks": len(chunks),
                    "facts_extracted": len(facts),
                    "entities_found": len(analysis.get("entities", []))
                },
                "processing_time_ms": processing_time,
                "tokens_used": 0  # Placeholder
            }
            
            logger.info(
                f"Summarize & Extract completed: "
                f"{len(facts)} facts, {len(analysis.get('entities', []))} entities"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Summarize & Extract workflow failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _get_document_chunks(self, source_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all chunks from the document."""
        if not self.context_manager:
            logger.warning("Context manager not available")
            return []
        
        # Retrieve all chunks (using high k value)
        retrieval_result = self.context_manager.retrieve_context(
            query="",  # Empty query to get all chunks
            k=100,
            use_mmr=False,
            source_filter=source_filter
        )
        
        if retrieval_result.success:
            return retrieval_result.chunks
        
        return []
    
    def _analyze_content(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze chunks for key information.
        
        Args:
            chunks: List of document chunks
            
        Returns:
            Analysis results
        """
        # Simple analysis implementation
        # In production, this would use LLM for deeper analysis
        
        entities = set()
        key_points = []
        themes = set()
        
        # Extract entities (capitalized words)
        for chunk in chunks:
            content = chunk.get("content", "")
            words = content.split()
            
            for word in words:
                cleaned = word.strip(".,!?;:")
                if cleaned and len(cleaned) > 1 and cleaned[0].isupper():
                    entities.add(cleaned)
        
        # Identify key points (first sentence of each chunk)
        for chunk in chunks[:10]:  # Limit to first 10 chunks
            content = chunk.get("content", "")
            sentences = content.split(".")
            if sentences:
                key_point = sentences[0].strip()
                if key_point and len(key_point) > 20:
                    key_points.append(key_point)
        
        # Identify themes (common words)
        word_freq = {}
        for chunk in chunks:
            content = chunk.get("content", "").lower()
            words = content.split()
            for word in words:
                if len(word) > 5:  # Only longer words
                    word_freq[word] = word_freq.get(word, 0) + 1
        
        # Top themes
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        themes = [word for word, freq in sorted_words[:10] if freq > 2]
        
        return {
            "entities": list(entities)[:50],  # Limit to 50
            "key_points": key_points,
            "themes": themes
        }
    
    def _generate_summary(
        self,
        chunks: List[Dict[str, Any]],
        analysis: Dict[str, Any],
        length: str
    ) -> str:
        """
        Generate summary from chunks and analysis.
        
        Args:
            chunks: Document chunks
            analysis: Content analysis
            length: Summary length
            
        Returns:
            Summary text
        """
        # Simple summary generation
        # In production, this would use LLM
        
        num_points = {"short": 3, "medium": 5, "long": 10}.get(length, 5)
        key_points = analysis.get("key_points", [])[:num_points]
        
        summary = "Document Summary:\n\n"
        
        if key_points:
            summary += "Key Points:\n"
            for i, point in enumerate(key_points, 1):
                summary += f"{i}. {point}\n"
        
        themes = analysis.get("themes", [])[:5]
        if themes:
            summary += f"\nMain Themes: {', '.join(themes)}\n"
        
        entities = analysis.get("entities", [])[:10]
        if entities:
            summary += f"\nKey Entities: {', '.join(entities)}\n"
        
        summary += f"\nDocument contains {len(chunks)} sections."
        
        return summary
    
    def _extract_facts(
        self,
        chunks: List[Dict[str, Any]],
        analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Extract facts from chunks.
        
        Args:
            chunks: Document chunks
            analysis: Content analysis
            
        Returns:
            List of extracted facts
        """
        facts = []
        
        # Simple fact extraction
        # In production, this would use LLM with structured output
        
        for i, chunk in enumerate(chunks[:20]):  # Limit to first 20 chunks
            content = chunk.get("content", "")
            
            # Look for sentences with numbers (potential facts)
            sentences = content.split(".")
            for sentence in sentences:
                if any(char.isdigit() for char in sentence):
                    fact = {
                        "content": sentence.strip(),
                        "type": "numerical",
                        "source": chunk.get("source_file", "Unknown"),
                        "page": chunk.get("page_number"),
                        "confidence": 0.7
                    }
                    facts.append(fact)
                    
                    if len(facts) >= 20:  # Limit total facts
                        break
            
            if len(facts) >= 20:
                break
        
        return facts
    
    def get_formatted_output(self, result: Dict[str, Any]) -> str:
        """
        Get formatted output from workflow result.
        
        Args:
            result: Workflow result dictionary
            
        Returns:
            Formatted output string
        """
        if not result.get("success"):
            return f"Error: {result.get('error', 'Unknown error')}"
        
        output = result.get("summary", "")
        
        facts = result.get("facts", [])
        if facts:
            output += "\n\nExtracted Facts:\n"
            for i, fact in enumerate(facts[:10], 1):
                output += f"{i}. {fact.get('content', '')}\n"
        
        return output
