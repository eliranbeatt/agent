"""Compare & Synthesize predefined workflow."""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

from ...context.context_manager import ContextManager
from ...state import ExecutionState
from ..workflow_models import WorkflowExecutionContext


logger = logging.getLogger(__name__)


class CompareSynthesizeWorkflow:
    """
    Predefined workflow for multi-document comparison and synthesis.
    
    Pipeline:
    1. Multi-doc processing - Process and index all documents
    2. Comparative analysis - Identify similarities and differences
    3. Synthesis generation - Synthesize insights across documents
    """
    
    def __init__(
        self,
        context_manager: Optional[ContextManager] = None,
        openai_api_key: Optional[str] = None
    ):
        """
        Initialize Compare & Synthesize workflow.
        
        Args:
            context_manager: Context manager instance
            openai_api_key: OpenAI API key
        """
        self.context_manager = context_manager
        self.openai_api_key = openai_api_key
        
        logger.info("Compare & Synthesize workflow initialized")
    
    def execute(
        self,
        user_request: str,
        execution_state: ExecutionState,
        workflow_context: WorkflowExecutionContext,
        source_files: Optional[List[str]] = None,
        comparison_topic: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute the Compare & Synthesize workflow.
        
        Args:
            user_request: User's request
            execution_state: Global execution state
            workflow_context: Workflow execution context
            source_files: Optional list of source files to compare
            comparison_topic: Optional specific topic to compare
            
        Returns:
            Workflow result dictionary
        """
        logger.info(f"Executing Compare & Synthesize workflow: {user_request}")
        
        start_time = datetime.now()
        
        try:
            # Step 1: Get chunks from all documents
            doc_chunks = self._get_multi_doc_chunks(source_files)
            
            if not doc_chunks or len(doc_chunks) < 2:
                return {
                    "success": False,
                    "error": "Need at least 2 documents for comparison"
                }
            
            # Step 2: Perform comparative analysis
            comparison = self._compare_documents(doc_chunks, comparison_topic or user_request)
            
            # Step 3: Generate synthesis
            synthesis = self._synthesize_insights(doc_chunks, comparison)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            result = {
                "success": True,
                "comparison": comparison,
                "synthesis": synthesis,
                "documents_compared": len(doc_chunks),
                "similarities": comparison.get("similarities", []),
                "differences": comparison.get("differences", []),
                "conflicts": comparison.get("conflicts", []),
                "processing_time_ms": processing_time,
                "tokens_used": 0  # Placeholder
            }
            
            logger.info(
                f"Compare & Synthesize completed: "
                f"{len(doc_chunks)} documents, "
                f"{len(comparison.get('similarities', []))} similarities, "
                f"{len(comparison.get('differences', []))} differences"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Compare & Synthesize workflow failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _get_multi_doc_chunks(
        self,
        source_files: Optional[List[str]] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get chunks organized by source document.
        
        Args:
            source_files: Optional list of source files
            
        Returns:
            Dictionary mapping source file to chunks
        """
        if not self.context_manager:
            logger.warning("Context manager not available")
            return {}
        
        # Retrieve chunks
        retrieval_result = self.context_manager.retrieve_context(
            query="",  # Empty query to get all chunks
            k=100,
            use_mmr=False
        )
        
        if not retrieval_result.success:
            return {}
        
        # Organize by source
        doc_chunks = {}
        for chunk in retrieval_result.chunks:
            source = chunk.get("source_file", "Unknown")
            
            # Filter by source files if specified
            if source_files and source not in source_files:
                continue
            
            if source not in doc_chunks:
                doc_chunks[source] = []
            doc_chunks[source].append(chunk)
        
        return doc_chunks
    
    def _compare_documents(
        self,
        doc_chunks: Dict[str, List[Dict[str, Any]]],
        topic: str
    ) -> Dict[str, Any]:
        """
        Compare documents to find similarities and differences.
        
        Args:
            doc_chunks: Chunks organized by document
            topic: Topic to focus comparison on
            
        Returns:
            Comparison results
        """
        # Simple comparison implementation
        # In production, this would use LLM for deeper analysis
        
        similarities = []
        differences = []
        conflicts = []
        
        # Extract key terms from each document
        doc_terms = {}
        for source, chunks in doc_chunks.items():
            terms = set()
            for chunk in chunks:
                content = chunk.get("content", "").lower()
                words = content.split()
                # Get significant words (length > 5)
                terms.update(word for word in words if len(word) > 5)
            doc_terms[source] = terms
        
        # Find common terms (similarities)
        if len(doc_terms) >= 2:
            sources = list(doc_terms.keys())
            common_terms = doc_terms[sources[0]].copy()
            for source in sources[1:]:
                common_terms &= doc_terms[source]
            
            if common_terms:
                similarities.append({
                    "type": "common_topics",
                    "description": f"Documents share {len(common_terms)} common topics",
                    "examples": list(common_terms)[:10]
                })
        
        # Find unique terms (differences)
        for source, terms in doc_terms.items():
            unique_terms = terms.copy()
            for other_source, other_terms in doc_terms.items():
                if other_source != source:
                    unique_terms -= other_terms
            
            if unique_terms:
                differences.append({
                    "source": source,
                    "type": "unique_topics",
                    "description": f"{source} has {len(unique_terms)} unique topics",
                    "examples": list(unique_terms)[:10]
                })
        
        # Placeholder for conflicts
        # In production, would detect contradictory statements
        conflicts.append({
            "type": "potential_conflict",
            "description": "Detailed conflict detection requires LLM analysis",
            "sources": list(doc_chunks.keys())
        })
        
        return {
            "similarities": similarities,
            "differences": differences,
            "conflicts": conflicts,
            "documents": list(doc_chunks.keys())
        }
    
    def _synthesize_insights(
        self,
        doc_chunks: Dict[str, List[Dict[str, Any]]],
        comparison: Dict[str, Any]
    ) -> str:
        """
        Synthesize insights across documents.
        
        Args:
            doc_chunks: Chunks organized by document
            comparison: Comparison results
            
        Returns:
            Synthesis text
        """
        # Simple synthesis generation
        # In production, this would use LLM
        
        synthesis = "Cross-Document Synthesis:\n\n"
        
        # Summarize documents
        synthesis += f"Analyzed {len(doc_chunks)} documents:\n"
        for source in doc_chunks.keys():
            synthesis += f"- {source}\n"
        
        # Summarize similarities
        similarities = comparison.get("similarities", [])
        if similarities:
            synthesis += f"\nCommon Themes ({len(similarities)}):\n"
            for sim in similarities[:5]:
                synthesis += f"- {sim.get('description', '')}\n"
        
        # Summarize differences
        differences = comparison.get("differences", [])
        if differences:
            synthesis += f"\nKey Differences ({len(differences)}):\n"
            for diff in differences[:5]:
                synthesis += f"- {diff.get('description', '')} ({diff.get('source', '')})\n"
        
        # Note conflicts
        conflicts = comparison.get("conflicts", [])
        if conflicts:
            synthesis += f"\nPotential Conflicts: {len(conflicts)} identified\n"
        
        synthesis += "\nNote: This is a preliminary synthesis. "
        synthesis += "Detailed analysis would require LLM-based comparison."
        
        return synthesis
    
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
        
        output = result.get("synthesis", "")
        
        # Add detailed comparison
        comparison = result.get("comparison", {})
        
        similarities = comparison.get("similarities", [])
        if similarities:
            output += "\n\nDetailed Similarities:\n"
            for i, sim in enumerate(similarities, 1):
                output += f"{i}. {sim.get('description', '')}\n"
                examples = sim.get("examples", [])
                if examples:
                    output += f"   Examples: {', '.join(examples[:5])}\n"
        
        differences = comparison.get("differences", [])
        if differences:
            output += "\n\nDetailed Differences:\n"
            for i, diff in enumerate(differences, 1):
                output += f"{i}. {diff.get('description', '')} ({diff.get('source', '')})\n"
        
        return output
