"""Citation management for RAG answers."""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from ..context.chunker import Chunk

logger = logging.getLogger(__name__)


@dataclass
class Citation:
    """Citation linking answer content to source chunks."""
    chunk_id: str
    source_file: str
    page_number: Optional[int]
    content_snippet: str
    relevance_score: float
    start_char: int = 0
    end_char: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def score(self) -> float:
        """Alias for relevance_score for API compatibility."""
        return self.relevance_score
    
    @property
    def content(self) -> str:
        """Alias for content_snippet for API compatibility."""
        return self.content_snippet
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert citation to dictionary."""
        return {
            "chunk_id": self.chunk_id,
            "source_file": self.source_file,
            "page_number": self.page_number,
            "content_snippet": self.content_snippet,
            "content": self.content_snippet,  # Include both for compatibility
            "relevance_score": self.relevance_score,
            "score": self.relevance_score,  # Include both for compatibility
            "start_char": self.start_char,
            "end_char": self.end_char,
            "metadata": self.metadata
        }


class CitationManager:
    """Manages citations for RAG-generated answers."""
    
    def __init__(self, snippet_length: int = 150):
        """
        Initialize citation manager.
        
        Args:
            snippet_length: Maximum length of content snippets
        """
        self.snippet_length = snippet_length
    
    def create_citations(
        self,
        chunks: List[Chunk],
        scores: List[float]
    ) -> List[Citation]:
        """
        Create citations from retrieved chunks.
        
        Args:
            chunks: Retrieved chunks
            scores: Relevance scores for each chunk
            
        Returns:
            List of citations
        """
        citations = []
        
        for chunk, score in zip(chunks, scores):
            # Create content snippet
            snippet = self._create_snippet(chunk.content)
            
            citation = Citation(
                chunk_id=chunk.id,
                source_file=chunk.source_file,
                page_number=chunk.page_number,
                content_snippet=snippet,
                relevance_score=score,
                start_char=chunk.start_char,
                end_char=chunk.end_char,
                metadata=chunk.metadata or {}
            )
            
            citations.append(citation)
        
        logger.debug(f"Created {len(citations)} citations")
        return citations
    
    def _create_snippet(self, content: str) -> str:
        """
        Create a snippet from content.
        
        Args:
            content: Full content text
            
        Returns:
            Truncated snippet
        """
        if len(content) <= self.snippet_length:
            return content
        
        # Truncate and add ellipsis
        snippet = content[:self.snippet_length].strip()
        
        # Try to end at a word boundary
        last_space = snippet.rfind(' ')
        if last_space > self.snippet_length * 0.8:  # If we can find a space in the last 20%
            snippet = snippet[:last_space]
        
        return snippet + "..."
    
    def format_citations(
        self,
        citations: List[Citation],
        format_type: str = "markdown"
    ) -> str:
        """
        Format citations for display.
        
        Args:
            citations: List of citations to format
            format_type: Format type (markdown, html, text)
            
        Returns:
            Formatted citation string
        """
        if not citations:
            return ""
        
        if format_type == "markdown":
            return self._format_markdown(citations)
        elif format_type == "html":
            return self._format_html(citations)
        else:
            return self._format_text(citations)
    
    def _format_markdown(self, citations: List[Citation]) -> str:
        """Format citations as markdown."""
        lines = ["## Sources\n"]
        
        for i, citation in enumerate(citations, 1):
            page_info = f", Page {citation.page_number}" if citation.page_number else ""
            lines.append(
                f"{i}. **{citation.source_file}**{page_info} "
                f"(Relevance: {citation.relevance_score:.2f})\n"
                f"   > {citation.content_snippet}\n"
            )
        
        return "\n".join(lines)
    
    def _format_html(self, citations: List[Citation]) -> str:
        """Format citations as HTML."""
        lines = ["<div class='citations'><h3>Sources</h3><ol>"]
        
        for citation in citations:
            page_info = f", Page {citation.page_number}" if citation.page_number else ""
            lines.append(
                f"<li><strong>{citation.source_file}</strong>{page_info} "
                f"(Relevance: {citation.relevance_score:.2f})<br>"
                f"<blockquote>{citation.content_snippet}</blockquote></li>"
            )
        
        lines.append("</ol></div>")
        return "\n".join(lines)
    
    def _format_text(self, citations: List[Citation]) -> str:
        """Format citations as plain text."""
        lines = ["Sources:\n"]
        
        for i, citation in enumerate(citations, 1):
            page_info = f", Page {citation.page_number}" if citation.page_number else ""
            lines.append(
                f"{i}. {citation.source_file}{page_info} "
                f"(Relevance: {citation.relevance_score:.2f})\n"
                f"   \"{citation.content_snippet}\"\n"
            )
        
        return "\n".join(lines)
    
    def group_by_source(self, citations: List[Citation]) -> Dict[str, List[Citation]]:
        """
        Group citations by source file.
        
        Args:
            citations: List of citations
            
        Returns:
            Dictionary mapping source files to their citations
        """
        grouped = {}
        
        for citation in citations:
            if citation.source_file not in grouped:
                grouped[citation.source_file] = []
            grouped[citation.source_file].append(citation)
        
        return grouped
    
    def deduplicate_citations(self, citations: List[Citation]) -> List[Citation]:
        """
        Remove duplicate citations based on chunk ID.
        
        Args:
            citations: List of citations
            
        Returns:
            Deduplicated list of citations
        """
        seen_ids = set()
        unique_citations = []
        
        for citation in citations:
            if citation.chunk_id not in seen_ids:
                seen_ids.add(citation.chunk_id)
                unique_citations.append(citation)
        
        return unique_citations
    
    def track_citation_usage(
        self,
        answer: str,
        citations: List[Citation]
    ) -> Dict[str, Any]:
        """
        Track which citations were actually used in the answer.
        
        This analyzes the answer text to determine which source chunks
        were likely referenced.
        
        Args:
            answer: Generated answer text
            citations: Available citations
            
        Returns:
            Dictionary with citation usage statistics
        """
        # Simple heuristic: check if citation content appears in answer
        used_citations = []
        unused_citations = []
        
        for citation in citations:
            # Check if any significant phrase from the citation appears in the answer
            # Use first 50 chars as a representative phrase
            representative_phrase = citation.content_snippet[:50].lower()
            words = representative_phrase.split()
            
            # Check if at least 3 consecutive words appear in answer
            found = False
            for i in range(len(words) - 2):
                phrase = " ".join(words[i:i+3])
                if phrase in answer.lower():
                    found = True
                    break
            
            if found:
                used_citations.append(citation)
            else:
                unused_citations.append(citation)
        
        return {
            "total_citations": len(citations),
            "used_citations": len(used_citations),
            "unused_citations": len(unused_citations),
            "usage_rate": len(used_citations) / len(citations) if citations else 0.0,
            "used_citation_ids": [c.chunk_id for c in used_citations],
            "unused_citation_ids": [c.chunk_id for c in unused_citations]
        }
