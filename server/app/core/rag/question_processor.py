"""Question processing for RAG pipeline."""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class ProcessedQuestion:
    """Processed question with metadata and context requirements."""
    original_question: str
    processed_question: str
    question_type: str  # factual, analytical, comparative, etc.
    keywords: List[str] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)
    context_requirements: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class QuestionProcessor:
    """Processes questions to optimize retrieval and answer generation."""
    
    def __init__(self):
        """Initialize question processor."""
        self.question_types = {
            "what": "factual",
            "who": "factual",
            "when": "factual",
            "where": "factual",
            "why": "analytical",
            "how": "analytical",
            "compare": "comparative",
            "difference": "comparative",
            "summarize": "summary",
            "explain": "analytical"
        }
    
    def process_question(
        self,
        question: str,
        context: Optional[Dict[str, Any]] = None
    ) -> ProcessedQuestion:
        """
        Process a question to extract metadata and optimize for retrieval.
        
        Args:
            question: Raw question text
            context: Optional context information
            
        Returns:
            Processed question with metadata
        """
        logger.debug(f"Processing question: {question}")
        
        # Clean and normalize question
        processed = self._normalize_question(question)
        
        # Identify question type
        q_type = self._identify_question_type(processed)
        
        # Extract keywords (simple implementation)
        keywords = self._extract_keywords(processed)
        
        # Extract entities (simple implementation)
        entities = self._extract_entities(processed)
        
        # Determine context requirements
        context_reqs = self._determine_context_requirements(q_type, processed)
        
        result = ProcessedQuestion(
            original_question=question,
            processed_question=processed,
            question_type=q_type,
            keywords=keywords,
            entities=entities,
            context_requirements=context_reqs,
            metadata=context or {}
        )
        
        logger.debug(f"Question type: {q_type}, Keywords: {keywords}")
        return result
    
    def _normalize_question(self, question: str) -> str:
        """
        Normalize question text.
        
        Args:
            question: Raw question
            
        Returns:
            Normalized question
        """
        # Remove extra whitespace
        normalized = " ".join(question.split())
        
        # Ensure question ends with question mark if it looks like a question
        if normalized and not normalized.endswith("?") and any(
            normalized.lower().startswith(word) 
            for word in ["what", "who", "when", "where", "why", "how", "is", "are", "can", "do", "does"]
        ):
            normalized += "?"
        
        return normalized
    
    def _identify_question_type(self, question: str) -> str:
        """
        Identify the type of question.
        
        Args:
            question: Processed question
            
        Returns:
            Question type
        """
        question_lower = question.lower()
        
        # Check for question type keywords
        for keyword, q_type in self.question_types.items():
            if keyword in question_lower:
                return q_type
        
        # Default to factual
        return "factual"
    
    def _extract_keywords(self, question: str) -> List[str]:
        """
        Extract keywords from question (simple implementation).
        
        Args:
            question: Processed question
            
        Returns:
            List of keywords
        """
        # Simple keyword extraction - remove common words
        stop_words = {
            "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "should",
            "could", "may", "might", "can", "what", "who", "when", "where", "why",
            "how", "which", "this", "that", "these", "those", "i", "you", "he",
            "she", "it", "we", "they", "me", "him", "her", "us", "them", "my",
            "your", "his", "its", "our", "their", "in", "on", "at", "to", "for",
            "of", "with", "from", "by", "about", "as", "into", "through", "during",
            "before", "after", "above", "below", "between", "under", "again",
            "further", "then", "once", "here", "there", "all", "both", "each",
            "few", "more", "most", "other", "some", "such", "no", "nor", "not",
            "only", "own", "same", "so", "than", "too", "very", "?"
        }
        
        # Tokenize and filter
        words = question.lower().split()
        keywords = [
            word.strip(".,!?;:") 
            for word in words 
            if word.strip(".,!?;:") not in stop_words and len(word) > 2
        ]
        
        return keywords
    
    def _extract_entities(self, question: str) -> List[str]:
        """
        Extract named entities from question (simple implementation).
        
        Args:
            question: Processed question
            
        Returns:
            List of entities
        """
        # Simple entity extraction - look for capitalized words
        words = question.split()
        entities = []
        
        for word in words:
            # Check if word starts with capital and is not at sentence start
            cleaned = word.strip(".,!?;:")
            if cleaned and cleaned[0].isupper() and len(cleaned) > 1:
                # Not a question word
                if cleaned.lower() not in ["what", "who", "when", "where", "why", "how"]:
                    entities.append(cleaned)
        
        return entities
    
    def _determine_context_requirements(
        self,
        question_type: str,
        question: str
    ) -> Dict[str, Any]:
        """
        Determine context requirements based on question type.
        
        Args:
            question_type: Type of question
            question: Processed question
            
        Returns:
            Context requirements
        """
        requirements = {
            "min_chunks": 3,
            "max_chunks": 10,
            "use_mmr": True,
            "diversity_weight": 0.5
        }
        
        # Adjust based on question type
        if question_type == "comparative":
            requirements["min_chunks"] = 5
            requirements["max_chunks"] = 12
            requirements["diversity_weight"] = 0.7  # More diversity for comparisons
        elif question_type == "summary":
            requirements["min_chunks"] = 8
            requirements["max_chunks"] = 15
            requirements["diversity_weight"] = 0.6
        elif question_type == "factual":
            requirements["min_chunks"] = 2
            requirements["max_chunks"] = 8
            requirements["diversity_weight"] = 0.3  # Less diversity, more relevance
        
        return requirements
    
    def generate_search_queries(self, processed_question: ProcessedQuestion) -> List[str]:
        """
        Generate multiple search queries from a processed question.
        
        Args:
            processed_question: Processed question
            
        Returns:
            List of search query variations
        """
        queries = [processed_question.processed_question]
        
        # Add keyword-based query
        if processed_question.keywords:
            keyword_query = " ".join(processed_question.keywords[:5])
            if keyword_query != processed_question.processed_question:
                queries.append(keyword_query)
        
        # Add entity-focused query
        if processed_question.entities:
            entity_query = " ".join(processed_question.entities)
            if entity_query and entity_query not in queries:
                queries.append(entity_query)
        
        return queries
