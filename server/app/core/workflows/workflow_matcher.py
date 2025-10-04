"""Workflow matching algorithm with confidence scoring."""

import logging
from typing import Dict, List, Optional, Tuple
import re

from ...config.models import WorkflowConfig


logger = logging.getLogger(__name__)


class WorkflowMatcher:
    """
    Handles matching user requests to predefined workflows with confidence scoring.
    
    Uses multiple matching strategies:
    - Exact trigger matching
    - Pattern-based matching
    - Semantic similarity
    - Context relevance
    """
    
    def __init__(self, workflows: Dict[str, WorkflowConfig]):
        """
        Initialize workflow matcher.
        
        Args:
            workflows: Dictionary of available workflows
        """
        self.workflows = workflows
        
        # Common patterns for different workflow types
        self.workflow_patterns = {
            "rag_qa": ["question", "ask", "what", "how", "why", "explain", "tell me about", "find"],
            "summarize": ["summarize", "summary", "brief", "overview", "key points", "main ideas"],
            "compare": ["compare", "difference", "versus", "vs", "contrast", "similarities"],
            "extract": ["extract", "find", "get", "retrieve", "pull out", "identify"],
            "analyze": ["analyze", "analysis", "examine", "study", "investigate", "review"],
            "ocr": ["read", "text from image", "ocr", "scan", "extract from image"]
        }
        
    def match_workflow(
        self, 
        user_request: str, 
        context: Optional[Dict] = None
    ) -> Tuple[Optional[str], float]:
        """
        Match user request to the best workflow using confidence scoring.
        
        Args:
            user_request: User's input request
            context: Optional context for enhanced matching
            
        Returns:
            Tuple of (workflow_name, confidence_score)
        """
        if not self.workflows:
            logger.warning("No workflows available for matching")
            return None, 0.0
            
        best_match = None
        best_confidence = 0.0
        
        # Convert request to lowercase for matching
        request_lower = user_request.lower()
        request_words = set(self._tokenize(request_lower))
        
        for workflow_name, workflow_config in self.workflows.items():
            if not workflow_config.enabled:
                logger.debug(f"Skipping disabled workflow: {workflow_name}")
                continue
                
            confidence = self._calculate_confidence(
                request_lower, 
                request_words, 
                workflow_config,
                context
            )
            
            logger.debug(f"Workflow '{workflow_name}' confidence: {confidence:.3f}")
            
            if confidence > best_confidence:
                best_confidence = confidence
                best_match = workflow_name
                
        logger.info(
            f"Best workflow match: {best_match or 'none'} "
            f"(confidence: {best_confidence:.3f})"
        )
        
        return best_match, best_confidence
    
    def _calculate_confidence(
        self,
        request: str,
        request_words: set,
        workflow: WorkflowConfig,
        context: Optional[Dict] = None
    ) -> float:
        """
        Calculate confidence score for a workflow match using multiple factors.
        
        Args:
            request: Lowercase user request
            request_words: Set of words in the request
            workflow: Workflow configuration
            context: Optional context for enhanced matching
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        confidence = 0.0
        
        # 1. Exact trigger matching (40% weight)
        trigger_score = self._calculate_trigger_score(request, workflow.triggers)
        confidence += trigger_score * 0.4
        
        # 2. Pattern matching (30% weight)
        pattern_score = self._calculate_pattern_score(request_words, workflow.name)
        confidence += pattern_score * 0.3
        
        # 3. Semantic similarity (20% weight)
        semantic_score = self._calculate_semantic_score(request, workflow)
        confidence += semantic_score * 0.2
        
        # 4. Context relevance (10% weight)
        context_score = self._calculate_context_score(request, workflow, context)
        confidence += context_score * 0.1
        
        # Apply workflow-specific confidence threshold adjustment
        if hasattr(workflow, 'confidence_threshold'):
            # Slightly boost workflows with lower thresholds (they're more general)
            threshold_factor = 1.0 - (workflow.confidence_threshold * 0.1)
            confidence *= threshold_factor
            
        return min(confidence, 1.0)
    
    def _calculate_trigger_score(self, request: str, triggers: List[str]) -> float:
        """
        Calculate score based on trigger word matches.
        
        Args:
            request: Lowercase user request
            triggers: List of trigger phrases
            
        Returns:
            Score between 0.0 and 1.0
        """
        if not triggers:
            return 0.0
            
        matches = 0
        partial_matches = 0
        
        for trigger in triggers:
            trigger_lower = trigger.lower()
            
            # Exact phrase match
            if trigger_lower in request:
                matches += 1
            else:
                # Partial word match
                trigger_words = set(self._tokenize(trigger_lower))
                request_words = set(self._tokenize(request))
                overlap = len(trigger_words.intersection(request_words))
                
                if overlap > 0:
                    partial_matches += overlap / len(trigger_words)
                    
        total_score = matches + (partial_matches * 0.5)
        return min(total_score / len(triggers), 1.0)
    
    def _calculate_pattern_score(self, request_words: set, workflow_name: str) -> float:
        """
        Calculate score based on workflow pattern matching.
        
        Args:
            request_words: Set of words in the request
            workflow_name: Name of the workflow
            
        Returns:
            Score between 0.0 and 1.0
        """
        workflow_type = workflow_name.lower().replace("_", " ")
        
        # Check if workflow name appears in request
        workflow_name_words = set(self._tokenize(workflow_type))
        if workflow_name_words.intersection(request_words):
            return 1.0
            
        # Check pattern keywords
        for pattern_type, keywords in self.workflow_patterns.items():
            if pattern_type in workflow_name.lower():
                keyword_set = set(keywords)
                matches = len(request_words.intersection(keyword_set))
                if matches > 0:
                    return min(matches / len(keyword_set), 1.0)
                    
        return 0.0
    
    def _calculate_semantic_score(self, request: str, workflow: WorkflowConfig) -> float:
        """
        Calculate semantic similarity score.
        
        Args:
            request: User request
            workflow: Workflow configuration
            
        Returns:
            Score between 0.0 and 1.0
        """
        if not workflow.description:
            return 0.0
            
        desc_words = set(self._tokenize(workflow.description.lower()))
        request_words = set(self._tokenize(request))
        
        # Calculate Jaccard similarity
        intersection = len(desc_words.intersection(request_words))
        union = len(desc_words.union(request_words))
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_context_score(
        self,
        request: str,
        workflow: WorkflowConfig,
        context: Optional[Dict] = None
    ) -> float:
        """
        Calculate context relevance score.
        
        Args:
            request: User request
            workflow: Workflow configuration
            context: Optional context information
            
        Returns:
            Score between 0.0 and 1.0
        """
        score = 0.5  # Neutral baseline
        
        # Check for file-related context
        file_indicators = ["file", "document", "pdf", "upload", "image", "picture"]
        has_file_context = any(indicator in request for indicator in file_indicators)
        
        # Check context dict if provided
        has_uploaded_files = False
        if context:
            has_uploaded_files = context.get("has_uploaded_files", False)
            
        # RAG workflows benefit from file context
        if "rag" in workflow.name.lower() or "qa" in workflow.name.lower():
            if has_file_context or has_uploaded_files:
                score = 0.9
            else:
                score = 0.3
                
        # Summarize workflows also benefit from file context
        elif "summarize" in workflow.name.lower() or "extract" in workflow.name.lower():
            if has_file_context or has_uploaded_files:
                score = 0.8
            else:
                score = 0.4
                
        # OCR workflows require image context
        elif "ocr" in workflow.name.lower() or "image" in workflow.name.lower():
            image_indicators = ["image", "picture", "photo", "scan", "screenshot"]
            has_image_context = any(indicator in request for indicator in image_indicators)
            if has_image_context:
                score = 0.95
            else:
                score = 0.2
                
        # Compare workflows benefit from multiple documents
        elif "compare" in workflow.name.lower():
            multi_doc_indicators = ["documents", "files", "multiple", "both", "all"]
            has_multi_doc = any(indicator in request for indicator in multi_doc_indicators)
            if has_multi_doc:
                score = 0.85
            else:
                score = 0.4
                
        return score
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words.
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of tokens
        """
        # Remove punctuation and split on whitespace
        text = re.sub(r'[^\w\s]', ' ', text)
        return [word for word in text.split() if word]
    
    def get_workflow_suggestions(
        self,
        user_request: str,
        top_k: int = 3,
        min_confidence: float = 0.3
    ) -> List[Tuple[str, float]]:
        """
        Get top workflow suggestions for a user request.
        
        Args:
            user_request: User's input request
            top_k: Number of suggestions to return
            min_confidence: Minimum confidence threshold
            
        Returns:
            List of (workflow_name, confidence) tuples
        """
        suggestions = []
        
        request_lower = user_request.lower()
        request_words = set(self._tokenize(request_lower))
        
        for workflow_name, workflow_config in self.workflows.items():
            if not workflow_config.enabled:
                continue
                
            confidence = self._calculate_confidence(
                request_lower,
                request_words,
                workflow_config,
                None
            )
            
            if confidence >= min_confidence:
                suggestions.append((workflow_name, confidence))
                
        # Sort by confidence descending
        suggestions.sort(key=lambda x: x[1], reverse=True)
        
        return suggestions[:top_k]
