"""Evaluator node for output verification and quality assurance."""

import logging
import os
import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from openai import OpenAI

from ..context.chunker import Chunk
from .citation_manager import Citation

logger = logging.getLogger(__name__)


@dataclass
class VerificationResult:
    """Result of answer verification."""
    passed: bool
    quality_score: float  # 0-1
    confidence: float  # 0-1
    issues: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    grounding_score: float = 0.0
    contradiction_score: float = 0.0
    completeness_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "passed": self.passed,
            "quality_score": self.quality_score,
            "confidence": self.confidence,
            "issues": self.issues,
            "suggestions": self.suggestions,
            "grounding_score": self.grounding_score,
            "contradiction_score": self.contradiction_score,
            "completeness_score": self.completeness_score,
            "metadata": self.metadata
        }


@dataclass
class GroundingCheck:
    """Result of source grounding verification."""
    is_grounded: bool
    grounding_score: float
    unsupported_claims: List[str] = field(default_factory=list)
    supported_claims: List[str] = field(default_factory=list)


@dataclass
class ContradictionCheck:
    """Result of contradiction detection."""
    has_contradictions: bool
    contradiction_score: float
    contradictions: List[Dict[str, str]] = field(default_factory=list)


class Evaluator:
    """
    Evaluator node for verifying RAG outputs against success criteria.
    
    Responsibilities:
    - Verify task completion against success criteria
    - Check answer grounding in retrieved sources
    - Detect logical contradictions
    - Generate quality scores and confidence metrics
    """
    
    def __call__(self, state):
        return state

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        quality_threshold: float = 0.6,
        grounding_threshold: float = 0.7
    ):
        """
        Initialize evaluator.
        
        Args:
            openai_api_key: OpenAI API key
            model: Model to use for evaluation
            quality_threshold: Minimum quality score to pass
            grounding_threshold: Minimum grounding score to pass
        """
        self.api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.quality_threshold = quality_threshold
        self.grounding_threshold = grounding_threshold
        
        if not self.api_key:
            logger.warning("No OpenAI API key provided. Evaluation will be limited.")
            self.client = None
        else:
            self.client = OpenAI(api_key=self.api_key)
    
    def verify_answer(
        self,
        question: str,
        answer: str,
        chunks: List[Chunk],
        citations: List[Citation],
        success_criteria: Optional[List[str]] = None
    ) -> VerificationResult:
        """
        Verify an answer against multiple criteria.
        
        Args:
            question: Original question
            answer: Generated answer
            chunks: Retrieved context chunks
            citations: Citations used
            success_criteria: Optional specific success criteria
            
        Returns:
            Verification result with scores and issues
        """
        logger.debug(f"Verifying answer for question: {question}")
        
        issues = []
        suggestions = []
        
        # Check 1: Source grounding
        grounding_check = self.check_source_grounding(answer, chunks)
        grounding_score = grounding_check.grounding_score
        
        if not grounding_check.is_grounded:
            issues.append("Answer contains unsupported claims")
            suggestions.append("Ensure all claims are backed by source documents")
        
        if grounding_check.unsupported_claims:
            issues.append(f"Found {len(grounding_check.unsupported_claims)} unsupported claims")
        
        # Check 2: Contradiction detection
        contradiction_check = self.detect_contradictions(answer, chunks)
        contradiction_score = 1.0 - contradiction_check.contradiction_score  # Invert so higher is better
        
        if contradiction_check.has_contradictions:
            issues.append("Answer contains logical contradictions")
            suggestions.append("Review and resolve contradictory statements")
        
        # Check 3: Completeness
        completeness_score = self._check_completeness(question, answer, chunks)
        
        if completeness_score < 0.5:
            issues.append("Answer may be incomplete")
            suggestions.append("Consider including more information from sources")
        
        # Check 4: Success criteria (if provided)
        criteria_score = 1.0
        if success_criteria:
            criteria_score = self._check_success_criteria(answer, success_criteria)
            if criteria_score < 0.7:
                issues.append("Answer does not fully meet success criteria")
        
        # Calculate overall quality score
        quality_score = (
            grounding_score * 0.4 +
            contradiction_score * 0.3 +
            completeness_score * 0.2 +
            criteria_score * 0.1
        )
        
        # Calculate confidence
        confidence = self._calculate_confidence(
            quality_score,
            len(chunks),
            len(citations),
            grounding_check,
            contradiction_check
        )
        
        # Determine if verification passed
        passed = (
            quality_score >= self.quality_threshold and
            grounding_score >= self.grounding_threshold and
            not contradiction_check.has_contradictions
        )
        
        result = VerificationResult(
            passed=passed,
            quality_score=quality_score,
            confidence=confidence,
            issues=issues,
            suggestions=suggestions,
            grounding_score=grounding_score,
            contradiction_score=contradiction_score,
            completeness_score=completeness_score,
            metadata={
                "unsupported_claims": len(grounding_check.unsupported_claims),
                "contradictions": len(contradiction_check.contradictions),
                "chunks_used": len(chunks),
                "citations_provided": len(citations)
            }
        )
        
        logger.info(
            f"Verification complete: passed={passed}, "
            f"quality={quality_score:.2f}, "
            f"grounding={grounding_score:.2f}"
        )
        
        return result
    
    def check_source_grounding(
        self,
        answer: str,
        chunks: List[Chunk]
    ) -> GroundingCheck:
        """
        Verify that answer claims are grounded in source documents.
        
        Args:
            answer: Generated answer
            chunks: Source chunks
            
        Returns:
            Grounding check result
        """
        if not self.client or not chunks:
            # Fallback to simple check
            return GroundingCheck(
                is_grounded=True,
                grounding_score=0.5,
                supported_claims=[],
                unsupported_claims=[]
            )
        
        try:
            # Build context from chunks
            context = "\n\n".join([f"Source {i+1}: {chunk.content}" for i, chunk in enumerate(chunks)])
            
            prompt = f"""Analyze if the answer is grounded in the provided sources.

Sources:
{context}

Answer:
{answer}

Task:
1. Identify key claims in the answer
2. For each claim, determine if it's supported by the sources
3. List any unsupported claims

Respond in this format:
SUPPORTED:
- [claim 1]
- [claim 2]

UNSUPPORTED:
- [claim 1]
- [claim 2]

GROUNDING_SCORE: [0.0-1.0]"""
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=500
            )
            
            result_text = response.choices[0].message.content
            
            # Parse response
            supported_claims = self._extract_claims(result_text, "SUPPORTED:")
            unsupported_claims = self._extract_claims(result_text, "UNSUPPORTED:")
            
            # Extract grounding score
            grounding_score = self._extract_score(result_text, "GROUNDING_SCORE:")
            
            is_grounded = grounding_score >= self.grounding_threshold and len(unsupported_claims) == 0
            
            logger.debug(
                f"Grounding check: score={grounding_score:.2f}, "
                f"supported={len(supported_claims)}, "
                f"unsupported={len(unsupported_claims)}"
            )
            
            return GroundingCheck(
                is_grounded=is_grounded,
                grounding_score=grounding_score,
                supported_claims=supported_claims,
                unsupported_claims=unsupported_claims
            )
            
        except Exception as e:
            logger.error(f"Source grounding check failed: {e}")
            return GroundingCheck(
                is_grounded=False,
                grounding_score=0.0,
                supported_claims=[],
                unsupported_claims=["Error during grounding check"]
            )
    
    def detect_contradictions(
        self,
        answer: str,
        chunks: List[Chunk]
    ) -> ContradictionCheck:
        """
        Detect logical contradictions in the answer.
        
        Args:
            answer: Generated answer
            chunks: Source chunks
            
        Returns:
            Contradiction check result
        """
        if not self.client:
            # Fallback to simple check
            return ContradictionCheck(
                has_contradictions=False,
                contradiction_score=0.0,
                contradictions=[]
            )
        
        try:
            # Build context from chunks
            context = "\n\n".join([f"Source {i+1}: {chunk.content}" for i, chunk in enumerate(chunks)])
            
            prompt = f"""Analyze the answer for logical contradictions or inconsistencies.

Sources:
{context}

Answer:
{answer}

Task:
1. Check for internal contradictions within the answer
2. Check for contradictions between the answer and sources
3. List any contradictions found

Respond in this format:
CONTRADICTIONS:
- Statement A: [statement]
  Contradicts: [contradicting statement]
  Type: [internal/source]

CONTRADICTION_SCORE: [0.0-1.0, where 0=no contradictions, 1=severe contradictions]"""
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=500
            )
            
            result_text = response.choices[0].message.content
            
            # Parse contradictions
            contradictions = self._parse_contradictions(result_text)
            
            # Extract contradiction score
            contradiction_score = self._extract_score(result_text, "CONTRADICTION_SCORE:")
            
            has_contradictions = contradiction_score > 0.3 or len(contradictions) > 0
            
            logger.debug(
                f"Contradiction check: score={contradiction_score:.2f}, "
                f"found={len(contradictions)}"
            )
            
            return ContradictionCheck(
                has_contradictions=has_contradictions,
                contradiction_score=contradiction_score,
                contradictions=contradictions
            )
            
        except Exception as e:
            logger.error(f"Contradiction detection failed: {e}")
            return ContradictionCheck(
                has_contradictions=False,
                contradiction_score=0.0,
                contradictions=[]
            )
    
    def _check_completeness(
        self,
        question: str,
        answer: str,
        chunks: List[Chunk]
    ) -> float:
        """
        Check if answer is complete relative to the question.
        
        Args:
            question: Original question
            answer: Generated answer
            chunks: Source chunks
            
        Returns:
            Completeness score (0-1)
        """
        # Simple heuristic-based completeness check
        score = 0.5  # Base score
        
        # Check answer length
        if len(answer) > 50:
            score += 0.1
        if len(answer) > 150:
            score += 0.1
        
        # Check if answer addresses question keywords
        question_words = set(question.lower().split())
        answer_words = set(answer.lower().split())
        
        # Remove common words
        stop_words = {"what", "who", "when", "where", "why", "how", "is", "are", "the", "a", "an"}
        question_keywords = question_words - stop_words
        
        if question_keywords:
            overlap = len(question_keywords & answer_words) / len(question_keywords)
            score += overlap * 0.2
        
        # Check if answer uses multiple sources
        if len(chunks) > 1:
            score += 0.1
        
        return min(1.0, score)
    
    def _check_success_criteria(
        self,
        answer: str,
        criteria: List[str]
    ) -> float:
        """
        Check answer against success criteria.
        
        Args:
            answer: Generated answer
            criteria: List of success criteria
            
        Returns:
            Criteria satisfaction score (0-1)
        """
        if not criteria:
            return 1.0
        
        # Simple keyword-based check
        satisfied = 0
        for criterion in criteria:
            # Check if criterion keywords appear in answer
            criterion_words = set(criterion.lower().split())
            answer_words = set(answer.lower().split())
            
            if criterion_words & answer_words:
                satisfied += 1
        
        return satisfied / len(criteria) if criteria else 1.0
    
    def _calculate_confidence(
        self,
        quality_score: float,
        num_chunks: int,
        num_citations: int,
        grounding_check: GroundingCheck,
        contradiction_check: ContradictionCheck
    ) -> float:
        """Calculate overall confidence in the answer."""
        confidence = quality_score * 0.5
        
        # Adjust based on number of sources
        if num_chunks >= 3:
            confidence += 0.1
        if num_chunks >= 5:
            confidence += 0.1
        
        # Adjust based on citations
        if num_citations > 0:
            confidence += 0.1
        
        # Penalize for issues
        if grounding_check.unsupported_claims:
            confidence -= 0.1
        if contradiction_check.has_contradictions:
            confidence -= 0.2
        
        return max(0.0, min(1.0, confidence))
    
    def _extract_claims(self, text: str, section_marker: str) -> List[str]:
        """Extract claims from a section of text."""
        claims = []
        
        if section_marker in text:
            section_start = text.find(section_marker) + len(section_marker)
            section_text = text[section_start:]
            
            # Find next section or end
            next_section = float('inf')
            for marker in ["SUPPORTED:", "UNSUPPORTED:", "GROUNDING_SCORE:", "CONTRADICTIONS:", "CONTRADICTION_SCORE:"]:
                if marker != section_marker and marker in section_text:
                    pos = section_text.find(marker)
                    if pos < next_section:
                        next_section = pos
            
            if next_section != float('inf'):
                section_text = section_text[:next_section]
            
            # Extract bullet points
            lines = section_text.strip().split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith('-') or line.startswith('•'):
                    claim = line.lstrip('-•').strip()
                    if claim:
                        claims.append(claim)
        
        return claims
    
    def _extract_score(self, text: str, score_marker: str) -> float:
        """Extract a score from text."""
        if score_marker in text:
            score_start = text.find(score_marker) + len(score_marker)
            score_text = text[score_start:score_start + 50].strip()
            
            # Extract first number
            match = re.search(r'(\d+\.?\d*)', score_text)
            if match:
                try:
                    score = float(match.group(1))
                    return max(0.0, min(1.0, score))
                except ValueError:
                    pass
        
        return 0.5  # Default score
    
    def _parse_contradictions(self, text: str) -> List[Dict[str, str]]:
        """Parse contradictions from text."""
        contradictions = []
        
        if "CONTRADICTIONS:" in text:
            section_start = text.find("CONTRADICTIONS:") + len("CONTRADICTIONS:")
            section_text = text[section_start:]
            
            # Find end of section
            if "CONTRADICTION_SCORE:" in section_text:
                section_text = section_text[:section_text.find("CONTRADICTION_SCORE:")]
            
            # Simple parsing - look for patterns
            lines = section_text.strip().split('\n')
            current_contradiction = {}
            
            for line in lines:
                line = line.strip()
                if line.startswith('-') or line.startswith('Statement'):
                    if current_contradiction:
                        contradictions.append(current_contradiction)
                        current_contradiction = {}
                
                if 'Statement A:' in line or 'Statement:' in line:
                    current_contradiction['statement_a'] = line.split(':', 1)[1].strip()
                elif 'Contradicts:' in line:
                    current_contradiction['statement_b'] = line.split(':', 1)[1].strip()
                elif 'Type:' in line:
                    current_contradiction['type'] = line.split(':', 1)[1].strip()
            
            if current_contradiction:
                contradictions.append(current_contradiction)
        
        return contradictions
