"""Answer generation for RAG pipeline."""

import logging
import os
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from openai import OpenAI

from ..context.chunker import Chunk
from .citation_manager import Citation

logger = logging.getLogger(__name__)


@dataclass
class GeneratedAnswer:
    """Generated answer with metadata."""
    answer: str
    question: str
    citations: List[Citation] = field(default_factory=list)
    confidence: float = 0.0
    tokens_used: int = 0
    model: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "answer": self.answer,
            "question": self.question,
            "citations": [c.to_dict() for c in self.citations],
            "confidence": self.confidence,
            "tokens_used": self.tokens_used,
            "model": self.model,
            "metadata": self.metadata
        }


class AnswerGenerator:
    """Generates answers using retrieved context and LLM."""
    
    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        temperature: float = 0.1,
        max_tokens: int = 1000
    ):
        """
        Initialize answer generator.
        
        Args:
            openai_api_key: OpenAI API key
            model: Model to use for generation
            temperature: Temperature for generation
            max_tokens: Maximum tokens in response
        """
        self.api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        if not self.api_key:
            logger.warning("No OpenAI API key provided. Answer generation will fail.")
            self.client = None
        else:
            self.client = OpenAI(api_key=self.api_key)
    
    def generate_answer(
        self,
        question: str,
        chunks: List[Chunk],
        citations: List[Citation],
        question_type: str = "factual"
    ) -> GeneratedAnswer:
        """
        Generate an answer using retrieved context.
        
        Args:
            question: User's question
            chunks: Retrieved context chunks
            citations: Citations for the chunks
            question_type: Type of question
            
        Returns:
            Generated answer with metadata
        """
        if not self.client:
            return GeneratedAnswer(
                answer="Error: OpenAI API key not configured",
                question=question,
                citations=[],
                confidence=0.0
            )
        
        try:
            # Build context from chunks
            context = self._build_context(chunks, citations)
            
            # Create system prompt
            system_prompt = self._create_system_prompt(question_type)
            
            # Create user prompt
            user_prompt = self._create_user_prompt(question, context)
            
            # Generate answer
            logger.debug(f"Generating answer for: {question}")
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            answer_text = response.choices[0].message.content
            tokens_used = response.usage.total_tokens if response.usage else 0
            
            # Estimate confidence based on finish reason and response
            confidence = self._estimate_confidence(response, chunks)
            
            logger.info(f"Generated answer ({tokens_used} tokens, confidence: {confidence:.2f})")
            
            return GeneratedAnswer(
                answer=answer_text,
                question=question,
                citations=citations,
                confidence=confidence,
                tokens_used=tokens_used,
                model=self.model,
                metadata={
                    "finish_reason": response.choices[0].finish_reason,
                    "chunks_used": len(chunks)
                }
            )
            
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            return GeneratedAnswer(
                answer=f"Error generating answer: {str(e)}",
                question=question,
                citations=[],
                confidence=0.0
            )
    
    def _build_context(self, chunks: List[Chunk], citations: List[Citation]) -> str:
        """
        Build context string from chunks with citations.
        
        Args:
            chunks: Retrieved chunks
            citations: Citations for chunks
            
        Returns:
            Formatted context string
        """
        context_parts = []
        
        for i, (chunk, citation) in enumerate(zip(chunks, citations), 1):
            source_info = f"[Source {i}: {citation.source_file}"
            if citation.page_number:
                source_info += f", Page {citation.page_number}"
            source_info += "]"
            
            context_parts.append(f"{source_info}\n{chunk.content}\n")
        
        return "\n".join(context_parts)
    
    def _create_system_prompt(self, question_type: str) -> str:
        """
        Create system prompt based on question type.
        
        Args:
            question_type: Type of question
            
        Returns:
            System prompt
        """
        base_prompt = """You are a helpful AI assistant that answers questions based on provided context.

Your responsibilities:
1. Answer questions accurately using ONLY the information in the provided context
2. If the context doesn't contain enough information, say so clearly
3. Cite sources by referencing [Source N] numbers in your answer
4. Be concise but complete in your answers
5. Do not make up information or use knowledge outside the provided context"""
        
        type_specific = {
            "factual": "\n6. Focus on providing specific facts and details from the sources",
            "analytical": "\n6. Provide analysis and explanation based on the context\n7. Connect related information from multiple sources",
            "comparative": "\n6. Compare and contrast information from different sources\n7. Highlight similarities and differences clearly",
            "summary": "\n6. Synthesize information from all sources\n7. Provide a comprehensive overview"
        }
        
        return base_prompt + type_specific.get(question_type, "")
    
    def _create_user_prompt(self, question: str, context: str) -> str:
        """
        Create user prompt with question and context.
        
        Args:
            question: User's question
            context: Retrieved context
            
        Returns:
            User prompt
        """
        return f"""Context:
{context}

Question: {question}

Please answer the question based on the context provided above. Remember to cite your sources using [Source N] references."""
    
    def _estimate_confidence(self, response: Any, chunks: List[Chunk]) -> float:
        """
        Estimate confidence in the generated answer.
        
        Args:
            response: OpenAI API response
            chunks: Retrieved chunks
            
        Returns:
            Confidence score (0-1)
        """
        confidence = 0.5  # Base confidence
        
        # Adjust based on finish reason
        if response.choices[0].finish_reason == "stop":
            confidence += 0.2
        elif response.choices[0].finish_reason == "length":
            confidence -= 0.1
        
        # Adjust based on number of chunks
        if len(chunks) >= 3:
            confidence += 0.1
        if len(chunks) >= 5:
            confidence += 0.1
        
        # Adjust based on average chunk relevance (if available)
        # This would require passing scores, simplified for now
        
        # Clamp to [0, 1]
        return max(0.0, min(1.0, confidence))
    
    def generate_followup_questions(
        self,
        question: str,
        answer: str,
        chunks: List[Chunk]
    ) -> List[str]:
        """
        Generate follow-up questions based on the answer.
        
        Args:
            question: Original question
            answer: Generated answer
            chunks: Retrieved chunks
            
        Returns:
            List of follow-up questions
        """
        if not self.client:
            return []
        
        try:
            prompt = f"""Based on this Q&A, suggest 3 relevant follow-up questions:

Question: {question}
Answer: {answer}

Generate 3 follow-up questions that would help the user learn more about this topic.
Return only the questions, one per line."""
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=200
            )
            
            followups = response.choices[0].message.content.strip().split("\n")
            # Clean up questions
            followups = [q.strip().lstrip("0123456789.-) ") for q in followups if q.strip()]
            
            return followups[:3]
            
        except Exception as e:
            logger.error(f"Failed to generate follow-up questions: {e}")
            return []
