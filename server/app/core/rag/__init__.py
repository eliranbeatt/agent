"""RAG (Retrieval-Augmented Generation) workflow components."""

from .rag_workflow import RAGWorkflow, RAGResult
from .question_processor import QuestionProcessor, ProcessedQuestion
from .answer_generator import AnswerGenerator, GeneratedAnswer
from .citation_manager import CitationManager, Citation
from .evaluator import Evaluator, VerificationResult, GroundingCheck, ContradictionCheck

__all__ = [
    "RAGWorkflow",
    "RAGResult",
    "QuestionProcessor",
    "ProcessedQuestion",
    "AnswerGenerator",
    "GeneratedAnswer",
    "CitationManager",
    "Citation",
    "Evaluator",
    "VerificationResult",
    "GroundingCheck",
    "ContradictionCheck",
]
