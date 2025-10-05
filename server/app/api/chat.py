"""Chat API endpoints."""
import asyncio
import logging
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from typing import AsyncGenerator, Any, Dict, List, Optional
import json
import uuid
import os
from datetime import datetime

from app.api.models import ChatRequest, ChatResponse, ErrorResponse
from app.core.orchestrator import MainOrchestrator
from openai import OpenAI
from app.core.graph_builder import create_execution_graph
from app.core.state import ExecutionState, ExecutionPath
from app.config.loader import ConfigLoader
from app.core.rag.rag_workflow import RAGWorkflow
from app.core.context.context_manager import ContextManager

router = APIRouter(prefix="/chat", tags=["chat"])

# Initialize execution graph
config_loader = ConfigLoader()
system_config = config_loader.load_config()
execution_graph = create_execution_graph(system_config)

# Initialize RAG workflow for direct question answering
context_manager = ContextManager(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    vector_db_path=system_config.context.vector_db_path,
    chunk_size=system_config.context.chunk_size,
    chunk_overlap=system_config.context.chunk_overlap,
    embedding_model=system_config.context.embedding_model,
    context_config=system_config.context
)

rag_workflow = RAGWorkflow(
    context_manager=context_manager,
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    answer_model="gpt-5-mini",
    retrieval_k=10,
    use_mmr=True,
    enable_verification=True
)

logger = logging.getLogger(__name__)

LIGHT_INTENT_MODEL = "gpt-5-mini"
INTENT_KEYWORDS = {
    "plan", "task", "upload", "analyze", "analysis", "workflow", "execute",
    "run", "agent", "document", "process", "generate", "summarize", "compare",
    "build", "compute", "diagnose"
}


class LightweightIntentAgent:
    """Lightweight intent classifier powered by gpt-5-mini with heuristic fallback."""

    def __init__(self, model: str = LIGHT_INTENT_MODEL, api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.client: Optional[OpenAI] = None
        if self.api_key:
            try:
                self.client = OpenAI(api_key=self.api_key)
            except Exception as exc:
                logger.warning("Failed to initialize gpt-5-mini intent client: %s", exc)
                self.client = None

    def _should_escalate(self, message: str, context: Dict[str, Any]) -> tuple[bool, List[str]]:
        reasons: List[str] = []
        normalized = message.lower()

        if context.get("use_rag") or context.get("has_uploaded_files"):
            reasons.append("context_requires_rag")
        if context.get("needs_file_processing"):
            reasons.append("context_requests_file_processing")
        if context.get("force_orchestrator"):
            reasons.append("context_force_orchestrator")

        for keyword in INTENT_KEYWORDS:
            if keyword in normalized:
                reasons.append(f"keyword:{keyword}")

        requires_followup = bool(reasons)
        return requires_followup, reasons

    def _build_heuristic_response(self, message: str, requires_followup: bool) -> str:
        trimmed = message.strip() or "your request"
        if requires_followup:
            return (
                "I understand you need deeper assistance with: "
                f'"{trimmed}". I\'ll route this to the full orchestrator and keep track of the next steps.'
            )
        return (
            "Here's a quick take: I'm on it. Right now there isn't any action required beyond this "
            f'clarification about "{trimmed}".'
        )

    def analyze(self, message: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        context = context or {}
        requires_followup, reasons = self._should_escalate(message, context)
        metadata: Dict[str, Any] = {
            "model": self.model,
            "analysis_reasons": reasons,
            "requires_followup": requires_followup
        }

        heuristic_response = self._build_heuristic_response(message, requires_followup)
        if not self.client:
            return {
                "response": heuristic_response,
                "requires_followup": requires_followup,
                "metadata": metadata
            }

        try:
            prompt_context = json.dumps({k: v for k, v in context.items() if isinstance(v, (str, int, float, bool, list, dict))})
            completion = self.client.chat.completions.create(
                model=self.model,
                temperature=0.2,
                max_tokens=256,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a super-fast intent analyst. Reply with a short, friendly sentence "
                            "that summarizes the user's request and whether you'll escalate it."
                        )
                    },
                    {
                        "role": "user",
                        "content": (
                            f"User message: {message}\n"
                            f"Structured context: {prompt_context}"
                        )
                    }
                ]
            )
            response_text = completion.choices[0].message.content.strip()
        except Exception as exc:
            logger.warning("Intent analysis via gpt-5-mini failed, using heuristic: %s", exc)
            metadata["model_error"] = str(exc)
            response_text = heuristic_response

        return {
            "response": response_text or heuristic_response,
            "requires_followup": requires_followup,
            "metadata": metadata
        }


orchestrator = MainOrchestrator(system_config)
intent_agent = LightweightIntentAgent()


PLACEHOLDER_RESPONSES = {
    '',
    'request processed',
    'request processed successfully',
    'execution completed',
    'request processed.'
}


def _normalize_execution_path(value: Optional[Any]) -> str:
    """Convert execution path value to a string representation."""
    if isinstance(value, ExecutionPath):
        return value.value
    if isinstance(value, str) and value.strip():
        return value
    return 'unknown'


def _is_placeholder_response(text: Optional[str]) -> bool:
    """Determine if the response text is effectively empty or a placeholder."""
    if not text:
        return True
    normalized = text.strip().lower()
    return normalized in PLACEHOLDER_RESPONSES


def _extract_final_result(final_state: Optional[Any]) -> Dict[str, Any]:
    """Extract the final result dictionary from a graph execution state."""
    if final_state is None:
        return {}
    values = getattr(final_state, 'values', final_state)
    if isinstance(values, ExecutionState):
        return values.final_result or {}
    if isinstance(values, dict):
        final_result = values.get('final_result')
        if isinstance(final_result, dict):
            return final_result
        expected_keys = {'response', 'sources', 'metadata', 'execution_path'}
        if expected_keys.intersection(values.keys()):
            return {key: values.get(key) for key in expected_keys if key in values}
    return {}


def _extract_execution_path_hint(final_state: Optional[Any]) -> Optional[Any]:
    """Extract the execution path value from a graph execution state."""
    if final_state is None:
        return None
    values = getattr(final_state, 'values', final_state)
    if isinstance(values, ExecutionState):
        return values.execution_path
    if isinstance(values, dict):
        return values.get('execution_path')
    return None


async def _generate_fallback_answer(message: str, context: Dict[str, Any]) -> tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
    """Generate an answer using the RAG workflow as a fallback path."""
    try:
        rag_result = await asyncio.to_thread(
            rag_workflow.process_question,
            question=message,
            source_filter=context.get('source_filter'),
            page_filter=context.get('page_filter'),
            custom_k=context.get('retrieval_k')
        )
    except Exception as exc:
        return (
            f'Unable to generate an answer right now: {exc}',
            [],
            {
                'fallback_strategy': 'rag_workflow',
                'error': str(exc)
            }
        )

    metadata: Dict[str, Any] = {
        'fallback_strategy': 'rag_workflow'
    }
    if rag_result.metadata:
        metadata.update(rag_result.metadata)

    if rag_result.success:
        sources = [
            {
                'chunk_id': citation.chunk_id,
                'content': citation.content,
                'source_file': citation.source_file,
                'page_number': citation.page_number,
                'score': citation.score
            }
            for citation in rag_result.citations
        ]
        metadata.update({
            'confidence': rag_result.confidence,
            'tokens_used': rag_result.tokens_used,
            'processing_time_ms': rag_result.processing_time_ms
        })
        if rag_result.verification:
            metadata['verification'] = rag_result.verification.to_dict()
        return rag_result.answer, sources, metadata

    message_text = rag_result.answer or rag_result.error or 'Unable to generate an answer.'
    metadata['error'] = rag_result.error
    return message_text, [], metadata


async def _resolve_final_response(
    message: str,
    session_id: str,
    context: Dict[str, Any],
    final_result: Dict[str, Any],
    execution_path_hint: Optional[Any]
) -> tuple[str, List[Dict[str, Any]], str, Dict[str, Any], bool]:
    """Ensure a non-empty assistant response, optionally invoking the fallback."""
    metadata = dict(final_result.get('metadata') or {})
    response_text = final_result.get('response', '')
    sources = final_result.get('sources', [])
    execution_path_value = final_result.get('execution_path')
    execution_path = _normalize_execution_path(execution_path_value or execution_path_hint)
    fallback_used = False

    if not isinstance(sources, list):
        sources = []

    if _is_placeholder_response(response_text):
        fallback_used = True
        fallback_response, fallback_sources, fallback_metadata = await _generate_fallback_answer(message, context)
        response_text = fallback_response
        sources = fallback_sources
        metadata.update(fallback_metadata)
        execution_path = 'rag_fallback'

    metadata.setdefault('session_id', session_id)
    metadata['fallback_used'] = fallback_used

    return response_text, sources, execution_path, metadata, fallback_used




async def generate_streaming_response(
    message: str,
    session_id: str,
    context: dict
) -> AsyncGenerator[str, None]:
    """Generate streaming response from execution graph with lightweight intent triage."""
    try:
        context = context or {}
        analysis = intent_agent.analyze(message, context)
        analysis_metadata = dict(analysis.get("metadata", {}))
        analysis_metadata.setdefault("model", LIGHT_INTENT_MODEL)

        if not analysis.get("requires_followup", False):
            lightweight_metadata = {
                **analysis_metadata,
                "initial_response": analysis["response"],
                "initial_model": LIGHT_INTENT_MODEL,
                "execution_stage": "light_intent",
                "fallback_used": False,
            }
            response_event = {
                "type": "response",
                "response": analysis["response"],
                "session_id": session_id,
                "sources": [],
                "execution_path": "light_intent",
                "metadata": lightweight_metadata,
            }
            yield f"data: {json.dumps(response_event)}\n\n"
            return

        progress_data = {
            "type": "progress",
            "node": "intent_analysis",
            "step": 0,
            "session_id": session_id,
            "status": "analysis_complete",
            "message": analysis["response"],
        }
        yield f"data: {json.dumps(progress_data)}\n\n"

        execution_state = ExecutionState(
            session_id=session_id,
            user_request=message,
            current_step=0,
            max_steps=system_config.orchestrator.max_iterations,
            tokens_used=0,
            token_budget=system_config.orchestrator.token_budget,
        )

        enriched_context = {
            **context,
            "intent_analysis": analysis_metadata,
            "analysis_message": analysis["response"],
        }
        execution_state.context.update(enriched_context)

        config = {"configurable": {"thread_id": session_id}}

        for step_output in execution_graph.stream(execution_state, config):
            node_name = list(step_output.keys())[0]
            state_dict = step_output[node_name]
            progress_update = {
                "type": "progress",
                "node": node_name,
                "step": state_dict.get("current_step", 0),
                "session_id": session_id,
            }
            yield f"data: {json.dumps(progress_update)}\n\n"

        final_state = execution_graph.get_state(config)
        final_result = _extract_final_result(final_state)
        execution_path_hint = _extract_execution_path_hint(final_state)

        response_text, sources, execution_path, metadata, fallback_used = await _resolve_final_response(
            message=message,
            session_id=session_id,
            context=enriched_context,
            final_result=final_result,
            execution_path_hint=execution_path_hint,
        )

        if fallback_used:
            fallback_progress = {
                "type": "progress",
                "node": "rag_fallback",
                "step": metadata.get("steps", 0),
                "session_id": session_id,
                "status": "fallback_invoked",
            }
            yield f"data: {json.dumps(fallback_progress)}\n\n"

        metadata = metadata or {}
        metadata.setdefault("analysis", analysis_metadata)
        metadata.setdefault("initial_response", analysis["response"])
        metadata.setdefault("initial_model", LIGHT_INTENT_MODEL)
        metadata.setdefault("execution_stage", "orchestrator")
        metadata["fallback_used"] = metadata.get("fallback_used", False) or fallback_used

        response_data = {
            "type": "response",
            "response": response_text,
            "session_id": session_id,
            "sources": sources,
            "execution_path": execution_path,
            "metadata": metadata,
        }

        yield f"data: {json.dumps(response_data)}\n\n"

    except Exception as e:
        error_data = {
            "type": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }
        yield f"data: {json.dumps(error_data)}\n\n"




@router.post("/", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """Process a chat message with lightweight intent analysis and optional orchestration."""
    try:
        session_id = request.session_id or str(uuid.uuid4())
        request_context = dict(request.context or {})

        analysis = intent_agent.analyze(request.message, request_context)
        analysis_metadata = dict(analysis.get("metadata", {}))
        analysis_metadata.setdefault("model", LIGHT_INTENT_MODEL)

        if not analysis.get("requires_followup", False):
            metadata = {
                **analysis_metadata,
                "initial_response": analysis["response"],
                "initial_model": LIGHT_INTENT_MODEL,
                "execution_stage": "light_intent",
                "fallback_used": False,
            }
            return ChatResponse(
                response=analysis["response"],
                session_id=session_id,
                sources=[],
                execution_path="light_intent",
                metadata=metadata,
            )

        request_context["intent_analysis"] = analysis_metadata
        request_context["analysis_message"] = analysis["response"]

        use_rag = bool(
            request_context.get("use_rag") or request_context.get("has_uploaded_files")
        )

        if use_rag:
            source_filter = request_context.get("source_filter")
            page_filter = request_context.get("page_filter")
            custom_k = request_context.get("retrieval_k")

            rag_result = rag_workflow.process_question(
                question=request.message,
                source_filter=source_filter,
                page_filter=page_filter,
                custom_k=custom_k,
            )

            if not rag_result.success:
                raise HTTPException(status_code=500, detail=rag_result.error)

            sources = [
                {
                    "chunk_id": citation.chunk_id,
                    "content": citation.content,
                    "source_file": citation.source_file,
                    "page_number": citation.page_number,
                    "score": citation.score,
                }
                for citation in rag_result.citations
            ]

            metadata = {
                "confidence": rag_result.confidence,
                "tokens_used": rag_result.tokens_used,
                "processing_time_ms": rag_result.processing_time_ms,
                "chunks_retrieved": rag_result.metadata.get("chunks_retrieved", 0),
                "question_type": rag_result.metadata.get("question_type"),
                "verification_passed": rag_result.verification.passed if rag_result.verification else None,
                "verification_confidence": rag_result.verification.confidence if rag_result.verification else None,
                **rag_result.metadata,
                "analysis": analysis_metadata,
                "initial_response": analysis["response"],
                "initial_model": LIGHT_INTENT_MODEL,
                "execution_stage": "rag_qa",
                "fallback_used": False,
            }

            return ChatResponse(
                response=rag_result.answer,
                session_id=session_id,
                sources=sources,
                execution_path="rag_qa",
                metadata=metadata,
            )

        execution_state = ExecutionState(
            session_id=session_id,
            user_request=request.message,
            current_step=0,
            max_steps=system_config.orchestrator.max_iterations,
            tokens_used=0,
            token_budget=system_config.orchestrator.token_budget,
        )

        orchestrator_result = orchestrator.process_request(
            user_request=request.message,
            context=request_context,
            execution_state=execution_state,
        )
        final_result = orchestrator_result if isinstance(orchestrator_result, dict) else {}

        response_text, sources, execution_path, metadata, fallback_used = await _resolve_final_response(
            message=request.message,
            session_id=session_id,
            context=request_context,
            final_result=final_result,
            execution_path_hint=final_result.get("execution_path"),
        )

        metadata = metadata or {}
        metadata.setdefault("analysis", analysis_metadata)
        metadata.setdefault("initial_response", analysis["response"])
        metadata.setdefault("initial_model", LIGHT_INTENT_MODEL)
        metadata.setdefault("execution_stage", "orchestrator")
        metadata["fallback_used"] = metadata.get("fallback_used", False) or fallback_used

        return ChatResponse(
            response=response_text,
            session_id=session_id,
            sources=sources,
            execution_path=execution_path,
            metadata=metadata,
        )

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))



@router.post("/stream")
async def chat_stream(request: ChatRequest):
    """
    Process a chat message and stream the response.
    
    - **message**: User message to process
    - **session_id**: Optional session ID for conversation continuity
    - **context**: Optional additional context
    """
    session_id = request.session_id or str(uuid.uuid4())
    
    return StreamingResponse(
        generate_streaming_response(request.message, session_id, request.context or {}),
        media_type="text/event-stream"
    )


@router.post("/rag", response_model=ChatResponse)
async def rag_question(request: ChatRequest) -> ChatResponse:
    """
    Process a question using RAG (Retrieval-Augmented Generation).
    
    This endpoint is optimized for document question answering with:
    - Semantic search over uploaded documents
    - Answer generation with citations
    - Source verification
    
    - **message**: Question to ask about documents
    - **session_id**: Optional session ID
    - **context**: Optional context with source_filter, page_filter, etc.
    """
    try:
        session_id = request.session_id or str(uuid.uuid4())
        
        # Extract RAG-specific parameters from context
        source_filter = request.context.get("source_filter") if request.context else None
        page_filter = request.context.get("page_filter") if request.context else None
        custom_k = request.context.get("retrieval_k") if request.context else None
        
        # Process question through RAG workflow
        rag_result = rag_workflow.process_question(
            question=request.message,
            source_filter=source_filter,
            page_filter=page_filter,
            custom_k=custom_k
        )
        
        if not rag_result.success:
            raise HTTPException(status_code=500, detail=rag_result.error)
        
        # Convert citations to sources format
        sources = [
            {
                "chunk_id": citation.chunk_id,
                "content": citation.content,
                "source_file": citation.source_file,
                "page_number": citation.page_number,
                "score": citation.score
            }
            for citation in rag_result.citations
        ]
        
        return ChatResponse(
            response=rag_result.answer,
            session_id=session_id,
            sources=sources,
            execution_path="rag_qa",
            metadata={
                "confidence": rag_result.confidence,
                "tokens_used": rag_result.tokens_used,
                "processing_time_ms": rag_result.processing_time_ms,
                "chunks_retrieved": rag_result.metadata.get("chunks_retrieved", 0),
                "question_type": rag_result.metadata.get("question_type"),
                "verification_passed": rag_result.verification.passed if rag_result.verification else None,
                "verification_confidence": rag_result.verification.confidence if rag_result.verification else None,
                **rag_result.metadata,
                "fallback_used": False
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
