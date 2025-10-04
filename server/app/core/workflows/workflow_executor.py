"""Workflow execution engine with step-by-step processing."""

import logging
import time
from typing import Dict, Any, Optional, List
from datetime import datetime

from ...config.models import WorkflowConfig, WorkflowStep as ConfigWorkflowStep
from ..state import ExecutionState
from .workflow_models import (
    WorkflowExecutionContext,
    WorkflowStepResult,
    WorkflowStepStatus
)


logger = logging.getLogger(__name__)


class WorkflowStepExecutor:
    """Executes individual workflow steps."""
    
    def __init__(self, context_manager=None, rag_workflow=None, memory_manager=None):
        """
        Initialize step executor with required components.
        
        Args:
            context_manager: Context manager for file operations
            rag_workflow: RAG workflow for QA operations
            memory_manager: Memory manager for persistence
        """
        self.context_manager = context_manager
        self.rag_workflow = rag_workflow
        self.memory_manager = memory_manager
        
    def execute_step(
        self,
        step: ConfigWorkflowStep,
        workflow_context: WorkflowExecutionContext,
        execution_state: ExecutionState
    ) -> WorkflowStepResult:
        """
        Execute a single workflow step.
        
        Args:
            step: Step configuration
            workflow_context: Workflow execution context
            execution_state: Global execution state
            
        Returns:
            WorkflowStepResult with execution outcome
        """
        start_time = time.time()
        logger.info(f"Executing workflow step: {step.name}")
        
        try:
            # Route to appropriate handler based on node type
            if step.node_type == "context_manager":
                output = self._execute_context_step(step, workflow_context, execution_state)
            elif step.node_type == "llm_generator":
                output = self._execute_llm_step(step, workflow_context, execution_state)
            elif step.node_type == "evaluator":
                output = self._execute_evaluator_step(step, workflow_context, execution_state)
            elif step.node_type == "analyzer":
                output = self._execute_analyzer_step(step, workflow_context, execution_state)
            elif step.node_type == "summarizer":
                output = self._execute_summarizer_step(step, workflow_context, execution_state)
            elif step.node_type == "fact_extractor":
                output = self._execute_fact_extractor_step(step, workflow_context, execution_state)
            elif step.node_type == "comparator":
                output = self._execute_comparator_step(step, workflow_context, execution_state)
            elif step.node_type == "synthesizer":
                output = self._execute_synthesizer_step(step, workflow_context, execution_state)
            elif step.node_type == "image_processor":
                output = self._execute_image_processor_step(step, workflow_context, execution_state)
            elif step.node_type == "ocr_engine":
                output = self._execute_ocr_step(step, workflow_context, execution_state)
            elif step.node_type == "text_processor":
                output = self._execute_text_processor_step(step, workflow_context, execution_state)
            elif step.node_type == "qa_engine":
                output = self._execute_qa_step(step, workflow_context, execution_state)
            else:
                raise ValueError(f"Unknown node type: {step.node_type}")
                
            # Check success criteria
            criteria_met, criteria_failed = self._check_success_criteria(
                step.success_criteria,
                output,
                workflow_context
            )
            
            execution_time = (time.time() - start_time) * 1000  # Convert to ms
            
            return WorkflowStepResult(
                step_name=step.name,
                status=WorkflowStepStatus.COMPLETED,
                output=output,
                execution_time_ms=execution_time,
                tokens_used=output.get("tokens_used", 0) if isinstance(output, dict) else 0,
                success_criteria_met=criteria_met,
                success_criteria_failed=criteria_failed
            )
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            logger.error(f"Step '{step.name}' failed: {str(e)}")
            
            return WorkflowStepResult(
                step_name=step.name,
                status=WorkflowStepStatus.FAILED,
                error=str(e),
                execution_time_ms=execution_time
            )
    
    def _execute_context_step(
        self,
        step: ConfigWorkflowStep,
        workflow_context: WorkflowExecutionContext,
        execution_state: ExecutionState
    ) -> Dict[str, Any]:
        """Execute context manager step (retrieval, chunking, etc.)."""
        params = step.parameters
        
        if "retrieval_k" in params:
            # This is a retrieval step
            query = workflow_context.user_request
            k = params.get("retrieval_k", 10)
            use_mmr = params.get("use_mmr", True)
            
            if self.context_manager:
                chunks = self.context_manager.retrieve_context(
                    query=query,
                    k=k,
                    use_mmr=use_mmr
                )
                
                # Store in execution state
                execution_state.retrieved_chunks = chunks
                
                return {
                    "chunks": chunks,
                    "num_chunks": len(chunks),
                    "query": query
                }
            else:
                logger.warning("Context manager not available")
                return {"chunks": [], "num_chunks": 0}
                
        else:
            # Generic context operation
            return {"status": "completed", "message": "Context step executed"}
    
    def _execute_llm_step(
        self,
        step: ConfigWorkflowStep,
        workflow_context: WorkflowExecutionContext,
        execution_state: ExecutionState
    ) -> Dict[str, Any]:
        """Execute LLM generation step."""
        params = step.parameters
        
        # Get retrieved chunks from previous step or state
        chunks = execution_state.retrieved_chunks
        
        if self.rag_workflow and chunks:
            # Use RAG workflow for answer generation
            result = self.rag_workflow.generate_answer(
                question=workflow_context.user_request,
                chunks=chunks,
                max_tokens=params.get("max_tokens", 1000),
                temperature=params.get("temperature", 0.1),
                include_citations=params.get("include_citations", True)
            )
            return result
        else:
            # Placeholder for direct LLM call
            return {
                "answer": "LLM response placeholder",
                "citations": [],
                "tokens_used": 100
            }
    
    def _execute_evaluator_step(
        self,
        step: ConfigWorkflowStep,
        workflow_context: WorkflowExecutionContext,
        execution_state: ExecutionState
    ) -> Dict[str, Any]:
        """Execute evaluator/verifier step."""
        params = step.parameters
        
        # Get answer from previous step
        answer_result = workflow_context.get_intermediate_result("answer_generation")
        
        if answer_result and self.rag_workflow:
            answer = answer_result.get("answer", "")
            chunks = execution_state.retrieved_chunks
            
            # Perform verification
            verification = self.rag_workflow.verify_answer(
                answer=answer,
                chunks=chunks,
                check_grounding=params.get("check_grounding", True),
                check_contradictions=params.get("check_contradictions", True)
            )
            
            return verification
        else:
            return {
                "is_grounded": True,
                "has_contradictions": False,
                "quality_score": 0.8
            }
    
    def _execute_analyzer_step(
        self,
        step: ConfigWorkflowStep,
        workflow_context: WorkflowExecutionContext,
        execution_state: ExecutionState
    ) -> Dict[str, Any]:
        """Execute content analyzer step."""
        # Placeholder for content analysis
        return {
            "entities": [],
            "facts": [],
            "themes": []
        }
    
    def _execute_summarizer_step(
        self,
        step: ConfigWorkflowStep,
        workflow_context: WorkflowExecutionContext,
        execution_state: ExecutionState
    ) -> Dict[str, Any]:
        """Execute summarizer step."""
        # Placeholder for summarization
        return {
            "summary": "Document summary placeholder",
            "key_points": [],
            "statistics": {}
        }
    
    def _execute_fact_extractor_step(
        self,
        step: ConfigWorkflowStep,
        workflow_context: WorkflowExecutionContext,
        execution_state: ExecutionState
    ) -> Dict[str, Any]:
        """Execute fact extractor step."""
        # Placeholder for fact extraction
        return {
            "facts": [],
            "fact_types": {},
            "confidence_scores": {}
        }
    
    def _execute_comparator_step(
        self,
        step: ConfigWorkflowStep,
        workflow_context: WorkflowExecutionContext,
        execution_state: ExecutionState
    ) -> Dict[str, Any]:
        """Execute document comparator step."""
        # Placeholder for comparison
        return {
            "similarities": [],
            "differences": [],
            "conflicts": []
        }
    
    def _execute_synthesizer_step(
        self,
        step: ConfigWorkflowStep,
        workflow_context: WorkflowExecutionContext,
        execution_state: ExecutionState
    ) -> Dict[str, Any]:
        """Execute synthesizer step."""
        # Placeholder for synthesis
        return {
            "synthesis": "Synthesized insights placeholder",
            "sources": [],
            "confidence": 0.8
        }
    
    def _execute_image_processor_step(
        self,
        step: ConfigWorkflowStep,
        workflow_context: WorkflowExecutionContext,
        execution_state: ExecutionState
    ) -> Dict[str, Any]:
        """Execute image preprocessing step."""
        # Placeholder for image processing
        return {
            "processed": True,
            "quality_improved": True,
            "text_regions": []
        }
    
    def _execute_ocr_step(
        self,
        step: ConfigWorkflowStep,
        workflow_context: WorkflowExecutionContext,
        execution_state: ExecutionState
    ) -> Dict[str, Any]:
        """Execute OCR extraction step."""
        # Placeholder for OCR
        return {
            "extracted_text": "OCR text placeholder",
            "confidence": 0.85,
            "structure_preserved": True
        }
    
    def _execute_text_processor_step(
        self,
        step: ConfigWorkflowStep,
        workflow_context: WorkflowExecutionContext,
        execution_state: ExecutionState
    ) -> Dict[str, Any]:
        """Execute text processing step."""
        # Placeholder for text processing
        return {
            "cleaned_text": "Processed text placeholder",
            "chunks": [],
            "ready_for_qa": True
        }
    
    def _execute_qa_step(
        self,
        step: ConfigWorkflowStep,
        workflow_context: WorkflowExecutionContext,
        execution_state: ExecutionState
    ) -> Dict[str, Any]:
        """Execute QA engine step."""
        # Placeholder for QA
        return {
            "answer": "QA answer placeholder",
            "confidence": 0.8,
            "sources": []
        }
    
    def _check_success_criteria(
        self,
        criteria: List[str],
        output: Any,
        workflow_context: WorkflowExecutionContext
    ) -> tuple[List[str], List[str]]:
        """
        Check if success criteria are met.
        
        Args:
            criteria: List of success criteria strings
            output: Step output to check
            workflow_context: Workflow execution context
            
        Returns:
            Tuple of (criteria_met, criteria_failed)
        """
        criteria_met = []
        criteria_failed = []
        
        if not criteria:
            return criteria_met, criteria_failed
            
        for criterion in criteria:
            # Simple heuristic checking
            if self._evaluate_criterion(criterion, output, workflow_context):
                criteria_met.append(criterion)
            else:
                criteria_failed.append(criterion)
                
        return criteria_met, criteria_failed
    
    def _evaluate_criterion(
        self,
        criterion: str,
        output: Any,
        workflow_context: WorkflowExecutionContext
    ) -> bool:
        """Evaluate a single success criterion."""
        criterion_lower = criterion.lower()
        
        if not isinstance(output, dict):
            return False
            
        # Check for common criterion patterns
        if "retrieved" in criterion_lower and "chunks" in criterion_lower:
            num_chunks = output.get("num_chunks", 0)
            return num_chunks >= 3
            
        if "similarity" in criterion_lower and "score" in criterion_lower:
            # Check if chunks have good similarity
            chunks = output.get("chunks", [])
            if chunks:
                avg_score = sum(c.get("score", 0) for c in chunks) / len(chunks)
                return avg_score > 0.7
            return False
            
        if "grounded" in criterion_lower:
            return output.get("is_grounded", False)
            
        if "citations" in criterion_lower:
            citations = output.get("citations", [])
            return len(citations) > 0
            
        if "quality score" in criterion_lower:
            quality = output.get("quality_score", 0)
            return quality >= 0.7
            
        if "contradictions" in criterion_lower:
            return not output.get("has_contradictions", True)
            
        # Default: assume criterion is met if output exists
        return True


class WorkflowExecutor:
    """
    Main workflow execution engine.
    
    Executes predefined workflows step-by-step with proper error handling,
    success criteria checking, and result tracking.
    """
    
    def __init__(
        self,
        workflow_config: WorkflowConfig,
        context_manager=None,
        rag_workflow=None,
        memory_manager=None
    ):
        """
        Initialize workflow executor.
        
        Args:
            workflow_config: Configuration for the workflow
            context_manager: Context manager for file operations
            rag_workflow: RAG workflow for QA operations
            memory_manager: Memory manager for persistence
        """
        self.workflow_config = workflow_config
        self.step_executor = WorkflowStepExecutor(
            context_manager=context_manager,
            rag_workflow=rag_workflow,
            memory_manager=memory_manager
        )
        
    def execute(
        self,
        user_request: str,
        execution_state: ExecutionState,
        input_data: Optional[Dict[str, Any]] = None
    ) -> WorkflowExecutionContext:
        """
        Execute the complete workflow.
        
        Args:
            user_request: User's request
            execution_state: Global execution state
            input_data: Optional input data for the workflow
            
        Returns:
            WorkflowExecutionContext with execution results
        """
        # Initialize workflow context
        workflow_context = WorkflowExecutionContext(
            workflow_name=self.workflow_config.name,
            workflow_description=self.workflow_config.description,
            user_request=user_request,
            total_steps=len(self.workflow_config.steps),
            input_data=input_data or {}
        )
        
        workflow_context.start_execution()
        
        logger.info(
            f"Starting workflow execution: {self.workflow_config.name} "
            f"({workflow_context.total_steps} steps)"
        )
        
        try:
            # Execute each step in sequence
            for step_index, step in enumerate(self.workflow_config.steps):
                workflow_context.current_step_index = step_index
                
                logger.info(
                    f"Step {step_index + 1}/{workflow_context.total_steps}: {step.name}"
                )
                
                # Execute the step
                step_result = self.step_executor.execute_step(
                    step=step,
                    workflow_context=workflow_context,
                    execution_state=execution_state
                )
                
                # Add result to context
                workflow_context.add_step_result(step_result)
                
                # Update execution state
                execution_state.increment_step()
                execution_state.add_tokens_used(step_result.tokens_used)
                
                # Log step completion
                execution_state.log_execution_step(
                    node_name="WorkflowExecutor",
                    action=f"step_completed:{step.name}",
                    details={
                        "status": step_result.status.value,
                        "execution_time_ms": step_result.execution_time_ms,
                        "tokens_used": step_result.tokens_used,
                        "success_criteria_met": len(step_result.success_criteria_met),
                        "success_criteria_failed": len(step_result.success_criteria_failed)
                    }
                )
                
                # Check if step failed
                if step_result.status == WorkflowStepStatus.FAILED:
                    error_msg = f"Step '{step.name}' failed: {step_result.error}"
                    logger.error(error_msg)
                    workflow_context.fail_execution(error_msg)
                    return workflow_context
                    
                # Check if critical success criteria failed
                if step_result.success_criteria_failed:
                    logger.warning(
                        f"Step '{step.name}' has failed criteria: "
                        f"{step_result.success_criteria_failed}"
                    )
                    
            # Check overall workflow success criteria
            workflow_success = self._check_workflow_success(workflow_context)
            
            if workflow_success:
                # Extract final output
                final_output = self._extract_final_output(workflow_context)
                workflow_context.complete_execution(final_output)
                
                logger.info(
                    f"Workflow '{self.workflow_config.name}' completed successfully "
                    f"(tokens: {workflow_context.total_tokens_used}, "
                    f"time: {workflow_context.total_execution_time_ms:.0f}ms)"
                )
            else:
                error_msg = "Workflow success criteria not met"
                logger.error(error_msg)
                workflow_context.fail_execution(error_msg)
                
        except Exception as e:
            error_msg = f"Workflow execution failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            workflow_context.fail_execution(error_msg)
            
        return workflow_context
    
    def _check_workflow_success(self, workflow_context: WorkflowExecutionContext) -> bool:
        """
        Check if overall workflow success criteria are met.
        
        Args:
            workflow_context: Workflow execution context
            
        Returns:
            True if workflow succeeded, False otherwise
        """
        # Check if all steps completed
        for result in workflow_context.step_results:
            if result.status != WorkflowStepStatus.COMPLETED:
                return False
                
        # Check workflow-level success criteria
        if self.workflow_config.success_criteria:
            for criterion in self.workflow_config.success_criteria:
                # Simple heuristic checking
                if not self._evaluate_workflow_criterion(criterion, workflow_context):
                    logger.warning(f"Workflow criterion not met: {criterion}")
                    return False
                    
        return True
    
    def _evaluate_workflow_criterion(
        self,
        criterion: str,
        workflow_context: WorkflowExecutionContext
    ) -> bool:
        """Evaluate a workflow-level success criterion."""
        criterion_lower = criterion.lower()
        
        # Check for common patterns
        if "answered" in criterion_lower or "answer" in criterion_lower:
            # Check if we have an answer
            for result in workflow_context.step_results:
                if result.output and isinstance(result.output, dict):
                    if "answer" in result.output:
                        return True
            return False
            
        if "citations" in criterion_lower or "sources" in criterion_lower:
            # Check if we have citations
            for result in workflow_context.step_results:
                if result.output and isinstance(result.output, dict):
                    if result.output.get("citations") or result.output.get("sources"):
                        return True
            return False
            
        if "verified" in criterion_lower or "accuracy" in criterion_lower:
            # Check if verification passed
            for result in workflow_context.step_results:
                if "verif" in result.step_name.lower() or "evaluat" in result.step_name.lower():
                    if result.status == WorkflowStepStatus.COMPLETED:
                        return True
            return False
            
        # Default: assume criterion is met
        return True
    
    def _extract_final_output(self, workflow_context: WorkflowExecutionContext) -> Dict[str, Any]:
        """
        Extract final output from workflow execution.
        
        Args:
            workflow_context: Workflow execution context
            
        Returns:
            Final output dictionary
        """
        final_output = {
            "workflow_name": workflow_context.workflow_name,
            "success": True,
            "total_tokens_used": workflow_context.total_tokens_used,
            "total_execution_time_ms": workflow_context.total_execution_time_ms,
            "steps_completed": len(workflow_context.step_results)
        }
        
        # Extract key results from steps
        for result in workflow_context.step_results:
            if result.output and isinstance(result.output, dict):
                # Merge important outputs
                if "answer" in result.output:
                    final_output["answer"] = result.output["answer"]
                if "citations" in result.output:
                    final_output["citations"] = result.output["citations"]
                if "summary" in result.output:
                    final_output["summary"] = result.output["summary"]
                if "facts" in result.output:
                    final_output["facts"] = result.output["facts"]
                if "synthesis" in result.output:
                    final_output["synthesis"] = result.output["synthesis"]
                    
        return final_output
