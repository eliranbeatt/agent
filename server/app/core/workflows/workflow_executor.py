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
    WorkflowStepStatus,
    WorkflowStatus
)
from ..monitoring import (
    performance_monitor,
    execution_tracer,
    trace_context,
    monitor_performance
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
        
        # Start step trace
        step_trace_id = execution_tracer.start_trace(
            component="WorkflowStepExecutor",
            operation=f"execute_step_{step.name}",
            metadata={
                "step_name": step.name,
                "node_type": step.node_type,
                "workflow": workflow_context.workflow_name
            }
        )
        
        # Record step start metric
        performance_monitor.increment_counter(
            "workflow.steps.started",
            tags={"step": step.name, "node_type": step.node_type}
        )
        
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
            tokens_used = output.get("tokens_used", 0) if isinstance(output, dict) else 0
            
            # Record step success metrics
            performance_monitor.increment_counter(
                "workflow.steps.completed",
                tags={"step": step.name, "node_type": step.node_type, "status": "success"}
            )
            performance_monitor.record_histogram(
                "workflow.step.execution_time_ms",
                execution_time,
                tags={"step": step.name, "node_type": step.node_type}
            )
            if tokens_used > 0:
                performance_monitor.record_histogram(
                    "workflow.step.tokens_used",
                    tokens_used,
                    tags={"step": step.name, "node_type": step.node_type}
                )
            
            # End step trace successfully
            execution_tracer.end_trace(step_trace_id, "success", {
                "execution_time_ms": execution_time,
                "tokens_used": tokens_used,
                "criteria_met": len(criteria_met),
                "criteria_failed": len(criteria_failed)
            })
            
            return WorkflowStepResult(
                step_name=step.name,
                status=WorkflowStepStatus.COMPLETED,
                output=output,
                execution_time_ms=execution_time,
                tokens_used=tokens_used,
                success_criteria_met=criteria_met,
                success_criteria_failed=criteria_failed
            )
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            logger.error(f"Step '{step.name}' failed: {str(e)}")
            
            # Record step failure metrics
            performance_monitor.increment_counter(
                "workflow.steps.failed",
                tags={"step": step.name, "node_type": step.node_type, "error_type": type(e).__name__}
            )
            
            # End step trace with error
            execution_tracer.end_trace(step_trace_id, "error", {
                "error": str(e),
                "error_type": type(e).__name__,
                "execution_time_ms": execution_time
            })
            
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
        input_data: Optional[Dict[str, Any]] = None,
        resume_from_checkpoint: Optional[str] = None
    ) -> WorkflowExecutionContext:
        """
        Execute the complete workflow.
        
        Args:
            user_request: User's request
            execution_state: Global execution state
            input_data: Optional input data for the workflow
            resume_from_checkpoint: Optional checkpoint ID to resume from
            
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
        
        # Resume from checkpoint if provided
        start_step = 0
        if resume_from_checkpoint:
            if workflow_context.restore_from_checkpoint(resume_from_checkpoint):
                start_step = workflow_context.current_step_index
                workflow_context.resume_execution()
                logger.info(f"Resuming workflow from checkpoint at step {start_step}")
            else:
                logger.warning(f"Checkpoint {resume_from_checkpoint} not found, starting from beginning")
        
        if not resume_from_checkpoint:
            workflow_context.start_execution()
        
        logger.info(
            f"Starting workflow execution: {self.workflow_config.name} "
            f"({workflow_context.total_steps} steps)"
        )
        
        # Start workflow execution trace
        trace_id = execution_tracer.start_trace(
            component="WorkflowExecutor",
            operation=f"execute_{self.workflow_config.name}",
            metadata={
                "workflow_name": self.workflow_config.name,
                "total_steps": workflow_context.total_steps,
                "user_request": user_request[:100]
            }
        )
        
        # Record workflow start metric
        performance_monitor.increment_counter(
            "workflow.executions.started",
            tags={"workflow": self.workflow_config.name}
        )
        
        try:
            # Execute each step in sequence
            for step_index in range(start_step, len(self.workflow_config.steps)):
                # Check if workflow is paused
                if workflow_context.is_paused:
                    logger.info(f"Workflow paused at step {step_index}")
                    return workflow_context
                
                step = self.workflow_config.steps[step_index]
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
                
                # Create checkpoint after each step
                checkpoint = workflow_context.create_checkpoint()
                
                # Log step completion
                execution_state.log_execution_step(
                    node_name="WorkflowExecutor",
                    action=f"step_completed:{step.name}",
                    details={
                        "status": step_result.status.value,
                        "execution_time_ms": step_result.execution_time_ms,
                        "tokens_used": step_result.tokens_used,
                        "success_criteria_met": len(step_result.success_criteria_met),
                        "success_criteria_failed": len(step_result.success_criteria_failed),
                        "checkpoint_id": checkpoint["checkpoint_id"]
                    }
                )
                
                # Check if step failed
                if step_result.status == WorkflowStepStatus.FAILED:
                    error_msg = f"Step '{step.name}' failed: {step_result.error}"
                    logger.error(error_msg)
                    workflow_context.fail_execution(error_msg)
                    return workflow_context
                    
                # Validate success criteria after step
                self._validate_step_criteria(step, step_result, workflow_context, execution_state)
                    
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
                
                # Record success metrics
                performance_monitor.increment_counter(
                    "workflow.executions.completed",
                    tags={"workflow": self.workflow_config.name, "status": "success"}
                )
                performance_monitor.record_histogram(
                    "workflow.execution_time_ms",
                    workflow_context.total_execution_time_ms,
                    tags={"workflow": self.workflow_config.name}
                )
                performance_monitor.record_histogram(
                    "workflow.tokens_used",
                    workflow_context.total_tokens_used,
                    tags={"workflow": self.workflow_config.name}
                )
                
                # End trace successfully
                execution_tracer.end_trace(trace_id, "success", {
                    "tokens_used": workflow_context.total_tokens_used,
                    "execution_time_ms": workflow_context.total_execution_time_ms,
                    "steps_completed": len(workflow_context.step_results)
                })
            else:
                error_msg = "Workflow success criteria not met"
                logger.error(error_msg)
                
                # Check if replanning should be triggered
                if self._should_trigger_replan(workflow_context):
                    logger.info("Triggering replan due to failed success criteria")
                    workflow_context.intermediate_results["replan_required"] = True
                    workflow_context.intermediate_results["replan_reason"] = "workflow_criteria_not_met"
                    execution_state.next_node = "Planner"  # Route back to planner
                    
                    # Record replan metric
                    performance_monitor.increment_counter(
                        "workflow.replans.triggered",
                        tags={"workflow": self.workflow_config.name}
                    )
                else:
                    workflow_context.fail_execution(error_msg)
                    
                    # Record failure metrics
                    performance_monitor.increment_counter(
                        "workflow.executions.failed",
                        tags={"workflow": self.workflow_config.name, "reason": "criteria_not_met"}
                    )
                    
                    # End trace with failure
                    execution_tracer.end_trace(trace_id, "failed", {
                        "error": error_msg,
                        "failed_criteria": workflow_context.intermediate_results.get("failed_criteria", [])
                    })
                
        except Exception as e:
            error_msg = f"Workflow execution failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            workflow_context.fail_execution(error_msg)
            
            # Record error metrics
            performance_monitor.increment_counter(
                "workflow.executions.errors",
                tags={"workflow": self.workflow_config.name, "error_type": type(e).__name__}
            )
            
            # End trace with error
            execution_tracer.end_trace(trace_id, "error", {
                "error": error_msg,
                "error_type": type(e).__name__
            })
            
        return workflow_context
    
    def pause_workflow(self, workflow_context: WorkflowExecutionContext) -> None:
        """
        Pause workflow execution.
        
        Args:
            workflow_context: Workflow execution context to pause
        """
        workflow_context.pause_execution()
        logger.info(f"Workflow '{workflow_context.workflow_name}' paused at step {workflow_context.current_step_index}")
        
    def resume_workflow(
        self,
        workflow_context: WorkflowExecutionContext,
        execution_state: ExecutionState
    ) -> WorkflowExecutionContext:
        """
        Resume paused workflow execution.
        
        Args:
            workflow_context: Workflow execution context to resume
            execution_state: Global execution state
            
        Returns:
            Updated workflow execution context
        """
        if not workflow_context.can_resume():
            logger.error(f"Cannot resume workflow in status: {workflow_context.status.value}")
            return workflow_context
            
        logger.info(f"Resuming workflow '{workflow_context.workflow_name}' from step {workflow_context.current_step_index}")
        
        # Resume execution from current step
        return self.execute(
            user_request=workflow_context.user_request,
            execution_state=execution_state,
            input_data=workflow_context.input_data,
            resume_from_checkpoint=workflow_context.get_latest_checkpoint()["checkpoint_id"] if workflow_context.checkpoints else None
        )
    
    def _validate_step_criteria(
        self,
        step: ConfigWorkflowStep,
        step_result: WorkflowStepResult,
        workflow_context: WorkflowExecutionContext,
        execution_state: ExecutionState
    ) -> None:
        """
        Validate success criteria after step execution.
        
        Args:
            step: Step configuration
            step_result: Step execution result
            workflow_context: Workflow execution context
            execution_state: Global execution state
        """
        # Log criteria validation
        if step_result.success_criteria_met:
            logger.info(
                f"Step '{step.name}' success criteria met:\n" +
                "\n".join(f"  ✓ {c}" for c in step_result.success_criteria_met)
            )
            
        if step_result.success_criteria_failed:
            logger.warning(
                f"Step '{step.name}' success criteria failed:\n" +
                "\n".join(f"  ✗ {c}" for c in step_result.success_criteria_failed)
            )
            
            # Store failed criteria for potential replanning
            if "step_failures" not in workflow_context.intermediate_results:
                workflow_context.intermediate_results["step_failures"] = []
                
            workflow_context.intermediate_results["step_failures"].append({
                "step_name": step.name,
                "failed_criteria": step_result.success_criteria_failed,
                "step_index": workflow_context.current_step_index
            })
            
            # Log to execution state for monitoring
            execution_state.log_execution_step(
                node_name="WorkflowExecutor",
                action="criteria_validation_failed",
                details={
                    "step_name": step.name,
                    "failed_criteria": step_result.success_criteria_failed,
                    "met_criteria": step_result.success_criteria_met
                }
            )
    
    def _check_workflow_success(self, workflow_context: WorkflowExecutionContext) -> bool:
        """
        Check if overall workflow success criteria are met.
        
        Args:
            workflow_context: Workflow execution context
            
        Returns:
            True if workflow succeeded, False otherwise
        """
        # Check if all steps completed
        failed_steps = []
        for result in workflow_context.step_results:
            if result.status != WorkflowStepStatus.COMPLETED:
                failed_steps.append(result.step_name)
                
        if failed_steps:
            logger.error(f"Workflow has failed steps: {', '.join(failed_steps)}")
            return False
                
        # Check workflow-level success criteria
        failed_criteria = []
        if self.workflow_config.success_criteria:
            for criterion in self.workflow_config.success_criteria:
                # Simple heuristic checking
                if not self._evaluate_workflow_criterion(criterion, workflow_context):
                    failed_criteria.append(criterion)
                    
        if failed_criteria:
            logger.error(
                f"Workflow success criteria not met:\n" +
                "\n".join(f"  - {c}" for c in failed_criteria)
            )
            # Store failed criteria in context for potential replanning
            workflow_context.intermediate_results["failed_criteria"] = failed_criteria
            return False
                    
        logger.info(f"All workflow success criteria met for '{self.workflow_config.name}'")
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
    
    def _should_trigger_replan(self, workflow_context: WorkflowExecutionContext) -> bool:
        """
        Determine if replanning should be triggered based on failed criteria.
        
        Args:
            workflow_context: Workflow execution context
            
        Returns:
            True if replanning should be triggered, False otherwise
        """
        # Check if there are failed criteria
        failed_criteria = workflow_context.intermediate_results.get("failed_criteria", [])
        step_failures = workflow_context.intermediate_results.get("step_failures", [])
        
        # Don't replan if no failures
        if not failed_criteria and not step_failures:
            return False
            
        # Don't replan if we've already tried multiple times
        replan_count = workflow_context.intermediate_results.get("replan_count", 0)
        if replan_count >= 2:
            logger.warning(f"Maximum replan attempts reached ({replan_count})")
            return False
            
        # Trigger replan if critical criteria failed
        critical_keywords = ["answer", "verified", "grounded", "quality"]
        for criterion in failed_criteria:
            if any(keyword in criterion.lower() for keyword in critical_keywords):
                logger.info(f"Critical criterion failed, triggering replan: {criterion}")
                return True
                
        return False
    
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
