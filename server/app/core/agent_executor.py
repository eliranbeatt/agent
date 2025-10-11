"""Sub-agent execution framework for Local Agent Studio."""

import json
import logging
import os
import threading
import time
import uuid

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

import openai
from openai import OpenAI

from ..config.models import SystemConfig
from .base_nodes import ProcessingNode
from .state import ExecutionState, Task, AgentSpec, TaskStatus


logger = logging.getLogger(__name__)


class AgentExecutionStatus(Enum):
    """Status of agent execution."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


@dataclass
class AgentExecutionResult:
    """Result of agent execution."""
    agent_id: str
    task_id: str
    status: AgentExecutionStatus
    result: Optional[Any] = None
    error: Optional[str] = None
    execution_time: float = 0.0
    steps_taken: int = 0
    tokens_used: int = 0
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    execution_log: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class RunningAgent:
    """Information about a currently running agent."""
    agent_id: str
    task_id: str
    agent_spec: AgentSpec
    started_at: datetime
    timeout_at: datetime
    current_step: int = 0
    tokens_used: int = 0
    is_cancelled: bool = False
    execution_thread: Optional[threading.Thread] = None
    result_future: Optional[Any] = None



class SubAgentRunner:
    """Handles the actual execution of individual sub-agents."""

    def __init__(self, config: SystemConfig):
        """
        Initialize sub-agent runner.

        Args:
            config: System configuration
        """
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.SubAgentRunner")
        self.api_key = config.openai_api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "OpenAI API key is required for agent execution. "
                "Set OPENAI_API_KEY in your environment or system configuration."
            )
        try:
            self.client = OpenAI(api_key=self.api_key)
        except Exception as exc:
            self.logger.error("Failed to initialize OpenAI client: %s", exc)
            raise

        self.default_model = (
            os.getenv("OPENAI_AGENT_MODEL")
            or os.getenv("OPENAI_MODEL")
            or "gpt-4o-mini"
        )
        try:
            self.default_temperature = float(os.getenv("OPENAI_AGENT_TEMPERATURE", "0.2"))
        except ValueError:
            self.default_temperature = 0.2

    def execute_agent(self, agent_spec: AgentSpec, task: Task, context: Dict[str, Any]) -> AgentExecutionResult:
        """
        Execute a single sub-agent.

        Args:
            agent_spec: Agent specification
            task: Task to execute
            context: Execution context

        Returns:
            Agent execution result
        """
        start_time = datetime.now()
        result = AgentExecutionResult(
            agent_id=agent_spec.id,
            task_id=task.id,
            status=AgentExecutionStatus.RUNNING,
            started_at=start_time
        )

        try:
            agent_output, usage_metadata = self._run_openai_agent(agent_spec, task, context)

            result.result = agent_output
            result.status = AgentExecutionStatus.COMPLETED
            result.completed_at = datetime.now()
            result.execution_time = (result.completed_at - start_time).total_seconds()

            total_tokens = usage_metadata.get("total_tokens")
            if total_tokens:
                result.tokens_used = total_tokens
            else:
                prompt_tokens = usage_metadata.get("prompt_tokens") or 0
                completion_tokens = usage_metadata.get("completion_tokens") or 0
                result.tokens_used = prompt_tokens + completion_tokens

            result.steps_taken = self._estimate_steps(agent_output)

            log_entry = {
                "event": "agent_completion",
                "model": usage_metadata.get("model"),
                "finish_reason": usage_metadata.get("finish_reason"),
                "prompt_tokens": usage_metadata.get("prompt_tokens"),
                "completion_tokens": usage_metadata.get("completion_tokens"),
                "timestamp": result.completed_at.isoformat(),
            }
            raw_preview = usage_metadata.get("raw_response")
            if raw_preview:
                log_entry["raw_response_preview"] = self._truncate_text(raw_preview, 500)
            result.execution_log.append(log_entry)

            self.logger.info(
                "Agent %s completed successfully in %.2fs (model=%s, tokens=%s)",
                agent_spec.id,
                result.execution_time,
                usage_metadata.get("model"),
                result.tokens_used,
            )

        except TimeoutError:
            result.status = AgentExecutionStatus.TIMEOUT
            result.error = f"Agent execution timed out after {agent_spec.limits.timeout_seconds}s"
            result.completed_at = datetime.now()
            result.execution_time = (result.completed_at - start_time).total_seconds()
            self.logger.warning("Agent %s timed out", agent_spec.id)

        except Exception as exc:
            result.status = AgentExecutionStatus.FAILED
            result.error = str(exc)
            result.completed_at = datetime.now()
            result.execution_time = (result.completed_at - start_time).total_seconds()
            self.logger.error("Agent %s failed: %s", agent_spec.id, exc, exc_info=True)

        return result

    def _run_openai_agent(self, agent_spec: AgentSpec, task: Task, context: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """Invoke OpenAI to execute the agent task."""
        messages, required_fields = self._build_messages(agent_spec, task, context)
        model_name = self._select_model(agent_spec, context)
        max_tokens = self._determine_max_tokens(agent_spec)
        timeout_seconds = max(agent_spec.limits.timeout_seconds, 5)

        self.logger.debug(
            "Executing agent %s with model %s (max_tokens=%s, timeout=%ss)",
            agent_spec.id,
            model_name,
            max_tokens,
            timeout_seconds,
        )

        try:
            response = self.client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=self.default_temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
                timeout=timeout_seconds,
            )
        except Exception as exc:
            if hasattr(openai, "APITimeoutError") and isinstance(exc, openai.APITimeoutError):
                raise TimeoutError(str(exc)) from exc
            raise

        usage = getattr(response, "usage", None)
        choice = response.choices[0] if getattr(response, "choices", None) else None
        raw_content = choice.message.content if choice else "{}"
        finish_reason = getattr(choice, "finish_reason", None)

        parsed = self._parse_agent_response(raw_content, agent_spec, task, required_fields)

        usage_metadata = {
            "model": getattr(response, "model", model_name),
            "finish_reason": finish_reason,
            "prompt_tokens": getattr(usage, "prompt_tokens", None) if usage else None,
            "completion_tokens": getattr(usage, "completion_tokens", None) if usage else None,
            "total_tokens": getattr(usage, "total_tokens", None) if usage else 0,
            "raw_response": raw_content,
        }

        return parsed, usage_metadata

    def _build_messages(self, agent_spec: AgentSpec, task: Task, context: Dict[str, Any]) -> tuple[List[Dict[str, Any]], List[str]]:
        """Build chat messages for the OpenAI request."""
        output_contract = agent_spec.output_contract or {}
        required_fields = output_contract.get("required_fields", [])
        if not isinstance(required_fields, list):
            required_fields = []

        baseline_fields = ["result", "status", "execution_summary"]
        for field in baseline_fields:
            if field not in required_fields:
                required_fields.append(field)

        required_fields = list(dict.fromkeys(required_fields))

        instructions = self._build_output_contract_instructions(output_contract, required_fields)
        system_prompt = agent_spec.system_prompt.strip() if agent_spec.system_prompt else (
            "You are a focused specialist agent within the Local Agent Studio."
        )

        system_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "system", "content": instructions},
        ]

        user_payload = {
            "task": {
                "id": task.id,
                "name": task.name,
                "description": task.description,
                "inputs": task.inputs,
                "expected_output": task.expected_output,
                "success_criteria": task.success_criteria,
            },
            "available_tools": agent_spec.tools,
            "limits": {
                "max_steps": agent_spec.limits.max_steps,
                "max_tokens": agent_spec.limits.max_tokens,
                "timeout_seconds": agent_spec.limits.timeout_seconds,
            },
            "execution_context": self._extract_relevant_context(context, task),
            "output_contract": output_contract,
        }

        user_message = {
            "role": "user",
            "content": json.dumps(user_payload, ensure_ascii=True, default=self._json_default),
        }

        return system_messages + [user_message], required_fields

    def _build_output_contract_instructions(self, output_contract: Dict[str, Any], required_fields: List[str]) -> str:
        """Create explicit formatting instructions for the LLM."""
        expected_output = output_contract.get("expected_output")
        success_criteria = output_contract.get("success_criteria") or []

        instructions = [
            "You must respond with a valid JSON object and nothing else.",
            "The JSON must include the following required fields: " + ", ".join(required_fields) + ".",
            "If the task completes successfully, set 'status' to 'completed'.",
            "If information is missing, explain it in 'execution_summary' and set status to 'needs_follow_up'.",
            "Return arrays or objects for structured data (lists of findings, citations, etc.).",
        ]

        if expected_output:
            instructions.append(f"Target output description: {expected_output}.")
        if success_criteria:
            instructions.append("Success criteria: " + "; ".join(str(item) for item in success_criteria) + ".")

        instructions.append("Do not include markdown or commentary outside the JSON response.")

        return " ".join(instructions)

    def _extract_relevant_context(self, context: Dict[str, Any], task: Task) -> Dict[str, Any]:
        """Extract a compact, serialisable view of the execution context."""
        if not isinstance(context, dict):
            return {}

        relevant: Dict[str, Any] = {}

        user_request = context.get("user_request")
        if user_request:
            relevant["user_request"] = self._truncate_text(str(user_request), 500)

        retrieved_chunks = context.get("retrieved_chunks")
        if isinstance(retrieved_chunks, list) and retrieved_chunks:
            trimmed_chunks = []
            for chunk in retrieved_chunks[:5]:
                if not isinstance(chunk, dict):
                    continue
                trimmed_chunks.append({
                    "chunk_id": chunk.get("chunk_id") or chunk.get("id"),
                    "content": self._truncate_text(chunk.get("content") or "", 600),
                    "source": chunk.get("source") or chunk.get("source_file"),
                    "score": chunk.get("score"),
                })
            if trimmed_chunks:
                relevant["retrieved_chunks"] = trimmed_chunks

        memory_hits = context.get("memory_hits")
        if isinstance(memory_hits, list) and memory_hits:
            trimmed_hits = []
            for hit in memory_hits[:5]:
                if not isinstance(hit, dict):
                    continue
                trimmed_hits.append({
                    "memory_id": hit.get("memory_id") or hit.get("id"),
                    "content": self._truncate_text(hit.get("content") or hit.get("value") or "", 400),
                    "metadata": hit.get("metadata"),
                })
            if trimmed_hits:
                relevant["memory_hits"] = trimmed_hits

        notes = context.get("notes") or context.get("summary")
        if notes:
            relevant["notes"] = self._truncate_text(str(notes), 600)

        if task.inputs:
            relevant["task_inputs"] = task.inputs

        uploads = context.get("uploaded_files") or context.get("uploads")
        if uploads:
            relevant["uploads"] = uploads

        return relevant

    def _select_model(self, agent_spec: AgentSpec, context: Dict[str, Any]) -> str:
        """Select the model to use for this agent."""
        output_contract = agent_spec.output_contract or {}
        contract_model = output_contract.get("model")
        if isinstance(contract_model, str) and contract_model.strip():
            return contract_model.strip()

        if isinstance(context, dict):
            overrides = context.get("agent_model_overrides")
            if isinstance(overrides, dict):
                override = overrides.get(agent_spec.id) or overrides.get(agent_spec.created_for_task)
                if isinstance(override, str) and override.strip():
                    return override.strip()
            generic_override = context.get("agent_model") or context.get("preferred_model")
            if isinstance(generic_override, str) and generic_override.strip():
                return generic_override.strip()

        return self.default_model

    def _determine_max_tokens(self, agent_spec: AgentSpec) -> int:
        """Determine the maximum tokens allowed for the completion."""
        max_tokens = agent_spec.limits.max_tokens if agent_spec.limits else None
        if not max_tokens or max_tokens <= 0:
            max_tokens = self.config.agent_generator.default_agent_max_tokens
        max_tokens = max(max_tokens, 256)
        return min(max_tokens, 4096)

    def _parse_agent_response(
        self,
        raw_content: str,
        agent_spec: AgentSpec,
        task: Task,
        required_fields: List[str],
    ) -> Dict[str, Any]:
        """Parse and validate the agent response from OpenAI."""
        try:
            parsed = json.loads(raw_content)
        except json.JSONDecodeError:
            self.logger.warning(
                "Agent %s returned non-JSON payload; wrapping raw response.",
                agent_spec.id,
            )
            parsed = {
                "result": raw_content,
                "status": "needs_review",
                "execution_summary": "Model returned non-JSON payload.",
            }

        if not isinstance(parsed, dict):
            parsed = {
                "result": parsed,
                "status": "completed",
                "execution_summary": f"Completed task '{task.name}'",
            }

        for field in required_fields:
            if field not in parsed or parsed[field] in (None, ""):
                parsed[field] = self._default_field_value(field, task)

        status = parsed.get("status")
        if isinstance(status, str):
            status = status.strip().lower()
        else:
            status = "completed"

        if status not in {"completed", "failed", "needs_review", "needs_follow_up"}:
            status = "completed"
        parsed["status"] = status

        if not parsed.get("execution_summary"):
            parsed["execution_summary"] = f"Completed task '{task.name}'"

        if not parsed.get("result"):
            parsed["result"] = parsed["execution_summary"]

        return parsed

    def _default_field_value(self, field: str, task: Task) -> Any:
        """Provide sensible defaults for required output fields."""
        list_fields = {
            "citations",
            "insights",
            "recommendations",
            "key_findings",
            "task_breakdown",
            "dependencies",
            "processing_errors",
            "actions",
        }
        dict_fields = {
            "metadata",
            "analysis_results",
            "evaluation_details",
            "quality_metrics",
            "resource_estimates",
        }

        if field in list_fields:
            return []
        if field in dict_fields:
            return {}
        if field in {"confidence_score", "quality_score"}:
            return 0.0
        if field == "result":
            return f"Completed task: {task.description or task.name}"
        if field == "execution_summary":
            return f"Completed task '{task.name}'"
        if field == "status":
            return "completed"
        return ""

    def _estimate_steps(self, agent_output: Any) -> int:
        """Estimate the number of steps performed by the agent."""
        if isinstance(agent_output, dict):
            for key in ("steps_taken", "step_count"):
                value = agent_output.get(key)
                if isinstance(value, int) and value > 0:
                    return value
            for key in ("reasoning_steps", "actions", "plan_steps"):
                value = agent_output.get(key)
                if isinstance(value, list) and value:
                    return len(value)
        return 1

    def _truncate_text(self, text: str, limit: int = 800) -> str:
        """Truncate text to avoid oversized prompts/logs."""
        if not isinstance(text, str):
            text = str(text)
        return text[:limit] + ("..." if len(text) > limit else "")

    def _json_default(self, obj: Any) -> str:
        """Fallback serializer for JSON encoding."""
        if isinstance(obj, datetime):
            return obj.isoformat()
        return str(obj)


class AgentLifecycleManager:
    """Manages the lifecycle of sub-agents including instantiation, monitoring, and cleanup."""
    
    def __init__(self, config: SystemConfig):
        """
        Initialize agent lifecycle manager.
        
        Args:
            config: System configuration
        """
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.AgentLifecycleManager")
        self.running_agents: Dict[str, RunningAgent] = {}
        self.completed_agents: Dict[str, AgentExecutionResult] = {}
        self.agent_runner = SubAgentRunner(config)
        self._shutdown_event = threading.Event()
        self._monitor_thread = None
        
    def start_monitoring(self):
        """Start the agent monitoring thread."""
        if self._monitor_thread is None or not self._monitor_thread.is_alive():
            self._shutdown_event.clear()
            self._monitor_thread = threading.Thread(target=self._monitor_agents, daemon=True)
            self._monitor_thread.start()
            self.logger.info("Agent monitoring started")
    
    def stop_monitoring(self):
        """Stop the agent monitoring thread."""
        self._shutdown_event.set()
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=5)
            self.logger.info("Agent monitoring stopped")
    
    def instantiate_agent(self, agent_spec: AgentSpec, task: Task, context: Dict[str, Any]) -> str:
        """
        Instantiate and start a sub-agent.
        
        Args:
            agent_spec: Agent specification
            task: Task to execute
            context: Execution context
            
        Returns:
            Agent execution ID
        """
        if len(self.running_agents) >= self.config.agent_generator.max_concurrent_agents:
            raise RuntimeError("Maximum concurrent agents limit reached")
        
        # Create running agent record
        now = datetime.now()
        timeout_at = now + timedelta(seconds=agent_spec.limits.timeout_seconds)
        
        running_agent = RunningAgent(
            agent_id=agent_spec.id,
            task_id=task.id,
            agent_spec=agent_spec,
            started_at=now,
            timeout_at=timeout_at
        )
        
        # Start execution in a separate thread
        def execute_wrapper():
            try:
                result = self.agent_runner.execute_agent(agent_spec, task, context)
                self._handle_agent_completion(agent_spec.id, result)
            except Exception as e:
                error_result = AgentExecutionResult(
                    agent_id=agent_spec.id,
                    task_id=task.id,
                    status=AgentExecutionStatus.FAILED,
                    error=str(e),
                    completed_at=datetime.now()
                )
                self._handle_agent_completion(agent_spec.id, error_result)
        
        execution_thread = threading.Thread(target=execute_wrapper, daemon=True)
        running_agent.execution_thread = execution_thread
        
        # Register and start the agent
        self.running_agents[agent_spec.id] = running_agent
        execution_thread.start()
        
        self.logger.info(f"Instantiated agent {agent_spec.id} for task {task.id}")
        return agent_spec.id
    
    def cancel_agent(self, agent_id: str) -> bool:
        """
        Cancel a running agent.
        
        Args:
            agent_id: ID of agent to cancel
            
        Returns:
            True if agent was cancelled, False if not found or already completed
        """
        if agent_id in self.running_agents:
            running_agent = self.running_agents[agent_id]
            running_agent.is_cancelled = True
            
            # Create cancellation result
            cancel_result = AgentExecutionResult(
                agent_id=agent_id,
                task_id=running_agent.task_id,
                status=AgentExecutionStatus.CANCELLED,
                completed_at=datetime.now(),
                execution_time=(datetime.now() - running_agent.started_at).total_seconds()
            )
            
            self._handle_agent_completion(agent_id, cancel_result)
            self.logger.info(f"Cancelled agent {agent_id}")
            return True
        
        return False
    
    def get_agent_status(self, agent_id: str) -> Optional[AgentExecutionStatus]:
        """
        Get the status of an agent.
        
        Args:
            agent_id: ID of agent
            
        Returns:
            Agent status or None if not found
        """
        if agent_id in self.running_agents:
            return AgentExecutionStatus.RUNNING
        elif agent_id in self.completed_agents:
            return self.completed_agents[agent_id].status
        else:
            return None
    
    def get_agent_result(self, agent_id: str) -> Optional[AgentExecutionResult]:
        """
        Get the result of a completed agent.
        
        Args:
            agent_id: ID of agent
            
        Returns:
            Agent execution result or None if not completed
        """
        return self.completed_agents.get(agent_id)
    
    def get_running_agents(self) -> List[str]:
        """Get list of currently running agent IDs."""
        return list(self.running_agents.keys())
    
    def get_completed_agents(self) -> List[str]:
        """Get list of completed agent IDs."""
        return list(self.completed_agents.keys())
    
    def cleanup_completed_agents(self, max_age_hours: int = 24):
        """
        Clean up old completed agent records.
        
        Args:
            max_age_hours: Maximum age in hours for completed agents
        """
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        
        to_remove = []
        for agent_id, result in self.completed_agents.items():
            if result.completed_at and result.completed_at < cutoff_time:
                to_remove.append(agent_id)
        
        for agent_id in to_remove:
            del self.completed_agents[agent_id]
        
        if to_remove:
            self.logger.info(f"Cleaned up {len(to_remove)} old agent records")
    
    def _handle_agent_completion(self, agent_id: str, result: AgentExecutionResult):
        """Handle completion of an agent."""
        existing_result = self.completed_agents.get(agent_id)
        if existing_result and existing_result.status in {AgentExecutionStatus.CANCELLED, AgentExecutionStatus.TIMEOUT}:
            self.logger.debug(
                "Ignoring completion update for agent %s because status is already %s",
                agent_id,
                existing_result.status.value,
            )
            return

        # Move from running to completed
        if agent_id in self.running_agents:
            del self.running_agents[agent_id]
        
        self.completed_agents[agent_id] = result
        
        self.logger.info(f"Agent {agent_id} completed with status: {result.status}")
    
    def _monitor_agents(self):
        """Monitor running agents for timeouts and cleanup."""
        while not self._shutdown_event.is_set():
            try:
                now = datetime.now()
                timed_out_agents = []
                
                # Check for timeouts
                for agent_id, running_agent in self.running_agents.items():
                    if now > running_agent.timeout_at:
                        timed_out_agents.append(agent_id)
                
                # Handle timeouts
                for agent_id in timed_out_agents:
                    running_agent = self.running_agents[agent_id]
                    timeout_result = AgentExecutionResult(
                        agent_id=agent_id,
                        task_id=running_agent.task_id,
                        status=AgentExecutionStatus.TIMEOUT,
                        error=f"Agent timed out after {running_agent.agent_spec.limits.timeout_seconds}s",
                        completed_at=now,
                        execution_time=(now - running_agent.started_at).total_seconds()
                    )
                    self._handle_agent_completion(agent_id, timeout_result)
                
                # Periodic cleanup
                if len(self.completed_agents) > 100:  # Arbitrary threshold
                    self.cleanup_completed_agents(max_age_hours=1)
                
                # Sleep before next check
                time.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                self.logger.error(f"Error in agent monitoring: {e}", exc_info=True)
                time.sleep(10)  # Wait longer on error


class ResultCollector:
    """Collects and integrates results from multiple sub-agents."""
    
    def __init__(self, config: SystemConfig):
        """
        Initialize result collector.
        
        Args:
            config: System configuration
        """
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.ResultCollector")
    
    def collect_results(self, agent_results: List[AgentExecutionResult], tasks: List[Task]) -> Dict[str, Any]:
        """
        Collect and integrate results from multiple agents.
        
        Args:
            agent_results: List of agent execution results
            tasks: List of tasks that were executed
            
        Returns:
            Integrated results
        """
        integrated_results = {
            "execution_summary": {
                "total_agents": len(agent_results),
                "successful_agents": len([r for r in agent_results if r.status == AgentExecutionStatus.COMPLETED]),
                "failed_agents": len([r for r in agent_results if r.status == AgentExecutionStatus.FAILED]),
                "total_execution_time": sum(r.execution_time for r in agent_results),
                "total_tokens_used": sum(r.tokens_used for r in agent_results)
            },
            "task_results": {},
            "agent_results": {},
            "errors": [],
            "warnings": []
        }
        
        # Process each agent result
        for result in agent_results:
            # Store agent-specific result
            integrated_results["agent_results"][result.agent_id] = {
                "status": result.status.value,
                "execution_time": result.execution_time,
                "tokens_used": result.tokens_used,
                "steps_taken": result.steps_taken
            }
            
            # Store task-specific result
            if result.task_id:
                integrated_results["task_results"][result.task_id] = {
                    "status": result.status.value,
                    "result": result.result,
                    "agent_id": result.agent_id
                }
            
            # Collect errors and warnings
            if result.error:
                integrated_results["errors"].append({
                    "agent_id": result.agent_id,
                    "task_id": result.task_id,
                    "error": result.error
                })
            
            if result.status in [AgentExecutionStatus.TIMEOUT, AgentExecutionStatus.CANCELLED]:
                integrated_results["warnings"].append({
                    "agent_id": result.agent_id,
                    "task_id": result.task_id,
                    "warning": f"Agent {result.status.value}"
                })
        
        # Synthesize cross-agent insights
        integrated_results["synthesis"] = self._synthesize_results(agent_results, tasks)
        
        return integrated_results
    
    def _synthesize_results(self, agent_results: List[AgentExecutionResult], tasks: List[Task]) -> Dict[str, Any]:
        """
        Synthesize insights across multiple agent results.
        
        Args:
            agent_results: List of agent execution results
            tasks: List of tasks
            
        Returns:
            Synthesized insights
        """
        synthesis = {
            "overall_success": True,
            "completion_rate": 0.0,
            "key_findings": [],
            "recommendations": [],
            "quality_metrics": {}
        }
        
        successful_results = [r for r in agent_results if r.status == AgentExecutionStatus.COMPLETED]
        synthesis["completion_rate"] = len(successful_results) / len(agent_results) if agent_results else 0.0
        synthesis["overall_success"] = synthesis["completion_rate"] >= 0.8  # 80% success threshold
        
        # Extract key findings from successful results
        for result in successful_results:
            if result.result and isinstance(result.result, dict):
                # Extract various types of findings
                if "key_findings" in result.result:
                    synthesis["key_findings"].extend(result.result["key_findings"])
                if "insights" in result.result:
                    synthesis["key_findings"].extend(result.result["insights"])
                if "analysis_results" in result.result:
                    if isinstance(result.result["analysis_results"], dict):
                        synthesis["key_findings"].extend(result.result["analysis_results"].get("key_findings", []))
        
        # Generate recommendations based on results
        if synthesis["completion_rate"] < 1.0:
            synthesis["recommendations"].append("Some agents failed - consider retrying failed tasks")
        
        if len(successful_results) > 1:
            synthesis["recommendations"].append("Multiple agents completed successfully - results may benefit from cross-validation")
        
        # Calculate quality metrics
        if successful_results:
            avg_execution_time = sum(r.execution_time for r in successful_results) / len(successful_results)
            avg_tokens_used = sum(r.tokens_used for r in successful_results) / len(successful_results)
            
            synthesis["quality_metrics"] = {
                "average_execution_time": avg_execution_time,
                "average_tokens_used": avg_tokens_used,
                "efficiency_score": min(1.0, 60.0 / avg_execution_time) if avg_execution_time > 0 else 1.0  # Higher score for faster execution
            }
        
        return synthesis


class AgentExecutor(ProcessingNode):
    """
    Agent Executor node that manages sub-agent instantiation, lifecycle, and result collection.
    
    Responsibilities:
    - Instantiate sub-agents based on agent specifications
    - Monitor agent execution and enforce limits
    - Collect and integrate results from multiple agents
    - Handle agent failures and timeouts gracefully
    """
    
    def __init__(self, config: SystemConfig):
        """
        Initialize the Agent Executor.
        
        Args:
            config: System configuration
        """
        super().__init__("AgentExecutor", config.agent_generator.__dict__)
        self.system_config = config
        self.lifecycle_manager = AgentLifecycleManager(config)
        self.result_collector = ResultCollector(config)
        
        # Start monitoring
        self.lifecycle_manager.start_monitoring()
    
    def execute(self, state: ExecutionState) -> ExecutionState:
        """
        Execute sub-agents and collect results.
        
        Args:
            state: Current execution state
            
        Returns:
            Updated execution state with agent results
        """
        generated_agents = state.context.get("generated_agents", [])
        
        if not generated_agents:
            self.logger.warning("No generated agents found in state")
            state.complete_execution("No agents to execute")
            return state
        
        self.logger.info(f"Starting execution of {len(generated_agents)} agents")
        
        # Start all agents
        running_agent_ids = []
        for agent_info in generated_agents:
            agent_id = agent_info["agent_id"]
            task_id = agent_info["task_id"]
            
            # Get agent spec and task from state
            agent_spec = state.agent_specs.get(agent_id)
            task = state.get_task_by_id(task_id)
            
            if not agent_spec or not task:
                self.logger.error(f"Missing agent spec or task for agent {agent_id}")
                continue
            
            try:
                # Start the agent
                execution_id = self.lifecycle_manager.instantiate_agent(
                    agent_spec, task, state.context
                )
                running_agent_ids.append(agent_id)
                
                # Mark task as in progress
                task.status = TaskStatus.IN_PROGRESS
                task.started_at = datetime.now()
                
                # Activate agent in state
                state.activate_agent(agent_id)
                
            except Exception as e:
                self.logger.error(f"Failed to start agent {agent_id}: {e}")
                state.mark_task_failed(task_id, str(e))
        
        # Wait for all agents to complete
        agent_results = self._wait_for_agents(running_agent_ids, state)
        
        # Process results
        self._process_agent_results(agent_results, state)
        
        # Collect and integrate results
        tasks = [state.get_task_by_id(info["task_id"]) for info in generated_agents]
        tasks = [t for t in tasks if t is not None]
        
        integrated_results = self.result_collector.collect_results(agent_results, tasks)
        
        # Store results in state
        state.context["agent_execution_results"] = integrated_results
        
        # Log execution summary
        state.log_execution_step(
            node_name=self.name,
            action="agent_execution_complete",
            details={
                "total_agents": len(generated_agents),
                "successful_agents": integrated_results["execution_summary"]["successful_agents"],
                "failed_agents": integrated_results["execution_summary"]["failed_agents"],
                "total_execution_time": integrated_results["execution_summary"]["total_execution_time"],
                "completion_rate": integrated_results["synthesis"]["completion_rate"]
            }
        )
        
        # Determine next node
        if integrated_results["synthesis"]["overall_success"]:
            state.next_node = "Evaluator"
        else:
            # Some agents failed - might need replanning
            state.next_node = "Planner"  # Could replan failed tasks
        
        return state
    
    def _wait_for_agents(self, agent_ids: List[str], state: ExecutionState) -> List[AgentExecutionResult]:
        """
        Wait for all agents to complete execution.
        
        Args:
            agent_ids: List of agent IDs to wait for
            state: Current execution state
            
        Returns:
            List of agent execution results
        """
        results = []
        max_wait_time = 300  # 5 minutes maximum wait
        check_interval = 1  # Check every second
        start_time = datetime.now()
        
        remaining_agents = set(agent_ids)
        
        while remaining_agents and (datetime.now() - start_time).total_seconds() < max_wait_time:
            completed_this_cycle = []
            
            for agent_id in remaining_agents:
                status = self.lifecycle_manager.get_agent_status(agent_id)
                
                if status in [AgentExecutionStatus.COMPLETED, AgentExecutionStatus.FAILED, 
                             AgentExecutionStatus.TIMEOUT, AgentExecutionStatus.CANCELLED]:
                    result = self.lifecycle_manager.get_agent_result(agent_id)
                    if result:
                        results.append(result)
                        completed_this_cycle.append(agent_id)
                        
                        # Deactivate agent in state
                        state.deactivate_agent(agent_id)
            
            # Remove completed agents from remaining set
            for agent_id in completed_this_cycle:
                remaining_agents.remove(agent_id)
            
            if remaining_agents:
                time.sleep(check_interval)
        
        # Handle any remaining agents (timeout)
        for agent_id in remaining_agents:
            self.logger.warning(f"Agent {agent_id} did not complete within timeout, cancelling")
            self.lifecycle_manager.cancel_agent(agent_id)
            
            # Get the cancellation result
            result = self.lifecycle_manager.get_agent_result(agent_id)
            if result:
                results.append(result)
            
            state.deactivate_agent(agent_id)
        
        return results
    
    def _process_agent_results(self, agent_results: List[AgentExecutionResult], state: ExecutionState):
        """
        Process agent results and update task statuses.
        
        Args:
            agent_results: List of agent execution results
            state: Current execution state
        """
        for result in agent_results:
            task = state.get_task_by_id(result.task_id)
            if not task:
                continue
            
            if result.status == AgentExecutionStatus.COMPLETED:
                state.mark_task_completed(result.task_id, result.result)
                self.logger.info(f"Task {result.task_id} completed successfully by agent {result.agent_id}")
            else:
                error_msg = result.error or f"Agent execution {result.status.value}"
                state.mark_task_failed(result.task_id, error_msg)
                self.logger.warning(f"Task {result.task_id} failed: {error_msg}")
    
    def validate_inputs(self, state: ExecutionState) -> bool:
        """
        Validate that required inputs are present.
        
        Args:
            state: Current execution state
            
        Returns:
            True if inputs are valid, False otherwise
        """
        generated_agents = state.context.get("generated_agents", [])
        
        if not generated_agents:
            return False
        
        # Check that all referenced agents and tasks exist
        for agent_info in generated_agents:
            agent_id = agent_info.get("agent_id")
            task_id = agent_info.get("task_id")
            
            if not agent_id or not task_id:
                return False
            
            if agent_id not in state.agent_specs:
                return False
            
            if not state.get_task_by_id(task_id):
                return False
        
        return True
    
    def get_next_node(self, state: ExecutionState) -> Optional[str]:
        """
        Determine the next node based on execution results.
        
        Args:
            state: Current execution state
            
        Returns:
            Name of next node
        """
        results = state.context.get("agent_execution_results", {})
        
        if not results:
            return None
        
        synthesis = results.get("synthesis", {})
        overall_success = synthesis.get("overall_success", False)
        
        if overall_success:
            return "Evaluator"
        else:
            # Could implement retry logic or replanning
            return "Planner"
    
    def is_critical_error(self, error: Exception) -> bool:
        """
        Determine if an error should stop execution.
        
        Args:
            error: The exception that occurred
            
        Returns:
            True if execution should stop, False to continue
        """
        # Agent execution errors are generally not critical for the overall system
        # Individual agent failures are handled gracefully
        return False
    
    def __del__(self):
        """Cleanup when executor is destroyed."""
        if hasattr(self, 'lifecycle_manager'):
            self.lifecycle_manager.stop_monitoring()