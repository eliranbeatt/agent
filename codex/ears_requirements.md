# Local Agent Studio – EARS Requirements

## Requirement 1 – Core Orchestration System (Status: Implemented)
- **R1.1** WHEN a user request matches a predefined workflow with confidence =0.7 THE orchestrator shall select the predefined workflow path before invoking planner nodes (`server/app/core/orchestrator.py:231`, `tests/test_acceptance.py:23`).
- **R1.2** WHEN a user request does not match any predefined workflow THE orchestrator shall route execution to the planner-driven path (`server/app/core/orchestrator.py:258`, `tests/test_acceptance.py:40`).
- **R1.3** WHEN executing any workflow THE orchestrator shall enforce configured iteration, token, and agent limits before allowing additional steps (`server/app/core/orchestrator.py:28`, `tests/test_acceptance.py:57`).
- **R1.4** WHEN workflow execution completes THE orchestrator shall log the chosen path, steps taken, tools used, and memory hits inside the execution state (`server/app/core/orchestrator.py:312`, `tests/test_acceptance.py:74`).
- **R1.5** IF execution limits are reached THE orchestrator shall terminate gracefully and emit an explanatory status (`server/app/core/orchestrator.py:61`, `tests/test_acceptance.py:90`).

## Requirement 2 – Dynamic Agent Generation (Status: Partial)
- **R2.1** WHEN the planner marks a task as requiring specialized handling THE agent generator shall create an AgentSpec with an appropriate type (`server/app/core/agent_generator.py:571`, `tests/test_agent_generator.py:296`).
- **R2.2** WHEN an agent is created THE agent generator shall include role definition, goals, allowed tools, limits, and required outputs in the system prompt (`server/app/core/agent_generator.py:368`).
- **R2.3** WHEN a sub-agent executes THE agent executor shall enforce the AgentLimits for steps, tokens, and timeouts (`server/app/core/agent_executor.py:70`).
- **R2.4** WHEN a sub-agent finishes THE agent executor shall attach its outputs back to the parent task and execution state (`server/app/core/agent_executor.py:279`).
- **R2.5** IF a sub-agent fails or exceeds limits THE executor shall mark the task failed and surface the error to the orchestrator (`server/app/core/agent_executor.py:208`).
  - *Gap*: Execution today is simulated, so integrations with real toolchains remain unproven (`server/app/core/agent_executor.py:125`).

## Requirement 3 – Comprehensive File Processing (Status: Partial)
- **R3.1** WHEN a user uploads Office documents THE file processor shall extract and normalize the content via Unstructured loaders (`server/app/core/context/file_processor.py:218`).
- **R3.2** WHEN a user uploads PDFs THE file processor shall extract text and compute page counts (`server/app/core/context/file_processor.py:248`).
- **R3.3** WHEN a user uploads images THE file processor shall perform OCR to return extracted text when Tesseract is available (`server/app/core/context/file_processor.py:309`).
- **R3.4** WHEN files are processed THE chunker shall segment content into 800–1200 token chunks with configured overlap (`server/app/core/context/chunker.py:71`).
- **R3.5** WHEN content is chunked THE vector store shall persist embeddings for later retrieval (`server/app/core/context/vector_store.py:58`).
  - *Gaps*: python-magic and Tesseract binaries are not bundled, causing failures on Windows (`tests/test_file_processor.py:39`).

## Requirement 4 – Intelligent Planning and Task Management (Status: Implemented)
- **R4.1** WHEN receiving a complex request THE planner shall decompose it into minimal tasks with dependencies (`server/app/core/planner.py:407`).
- **R4.2** WHEN creating tasks THE planner shall assign inputs, expected outputs, and success criteria (`server/app/core/planner.py:457`).
- **R4.3** WHEN tasks have dependencies THE dependency resolver shall enforce the correct execution order via topological sort (`server/app/core/planner.py:602`).
- **R4.4** WHEN tasks start executing THE task identifier shall map each task to the most suitable tools and context (`server/app/core/task_identifier.py:420`).
- **R4.5** IF task dependencies cannot be resolved THE planner shall flag the issue for additional context before continuing (`server/app/core/planner.py:663`).

## Requirement 5 – Persistent Memory Management (Status: Partial)
- **R5.1** WHEN a user interacts THE memory manager shall maintain a profile and preferences in persistent storage (`server/app/core/memory/memory_manager.py:203`).
- **R5.2** WHEN new facts are learned THE memory manager shall store them with timestamps and sources (`server/app/core/memory/memory_manager.py:258`).
- **R5.3** WHEN conversations occur THE conversation manager shall append turns and maintain summaries (`server/app/core/memory/conversation_manager.py:32`).
- **R5.4** WHEN RAG operations run THE manager shall store trace metadata for future optimization (`server/app/core/memory/memory_manager.py:542`).
- **R5.5** WHEN retention limits are reached THE policy manager shall prune content according to TTL/LRU/FIFO rules (`server/app/core/memory/memory_manager.py:729`).
  - *Gap*: Windows tests cannot clean up embedded Chroma stores, leaving teardown failures (`tests/test_task_6_verification.py:27`).

## Requirement 6 – RAG and Knowledge Retrieval (Status: Partial)
- **R6.1** WHEN a question references uploaded documents THE context manager shall retrieve the top-k relevant chunks via semantic search (`server/app/core/context/context_manager.py:233`).
- **R6.2** WHEN generating answers THE RAG workflow shall include citations with chunk IDs and sources (`server/app/core/rag/rag_workflow.py:322`).
- **R6.3** WHEN retrieving information THE vector store shall support configurable k and MMR diversification (`server/app/core/context/vector_store.py:228`).
- **R6.4** WHEN providing answers THE evaluator shall verify claims against retrieved sources and compute quality metrics (`server/app/core/rag/evaluator.py:64`).
- **R6.5** IF retrieval is insufficient THE workflow shall trigger replanning or request more context (`server/app/core/rag/rag_workflow.py:519`).
  - *Gap*: Without an OpenAI embedding key retrieval returns failure immediately (`server/app/core/context/context_manager.py:212`).

## Requirement 7 – Local Execution and Privacy (Status: Implemented)
- **R7.1** WHEN the system starts THE backend and vector store shall run entirely on the local machine (`server/app/config/loader.py:176`).
- **R7.2** WHEN processing files THE system shall keep all data under the `data/` workspace (`config/agents.yaml:28`).
- **R7.3** WHEN using AI capabilities THE system shall rely only on OpenAI API calls and no other external services (`server/scripts/health_check.py:96`).
- **R7.4** WHEN storing embeddings THE system shall use embedded ChromaDB by default (`server/app/core/context/vector_store.py:58`).
- **R7.5** IF the network is lost THE execution graph continues operating with cached state until embeddings are required (fallback needed for full compliance).

## Requirement 8 – Configuration-Driven Architecture (Status: Implemented)
- **R8.1** WHEN configuring agents THE system shall load parameters from `agents.yaml` into dataclasses with validation (`server/app/config/models.py:12`).
- **R8.2** WHEN defining workflows THE loader shall hydrate `WorkflowConfig` instances from `workflows.yaml` (`server/app/config/loader.py:206`).
- **R8.3** WHEN the system boots THE configuration loader shall parse YAML/JSON and apply defaults before building `SystemConfig` (`server/app/config/loader.py:102`).
- **R8.4** WHEN configuration changes are detected THE hot-reload watcher shall refresh in-memory config (`server/app/config/loader.py:259`).
- **R8.5** IF configuration validation fails THE loader shall fall back to defaults and log explicit errors (`server/app/config/loader.py:160`).

## Requirement 9 – User Interface and Interaction (Status: Partial)
- **R9.1** WHEN users open the interface THE chat view shall render history and the active conversation (`ui/src/components/ChatInterface.tsx:24`).
- **R9.2** WHEN users upload files THE UI shall provide drag-and-drop with progress indicators (`ui/src/components/FileUpload.tsx:18`).
- **R9.3** WHEN agents execute THE inspector panel shall expose real-time status, tasks, and resource usage (`ui/src/components/InspectorPanel.tsx:20`).
- **R9.4** WHEN results include citations THE source panel shall surface retrieved chunks and metadata (`ui/src/components/SourcePanel.tsx:18`).
- **R9.5** WHEN execution runs THE UI shall stream tokens via the services API/WebSocket client (`ui/src/services/api.ts:86`).
  - *Gap*: Playwright automation cannot connect because backend/frontend servers are not launched during tests (`tests/test_ui.py:14`).

## Requirement 10 – Quality Assurance and Verification (Status: Partial)
- **R10.1** WHEN tasks finish THE evaluator shall check outputs against success criteria and quality thresholds (`server/app/core/rag/evaluator.py:60`).
- **R10.2** WHEN generating responses THE evaluator shall confirm grounding against retrieved sources (`server/app/core/rag/evaluator.py:182`).
- **R10.3** WHEN contradictions are detected THE evaluator shall document them and suggest corrections (`server/app/core/rag/evaluator.py:281`).
- **R10.4** WHEN verification fails THE workflow shall attempt replanning or escalate (`server/app/core/rag/rag_workflow.py:519`).
- **R10.5** IF quality thresholds are not met THE orchestrator shall prevent final delivery and surface errors (`server/app/core/orchestrator.py:317`).
  - *Gap*: Several QA-related tests currently fail due to dependency and mocking mismatches (`tests/test_embeddings.py:87`, `tests/test_websocket.py:140`).
