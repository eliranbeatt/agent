# Local Agent Studio Detailed Design

## 1. System Overview
- Local Agent Studio couples a FastAPI backend with a Next.js frontend to deliver a local-first multi-agent orchestration environment.
- The backend wraps a LangGraph execution graph when the library is available, but transparently falls back to a custom `SimpleGraphExecutor` for compatibility (`server/app/core/graph_builder.py:44`, `server/app/core/simple_graph_executor.py:33`).
- Core flows: user request ? orchestrator routing ? planner/task decomposition ? agent generation/execution ? context retrieval via RAG pipeline ? evaluator verification ? response assembly ? memory persistence.

## 2. Component Architecture
### Orchestrator
- `MainOrchestrator` selects execution paths based on workflow confidence and enforces global limits through `ResourceLimitEnforcer` (`server/app/core/orchestrator.py:18`).
- Execution metadata, logs, and context are persisted in the `ExecutionState` dataclass (`server/app/core/state.py:78`).

### Planner & Task Management
- `Planner` analyses requests, decomposes them into `Task` objects, and resolves dependencies (`server/app/core/planner.py:407`).
- `TaskIdentifier` maps tasks to tools/context, providing inputs for agent generation (`server/app/core/task_identifier.py:420`).

### Agent Generator & Executor
- `AgentGenerator` builds `AgentSpec` instances from templates stored in `config/templates/agent_prompts.yaml`, applying limits per complexity tier (`server/app/core/agent_generator.py:547`).
- `SubAgentRunner` currently simulates execution, returning fabricated results according to output contracts; no real tool or LLM invocation occurs (`server/app/core/agent_executor.py:125`).

### Workflow Execution
- `WorkflowExecutor` runs predefined workflows defined in `config/workflows.yaml`, with checkpoints and pause/resume support (`server/app/core/workflows/workflow_executor.py:52`).
- `WorkflowMatcher` scores triggers to decide when to use RAG, summary, compare, or OCR workflows (`server/app/core/workflows/workflow_matcher.py:19`).

### Context & RAG Layer
- `ContextManager` orchestrates file ingestion, chunking, embeddings, and vector search using Unstructured, tiktoken, OpenAI embeddings, and ChromaDB (`server/app/core/context/context_manager.py:35`).
- `FileProcessor` routes document types to specialized loaders and optional OCR (`server/app/core/context/file_processor.py:70`).
- `ContentChunker` provides paragraph/sentence/token strategies with configurable chunk size/overlap (`server/app/core/context/chunker.py:71`).
- `VectorStore` persists embeddings in a local Chroma collection and implements MMR selection (`server/app/core/context/vector_store.py:58`).
- `RAGWorkflow` stitches question processing, retrieval, answer generation, verification, and replanning loops (`server/app/core/rag/rag_workflow.py:37`).

### Memory Subsystem
- `MemoryManager` integrates Mem0 (when OpenAI credentials are present) with JSON-based local storage for profiles, facts, conversations, and RAG traces (`server/app/core/memory/memory_manager.py:28`).
- `RetentionPolicyManager` applies TTL, LRU, FIFO, or composite rules to bounded stores (`server/app/core/memory/memory_manager.py:729`).

### API Layer
- FastAPI routers expose chat, files, memory, config, and WebSocket routes (`server/app/api/chat.py:18`, `server/app/api/files.py:14`, `server/app/api/memory.py:14`, `server/app/api/config.py:16`, `server/app/api/websocket.py:18`).
- Configuration loader is initialized once and injected into route handlers (`server/app/api/chat.py:24`).

### Frontend
- `ChatInterface` orchestrates chat history, streaming responses, file uploads, inspector, and source panels (`ui/src/components/ChatInterface.tsx:1`).
- API client handles REST + SSE + WebSocket interactions (`ui/src/services/api.ts:17`).
- Execution telemetry (tasks, agents, resource usage) is rendered via `InspectorPanel`, while citations appear in `SourcePanel` (`ui/src/components/InspectorPanel.tsx:20`, `ui/src/components/SourcePanel.tsx:18`).

## 3. Data Flow
1. **Request initiation**: `POST /chat/` or `/chat/stream` captures the message and optional session context (`server/app/api/chat.py:126`).
2. **Execution graph**: The orchestrator updates `ExecutionState`, decides workflow vs planner, and executes nodes sequentially via LangGraph bindings or the simple executor (`server/app/core/graph_builder.py:102`).
3. **Agent lifecycle**: Task mappings feed the generator ? agent specs recorded in state ? executor simulates tool runs and stores results (`server/app/core/agent_generator.py:571`, `server/app/core/agent_executor.py:279`).
4. **Context pipeline**: Uploaded files are processed/chunked/embedded, stored in Chroma, and retrieved on demand (`server/app/core/context/context_manager.py:140`).
5. **Answer synthesis**: `RAGWorkflow` drives retrieval, answer generation, evaluator verification, and optional replanning before returning a final payload (`server/app/core/rag/rag_workflow.py:322`).
6. **Persistence**: Responses and traces are written into memory stores (`server/app/core/memory/memory_manager.py:482`).
7. **Response**: `ResultAssembler` projects final response, citations, and metadata back through the API (`server/app/core/graph_builder.py:310`).

## 4. Configuration & Deployment
- YAML files under `config/` load into typed dataclasses with validation and hot reload support (`server/app/config/loader.py:37`).
- Environment variables (via `.env`) supply OpenAI keys and other runtime flags (`server/app/config/loader.py:50`).
- Startup scripts (`start-dev.bat`, `start-dev.sh`) bootstrap virtual environments, dependencies, and concurrent backend/frontend processes.

## 5. Error Handling & Observability
- Circuit breakers, retries, and categorized recovery actions live in `error_handling.py` (`server/app/core/error_handling.py:21`).
- Structured logging, metrics, traces, and alerting are provided by `monitoring.py` (`server/app/core/monitoring.py:34`).
- Health check script validates environment, config, OpenAI connectivity, and vector store readiness (`server/scripts/health_check.py:29`).

## 6. External Dependencies
- **OpenAI**: Required for embeddings and evaluation; absence results in `ContextManager` retrieval failure (`server/app/core/context/context_manager.py:212`).
- **Mem0 + ChromaDB**: Combined for long-term memory; Mem0 requires OpenAI credentials, while Chroma persists under `data/` and may lock files on Windows (`tests/test_task_6_verification.py:27`).
- **Unstructured, Tesseract, python-magic**: Power file ingestion and OCR; they are optional but heavily influence capabilities (`server/app/core/context/file_processor.py:392`).
- **Playwright**: Used for UI smoke tests but needs frontend/backend servers running to succeed (`tests/test_ui.py:14`).

## 7. Limitations & Open Issues
- Sub-agent execution is a stub; true LangChain/LangGraph agents are not integrated (`server/app/core/agent_executor.py:125`).
- Retrieval hard-fails without embeddings; a local fallback is necessary for offline compliance (`server/app/core/context/context_manager.py:212`).
- Multiple tests fail on Windows because of locked temp files and dependency assumptions (`tests/test_file_processor.py:39`, `tests/test_vector_store.py:26`).
- UI strings rely on placeholder glyphs and Playwright automation lacks environment orchestration (`ui/src/components/ChatInterface.tsx:120`, `tests/test_ui.py:14`).
- pytest-asyncio strict mode clashes with WebSocket helper tests; fixtures must be revisited (`tests/test_websocket.py:140`).
