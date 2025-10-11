# Implementation Plan

## 1. Replace Simulated Agent Execution (High)
- Implement LangChain/LangGraph-backed agent execution to replace `_simulate_agent_execution` and surface real tool results (`server/app/core/agent_executor.py:125`).
- Extend agent specs with tool invocation metadata and streaming callbacks so UI receives live updates (`ui/src/services/api.ts:157`).
- Update agent-related tests to cover real execution branches and adapt fixtures for asynchronous behaviour (`tests/test_agent_generator.py:296`).

## 2. Align Agent Templates and Classifiers (High)
- Add missing task mappings (e.g., data analysis) or adjust fixtures to match available templates (`tests/test_agent_generator.py:296`).
- Enrich `PromptTemplateManager` to return base template fallbacks when specialized prompts are missing (`server/app/core/agent_generator.py:368`).
- Re-run generator tests to confirm coverage across all supported task archetypes (`tests/test_agent_generator.py:200`).

## 3. Stabilize File Processing Toolchain (High)
- Bundle or document installation of python-magic and Tesseract across platforms to avoid runtime warnings (`server/app/core/context/file_processor.py:124`).
- Ensure all file handles close before deletion to satisfy Windows cleanup semantics (`server/app/core/context/file_processor.py:175`, `tests/test_file_processor.py:39`).
- Add integration tests that mock OCR and partition libraries without requiring binaries, allowing CI to validate logic (`tests/test_file_processor.py:95`).

## 4. Provide Embedding Fallback (High)
- Introduce a local embedding strategy (e.g., sentence-transformers, FAISS) used when OpenAI credentials are absent (`server/app/core/context/context_manager.py:212`).
- Surface status in health checks and acceptance tests to confirm retrieval works in offline mode (`server/scripts/health_check.py:123`).

## 5. Harden Memory/Vector Store Cleanup (Medium)
- Wrap Chroma client usage with explicit close/shutdown hooks to release locks on Windows (`server/app/core/context/vector_store.py:62`).
- Isolate Mem0 storage into per-test temporary directories via context managers to avoid PermissionErrors (`tests/test_task_6_verification.py:27`).

## 6. Orchestrate UI & API for Playwright (Medium)
- Provide pytest fixtures that launch FastAPI and Next.js (or mocked endpoints) before Playwright runs (`tests/test_ui.py:14`).
- Replace placeholder glyphs in UI components with accessible icons/text (`ui/src/components/ChatInterface.tsx:120`).
- Expand Playwright scenarios to cover file upload, streaming, and inspector panels once servers are live (`ui/__tests__/README.md:4`).

## 7. Relax Async Test Harness (Medium)
- Switch pytest-asyncio to `mode=AUTO` or create an event-loop fixture to prevent `RuntimeError: This event loop is already running` errors (`tests/test_websocket.py:140`).
- Add deterministic assertions for WebSocket helper functions using mock websockets rather than real loops (`server/app/api/websocket.py:76`).

## 8. Update OpenAI Error Handling (Medium)
- Adapt embedding generator mocks to the new OpenAI client signatures so rate-limit and API error tests pass (`server/app/core/context/embeddings.py:64`, `tests/test_embeddings.py:87`).
- Encapsulate OpenAI exceptions to allow offline or mocked environments to simulate failure conditions without exploding tests.

## 9. Documentation & Dev Experience (Low)
- Refresh docs to highlight required optional dependencies, Windows caveats, and embedding fallbacks (`INSTALLATION.md:33`).
- Add quickstart scripts for running combined backend/frontend with Playwright smoke tests locally.
- Capture troubleshooting steps for vector store locks and mem0 cleanup in README (`README.md:112`).
