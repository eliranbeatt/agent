# Gemini Code Assistant Context

This document provides context for the Gemini Code Assistant to understand the "Local Agent Studio" project.

## Project Overview

The "Local Agent Studio" is a local, hybrid multi-agent system that provides full control over agent architecture, system prompts, workflows, iterations, and memory. It is built with a FastAPI backend, a Next.js frontend, and a variety of AI and machine learning libraries.

**Key Technologies:**

*   **Backend:** FastAPI, Python 3.11
*   **Frontend:** Next.js, React, TypeScript
*   **AI/ML:** LangGraph, LangChain, ChromaDB, Mem0, OpenAI
*   **Configuration:** YAML

**Architecture:**

The application follows a client-server architecture:

*   **`ui` directory:** Contains the Next.js frontend, which provides the user interface for interacting with the system.
*   **`server` directory:** Contains the FastAPI backend, which exposes a REST/WebSocket API for the frontend to consume. The backend is responsible for orchestrating agent workflows, managing memory, and processing files.

The system is highly configurable through YAML files located in the `config` directory. These files control the behavior of agents, workflows, and memory.

## Building and Running

**Prerequisites:**

*   Python 3.11 or higher
*   Node.js 18 or higher
*   OpenAI API key

**Installation and Execution:**

1.  **Set up environment:**
    ```bash
    cp .env.example .env
    # Edit .env and add your OPENAI_API_KEY
    ```

2.  **Initialize the system:**
    ```bash
    cd server
    python scripts/init_db.py
    ```

3.  **Start the application:**
    *   **Windows:** `start-dev.bat`
    *   **Linux/Mac:** `./start-dev.sh`

**Testing:**

*   **Run the test suite:**
    ```bash
    cd server
    python -m pytest tests/ -v
    ```

*   **Run acceptance tests:**
    ```bash
    python -m pytest tests/test_acceptance.py -v
    ```

*   **Run health check:**
    ```bash
    python scripts/health_check.py
    ```

## Development Conventions

*   **Configuration-driven:** The application's behavior is heavily driven by YAML configuration files. When adding new features, consider whether they can be implemented through configuration.
*   **Modular architecture:** The backend is organized into modules for API, core logic, and configuration. Maintain this separation of concerns when adding new code.
*   **Testing:** All new features should be accompanied by tests. The `tests` directory in the `server` directory contains a comprehensive test suite.
*   **Frontend components:** The frontend is built with React components. See the `ui/src/components` directory for existing components.
