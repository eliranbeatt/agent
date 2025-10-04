# Local Agent Studio (LAS)

A local, hybrid multi-agent system that provides full control over agent architecture, system prompts, workflows, iterations, and memory. Built with FastAPI, LangGraph, and Next.js.

## 🌟 Features

- **Hybrid Multi-Agent System**: Intelligently routes between predefined workflows and dynamic agent generation
- **Local-First**: Runs entirely on your machine with only OpenAI API requirement
- **Comprehensive File Processing**: Supports Office documents, PDFs, and images with OCR
- **RAG Question Answering**: Ask questions about your documents with citations
- **Persistent Memory**: Maintains context across sessions with Mem0
- **Configuration-Driven**: Customize agents, workflows, and behavior via YAML files
- **Real-Time Monitoring**: Track execution, resource usage, and agent activity
- **Privacy-Focused**: All data stays on your local machine

## 🚀 Quick Start

### Prerequisites

- Python 3.11 or higher
- Node.js 18 or higher
- OpenAI API key

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd local-agent-studio
   ```

2. **Set up environment**:
   ```bash
   cp .env.example .env
   # Edit .env and add your OPENAI_API_KEY
   ```

3. **Initialize the system**:
   ```bash
   cd server
   python scripts/init_db.py
   ```

4. **Start the application**:
   
   **Windows**:
   ```bash
   start-dev.bat
   ```
   
   **Linux/Mac**:
   ```bash
   chmod +x start-dev.sh
   ./start-dev.sh
   ```

5. **Access the application**:
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000
   - API Docs: http://localhost:8000/docs

## 📖 Documentation

- [Installation Guide](INSTALLATION.md) - Detailed setup instructions
- [Task 13 Summary](TASK_13_IMPLEMENTATION_SUMMARY.md) - System integration details
- [API Documentation](http://localhost:8000/docs) - Interactive API docs (when running)

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      User Interface (Next.js)                │
│  Chat │ File Upload │ Source Panel │ Inspector │ Controls   │
└────────────────────────┬────────────────────────────────────┘
                         │ REST/WebSocket
┌────────────────────────┴────────────────────────────────────┐
│                    FastAPI Backend                           │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              Main Orchestrator                        │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌────────────┐ │  │
│  │  │  Predefined  │  │   Planner    │  │  Context   │ │  │
│  │  │  Workflows   │  │   Driven     │  │  Manager   │ │  │
│  │  └──────┬───────┘  └──────┬───────┘  └─────┬──────┘ │  │
│  │         │                  │                 │        │  │
│  │         ├──────────────────┴─────────────────┤        │  │
│  │         │        Agent Generator              │        │  │
│  │         │        Agent Executor               │        │  │
│  │         └──────────────┬─────────────────────┘        │  │
│  │                        │                               │  │
│  │                   Evaluator                            │  │
│  │                        │                               │  │
│  │                 Result Assembler                       │  │
│  │                        │                               │  │
│  │                 Memory Manager                         │  │
│  └────────────────────────┴───────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                         │
┌────────────────────────┴────────────────────────────────────┐
│              Data Layer (Local Storage)                      │
│  ChromaDB │ Mem0 │ File System │ Configuration Files        │
└─────────────────────────────────────────────────────────────┘
```

## 🎯 Use Cases

1. **Document Q&A**: Upload documents and ask questions with cited answers
2. **Content Summarization**: Extract key points and summaries from documents
3. **Multi-Document Analysis**: Compare and synthesize information across files
4. **Image OCR**: Process image-based documents with text extraction
5. **Custom Workflows**: Define your own agent workflows via configuration

## ⚙️ Configuration

### Agent Configuration (`config/agents.yaml`)

```yaml
orchestrator:
  max_iterations: 6
  token_budget: 50000
  workflow_confidence_threshold: 0.7

agent_generator:
  max_concurrent_agents: 5
  default_max_steps: 4
```

### Workflow Configuration (`config/workflows.yaml`)

```yaml
workflows:
  rag_qa:
    name: "RAG Question Answering"
    enabled: true
    triggers:
      - "what"
      - "how"
      - "why"
```

### Memory Configuration (`config/memory.yaml`)

```yaml
memory:
  provider: "mem0"
  ttl_days: 30
  max_entries: 10000
```

## 🧪 Testing

Run the test suite:

```bash
cd server
python -m pytest tests/ -v
```

Run acceptance tests:

```bash
python -m pytest tests/test_acceptance.py -v
```

Run health check:

```bash
python scripts/health_check.py
```

## 📊 System Requirements

### Minimum
- CPU: 2 cores
- RAM: 4 GB
- Disk: 2 GB free space
- OS: Windows 10+, macOS 10.15+, or Linux

### Recommended
- CPU: 4+ cores
- RAM: 8+ GB
- Disk: 10+ GB free space
- OS: Windows 11, macOS 12+, or Ubuntu 20.04+

## 🔒 Security & Privacy

- All data stored locally in `data/` directory
- No external services except OpenAI API
- API key stored securely in `.env` file
- CORS configured for localhost only
- File uploads validated and sanitized

## 🛠️ Development

### Project Structure

```
local-agent-studio/
├── server/              # FastAPI backend
│   ├── app/
│   │   ├── api/        # API endpoints
│   │   ├── core/       # Core logic (orchestrator, agents, etc.)
│   │   └── config/     # Configuration management
│   ├── scripts/        # Utility scripts
│   └── tests/          # Test suite
├── ui/                 # Next.js frontend
│   ├── src/
│   │   ├── components/ # React components
│   │   ├── services/   # API services
│   │   └── types/      # TypeScript types
│   └── __tests__/      # Frontend tests
├── config/             # Configuration files
├── data/               # Local data storage
└── docs/               # Documentation
```

### Adding New Workflows

1. Define workflow in `config/workflows.yaml`
2. Implement workflow logic in `server/app/core/workflows/predefined/`
3. Register workflow in workflow executor
4. Test with acceptance tests

### Adding New Agents

1. Configure agent template in `config/agents.yaml`
2. Implement agent logic in `server/app/core/`
3. Register with agent generator
4. Add tests

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📝 License

See LICENSE file for details.

## 🙏 Acknowledgments

Built with:
- [FastAPI](https://fastapi.tiangolo.com/) - Backend framework
- [LangGraph](https://github.com/langchain-ai/langgraph) - Agent orchestration
- [Next.js](https://nextjs.org/) - Frontend framework
- [ChromaDB](https://www.trychroma.com/) - Vector database
- [Mem0](https://mem0.ai/) - Memory management
- [OpenAI](https://openai.com/) - Language models

## 📞 Support

For issues and questions:
- Check [INSTALLATION.md](INSTALLATION.md) for setup help
- Run `python scripts/health_check.py` for diagnostics
- Review logs in terminal output
- Check API docs at http://localhost:8000/docs

## 🗺️ Roadmap

- [ ] Docker deployment support
- [ ] User authentication
- [ ] Additional LLM providers
- [ ] Enhanced monitoring and metrics
- [ ] Mobile-responsive UI
- [ ] Plugin system for custom tools
- [ ] Distributed execution support

---

**Local Agent Studio** - Your local AI agent platform with full control and privacy.
