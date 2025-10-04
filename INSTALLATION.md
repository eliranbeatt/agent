# Local Agent Studio - Installation Guide

This guide will help you install and set up Local Agent Studio on your local machine.

## Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.11 or higher** - [Download Python](https://www.python.org/downloads/)
- **Node.js 18 or higher** - [Download Node.js](https://nodejs.org/)
- **Git** - [Download Git](https://git-scm.com/downloads)
- **OpenAI API Key** - [Get API Key](https://platform.openai.com/api-keys)

### Optional Dependencies

For full functionality, you may also want to install:

- **Tesseract OCR** - For image-based document processing
  - Windows: Download from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)
  - Mac: `brew install tesseract`
  - Linux: `sudo apt-get install tesseract-ocr`

## Installation Steps

### 1. Clone the Repository

```bash
git clone <repository-url>
cd local-agent-studio
```

### 2. Set Up Environment Variables

Create a `.env` file in the root directory:

```bash
# Copy the example file
cp .env.example .env

# Edit the file and add your OpenAI API key
# .env
OPENAI_API_KEY=your-api-key-here
```

### 3. Initialize the System

Run the initialization script to set up directories and configuration:

#### Windows

```bash
cd server
python scripts\init_db.py
```

#### Linux/Mac

```bash
cd server
python scripts/init_db.py
```

This script will:
- Create necessary data directories
- Generate default configuration files
- Initialize the vector database
- Set up the memory system

### 4. Verify Installation

Run the health check to ensure everything is configured correctly:

#### Windows

```bash
cd server
python scripts\health_check.py
```

#### Linux/Mac

```bash
cd server
python scripts/health_check.py
```

All checks should pass. If any fail, follow the error messages to fix the issues.

## Starting the Application

### Development Mode

#### Windows

Double-click `start-dev.bat` or run from command line:

```bash
start-dev.bat
```

#### Linux/Mac

```bash
chmod +x start-dev.sh
./start-dev.sh
```

This will start both the backend and frontend servers:
- **Backend API**: http://localhost:8000
- **Frontend UI**: http://localhost:3000
- **API Documentation**: http://localhost:8000/docs

### Manual Start (Alternative)

If you prefer to start services manually:

#### Backend

```bash
cd server
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate

pip install -e .
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

#### Frontend

```bash
cd ui
npm install
npm run dev
```

## Configuration

### Agent Configuration

Edit `config/agents.yaml` to customize agent behavior:

```yaml
orchestrator:
  max_iterations: 6
  token_budget: 50000
  workflow_confidence_threshold: 0.7
  timeout_seconds: 300

planner:
  max_tasks: 10
  decomposition_strategy: "minimal"

agent_generator:
  max_concurrent_agents: 5
  default_max_steps: 4
  default_token_limit: 2000
```

### Workflow Configuration

Edit `config/workflows.yaml` to define or modify workflows:

```yaml
workflows:
  rag_qa:
    name: "RAG Question Answering"
    description: "Answer questions using retrieved context"
    enabled: true
    triggers:
      - "what"
      - "how"
      - "why"
```

### Memory Configuration

Edit `config/memory.yaml` to configure memory settings:

```yaml
memory:
  provider: "mem0"
  ttl_days: 30
  max_entries: 10000
```

## Troubleshooting

### Common Issues

#### 1. Python Version Error

**Error**: `Python version 3.11 or higher required`

**Solution**: Install Python 3.11+ and ensure it's in your PATH

#### 2. OpenAI API Key Not Set

**Error**: `OPENAI_API_KEY not found`

**Solution**: Edit `.env` file and add your API key

#### 3. Port Already in Use

**Error**: `Address already in use: 8000` or `3000`

**Solution**: Stop other services using these ports or change ports in configuration

#### 4. Module Not Found Errors

**Error**: `ModuleNotFoundError: No module named 'fastapi'`

**Solution**: Ensure virtual environment is activated and dependencies are installed:

```bash
cd server
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -e .
```

#### 5. Tesseract Not Found

**Warning**: `Tesseract OCR not available`

**Solution**: This is optional. Install Tesseract OCR if you need image-based document processing.

### Getting Help

If you encounter issues not covered here:

1. Check the logs in the terminal windows
2. Run the health check: `python scripts/health_check.py`
3. Check the API documentation at http://localhost:8000/docs
4. Review the error messages carefully

## Next Steps

Once installed, you can:

1. **Upload Documents**: Use the file upload interface to add documents
2. **Ask Questions**: Type questions about your documents in the chat
3. **Monitor Execution**: Use the Inspector panel to see how requests are processed
4. **Customize Workflows**: Edit configuration files to tailor the system to your needs

## Updating

To update to the latest version:

```bash
git pull
cd server
pip install -e . --upgrade
cd ../ui
npm install
```

## Uninstalling

To remove Local Agent Studio:

1. Stop all running services
2. Delete the project directory
3. Remove the virtual environment: `rm -rf server/venv`
4. Remove data directory: `rm -rf data`

## System Requirements

### Minimum Requirements

- **CPU**: 2 cores
- **RAM**: 4 GB
- **Disk**: 2 GB free space
- **OS**: Windows 10+, macOS 10.15+, or Linux

### Recommended Requirements

- **CPU**: 4+ cores
- **RAM**: 8+ GB
- **Disk**: 10+ GB free space (for document storage)
- **OS**: Windows 11, macOS 12+, or Ubuntu 20.04+

## Security Notes

- Keep your `.env` file secure and never commit it to version control
- Your OpenAI API key should be kept private
- All data is stored locally in the `data/` directory
- No data is sent to external services except OpenAI API calls

## License

See LICENSE file for details.
