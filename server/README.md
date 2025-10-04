# Local Agent Studio Server

This FastAPI backend powers the Local Agent Studio orchestrator and supporting services.

## Development setup

1. Create a Python virtual environment:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
2. Install dependencies with `pip` (uses the `pyproject.toml` metadata):
   ```powershell
   pip install -U pip setuptools wheel
   pip install -e .[dev]
   ```
3. Run the development server:
   ```powershell
   uvicorn app.main:app --reload --port 8000
   ```

## Testing

Run the unit test suite with `pytest`:
```powershell
pytest
```
