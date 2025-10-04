from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import chat, files, memory, config, websocket

app = FastAPI(title="Local Agent Studio API", version="0.1.0")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Next.js default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(chat.router)
app.include_router(files.router)
app.include_router(memory.router)
app.include_router(config.router)
app.include_router(websocket.router)


@app.get("/healthz", tags=["system"])
async def health_check() -> dict[str, str]:
    """Lightweight health check endpoint."""
    return {"status": "ok"}
