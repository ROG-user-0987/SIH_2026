import logging
import os
import uuid

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from .websocket_handler import WebSocketHandler

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

MOCK_MODE = os.getenv("MOCK_MODE", "true").lower() == "true"
MODEL_DIR = os.getenv("MODEL_DIR", None)
REMOTE_INFERENCE_URL = os.getenv("REMOTE_INFERENCE_URL", None)
USE_SILERO_VAD = os.getenv("USE_SILERO_VAD", "false").lower() == "true"
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

app = FastAPI(
    title="Voice Cloning Detection API",
    description="Real-time voice cloning detection over WebSocket",
    version="0.3.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

handler = WebSocketHandler(
    mock_mode=MOCK_MODE,
    model_dir=MODEL_DIR,
    remote_url=REMOTE_INFERENCE_URL,
    use_silero=USE_SILERO_VAD,
)


@app.get("/")
async def root():
    return {
        "service": "voice-cloning-detection",
        "version": "0.3.0",
        "status": "running",
        "mock_mode": MOCK_MODE,
    }


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "mock_mode": MOCK_MODE,
        "remote_inference": bool(REMOTE_INFERENCE_URL),
    }


@app.websocket("/ws/analyze")
async def websocket_analyze(websocket: WebSocket):
    await websocket.accept()
    session_id = str(uuid.uuid4())[:8]
    logger.info(f"New WebSocket connection: {session_id}")

    try:
        await handler.process_session(websocket)
    except WebSocketDisconnect:
        logger.info(f"Client disconnected: {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error for {session_id}: {e}")
    finally:
        logger.info(f"Session {session_id} closed")


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")
    uvicorn.run(
        "backend.main:app",
        host=host,
        port=port,
        reload=os.getenv("RELOAD", "false").lower() == "true",
    )
