import logging
import os
import uuid

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from .websocket_handler import WebSocketHandler
from .detectors.speaker import SpeakerVerifier

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
SPEAKERS_AUTOLOAD = os.getenv("SPEAKERS_AUTOLOAD", "false").lower() == "true"
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
    speak_autoload=SPEAKERS_AUTOLOAD,
)

speaker_verifier = handler.speaker_verifier


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
        "speaker": {
            "loaded": speaker_verifier.is_loaded,
            "enrolled": speaker_verifier.enrolled,
        },
    }


@app.post("/enroll")
async def enroll_speaker(request: Request):
    """Enroll a known speaker. Body: {"speaker_id": str, "audio_b64": str} (int16 PCM mono 16kHz)."""
    import base64
    import json as _json
    import numpy as np

    if speaker_verifier is None or not speaker_verifier.is_loaded:
        raise HTTPException(status_code=503, detail="Speaker encoder unavailable")

    try:
        body = _json.loads(await request.body())
        speaker_id = body.get("speaker_id")
        audio_b64 = body.get("audio_b64")
        sr = int(body.get("sample_rate", 16000))
        if not speaker_id or not audio_b64:
            raise ValueError("speaker_id and audio_b64 required")
        pcm = base64.b64decode(audio_b64)
        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        if sr != 16000:
            import scipy.signal as sg
            audio = sg.resample_poly(audio, 16000, sr)
        ok = speaker_verifier.enroll(speaker_id, audio)
        if not ok:
            raise HTTPException(status_code=400, detail="Enrollment failed")
        return {"speaker_id": speaker_id, "enrolled": speaker_verifier.enrolled}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/speakers")
async def list_speakers():
    if speaker_verifier is None:
        return {"speakers": []}
    return {"speakers": speaker_verifier.enrolled}


@app.delete("/speakers/{speaker_id}")
async def delete_speaker(speaker_id: str):
    if speaker_verifier is None:
        raise HTTPException(status_code=503, detail="Speaker encoder unavailable")
    if not speaker_verifier.delete_speaker(speaker_id):
        raise HTTPException(status_code=404, detail="Speaker not found")
    return {"deleted": speaker_id, "enrolled": speaker_verifier.enrolled}


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
