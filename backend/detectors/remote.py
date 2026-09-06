from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class RemoteDetector:
    def __init__(self, url: Optional[str] = None, timeout: float = 15.0):
        self.url = (url or "").rstrip("/")
        self.timeout = timeout
        self._client = None
        self.last_error: Optional[str] = None
        self.model_info: str = ""
        if self.url:
            logger.info(f"Remote inference detector configured: {self.url}")
        else:
            logger.info("No REMOTE_INFERENCE_URL set — remote detector inactive")

    @property
    def is_loaded(self) -> bool:
        return bool(self.url)

    async def predict(self, audio: np.ndarray) -> dict:
        if not self.url:
            return self._unavailable(None)

        try:
            if self._client is None:
                import httpx
                self._client = httpx.AsyncClient(timeout=self.timeout)
            audio = np.asarray(audio, dtype=np.float32)
            resp = await self._client.post(
                f"{self.url}/analyze",
                params={"fmt": "float32", "sr": 16000},
                content=audio.tobytes(),
            )
            resp.raise_for_status()
            data = resp.json()
            spoof_prob = float(data.get("spoof_prob", 0.5))
            self.model_info = str(data.get("model", "remote"))
            return {
                "name": "aasist",
                "category": "spectral",
                "score": float(np.clip(spoof_prob, 0.0, 1.0)),
                "loaded": True,
                "model": self.model_info,
                "latency_ms": float(data.get("latency_ms", 0.0)),
            }
        except Exception as e:
            logger.error(f"Remote inference request failed: {e}")
            self.last_error = str(e)
            return self._unavailable(str(e))

    def _unavailable(self, error: Optional[str]) -> dict:
        score = 0.5
        reason = error or (None if self.url else "remote not configured")
        result = {
            "name": "aasist",
            "category": "spectral",
            "score": score,
            "loaded": False,
        }
        if reason:
            result["error"] = reason
        return result