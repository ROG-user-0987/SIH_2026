from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
WINDOW_SIZE_SAMPLES = 512
THRESHOLD = 0.5


class SileroVAD:
    def __init__(
        self,
        threshold: float = THRESHOLD,
        sample_rate: int = SAMPLE_RATE,
        use_torch: bool = True,
    ):
        self.threshold = threshold
        self.sample_rate = sample_rate
        self.use_torch = use_torch
        self._model = None
        self._load_model()

    def _load_model(self) -> None:
        if not self.use_torch:
            logger.info("Silero VAD disabled — using energy-based VAD (no downloads)")
            self._model = None
            return
        try:
            import torch
            model, utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                onnx=False,
            )
            self._model = model
            self._get_speech_timestamps = utils[0]
            self._get_audio_tensor = utils[2]
            logger.info("Silero VAD model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load Silero VAD: {e}. Using energy-based fallback.")
            self._model = None

    def reset_states(self) -> None:
        if self._model is not None:
            self._model.reset_states()

    def is_speech(self, audio_chunk: np.ndarray) -> bool:
        if len(audio_chunk) == 0:
            return False

        if self._model is not None:
            return self._infer_torch(audio_chunk)
        return self._infer_energy(audio_chunk)

    def _infer_torch(self, audio: np.ndarray) -> bool:
        import torch

        tensor = torch.from_numpy(audio).float()
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
        prob = self._model(tensor, self.sample_rate).item()
        return prob > self.threshold

    def _infer_energy(self, audio: np.ndarray) -> bool:
        rms = np.sqrt(np.mean(audio.astype(np.float32) ** 2))
        return rms > 0.01

    def process_chunk(self, audio_chunk: np.ndarray) -> dict:
        is_speech = self.is_speech(audio_chunk)
        energy = float(np.sqrt(np.mean(audio_chunk.astype(np.float32) ** 2)))
        return {
            "is_speech": is_speech,
            "energy": energy,
            "threshold": self.threshold,
        }
