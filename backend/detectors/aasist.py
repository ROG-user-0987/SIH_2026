from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class AASISTDetector:
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self._session = None
        self._loaded = False
        self._load_model()

    def _load_model(self) -> None:
        if self.model_path is None:
            logger.info("No AASIST model path provided — using mock mode")
            return
        try:
            import onnxruntime as ort
            self._session = ort.InferenceSession(self.model_path)
            self._loaded = True
            logger.info("AASIST ONNX model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load AASIST model: {e}. Running in mock mode.")

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def predict(self, audio: np.ndarray) -> dict:
        if not self._loaded or self._session is None:
            return self._mock_predict(audio)

        try:
            import torch
            features = self._extract_features(audio)
            input_tensor = features.reshape(1, 1, -1).astype(np.float32)
            outputs = self._session.run(None, {"input": input_tensor})
            spoof_prob = float(outputs[0][0][0])
            spoof_prob = np.clip(spoof_prob, 0.0, 1.0)
            return {
                "name": "aasist",
                "category": "spectral",
                "score": float(spoof_prob),
                "loaded": True,
            }
        except Exception as e:
            logger.error(f"AASIST inference failed: {e}")
            return self._mock_predict(audio)

    def _extract_features(self, audio: np.ndarray) -> np.ndarray:
        try:
            import torch
            import torchaudio

            waveform = torch.from_numpy(audio).float().unsqueeze(0)
            mel_spec = torchaudio.transforms.MelSpectrogram(
                sample_rate=16000, n_mels=80, n_fft=512, hop_length=160
            )(waveform)
            log_mel = torch.log(mel_spec + 1e-9)
            return log_mel.numpy()
        except ImportError:
            return self._dct_features(audio)

    def _dct_features(self, audio: np.ndarray) -> np.ndarray:
        from scipy.fftpack import dct
        frame_length = 512
        hop_length = 160
        n_frames = 1 + (len(audio) - frame_length) // hop_length
        frames = np.zeros((n_frames, frame_length))
        for i in range(n_frames):
            start = i * hop_length
            frames[i] = audio[start:start + frame_length]
        windowed = frames * np.hanning(frame_length)
        mag = np.abs(np.fft.rfft(windowed))
        features = dct(mag, type=2, axis=1, norm="ortho")[:, :80]
        return features.T

    def _mock_predict(self, audio: np.ndarray) -> dict:
        energy = float(np.sqrt(np.mean(audio.astype(np.float32) ** 2)))
        spoof_prob = min(1.0, max(0.0, energy * 10.0 + np.random.uniform(-0.1, 0.1)))
        return {
            "name": "aasist",
            "category": "spectral",
            "score": spoof_prob,
            "loaded": False,
        }
