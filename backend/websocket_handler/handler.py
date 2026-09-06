from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Optional

import numpy as np

from ..vad import SileroVAD
from ..detectors import AASISTDetector, SSLAASISTDetector, ProsodyDetector, RemoteDetector
from ..fusion import FusionModel
from ..explainability import TagVocabulary, WeightNormalizer
from ..sandbox import MockProcessor

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
WINDOW_SIZE_SECONDS = 4
WINDOW_SIZE_SAMPLES = SAMPLE_RATE * WINDOW_SIZE_SECONDS


class WebSocketHandler:
    def __init__(
        self,
        mock_mode: bool = False,
        model_dir: Optional[str] = None,
        remote_url: Optional[str] = None,
        use_silero: bool = False,
    ):
        self.mock_mode = mock_mode
        self.remote_mode = bool(remote_url)
        self.mock_processor = MockProcessor() if mock_mode else None

        if not mock_mode:
            self.vad = SileroVAD(use_torch=use_silero)
            self.ssl_aasist = SSLAASISTDetector(
                model_path=f"{model_dir}/ssl_aasist.onnx" if model_dir else None
            )
            if self.remote_mode:
                self.aasist = RemoteDetector(url=remote_url)
                logger.info("AASIST slot served by REMOTE inference service")
            else:
                self.aasist = AASISTDetector(
                    model_path=f"{model_dir}/aasist.onnx" if model_dir else None
                )
                if self.aasist.is_loaded:
                    logger.info("AASIST served by local ONNX checkpoint")
                else:
                    logger.info("AASIST running in fallback mode (no model available)")
            self.prosody = ProsodyDetector(
                model_path=f"{model_dir}/prosody_xgb.json" if model_dir else None
            )
            self.fusion = FusionModel(
                model_path=f"{model_dir}/fusion_lr.pkl" if model_dir else None
            )
        self.tag_vocabulary = TagVocabulary()
        self.weight_normalizer = WeightNormalizer()

    async def process_session(self, websocket) -> None:
        session_id = None
        sample_rate = SAMPLE_RATE
        audio_buffer = np.array([], dtype=np.float32)
        window_count = 0
        vad = SileroVAD(use_torch=False) if not self.mock_mode else None

        try:
            raw_msg = await websocket.receive_text()
            init_msg = self._parse_json(raw_msg)
            if not init_msg or init_msg.get("type") != "session_init":
                await websocket.close(code=4001, reason="Expected session_init")
                return
            session_id = init_msg.get("session_id", "unknown")
            sample_rate = init_msg.get("sample_rate", SAMPLE_RATE)
            await websocket.send_json({
                "type": "session_ack",
                "session_id": session_id,
                "status": "ready",
            })
            logger.info(f"Session {session_id} initialized (sample_rate={sample_rate})")

            while True:
                message = await websocket.receive()

                if message["type"] == "websocket.disconnect":
                    break

                if "bytes" in message and message["bytes"] is not None:
                    audio_bytes = message["bytes"]
                    audio_chunk = self._decode_audio(audio_bytes, sample_rate)
                    audio_buffer = np.concatenate([audio_buffer, audio_chunk])

                    while len(audio_buffer) >= WINDOW_SIZE_SAMPLES:
                        window = audio_buffer[:WINDOW_SIZE_SAMPLES]
                        audio_buffer = audio_buffer[WINDOW_SIZE_SAMPLES:]
                        window_count += 1

                        if self.mock_mode:
                            result = self.mock_processor.process(len(window))
                        else:
                            result = await self._process_window(window, sample_rate)

                        result["session_id"] = session_id
                        await websocket.send_json(result)

                elif "text" in message and message["text"]:
                    text = message["text"]
                    text_data = self._parse_json(text)
                    if text_data and text_data.get("type") == "stop":
                        break

        except Exception as e:
            logger.error(f"Session {session_id} error: {e}")
            try:
                await websocket.send_json({
                    "type": "error",
                    "error": "processing_error",
                    "message": str(e),
                })
            except Exception:
                pass
        finally:
            logger.info(
                f"Session {session_id} ended after {window_count} windows"
            )

    async def _process_window(
        self, window: np.ndarray, sample_rate: int
    ) -> dict:
        start_time = time.monotonic()

        if not self.mock_mode:
            speech_info = self.vad.process_chunk(window)
            if not speech_info["is_speech"]:
                return {
                    "type": "result",
                    "window_ts_start": datetime.now(timezone.utc).isoformat(),
                    "window_ts_end": datetime.now(timezone.utc).isoformat(),
                    "risk_score": 0,
                    "verdict": "INSUFFICIENT_SPEECH",
                    "confidence": 0.0,
                    "signals": [],
                    "optional_signals": {},
                    "explanation": {
                        "dominant_signal": "none",
                        "dominant_category": "none",
                        "rationale_tags": [],
                        "explainability_version": self.tag_vocabulary.VERSION,
                    },
                    "model_versions": {},
                }

        if self.remote_mode:
            aasist_result = await self.aasist.predict(window)
            ssl_result = dict(aasist_result)
            ssl_result["name"] = "ssl_aasist"
        else:
            ssl_result = self.ssl_aasist.predict(window)
            aasist_result = self.aasist.predict(window)
        prosody_result = self.prosody.predict(window)

        signals = [
            {
                "name": ssl_result["name"],
                "category": ssl_result["category"],
                "score": ssl_result["score"],
                "weight": 0.0,
            },
            {
                "name": aasist_result["name"],
                "category": aasist_result["category"],
                "score": aasist_result["score"],
                "weight": 0.0,
            },
            {
                "name": prosody_result["name"],
                "category": prosody_result["category"],
                "score": prosody_result["score"],
                "weight": 0.0,
                "top_feature": prosody_result.get("top_feature", "unknown"),
            },
        ]

        fusion_result = self.fusion.fuse(
            ssl_aasist_score=ssl_result["score"],
            aasist_score=aasist_result["score"],
            prosody_score=prosody_result["score"],
        )

        weights = fusion_result.get("weights", {})
        for sig in signals:
            sig["weight"] = weights.get(sig["name"], 0.25)

        explanation_data = self.tag_vocabulary.generate_rationale_tags(
            signals=signals,
            fusion_result=fusion_result,
        )

        dominant_info = self.weight_normalizer.find_dominant(signals)

        elapsed_ms = (time.monotonic() - start_time) * 1000
        now = datetime.now(timezone.utc).isoformat()

        optional = {}
        if self.remote_mode:
            optional["remote_model"] = aasist_result.get("model", "unknown")
            optional["remote_latency_ms"] = aasist_result.get("latency_ms", 0.0)
            optional["remote_loaded"] = aasist_result.get("loaded", False)

        return {
            "type": "result",
            "window_ts_start": now,
            "window_ts_end": now,
            "risk_score": fusion_result["risk_score"],
            "verdict": fusion_result["verdict"],
            "confidence": fusion_result["confidence"],
            "signals": signals,
            "optional_signals": optional,
            "explanation": {
                "dominant_signal": dominant_info["name"],
                "dominant_category": dominant_info["category"],
                "rationale_tags": explanation_data,
                "explainability_version": self.tag_vocabulary.VERSION,
            },
            "model_versions": {
                "fusion": self.fusion.version,
                "ssl_aasist": "ssl-aasist-v0.1",
                "aasist": "aasist-v0.1",
                "prosody": "prosody-v0.1",
            },
        }

    def _decode_audio(
        self, audio_bytes: bytes, source_sample_rate: int
    ) -> np.ndarray:
        audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        if source_sample_rate != SAMPLE_RATE:
            audio = self._resample(audio, source_sample_rate, SAMPLE_RATE)
        return audio

    def _resample(
        self, audio: np.ndarray, orig_sr: int, target_sr: int
    ) -> np.ndarray:
        if orig_sr == target_sr:
            return audio
        try:
            import torchaudio
            waveform = torch.from_numpy(audio).unsqueeze(0)
            resampler = torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=target_sr)
            resampled = resampler(waveform)
            return resampled.squeeze(0).numpy()
        except ImportError:
            pass
        ratio = target_sr / orig_sr
        target_len = int(len(audio) * ratio)
        indices = np.linspace(0, len(audio) - 1, target_len)
        return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)

    def _parse_json(self, text: str) -> Optional[dict]:
        import json
        try:
            return json.loads(text)
        except (json.JSONDecodeError, TypeError):
            return None
