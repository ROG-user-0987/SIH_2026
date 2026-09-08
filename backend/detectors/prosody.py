from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000

# librosa prosody extraction is accurate but slow (pyin on 4s windows can take
# seconds on CPU). Off by default to keep the real-time path responsive.
USE_LIBROSA_PROSODY = os.getenv("USE_LIBROSA_PROSODY", "false").lower() == "true"


class ProsodyDetector:
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self._model = None
        self._loaded = False
        self._load_model()

    def _load_model(self) -> None:
        if self.model_path is None:
            logger.info("No XGBoost model path provided — using mock mode")
            return
        try:
            import xgboost as xgb
            self._model = xgb.XGBClassifier()
            self._model.load_model(self.model_path)
            self._loaded = True
            logger.info("XGBoost prosody model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load XGBoost model: {e}. Running in mock mode.")

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def predict(self, audio: np.ndarray) -> dict:
        features = self.extract_features(audio)
        feature_names = self._get_feature_names()
        top_feature_idx = int(np.argmax(np.abs(features)))
        top_feature_name = (
            feature_names[top_feature_idx]
            if top_feature_idx < len(feature_names)
            else "unknown"
        )

        if self._loaded and self._model is not None:
            try:
                prob = self._model.predict_proba(features.reshape(1, -1))[0][1]
                return {
                    "name": "prosody",
                    "category": "prosody",
                    "score": float(prob),
                    "top_feature": top_feature_name,
                    "feature_values": features.tolist(),
                    "feature_names": feature_names[:len(features)],
                    "loaded": True,
                }
            except Exception as e:
                logger.error(f"XGBoost inference failed: {e}")

        spoof_prob = self._heuristic_score(features, feature_names, top_feature_idx)
        return {
            "name": "prosody",
            "category": "prosody",
            "score": spoof_prob,
            "top_feature": top_feature_name,
            "feature_values": features.tolist(),
            "feature_names": feature_names[:len(features)],
            "loaded": False,
        }

    def _heuristic_score(
        self, features: np.ndarray, feature_names: list, top_idx: int
    ) -> float:
        score = 0.5
        if top_idx < len(feature_names):
            name = feature_names[top_idx]
            val = float(features[top_idx])
            if "pitch" in name and "variance" in name:
                score += 0.15 if val < 0.3 else -0.1
            elif "jitter" in name:
                score += 0.1 if val < 0.01 else -0.1
            elif "shimmer" in name:
                score += 0.1 if val < 0.05 else -0.1
            elif "speech_rate" in name:
                score += 0.1 if val < 3.0 or val > 6.0 else -0.05
        return float(np.clip(score, 0.0, 1.0))

    def extract_features(self, audio: np.ndarray) -> np.ndarray:
        f0 = self._extract_f0(audio)
        jitter = self._calc_jitter(f0)
        shimmer = self._calc_shimmer(audio)
        spectral_centroid = self._spectral_centroid(audio)
        spectral_rolloff = self._spectral_rolloff(audio)
        spectral_bandwidth = self._spectral_bandwidth(audio)
        zcr = self._zero_crossing_rate(audio)
        rms = self._rms_energy(audio)
        speech_rate = self._estimate_speech_rate(audio)
        mfccs = self._extract_mfccs(audio)

        features = np.array([
            float(np.mean(f0)) if len(f0) > 0 else 0.0,
            float(np.std(f0)) if len(f0) > 0 else 0.0,
            float(np.median(f0)) if len(f0) > 0 else 0.0,
            float(np.percentile(f0, 5)) if len(f0) > 0 else 0.0,
            float(np.percentile(f0, 95)) if len(f0) > 0 else 0.0,
            jitter,
            shimmer,
            float(np.mean(spectral_centroid)),
            float(np.std(spectral_centroid)),
            float(np.mean(spectral_rolloff)),
            float(np.mean(spectral_bandwidth)),
            float(np.mean(zcr)),
            float(np.std(zcr)),
            float(np.mean(rms)),
            float(np.std(rms)),
            speech_rate,
            float(np.std(audio)),
            float(np.max(np.abs(audio))),
        ])

        features = np.concatenate([features, mfccs])
        return features

    def _get_feature_names(self) -> list:
        base = [
            "pitch_mean", "pitch_variance", "pitch_median",
            "pitch_p5", "pitch_p95", "jitter", "shimmer",
            "spectral_centroid_mean", "spectral_centroid_std",
            "spectral_rolloff_mean", "spectral_bandwidth_mean",
            "zcr_mean", "zcr_std", "rms_energy_mean", "rms_energy_std",
            "speech_rate", "amplitude_std", "amplitude_max",
        ]
        mfcc_names = [f"mfcc_{i}" for i in range(13)]
        return base + mfcc_names

    def _extract_f0(self, audio: np.ndarray) -> np.ndarray:
        if USE_LIBROSA_PROSODY:
            try:
                import librosa
                f0, voiced_flag, voiced_probs = librosa.pyin(
                    audio, fmin=50, fmax=500, sr=SAMPLE_RATE
                )
                return f0[~np.isnan(f0)] if f0 is not None else np.array([])
            except ImportError:
                pass
        return self._autocorr_f0(audio)

    def _autocorr_f0(self, audio: np.ndarray) -> np.ndarray:
        frame_length = 800
        hop_length = 400
        f0_values = []
        for i in range(0, len(audio) - frame_length, hop_length):
            frame = audio[i:i + frame_length]
            corr = np.correlate(frame, frame, mode="full")
            corr = corr[len(corr) // 2:]
            min_lag = int(SAMPLE_RATE / 500)
            max_lag = int(SAMPLE_RATE / 50)
            if max_lag > len(corr):
                continue
            search = corr[min_lag:max_lag]
            if len(search) == 0:
                continue
            lag = np.argmax(search) + min_lag
            if corr[lag] > 0.3 * corr[0]:
                f0_values.append(SAMPLE_RATE / lag)
        return np.array(f0_values)

    def _calc_jitter(self, f0: np.ndarray) -> float:
        if len(f0) < 2:
            return 0.0
        diffs = np.abs(np.diff(f0))
        return float(np.mean(diffs) / np.mean(f0)) if np.mean(f0) > 0 else 0.0

    def _calc_shimmer(self, audio: np.ndarray) -> float:
        frame_length = 800
        hop_length = 400
        amplitudes = []
        for i in range(0, len(audio) - frame_length, hop_length):
            frame = audio[i:i + frame_length]
            amplitudes.append(float(np.max(np.abs(frame))))
        amplitudes = np.array(amplitudes)
        if len(amplitudes) < 2:
            return 0.0
        diffs = np.abs(np.diff(amplitudes))
        mean_amp = np.mean(amplitudes)
        return float(np.mean(diffs) / mean_amp) if mean_amp > 0 else 0.0

    def _spectral_centroid(self, audio: np.ndarray) -> np.ndarray:
        try:
            import librosa
            return librosa.feature.spectral_centroid(y=audio, sr=SAMPLE_RATE)[0]
        except ImportError:
            return np.array([3000.0])

    def _spectral_rolloff(self, audio: np.ndarray) -> np.ndarray:
        try:
            import librosa
            return librosa.feature.spectral_rolloff(y=audio, sr=SAMPLE_RATE)[0]
        except ImportError:
            return np.array([5000.0])

    def _spectral_bandwidth(self, audio: np.ndarray) -> np.ndarray:
        try:
            import librosa
            return librosa.feature.spectral_bandwidth(y=audio, sr=SAMPLE_RATE)[0]
        except ImportError:
            return np.array([2000.0])

    def _zero_crossing_rate(self, audio: np.ndarray) -> np.ndarray:
        try:
            import librosa
            return librosa.feature.zero_crossing_rate(audio)[0]
        except ImportError:
            signs = np.sign(audio)
            return np.array([np.mean(np.abs(np.diff(signs)))])

    def _rms_energy(self, audio: np.ndarray) -> np.ndarray:
        frame_length = 512
        hop_length = 256
        rms_values = []
        for i in range(0, len(audio) - frame_length, hop_length):
            frame = audio[i:i + frame_length]
            rms_values.append(float(np.sqrt(np.mean(frame ** 2))))
        return np.array(rms_values) if rms_values else np.array([0.0])

    def _estimate_speech_rate(self, audio: np.ndarray) -> float:
        try:
            import librosa
            onset_env = librosa.onset.onset_strength(y=audio, sr=SAMPLE_RATE)
            tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=SAMPLE_RATE)
            return float(tempo[0]) if len(tempo) > 0 else 4.0
        except (ImportError, Exception):
            return 4.0

    def _extract_mfccs(self, audio: np.ndarray, n_mfcc: int = 13) -> np.ndarray:
        try:
            import librosa
            mfccs = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=n_mfcc)
            return np.mean(mfccs, axis=1)
        except ImportError:
            return np.zeros(n_mfcc)
