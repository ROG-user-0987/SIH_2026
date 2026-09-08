from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

MODEL_VERSION = "fusion-v0.3"


class FusionModel:
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self._model = None
        self._loaded = False
        self._weights = np.array([0.35, 0.30, 0.15, 0.20])
        self._bias = 0.0
        self._load_model()

    def _load_model(self) -> None:
        if self.model_path is None:
            logger.info("No fusion model path provided — using default weights")
            return
        try:
            from sklearn.linear_model import LogisticRegression
            import joblib
            self._model = joblib.load(self.model_path)
            self._loaded = True
            if hasattr(self._model, "coef_"):
                self._weights = self._model.coef_[0]
                self._bias = float(self._model.intercept_[0])
            logger.info("Fusion model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load fusion model: {e}. Using default weights.")

    @property
    def version(self) -> str:
        return MODEL_VERSION

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def fuse(
        self,
        ssl_aasist_score: float,
        aasist_score: float,
        prosody_score: float,
        speaker_score: float = 0.5,
    ) -> dict:
        scores = np.array([ssl_aasist_score, aasist_score, prosody_score, speaker_score])

        if self._loaded and self._model is not None:
            try:
                import joblib
                proba = self._model.predict_proba(scores.reshape(1, -1))[0]
                spoof_prob = float(proba[1])
                contribution_weights = self._calculate_contribution_weights(
                    scores, self._weights
                )
                risk_score = int(round(spoof_prob * 100))
                return self._build_result(risk_score, spoof_prob, contribution_weights)
            except Exception as e:
                logger.error(f"Fusion model inference failed: {e}")

        raw_logit = float(np.dot(self._weights, scores) + self._bias)
        spoof_prob = 1.0 / (1.0 + np.exp(-raw_logit))
        contribution_weights = self._calculate_contribution_weights(scores, self._weights)
        risk_score = int(round(spoof_prob * 100))
        return self._build_result(risk_score, spoof_prob, contribution_weights)

    def _calculate_contribution_weights(
        self, scores: np.ndarray, weights: np.ndarray
    ) -> list:
        signed_contributions = np.abs(weights * scores)
        total = np.sum(signed_contributions)
        if total > 0:
            normalized = signed_contributions / total
        else:
            normalized = np.ones_like(signed_contributions) / len(signed_contributions)
        return normalized.tolist()

    def _build_result(
        self, risk_score: int, spoof_prob: float, contribution_weights: list
    ) -> dict:
        if risk_score >= 70:
            verdict = "FAKE"
        elif risk_score >= 30:
            verdict = "REAL"
        else:
            verdict = "REAL"

        if spoof_prob > 0.7:
            confidence = 0.5 + 0.5 * (spoof_prob - 0.7) / 0.3
        elif spoof_prob < 0.3:
            confidence = 0.5 + 0.5 * (0.3 - spoof_prob) / 0.3
        else:
            confidence = 0.3 + 0.2 * (1.0 - abs(spoof_prob - 0.5) * 2)
        confidence = float(np.clip(confidence, 0.3, 0.99))

        return {
            "risk_score": risk_score,
            "verdict": verdict,
            "confidence": round(confidence, 3),
            "spoof_prob": round(spoof_prob, 4),
            "weights": {
                "ssl_aasist": round(float(contribution_weights[0]), 4),
                "aasist": round(float(contribution_weights[1]), 4),
                "prosody": round(float(contribution_weights[2]), 4),
                "speaker": round(float(contribution_weights[3]), 4),
            },
        }
