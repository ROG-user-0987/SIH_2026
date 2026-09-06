from __future__ import annotations

import numpy as np


class WeightNormalizer:
    @staticmethod
    def normalize_contribution_weights(
        ssl_aasist_score: float,
        aasist_score: float,
        prosody_score: float,
        ssl_weight: float,
        aasist_weight: float,
        prosody_weight: float,
    ) -> dict:
        signed = np.array([
            ssl_weight * ssl_aasist_score,
            aasist_weight * aasist_score,
            prosody_weight * prosody_score,
        ])
        abs_sum = np.sum(np.abs(signed))
        if abs_sum > 0:
            normalized = np.abs(signed) / abs_sum
        else:
            normalized = np.ones(3) / 3.0
        return {
            "ssl_aasist": round(float(normalized[0]), 4),
            "aasist": round(float(normalized[1]), 4),
            "prosody": round(float(normalized[2]), 4),
        }

    @staticmethod
    def rank_signals(signals: list[dict]) -> list[dict]:
        return sorted(signals, key=lambda s: abs(s.get("score", 0)) * abs(s.get("weight", 0)), reverse=True)

    @staticmethod
    def find_dominant(signals: list[dict]) -> dict:
        if not signals:
            return {"name": "unknown", "category": "unknown"}
        ranked = WeightNormalizer.rank_signals(signals)
        top = ranked[0]
        return {"name": top["name"], "category": top.get("category", "unknown")}
