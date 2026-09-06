from __future__ import annotations

import random
from datetime import datetime, timezone

MOCK_RATIOS = {
    "REAL": 0.6,
    "FAKE": 0.4,
}


class MockProcessor:
    def process(self, audio_length_samples: int = 64000) -> dict:
        verdict = random.choices(
            list(MOCK_RATIOS.keys()),
            weights=list(MOCK_RATIOS.values()),
        )[0]

        if verdict == "FAKE":
            spoof_prob = random.uniform(0.6, 0.95)
        else:
            spoof_prob = random.uniform(0.05, 0.35)

        risk_score = int(round(spoof_prob * 100))
        confidence = random.uniform(0.7, 0.95)

        ssl_score = spoof_prob + random.uniform(-0.1, 0.1)
        aasist_score = spoof_prob + random.uniform(-0.15, 0.15)
        prosody_score = spoof_prob + random.uniform(-0.2, 0.2)

        ssl_score = max(0.0, min(1.0, ssl_score))
        aasist_score = max(0.0, min(1.0, aasist_score))
        prosody_score = max(0.0, min(1.0, prosody_score))

        total = ssl_score + aasist_score + prosody_score
        if total > 0:
            w_ssl = ssl_score / total
            w_aasist = aasist_score / total
            w_prosody = prosody_score / total
        else:
            w_ssl = w_aasist = w_prosody = 1.0 / 3.0

        feature_options = [
            "pitch_variance", "jitter", "shimmer", "speech_rate",
            "spectral_centroid_mean", "zcr_mean",
        ]
        top_feature = random.choice(feature_options)

        signals = [
            {"name": "ssl_aasist", "category": "spectral", "score": round(ssl_score, 4), "weight": round(w_ssl, 4)},
            {"name": "aasist", "category": "spectral", "score": round(aasist_score, 4), "weight": round(w_aasist, 4)},
            {"name": "prosody", "category": "prosody", "score": round(prosody_score, 4), "weight": round(w_prosody, 4), "top_feature": top_feature},
        ]

        if verdict == "FAKE":
            dominant = random.choice(["ssl_aasist", "aasist"])
            dominant_category = "spectral"
            rationale_tags = random.sample(
                ["unnatural_spectral_pattern", "high_spectral_artifact"], k=min(2, 2)
            )
        else:
            dominant = "prosody"
            dominant_category = "prosody"
            rationale_tags = random.sample(
                ["natural_spectral_pattern", "healthy_pitch_variation"], k=min(2, 2)
            )

        now = datetime.now(timezone.utc).isoformat()
        return {
            "type": "result",
            "window_ts_start": now,
            "window_ts_end": now,
            "risk_score": risk_score,
            "verdict": verdict,
            "confidence": round(confidence, 3),
            "signals": signals,
            "optional_signals": {},
            "explanation": {
                "dominant_signal": dominant,
                "dominant_category": dominant_category,
                "rationale_tags": rationale_tags,
                "explainability_version": "explain-v0.1",
            },
            "model_versions": {
                "fusion": "mock-fusion-v0.1",
                "ssl_aasist": "mock-ssl-aasist-v0.1",
                "aasist": "mock-aasist-v0.1",
                "prosody": "mock-prosody-v0.1",
            },
        }
