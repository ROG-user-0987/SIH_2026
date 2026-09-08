from __future__ import annotations

from typing import Optional


TAG_DEFINITIONS = {
    "unnatural_spectral_pattern": {
        "category": "spectral",
        "plain_language": "Unusual spectral pattern typical of synthetic speech",
        "triggers": {
            "ssl_aasist": lambda score, _: score > 0.65,
            "aasist": lambda score, _: score > 0.65,
        },
    },
    "low_pitch_variance": {
        "category": "prosody",
        "plain_language": "Unnaturally flat pitch variance",
        "triggers": {
            "prosody": lambda score, features: features.get("top_feature") == "pitch_variance",
        },
    },
    "high_spectral_artifact": {
        "category": "spectral",
        "plain_language": "High-frequency spectral artifacts detected",
        "triggers": {
            "ssl_aasist": lambda score, _: score > 0.8,
            "aasist": lambda score, _: score > 0.75,
        },
    },
    "jitter_anomaly": {
        "category": "prosody",
        "plain_language": "Unnatural pitch perturbation (jitter)",
        "triggers": {
            "prosody": lambda score, features: features.get("top_feature") == "jitter",
        },
    },
    "shimmer_anomaly": {
        "category": "prosody",
        "plain_language": "Unnatural amplitude perturbation (shimmer)",
        "triggers": {
            "prosody": lambda score, features: features.get("top_feature") == "shimmer",
        },
    },
    "natural_spectral_pattern": {
        "category": "spectral",
        "plain_language": "Spectral pattern consistent with natural speech",
        "triggers": {
            "ssl_aasist": lambda score, _: score < 0.35,
            "aasist": lambda score, _: score < 0.35,
        },
    },
    "healthy_pitch_variation": {
        "category": "prosody",
        "plain_language": "Pitch variation within normal range",
        "triggers": {
            "prosody": lambda score, features: features.get("top_feature") == "pitch_variance",
        },
    },
    "inconsistent_detector_signals": {
        "category": "spectral",
        "plain_language": "Detectors disagree on authenticity",
        "triggers": {
            "_cross_detector": lambda scores: abs(scores.get("ssl_aasist", 0) - scores.get("aasist", 0)) > 0.4,
        },
    },
    "speech_rate_anomaly": {
        "category": "prosody",
        "plain_language": "Unusual speech rate pattern",
        "triggers": {
            "prosody": lambda score, features: features.get("top_feature") == "speech_rate",
        },
    },
    "low_confidence_signal": {
        "category": "meta",
        "plain_language": "Weak signal strength — result may be less reliable",
        "triggers": {
            "_overall": lambda fusion_result: 0.35 <= fusion_result.get("spoof_prob", 0.5) <= 0.65,
        },
    },
}

CATEGORY_TO_PLAIN_LANGUAGE = {
    "spectral": "Spectral pattern",
    "prosody": "Voice rhythm & pitch",
    "speaker_match": "Speaker identity",
    "transcript_risk": "Message content",
    "meta": "Overall assessment",
}


class TagVocabulary:
    VERSION = "explain-v0.1"

    def generate_rationale_tags(
        self,
        signals: list[dict],
        fusion_result: dict,
        optional_signals: Optional[dict] = None,
    ) -> dict:
        signal_map = {s["name"]: s for s in signals}
        scores = {name: data["score"] for name, data in signal_map.items()}

        tags = []
        for tag_name, definition in TAG_DEFINITIONS.items():
            for trigger_source, trigger_fn in definition["triggers"].items():
                if trigger_source.startswith("_"):
                    if trigger_source == "_cross_detector" and trigger_fn(scores):
                        tags.append(tag_name)
                        break
                    elif trigger_source == "_overall" and trigger_fn(fusion_result):
                        tags.append(tag_name)
                        break
                    continue
                if trigger_source in signal_map:
                    signal = signal_map[trigger_source]
                    if trigger_fn(signal["score"], signal):
                        tags.append(tag_name)
                        break

        return tags

    def resolve_category(self, category: str) -> str:
        return CATEGORY_TO_PLAIN_LANGUAGE.get(category, category)

    def resolve_tag(self, tag: str) -> str:
        definition = TAG_DEFINITIONS.get(tag)
        if definition:
            return definition["plain_language"]
        return tag.replace("_", " ").title()
