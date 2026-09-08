from __future__ import annotations

import logging
import os
import threading
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
USE_FAISS = True
try:
    import faiss
except ImportError:
    USE_FAISS = False
    faiss = None

SPEAKERS_DIR = os.getenv(
    "SPEAKERS_DIR", os.path.join(os.path.dirname(__file__), "..", "models", "speakers")
)
ENROLL_AUTOLOAD = os.getenv("SPEAKERS_AUTOLOAD", "false").lower() == "true"


class SpeakerVerifier:
    """Speaker recognition layer: ECAPA-TDNN embeddings (SpeechBrain) + FAISS
    nearest-neighbour match.

    Score semantics (SPOOF probability, 0-1, aligned with other detectors):
      - No speakers enrolled        -> 0.50 (neutral, 'not_enrolled')
      - Enrolled & matched          -> 0.05 (highly consistent)
      - Enrolled & no match         -> 0.90 (impostor / cloned voice)
    """

    VERSION = "speaker-v0.1"

    def __init__(
        self,
        embed_dir: Optional[str] = None,
        enroll_autoload: bool = ENROLL_AUTOLOAD,
        threshold: float = 0.45,
    ):
        self.embed_dir = embed_dir or SPEAKERS_DIR
        self.threshold = threshold
        self._encoder = None
        self._index = None
        self._speaker_ids: list[str] = []
        self._embeddings: list[np.ndarray] = []
        self._lock = threading.Lock()
        if USE_FAISS:
            self._index = faiss.IndexFlatIP(192)
        self._load_encoder()
        if enroll_autoload:
            self._autoload_registered()

    # ------------------------------------------------------------------
    # Encoder (SpeechBrain ECAPA-TDNN)
    # ------------------------------------------------------------------

    def _load_encoder(self) -> None:
        try:
            from speechbrain.inference.speaker import EncoderClassifier
            self._encoder = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir=os.path.join(self.embed_dir, "ecapa"),
                run_opts={"device": "cuda"},
            )
            logger.info("ECAPA-TDNN speaker encoder loaded (cuda)")
        except Exception as e:
            logger.warning(f"ECAPA-TDNN encoder load failed, retry on cpu: {e}")
            try:
                from speechbrain.inference.speaker import EncoderClassifier
                self._encoder = EncoderClassifier.from_hparams(
                    source="speechbrain/spkrec-ecapa-voxceleb",
                    savedir=os.path.join(self.embed_dir, "ecapa"),
                    run_opts={"device": "cpu"},
                )
                logger.info("ECAPA-TDNN speaker encoder loaded (cpu)")
            except Exception as e2:
                logger.error(f"ECAPA-TDNN encoder unavailable: {e2}")
                self._encoder = None

    @property
    def is_loaded(self) -> bool:
        return self._encoder is not None

    def extract_embedding(self, audio: np.ndarray) -> Optional[np.ndarray]:
        if self._encoder is None:
            return None
        try:
            import torch
            wav = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)
            emb = self._encoder.encode_batch(wav)
            emb = emb.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float32)
            norm = np.linalg.norm(emb) + 1e-8
            return emb / norm
        except Exception as e:
            logger.error(f"Embedding extraction failed: {e}")
            return None

    # ------------------------------------------------------------------
    # Enrollment
    # ------------------------------------------------------------------

    def enroll(self, speaker_id: str, audio: np.ndarray) -> bool:
        emb = self.extract_embedding(audio)
        if emb is None:
            return False
        with self._lock:
            if speaker_id in self._speaker_ids:
                idx = self._speaker_ids.index(speaker_id)
                self._embeddings[idx] = emb
                self._rebuild_index()
                action = "Updated"
            else:
                self._speaker_ids.append(speaker_id)
                self._embeddings.append(emb)
                if USE_FAISS and self._index is not None:
                    self._index.add(emb.reshape(1, -1))
                else:
                    self._index = None
                action = "Enrolled"
        logger.info(f"{action} speaker '{speaker_id}' ({len(self._speaker_ids)} total)")
        self.save()
        return True

    def delete_speaker(self, speaker_id: str) -> bool:
        with self._lock:
            if speaker_id not in self._speaker_ids:
                return False
            idx = self._speaker_ids.index(speaker_id)
            self._speaker_ids.pop(idx)
            self._embeddings.pop(idx)
            self._rebuild_index()
        self.save()
        return True

    def _rebuild_index(self) -> None:
        if USE_FAISS and self._embeddings:
            self._index = faiss.IndexFlatIP(192)
            self._index.add(np.stack(self._embeddings))
        else:
            self._index = None

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------

    def verify(self, audio: np.ndarray) -> dict:
        if len(self._speaker_ids) == 0:
            return {
                "name": "speaker",
                "category": "speaker",
                "score": 0.5,
                "matched_speaker": None,
                "similarity": None,
                "enrolled_count": 0,
            }

        emb = self.extract_embedding(audio)
        if emb is None:
            return {
                "name": "speaker",
                "category": "speaker",
                "score": 0.5,
                "matched_speaker": None,
                "similarity": None,
                "enrolled_count": len(self._speaker_ids),
            }

        with self._lock:
            if USE_FAISS and self._index is not None:
                scores, ids = self._index.search(emb.reshape(1, -1), 1)
                similarity = float(scores[0][0])
                match_id = self._speaker_ids[int(ids[0][0])]
            else:
                sims = [float(np.dot(emb, e) / (np.linalg.norm(emb) * np.linalg.norm(e) + 1e-8))
                        for e in self._embeddings]
                best = int(np.argmax(sims))
                similarity = float(sims[best])
                match_id = self._speaker_ids[best]

        score = 0.05 if similarity >= self.threshold else 0.90
        return {
            "name": "speaker",
            "category": "speaker",
            "score": round(score, 4),
            "matched_speaker": match_id,
            "similarity": round(float(similarity), 4),
            "enrolled_count": len(self._speaker_ids),
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Optional[str] = None) -> bool:
        if not self._embeddings:
            return False
        path = path or os.path.join(self.embed_dir, "speaker_index.npz")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            np.savez(
                path,
                ids=np.array(self._speaker_ids),
                embeddings=np.stack(self._embeddings),
            )
            return True
        except Exception as e:
            logger.error(f"Failed to save speaker index: {e}")
            return False

    def _autoload_registered(self) -> None:
        path = os.path.join(self.embed_dir, "speaker_index.npz")
        if not os.path.exists(path):
            logger.info("No persisted speaker index to autoload")
            return
        try:
            data = np.load(path, allow_pickle=True)
            ids = [str(i) for i in data["ids"]]
            embs = [e for e in data["embeddings"]]
            with self._lock:
                self._speaker_ids = ids
                self._embeddings = embs
                self._rebuild_index()
            logger.info(f"Autoloaded {len(ids)} enrolled speakers from {path}")
        except Exception as e:
            logger.error(f"Failed to autoload speaker index: {e}")

    @property
    def enrolled(self) -> list[str]:
        return list(self._speaker_ids)