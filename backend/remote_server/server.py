from __future__ import annotations

import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException, Request

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("remote_inference")

NB_SAMP = 64600
SAMPLE_RATE = 16000

SERVE_MODEL = os.getenv("MODEL_NAME", "spectra")  # spectra | aasist_local
AASIST_LOCAL_CODE = os.getenv(
    "AASIST_LOCAL_CODE", str(Path(__file__).resolve().parent.parent / "detectors" / "aasist")
)
AASIST_LOCAL_CKPT = os.getenv(
    "AASIST_LOCAL_CKPT", str(Path(__file__).resolve().parent.parent / "models" / "aasist" / "AASIST.pth")
)


def _pad_trim(w: np.ndarray, n: int = NB_SAMP) -> np.ndarray:
    if len(w) >= n:
        return w[:n].astype(np.float32, copy=False)
    return np.pad(w, (0, n - len(w))).astype(np.float32)


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Model wrappers
# ---------------------------------------------------------------------------

class BaseWrapper:
    name = "base"
    display = "base"
    spoof_idx = 1

    def __init__(self):
        self.device = _device()
        self.model = None
        self.lock = threading.Lock()

    def load(self):
        raise NotImplementedError

    def infer_logits(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class SpectraWrapper(BaseWrapper):
    name = "spectra-aasist3"
    display = "Spectra-AASIST3 (XLS-R 300M, KAN, ASVspoof5/MLAAD-trained)"
    spoof_idx = 1

    def load(self):
        import sys
        here = Path(__file__).resolve().parent
        sys.path.insert(0, str(here))
        from spectra_aasist3_model import SpectraAASIST3  # noqa: F401
        logger.info("Downloading Spectra-AASIST3 weights + XLS-R-300M on first start (one-time, on THIS machine)...")
        self.model = SpectraAASIST3.from_pretrained("lab260/Spectra-AASIST3")
        self.model.eval().to(self.device)
        logger.info(f"Spectra-AASIST3 loaded on {self.device}")

    def infer_logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class LocalAASISTWrapper(BaseWrapper):
    name = "aasist-2019la"
    display = "AASIST (ASVspoof2019-LA, clovaai)"
    spoof_idx = 0

    def load(self):
        import sys
        sys.path.insert(0, AASIST_LOCAL_CODE)
        from AASIST import Model as AASISTModel  # noqa: F401
        d_args = {
            "filts": [70, [1, 32], [32, 32], [32, 64], [64, 64]],
            "gat_dims": [64, 32],
            "pool_ratios": [0.5, 0.5, 0.5],
            "temperatures": [2.0, 5.0, 100.0],
            "first_conv": 1024,
        }
        model = AASISTModel(d_args)
        sd = torch.load(AASIST_LOCAL_CKPT, map_location="cpu", weights_only=False)
        model.load_state_dict(sd)
        self.model = model.eval().to(self.device)
        logger.info(f"Local AASIST loaded on {self.device}")

    def infer_logits(self, x: torch.Tensor) -> torch.Tensor:
        _, logits = self.model(x)
        return logits


def build_wrapper() -> BaseWrapper:
    if SERVE_MODEL == "spectra":
        w = SpectraWrapper()
    elif SERVE_MODEL == "aasist_local":
        w = LocalAASISTWrapper()
    else:
        raise RuntimeError(f"Unknown MODEL_NAME={SERVE_MODEL!r} (expected 'spectra' or 'aasist_local')")
    w.load()
    return w


_wrapper = None
_wrapper_lock = threading.Lock()


@asynccontextmanager
async def lifespan(app_: FastAPI):
    global _wrapper
    with _wrapper_lock:
        try:
            _wrapper = build_wrapper()
            logger.info(f"Serving model: {_wrapper.display} on {_wrapper.device}")
        except Exception as e:
            logger.exception("Failed to load model at startup")
            _wrapper = None
    yield
    with _wrapper_lock:
        _wrapper = None


app = FastAPI(
    title="Voice Cloning Remote Inference",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/")
def root():
    if _wrapper is None:
        return {"status": "starting_or_error", "model": SERVE_MODEL}
    return {
        "service": "voice-cloning-remote-inference",
        "model": _wrapper.name,
        "device": str(_wrapper.device),
    }


@app.get("/health")
def health():
    if _wrapper is None:
        return {"status": "starting_or_error", "detail": "model not loaded yet"}
    return {
        "status": "ok",
        "model": _wrapper.name,
        "device": str(_wrapper.device),
        "torch_cuda": torch.cuda.is_available(),
    }


@app.post("/analyze")
async def analyze(request: Request):
    if _wrapper is None:
        raise HTTPException(status_code=503, detail="model not loaded")

    query = dict(request.query_params)
    fmt = query.get("fmt", "float32")
    try:
        sr = int(query.get("sr", "16000"))
    except ValueError:
        raise HTTPException(status_code=422, detail="sr must be an integer")
    if sr != SAMPLE_RATE:
        logger.warning(f"Unsupported sample rate {sr}; server expects {SAMPLE_RATE}")

    payload = await request.body()
    if not payload:
        raise HTTPException(status_code=422, detail="empty audio body")

    if fmt == "int16":
        audio = np.frombuffer(payload, dtype=np.int16).astype(np.float32) / 32768.0
    elif fmt == "float32":
        audio = np.frombuffer(payload, dtype=np.float32)
    else:
        raise HTTPException(status_code=422, detail="fmt must be float32 or int16")

    x_in = torch.from_numpy(_pad_trim(audio)).view(1, -1).to(_wrapper.device)

    start = time.perf_counter()
    with _wrapper.lock:
        with torch.inference_mode():
            logits = _wrapper.infer_logits(x_in)
    latency_ms = (time.perf_counter() - start) * 1000.0

    logits = logits.float().cpu().view(-1).tolist()
    probs = F.softmax(torch.tensor(logits), dim=0).tolist()
    spoof_prob = float(probs[_wrapper.spoof_idx])

    return {
        "model": _wrapper.name,
        "model_display": _wrapper.display,
        "device": str(_wrapper.device),
        "latency_ms": round(latency_ms, 1),
        "nb_samp": NB_SAMP,
        "len_in": int(len(audio)),
        "logits": [round(v, 4) for v in logits],
        "probs": [round(v, 4) for v in probs],
        "spoof_prob": round(spoof_prob, 4),
    }


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8001"))
    host = os.getenv("HOST", "0.0.0.0")
    uvicorn.run("server:app", host=host, port=port, workers=1)