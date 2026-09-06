"""Run the remote inference server on Colab (free T4) and expose it publicly.

Usage (in a Colab cell, after mounting/uploading this repo or fetching the files):
    %pip install -q -r backend/remote_server/requirements-remote.txt
    %run backend/remote_server/colab_run.py

The server boots on port 8001 and a public HTTPS URL is printed (via cloudflared).
Paste that URL into the local app's backend/.env as REMOTE_INFERENCE_URL.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8001"))
MODEL = os.getenv("MODEL_NAME", "spectra")  # spectra | aasist_local


def boot_uvicorn():
    import threading
    import uvicorn

    def run():
        uvicorn.run("server:app", host=HOST, port=PORT, log_level="info")

    t = threading.Thread(target=run, daemon=True)
    t.start()
    return t


def find_cloudflared() -> str | None:
    which = subprocess.run(["which", "cloudflared"], capture_output=True, text=True)
    if which.returncode == 0:
        return which.stdout.strip()
    if Path("/root/cloudflared").exists():
        return "/root/cloudflared"
    return None


def setup_cloudflared() -> str | None:
    existing = find_cloudflared()
    if existing:
        return existing
    url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
    print("Downloading cloudflared...")
    subprocess.run(
        ["curl", "-L", "-o", "/root/cloudflared", url], check=True, capture_output=True
    )
    subprocess.run(["chmod", "+x", "/root/cloudflared"], check=True)
    return "/root/cloudflared"


def main():
    os.chdir(Path(__file__).resolve().parent)
    os.environ["MODEL_NAME"] = MODEL
    boot_uvicorn()
    print(f"Uvicorn booting on {HOST}:{PORT} (model={MODEL})...")
    time.sleep(5)

    proc = None
    try:
        cf = setup_cloudflared()
        print("Starting cloudflared tunnel...")
        proc = subprocess.Popen(
            [cf, "tunnel", "--no-autoupdate", "--url", f"http://{HOST}:{PORT}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        url = None
        for _ in range(120):
            line = proc.stdout.readline()
            if line:
                print(line, end="")
                if "trycloudflare.com" in line:
                    url = line.strip()
                    break
            else:
                time.sleep(2)
        if url:
            print("\n===== PUBLIC URL (paste into backend/.env) =====")
            print(url)
            print("================================================")
        else:
            print("WARNING: could not read tunnel URL. Check output above.")
    except Exception as e:
        print(f"cloudflared failed ({e}); the server still runs locally at "
              f"http://127.0.0.1:{PORT} but is not publicly reachable.")

    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        if proc:
            proc.terminate()


if __name__ == "__main__":
    main()