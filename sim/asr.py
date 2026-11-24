"""On-robot speech-to-text. Each robot has a microphone and runs ASR itself, then sends only the
text to the console, so raw audio never leaves the robot."""
from __future__ import annotations

import atexit
import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TextASR:
    """For the simulation: the dashboard's typed sentence stands in for what the microphone heard."""

    name = "text"
    audio = False

    def transcribe(self, audio):
        return str(audio).strip()


class WhisperASR:
    """Whisper on this computer (tools/asr_server.py, mlx-whisper on the Mac's GPU).

    In the simulation the service plays the robot's on-board ASR: audio recorded at the dashboard goes
    in, text comes out. Typed text passes straight through, as with TextASR.
    """

    audio = True

    def __init__(self, model, url):
        self.model, self.url = model, url.rstrip("/")
        self.name = f"whisper:{model.rsplit('/', 1)[-1]}"

    def transcribe(self, audio):
        if not isinstance(audio, (bytes, bytearray)):
            return str(audio).strip()
        req = urllib.request.Request(f"{self.url}/transcribe", data=bytes(audio),
                                     headers={"Content-Type": "application/octet-stream"})
        with urllib.request.urlopen(req, timeout=30) as resp:  # raises if the service isn't running
            return json.loads(resp.read())["text"]


def make_asr(cfg):
    if cfg.get("backend") == "whisper":
        return WhisperASR(cfg.get("model", "mlx-community/whisper-large-v3-turbo"),
                          cfg.get("url", "http://127.0.0.1:8791"))
    return TextASR()


def start_service(cfg):
    """Start tools/asr_server.py in .venv-asr if Whisper is configured and it isn't running yet."""
    if cfg.get("backend") != "whisper":
        return None
    url = cfg.get("url", "http://127.0.0.1:8791").rstrip("/")
    try:
        urllib.request.urlopen(f"{url}/health", timeout=1)
        print(f"[asr] using the Whisper service already running at {url}")
        return None
    except OSError:
        pass
    py = os.path.join(ROOT, ".venv-asr", "bin", "python")
    if not os.path.exists(py):
        print("[asr] .venv-asr not found: python3 -m venv .venv-asr && .venv-asr/bin/pip install mlx-whisper")
        return None
    port = url.rsplit(":", 1)[-1]
    proc = subprocess.Popen([py, os.path.join(ROOT, "tools", "asr_server.py"),
                             "--model", cfg.get("model", "mlx-community/whisper-large-v3-turbo"), "--port", port])
    atexit.register(proc.terminate)
    exit_on_sigterm()
    for _ in range(120):  # the model loads in a few seconds once downloaded
        try:
            urllib.request.urlopen(f"{url}/health", timeout=1)
            break
        except OSError:
            time.sleep(0.5)
    return proc


def exit_on_sigterm():
    """Being stopped with SIGTERM normally skips atexit, which would leave helper processes running."""
    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
