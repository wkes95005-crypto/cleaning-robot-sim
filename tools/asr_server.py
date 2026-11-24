"""Local speech-to-text service: Whisper on the Mac's GPU through mlx-whisper.

It stands in for the ASR model running on each robot. The dashboard records the microphone, the
simulator sends the audio here, and only the text goes on to the console's planner. Nothing leaves
the computer.

    .venv-asr/bin/python tools/asr_server.py [--model mlx-community/whisper-large-v3-turbo] [--port 8791]

run.py and the Webots fleet_bridge start it automatically when [asr] backend = "whisper".

    POST /transcribe   body: audio in any format ffmpeg reads (webm/opus from the browser, wav ...)
                       -> {"text": "...", "ms": 480, "seconds": 2.1}
    GET  /health       -> {"ok": true, "model": "..."}
"""
import argparse
import json
import os
import subprocess
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Lock

import mlx_whisper
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sim.world import ZONES  # noqa: E402

RATE = 16000
# Listing the words the robots expect makes Whisper spell zone names right (生鮮區, not 生先區).
PROMPT = "賣場清潔機器人指令：" + "、".join(ZONES) + "、掃地、拖地、紙箱、推車、電線、充電。"


def decode(data):
    """Any audio bytes -> 16 kHz mono float32, via ffmpeg."""
    out = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", "pipe:0", "-f", "f32le", "-ac", "1",
                          "-ar", str(RATE), "pipe:1"], input=data, capture_output=True, check=True)
    return np.frombuffer(out.stdout, np.float32)


def transcribe(audio, model):
    seconds = len(audio) / RATE
    # Whisper invents text on silence ("詞・作曲 ..."), so don't give it silence at all
    if seconds < 0.3 or float(np.sqrt(np.mean(audio ** 2))) < 0.003:
        return "", seconds
    r = mlx_whisper.transcribe(audio, path_or_hf_repo=model, language="zh", initial_prompt=PROMPT,
                               condition_on_previous_text=False, temperature=0.0)
    keep = [s["text"] for s in r["segments"]
            if s["compression_ratio"] < 2.0                                  # "銷匙、銷匙、銷匙…" loops
            and not (s["no_speech_prob"] > 0.5 and s["avg_logprob"] < -0.8)]  # noise, not speech
    text = "".join(keep).strip()
    if "指令：" in text or text.count("、") > 8:  # read the prompt back instead of hearing anything
        text = ""
    return text, seconds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/whisper-large-v3-turbo")
    ap.add_argument("--port", type=int, default=8791)
    args = ap.parse_args()

    t0 = time.time()
    transcribe(np.random.default_rng(0).normal(0, 0.05, RATE).astype(np.float32), args.model)  # load + warm up
    print(f"[asr] {args.model} ready in {time.time() - t0:.1f} s on http://127.0.0.1:{args.port}", flush=True)
    gpu = Lock()  # one transcription at a time on the GPU

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _json(self, obj, code=200):
            body = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                return self._json({"ok": True, "model": args.model})
            self._json({"ok": False}, 404)

        def do_POST(self):
            if self.path != "/transcribe":
                return self._json({"ok": False}, 404)
            data = self.rfile.read(int(self.headers.get("Content-Length") or 0))
            t = time.perf_counter()
            try:
                audio = decode(data)
                with gpu:
                    text, seconds = transcribe(audio, args.model)
            except subprocess.CalledProcessError:
                return self._json({"ok": False, "error": "unreadable audio"}, 400)
            ms = round((time.perf_counter() - t) * 1000)
            print(f"[asr] {seconds:.1f} s audio -> {ms} ms: {text}", flush=True)
            self._json({"ok": True, "text": text, "ms": ms, "seconds": round(seconds, 2)})

    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
