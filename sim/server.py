"""Local web server for the dashboard: runs the simulation in a background thread and serves JSON."""
from __future__ import annotations

import json
import os
import re
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .engine import Simulation

WEB_DIR = Path(__file__).resolve().parent.parent / "web"


class Runner:
    def __init__(self, cfg):
        self.cfg = cfg
        self.lock = threading.Lock()
        self.tps = float(cfg.get("sim", {}).get("ticks_per_second", 6))
        self.paused = False
        self.sim = Simulation(cfg)
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while True:
            if not self.paused:
                with self.lock:
                    self.sim.step()
            time.sleep(1.0 / max(self.tps, 0.5))

    def reset(self):
        with self.lock:
            self.sim = Simulation(self.cfg)


def make_handler(runner):
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=str(WEB_DIR), **kw)

        def log_message(self, *args):
            pass

        def _json(self, obj, code=200):
            body = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _body(self):
            n = int(self.headers.get("Content-Length") or 0)
            return json.loads(self.rfile.read(n) or b"{}")

        def _camera(self, rid, fps):
            """Latest JPEG from a robot's camera (Webots only). Asking for it keeps that camera on."""
            frames = getattr(runner, "frames_dir", None)
            if not frames:
                return self._json({"ok": False, "error": "no cameras in this simulator"}, 404)
            if not runner.sim.robot(rid):
                return self._json({"ok": False, "error": "unknown robot"}, 404)
            with open(os.path.join(frames, f"{rid}.want"), "w") as f:
                f.write(str(fps))
            try:
                with open(os.path.join(frames, f"{rid}.jpg"), "rb") as f:
                    mtime = os.fstat(f.fileno()).st_mtime_ns
                    body = f.read()
            except OSError:
                return self._json({"ok": False, "error": "no frame yet"}, 404)
            det = self._detections(frames, rid, mtime)
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("X-Detections", json.dumps(det) if det else "off")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        @staticmethod
        def _detections(frames, rid, mtime):
            """YOLO boxes for exactly this frame (the worker needs a few ms, so wait briefly for it).
            None when no detector is running."""
            path = os.path.join(frames, f"{rid}.det.json")
            for _ in range(10):
                try:
                    if time.time() - os.path.getmtime(path) > 5:
                        return None  # worker not running
                    with open(path) as f:
                        det = json.load(f)
                except (OSError, ValueError):
                    return None
                if det.get("frame") == mtime:
                    return det
                time.sleep(0.01)
            return {**det, "stale": True}

        def do_GET(self):
            url = urlparse(self.path)
            cam = re.fullmatch(r"/api/camera/(\w+)\.jpg", url.path)
            if cam:
                fps = float(parse_qs(url.query).get("fps", ["2"])[0])
                return self._camera(cam.group(1), max(0.5, min(15.0, fps)))
            if self.path == "/api/map":
                with runner.lock:
                    info = runner.sim.map_info()
                info["cameras"] = bool(getattr(runner, "frames_dir", None))
                return self._json(info)
            if self.path == "/api/state":
                with runner.lock:
                    snap = runner.sim.snapshot()
                snap.update(paused=runner.paused, tps=runner.tps)
                return self._json(snap)
            return super().do_GET()

        def do_POST(self):
            data = self._body()
            if self.path == "/api/command":
                text = (data.get("text") or "").strip()
                if not text:
                    return self._json({"ok": False, "error": "empty"}, 400)
                with runner.lock:
                    rid, heard = runner.sim.hear(text, data.get("robot") or "S1")
                return self._json({"ok": True, "robot": rid, "text": heard})
            if self.path == "/api/spawn":
                with runner.lock:
                    ok = runner.sim.spawn(data.get("kind"), data.get("type"), (int(data["x"]), int(data["y"])))
                return self._json({"ok": ok})
            if self.path == "/api/control":
                action = data.get("action")
                if action == "pause":
                    runner.paused = True
                elif action == "resume":
                    runner.paused = False
                elif action == "speed":
                    runner.tps = min(40.0, max(0.5, float(data.get("value", 6))))
                elif action == "step":
                    with runner.lock:
                        runner.sim.step()
                elif action == "reset":
                    runner.reset()
                return self._json({"ok": True, "paused": runner.paused, "tps": runner.tps})
            return self._json({"ok": False, "error": "not found"}, 404)

    return Handler


def serve(cfg, host="127.0.0.1", port=8780):
    runner = Runner(cfg)
    httpd = ThreadingHTTPServer((host, port), make_handler(runner))
    print(f"清潔機器人調度模擬器：http://{host}:{port}")
    httpd.serve_forever()
