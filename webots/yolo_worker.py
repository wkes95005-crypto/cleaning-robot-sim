"""Run the YOLO dirt detector on the robots' camera frames.

camera_feed writes webots/.frames/<id>.jpg while someone watches a robot's camera; this worker runs
YOLO on every new frame and writes <id>.det.json next to it, which the dashboard server attaches to
the image so the boxes are drawn on exactly the frame they were found in. fleet_bridge starts it
automatically when .venv-yolo and the weights exist; it can also be run by hand:

    .venv-yolo/bin/python webots/yolo_worker.py
"""
import json
import os
import sys
import time

import torch
from ultralytics import YOLO

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRAMES = os.environ.get("CLEANSIM_FRAMES", os.path.join(ROOT, "webots", ".frames"))
WEIGHTS = os.environ.get("CLEANSIM_WEIGHTS", os.path.join(ROOT, "weights", "dirt_yolov8n.pt"))
CONF = float(os.environ.get("CLEANSIM_YOLO_CONF", "0.35"))

model = YOLO(WEIGHTS)
device = "mps" if torch.backends.mps.is_available() else "cpu"
names = model.names
seen = {}  # robot id -> mtime of the last frame processed
print(f"[yolo] {WEIGHTS} on {device}, watching {FRAMES}", flush=True)

while True:
    busy = False
    for f in os.listdir(FRAMES) if os.path.isdir(FRAMES) else []:
        if not f.endswith(".jpg") or ".part" in f:
            continue
        rid, path = f[:-4], os.path.join(FRAMES, f)
        try:
            mtime = os.stat(path).st_mtime_ns
        except FileNotFoundError:
            continue
        if seen.get(rid) == mtime:
            continue
        seen[rid] = mtime
        busy = True
        t0 = time.perf_counter()
        try:
            r = model.predict(path, imgsz=320, conf=CONF, device=device, verbose=False)[0]
        except Exception as e:  # frame replaced mid-read
            print(f"[yolo] {rid}: {e}", file=sys.stderr)
            continue
        boxes = [{"cls": names[int(c)], "conf": round(float(p), 2), "xyxy": [round(float(v), 1) for v in b]}
                 for b, c, p in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist())]
        det = {"frame": mtime, "ms": round((time.perf_counter() - t0) * 1000, 1), "size": list(r.orig_shape[::-1]),
               "boxes": boxes}
        tmp = os.path.join(FRAMES, f"{rid}.det.part")
        with open(tmp, "w") as out:
            json.dump(det, out)
        os.replace(tmp, os.path.join(FRAMES, f"{rid}.det.json"))
    if not busy:
        time.sleep(0.03)
