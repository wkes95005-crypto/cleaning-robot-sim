"""Train the dirt detector on the dataset made by the dataset_maker controller.

    .venv-yolo/bin/python webots/train_yolo.py [--epochs 40]

Starts from the COCO-pretrained YOLOv8n weights (downloaded once by ultralytics) and writes the best
model to weights/dirt_yolov8n.pt, the path config.toml's [detector] section points to.
"""
import argparse
import os
import shutil

import torch
from ultralytics import YOLO

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "webots", "dataset", "data.yaml")
OUT = os.path.join(ROOT, "weights", "dirt_yolov8n.pt")

ap = argparse.ArgumentParser()
ap.add_argument("--epochs", type=int, default=40)
ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
args = ap.parse_args()

model = YOLO("yolov8n.pt")
model.train(data=DATA, imgsz=320, epochs=args.epochs, batch=32, device=args.device, workers=0,
            project=os.path.join(ROOT, "webots", "runs"), name="dirt", exist_ok=True, plots=False,
            fliplr=0.5, mosaic=1.0, verbose=False)
best = os.path.join(ROOT, "webots", "runs", "dirt", "weights", "best.pt")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
shutil.copy(best, OUT)
metrics = YOLO(OUT).val(data=DATA, imgsz=320, device=args.device, plots=False, verbose=False)
print(f"saved {OUT}")
print(f"mAP50 {metrics.box.map50:.3f}  mAP50-95 {metrics.box.map:.3f}")
for i, name in model.names.items():
    print(f"  {name:<6} AP50 {metrics.box.ap50[i]:.3f}")
