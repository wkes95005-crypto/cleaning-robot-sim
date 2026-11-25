"""Dirt detectors carried on each robot's camera.

The 2D sim has no rendered camera images, so `SimDetector` stands in for a trained model: it reads
the dirt really in view and mislabels some of it, more often from far away, the way a small YOLO
model does. Swap in `YoloDetector` once there are real (or Gazebo / Webots) frames to run it on.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

from .pathfinding import chebyshev

LABELS = ("dry", "wet", "mixed")


@dataclass
class Detection:
    cell: tuple
    label: str
    confidence: float
    near: bool  # seen from at most one cell away, i.e. a close, reliable look


class SimDetector:
    name = "sim"

    def __init__(self, seed=0, near_accuracy=0.98, far_accuracy=0.85):
        self.seed, self.near_accuracy, self.far_accuracy = seed, near_accuracy, far_accuracy

    def detect(self, robot_cell, dirt_in_view):
        out = []
        for d in dirt_in_view:
            near = chebyshev(robot_cell, d.cell) <= 1
            # same cell, kind and distance band always give the same answer, so reports don't flicker
            rng = random.Random(hash((self.seed, d.cell, LABELS.index(d.kind), near)))
            correct = rng.random() < (self.near_accuracy if near else self.far_accuracy)
            if correct:
                out.append(Detection(d.cell, d.kind, round(rng.uniform(0.82, 0.99), 2), near))
            else:
                wrong = rng.choice([k for k in LABELS if k != d.kind])
                out.append(Detection(d.cell, wrong, round(rng.uniform(0.45, 0.7), 2), near))
        return out


# Front camera of the Webots robot (CleaningRobot.proto): where it sits on the robot and how it looks.
CAMERA = {"x": 0.165, "z": 0.12, "pitch": 0.22, "fov": 1.2}


def pixel_to_floor(u, v, width, height, pose, cam=CAMERA):
    """Where the camera ray through pixel (u, v) meets the floor, in world metres, or None (at/above
    the horizon). `pose` is the robot's (x, y, yaw) when the frame was taken; the camera looks along
    the robot's x axis, tilted down by `pitch`, with Webots' convention (x forward, y left, z up)."""
    f = (width / 2) / math.tan(cam["fov"] / 2)
    cx, cy, cz = 1.0, -(u - width / 2) / f, -(v - height / 2) / f
    c, s = math.cos(cam["pitch"]), math.sin(cam["pitch"])
    rx, ry, rz = c * cx + s * cz, cy, -s * cx + c * cz
    if rz > -1e-3:
        return None
    t = -cam["z"] / rz
    px, py = cam["x"] + t * rx, t * ry  # on the floor, in the robot's frame
    x, y, yaw = pose
    return x + px * math.cos(yaw) - py * math.sin(yaw), y + px * math.sin(yaw) + py * math.cos(yaw), math.hypot(px, py)


def yolo_detections(det, to_cell, max_dist=4.0, near_dist=1.0, min_conf=0.5):
    """YOLO output for one frame (boxes + the pose it was taken from) -> Detections on map cells.

    Dirt lies flat on the floor, so the centre of its box is a point on the floor: project it. Several
    boxes on one cell keep the most confident; boxes further than `max_dist` are too rough to place, and
    ones below `min_conf` are left on screen only (in tests almost all false alarms scored 0.35-0.45)."""
    pose = det.get("pose")
    if not pose:
        return []
    w, h = det["size"]
    best = {}
    for b in det["boxes"]:
        x1, y1, x2, y2 = b["xyxy"]
        hit = pixel_to_floor((x1 + x2) / 2, (y1 + y2) / 2, w, h, (pose["x"], pose["y"], pose["yaw"]))
        if b["conf"] < min_conf or not hit or hit[2] > max_dist or b["cls"] not in LABELS:
            continue
        cell = to_cell(hit[0], hit[1])
        if cell not in best or b["conf"] > best[cell].confidence:
            best[cell] = Detection(cell, b["cls"], b["conf"], hit[2] <= near_dist)
    return list(best.values())


class YoloDetector:
    """Ultralytics YOLO weights trained on dry / wet / mixed dirt. Needs camera frames, not grid cells."""

    def __init__(self, weights):
        from ultralytics import YOLO  # optional dependency
        self.model = YOLO(weights)
        self.name = f"yolo:{weights}"

    def detect(self, robot_cell, dirt_in_view):
        raise NotImplementedError("YOLO needs camera frames; connect it to a renderer that produces images")


def make_detector(cfg, seed=0):
    if cfg.get("backend") == "yolo":
        try:
            return YoloDetector(cfg["weights"])
        except Exception as e:
            print(f"[detector] YOLO unavailable ({e}); using SimDetector")
    return SimDetector(seed, cfg.get("near_accuracy", 0.98), cfg.get("far_accuracy", 0.85))
