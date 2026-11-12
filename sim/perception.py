"""Dirt detectors carried on each robot's camera.

The 2D sim has no rendered camera images, so `SimDetector` stands in for a trained model: it reads
the dirt really in view and mislabels some of it, more often from far away, the way a small YOLO
model does. Swap in `YoloDetector` once there are real (or Gazebo / Webots) frames to run it on.
"""
from __future__ import annotations

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
