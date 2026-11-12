"""Shared between the world generator and the Webots controllers: grid cell <-> Webots metres.

The 2D grid is 48 x 32 cells of 0.5 m. In Webots the floor is centred on the origin, x points
east and y points north (grid y grows southwards, so it is flipped).
"""
import math
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from sim.world import CELL_M, H, W  # noqa: E402

FRAMES_DIR = os.path.join(ROOT, "webots", ".frames")  # robot camera images for the dashboard


def cell_to_xy(c):
    x, y = c
    return ((x + 0.5) * CELL_M - W * CELL_M / 2, H * CELL_M / 2 - (y + 0.5) * CELL_M)


def xy_to_cell(px, py):
    return (int((px + W * CELL_M / 2) // CELL_M), int((H * CELL_M / 2 - py) // CELL_M))


def dirt_def(c, kind):
    """DEF name of a dirt patch; the kind is part of the name so controllers can read it from the scene."""
    return f"DIRT_{c[0]}_{c[1]}_{kind}"


def parse_dirt_def(name):
    _, x, y, kind = name.split("_")
    return (int(x), int(y)), kind


DIRT_CLASSES = ["dry", "wet", "mixed"]  # YOLO class ids 0, 1, 2
PAPER = ["0.95 0.95 0.92", "0.86 0.78 0.6", "0.62 0.45 0.25", "0.92 0.82 0.35", "0.75 0.8 0.85"]
WATER = ["0.62 0.74 0.86", "0.58 0.66 0.74", "0.5 0.65 0.82"]
JUICE = ["0.85 0.55 0.15", "0.55 0.32 0.18", "0.72 0.22 0.25", "0.9 0.75 0.3"]
GLASS = ["0.7 0.9 0.75", "0.85 0.92 0.95", "0.45 0.6 0.35"]
OBSTACLE_LOOK = {"box": (0.4, 0.4, 0.4, "0.72 0.52 0.3"), "cart": (0.45, 0.32, 0.9, "0.6 0.62 0.66"),
                 "pallet": (0.45, 0.45, 0.14, "0.55 0.42 0.26"), "cable": (0.45, 0.05, 0.02, "0.1 0.1 0.1")}


def _scrap(dx, dy, sx, sy, yaw, color, h=0.004):
    return (f"Pose {{ translation {dx:.3f} {dy:.3f} {h / 2:.4f} rotation 0 0 1 {yaw:.2f} children [ Shape {{ "
            f"appearance PBRAppearance {{ baseColor {color} roughness 0.7 metalness 0 }} "
            f"geometry Box {{ size {sx:.3f} {sy:.3f} {h} }} }} ] }}")


def _puddle(dx, dy, r, color, transparency):
    return (f"Pose {{ translation {dx:.3f} {dy:.3f} 0.0015 children [ Shape {{ appearance PBRAppearance {{ "
            f"baseColor {color} transparency {transparency:.2f} roughness 0.05 metalness 0 }} "
            f"geometry Cylinder {{ height 0.003 radius {r:.3f} subdivision 20 }} }} ] }}")


def dirt_node(c, kind, seed=None, jitter=0.0, rec_color="1 0 0"):
    """Webots node text for a dirt patch. Visual only (no bounding object, robots drive over it), with a
    `model` and recognitionColors so the camera's Recognition can label it for YOLO training.

    The look is varied but repeatable: the same cell and kind always look the same unless `seed` is
    given (the dataset generator passes a random seed, some position jitter and a unique recognition
    colour per patch, so each patch can be found in the camera's segmentation image)."""
    rng = random.Random(seed if seed is not None else c[0] * 1000 + c[1] * 7 + DIRT_CLASSES.index(kind))
    x, y = cell_to_xy(c)
    x, y = x + rng.uniform(-jitter, jitter), y + rng.uniform(-jitter, jitter)
    parts = []
    if kind == "dry":  # paper scraps, crumbs, a receipt
        for _ in range(rng.randint(3, 7)):
            parts.append(_scrap(rng.uniform(-0.15, 0.15), rng.uniform(-0.15, 0.15), rng.uniform(0.03, 0.11),
                                rng.uniform(0.02, 0.08), rng.uniform(0, math.pi), rng.choice(PAPER)))
    elif kind == "wet":  # water / clear liquid: glossy, half transparent
        color = rng.choice(WATER)
        for _ in range(rng.randint(1, 3)):
            parts.append(_puddle(rng.uniform(-0.08, 0.08), rng.uniform(-0.08, 0.08), rng.uniform(0.08, 0.2),
                                 color, rng.uniform(0.25, 0.45)))
    else:  # mixed: a coloured drink with shards of the bottle in it
        color = rng.choice(JUICE)
        for _ in range(rng.randint(1, 2)):
            parts.append(_puddle(rng.uniform(-0.06, 0.06), rng.uniform(-0.06, 0.06), rng.uniform(0.1, 0.19),
                                 color, rng.uniform(0.05, 0.25)))
        for _ in range(rng.randint(2, 5)):
            parts.append(_scrap(rng.uniform(-0.13, 0.13), rng.uniform(-0.13, 0.13), rng.uniform(0.02, 0.06),
                                rng.uniform(0.015, 0.04), rng.uniform(0, math.pi), rng.choice(GLASS), h=0.008))
    return (f"DEF {dirt_def(c, kind)} Solid {{ translation {x:.3f} {y:.3f} 0.001 "
            f"rotation 0 0 1 {rng.uniform(0, 2 * math.pi):.2f} children [ {' '.join(parts)} ] "
            f"name \"dirt {c[0]} {c[1]}\" model \"{kind}\" recognitionColors [ {rec_color} ] }}")


def obstacle_node(oid, kind, c):
    x, y = cell_to_xy(c)
    sx, sy, sz, color = OBSTACLE_LOOK[kind]
    return (f"DEF OBS_{oid} Solid {{ translation {x:.3f} {y:.3f} {sz / 2:.3f} children [ Shape {{ "
            f"appearance PBRAppearance {{ baseColor {color} roughness 0.8 metalness 0 }} "
            f"geometry Box {{ size {sx} {sy} {sz} }} }} ] name \"{kind} {oid}\" "
            f"boundingObject Box {{ size {sx} {sy} {sz} }} }}")
