"""Generate a labelled YOLO dataset of floor dirt inside the Webots supermarket (world: dirt_dataset.wbt).

The labels come from the camera's Recognition segmentation image: every patch gets its own
recognition colour, and the box is the extent of that colour's pixels. That gives tight boxes for flat
patches (Recognition's own boxes come from a bounding sphere and are far too tall) and respects
occlusion by shelves and counters.

1. scatter 25-45 random dirt patches (random kind, look, position and rotation) over the floor,
2. put the camera rig at a random pose, mostly a short distance from some dirt and looking at it,
3. save the camera image and one YOLO label line per visible patch,
4. re-scatter every 60 poses, until N images are saved.

Output: webots/dataset/{images,labels}/{train,val}/ and webots/dataset/data.yaml.
Environment: CLEANSIM_DATASET_N (default 1800), CLEANSIM_DATASET_DIR, CLEANSIM_DATASET_SEED.
"""
import math
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from controller import Supervisor  # noqa: E402
from layout import DIRT_CLASSES, ROOT, cell_to_xy, dirt_node, xy_to_cell  # noqa: E402
from sim.world import H, W, World  # noqa: E402

N = int(os.environ.get("CLEANSIM_DATASET_N", "1800"))
OUT = os.environ.get("CLEANSIM_DATASET_DIR", os.path.join(ROOT, "webots", "dataset"))
VAL_EVERY = 6           # every 6th image goes to the validation set
NEGATIVE_KEEP = 0.25    # share of dirt-free views kept, so the model also learns "nothing here"
MIN_BOX_PX = (5, 3)     # skip slivers at the edge of the image or far away

sup = Supervisor()
dt = int(sup.getBasicTimeStep())
cam = sup.getDevice("camera")
cam.enable(dt)
cam.recognitionEnable(dt)
cam.enableRecognitionSegmentation()
iw, ih = cam.getWidth(), cam.getHeight()
SEG_G = 128  # every patch's colour is (k * 5, 128, 64) / 255, so the red byte identifies it
me = sup.getSelf()
pos_f, rot_f = me.getField("translation"), me.getField("rotation")
root = sup.getRoot().getField("children")
rng = random.Random(int(os.environ.get("CLEANSIM_DATASET_SEED", "1")))

world = World()
floor = [(x, y) for x in range(W) for y in range(H) if world.is_floor((x, y)) and (x, y) not in world.obstacle_cells]
for split in ("train", "val"):
    os.makedirs(os.path.join(OUT, "images", split), exist_ok=True)
    os.makedirs(os.path.join(OUT, "labels", split), exist_ok=True)


patches = {}  # red byte of the segmentation colour -> class name


def scatter():
    for i in range(root.getCount() - 1, -1, -1):
        if root.getMFNode(i).getDef().startswith("DIRT_"):
            root.getMFNode(i).remove()
    patches.clear()
    cells = rng.sample(floor, rng.randint(25, 45))
    for k, c in enumerate(cells, start=1):
        kind = rng.choice(DIRT_CLASSES)
        patches[k * 5] = kind
        color = f"{k * 5 / 255:.6f} {SEG_G / 255:.6f} {64 / 255:.6f}"
        root.importMFNodeFromString(-1, dirt_node(c, kind, seed=rng.random(), jitter=0.15, rec_color=color))
    return cells


def place_rig(targets):
    """Mostly 0.4-2.5 m from a dirt patch and roughly facing it; sometimes anywhere."""
    for _ in range(50):
        if rng.random() < 0.8:
            tx, ty = cell_to_xy(rng.choice(targets))
            d, a = rng.uniform(0.4, 2.2), rng.uniform(0, 2 * math.pi)
            px, py = tx + d * math.cos(a), ty + d * math.sin(a)
            heading = math.atan2(ty - py, tx - px) + rng.uniform(-0.45, 0.45)
        else:
            px, py = cell_to_xy(rng.choice(floor))
            heading = rng.uniform(0, 2 * math.pi)
        cell = xy_to_cell(px, py)
        if world.is_floor(cell) and cell not in world.obstacle_cells:
            pos_f.setSFVec3f([px, py, 0.0])
            rot_f.setSFRotation([0, 0, 1, heading])
            me.resetPhysics()
            return


def labels():
    seg = cam.getRecognitionSegmentationImage()  # BGRA bytes
    boxes = {}
    for i in range(0, len(seg), 4):
        if seg[i + 1] != SEG_G or seg[i + 2] not in patches:
            continue
        p = i // 4
        x, y = p % iw, p // iw
        b = boxes.setdefault(seg[i + 2], [x, y, x, y])
        b[0], b[1], b[2], b[3] = min(b[0], x), min(b[1], y), max(b[2], x), max(b[3], y)
    out = []
    for red, (x1, y1, x2, y2) in boxes.items():
        x2, y2 = x2 + 1, y2 + 1
        if x2 - x1 < MIN_BOX_PX[0] or y2 - y1 < MIN_BOX_PX[1]:
            continue
        out.append(f"{DIRT_CLASSES.index(patches[red])} {(x1 + x2) / 2 / iw:.6f} {(y1 + y2) / 2 / ih:.6f} "
                   f"{(x2 - x1) / iw:.6f} {(y2 - y1) / ih:.6f}")
    return out


saved, tries, targets = 0, 0, scatter()
while saved < N and sup.step(dt) != -1:
    tries += 1
    if tries % 60 == 0:
        targets = scatter()
    place_rig(targets)
    sup.step(dt)  # one more step so the image and recognition reflect the new pose
    lines = labels()
    if not lines and rng.random() > NEGATIVE_KEEP:
        continue
    split = "val" if saved % VAL_EVERY == 0 else "train"
    name = f"{saved:05d}"
    cam.saveImage(os.path.join(OUT, "images", split, name + ".jpg"), 92)
    with open(os.path.join(OUT, "labels", split, name + ".txt"), "w") as f:
        f.write("\n".join(lines))
    saved += 1
    if saved % 200 == 0:
        print(f"[dataset] {saved}/{N} images")

with open(os.path.join(OUT, "data.yaml"), "w") as f:
    f.write(f"path: {OUT}\ntrain: images/train\nval: images/val\nnames:\n"
            + "".join(f"  {i}: {n}\n" for i, n in enumerate(DIRT_CLASSES)))
print(f"[dataset] done: {saved} images in {OUT}")
sup.simulationQuit(0)
