"""Phase-1 Webots controller: each robot cleans the dirt it is built for, on its own.

Sweepers (S*) take dry and mixed dirt; sweeping a mixed spill leaves a wet residue for the mopper.
The mopper (M*) takes wet dirt. The obstacle robot (A*) waits at its dock for now.
Paths come from the same A* and grid map as the 2D simulator; the lidar stops the robot if something
is right in front of it. Dirt lives in the scene as DEF DIRT_x_y_kind nodes, which stand in for the
console's shared map until the console is connected.

Environment variables (optional, for unattended runs):
    CLEANSIM_SNAP_DIR   save the 3D view as PNG every 20 s of simulated time (robot S1 only)
    CLEANSIM_QUIT_AT    quit the simulation after this many simulated seconds
"""
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from controller import Supervisor  # noqa: E402
from layout import cell_to_xy, dirt_def, parse_dirt_def, xy_to_cell  # noqa: E402
from sim.pathfinding import astar, chebyshev, neighbors8  # noqa: E402
from sim.world import World, load_default_scenario  # noqa: E402

WHEEL_R, AXLE = 0.035, 0.26
MAX_V, MAX_W = 0.45, 2.0          # m/s, rad/s
CLEAN_S = 1.5                     # seconds spent cleaning at each spot
WANT = {"sweeper": ("dry", "mixed"), "mopper": ("wet",), "arm": ()}

robot = Supervisor()
dt = int(robot.getBasicTimeStep())
rid = sys.argv[1] if len(sys.argv) > 1 else robot.getName()
role = {"S": "sweeper", "M": "mopper", "A": "arm"}[rid[0]]
me = robot.getSelf()
root = robot.getRoot().getField("children")

left, right = robot.getDevice("left wheel motor"), robot.getDevice("right wheel motor")
for m in (left, right):
    m.setPosition(float("inf"))
    m.setVelocity(0)
lidar = robot.getDevice("lidar")
lidar.enable(dt)
robot.getDevice("camera").enable(dt * 4)  # for the dirt detector, next phase

world = World()
load_default_scenario(world)
blocked = set(world.walls) | set(world.obstacle_cells)
hazard = {n for o in world.obstacles.values() if o.kind == "cable" for n in neighbors8(o.cell)} - blocked
home = cell_to_xy(world.docks[["A1", "M1", "S1", "S2"].index(rid)]) if rid in ("A1", "M1", "S1", "S2") else None

snap_dir = os.environ.get("CLEANSIM_SNAP_DIR") if rid == "S1" else None
quit_at = float(os.environ.get("CLEANSIM_QUIT_AT", "0"))


def pose():
    p, r = me.getPosition(), me.getOrientation()
    return p[0], p[1], math.atan2(r[3], r[0])


def dirt_in_scene():
    out = {}
    for i in range(root.getCount()):
        name = root.getMFNode(i).getDef()
        if name.startswith("DIRT_"):
            cell, kind = parse_dirt_def(name)
            out[cell] = (kind, root.getMFNode(i))
    return out


def plan(start):
    targets = {c for c, (k, _) in dirt_in_scene().items() if k in WANT[role]}
    if not targets:
        return None, None
    goal = min(targets, key=lambda c: abs(c[0] - start[0]) + abs(c[1] - start[1]))
    path = astar(start, {goal}, lambda c: c not in blocked, lambda c: 8 if c in hazard else 0)
    return goal, path


def clean_around(cell):
    """Remove (or, for a sweeper on a mixed spill, turn into wet residue) dirt in the 3x3 footprint."""
    n = 0
    for c, (kind, node) in dirt_in_scene().items():
        if chebyshev(c, cell) > 1 or kind not in WANT[role]:
            continue
        node.remove()
        n += 1
        if kind == "mixed":
            x, y = cell_to_xy(c)
            root.importMFNodeFromString(-1, (
                f"DEF {dirt_def(c, 'wet')} Pose {{ translation {x:.3f} {y:.3f} 0.004 children [ Shape {{ "
                "appearance PBRAppearance { baseColor 0.25 0.6 1.0 roughness 0.3 metalness 0 } "
                "geometry Cylinder { height 0.003 radius 0.24 } } ] }"))
    if n:
        print(f"[{rid}] cleaned {n} spot(s) at {cell}")
    return n


def drive(v, w):
    left.setVelocity(max(-20, min(20, (v - w * AXLE / 2) / WHEEL_R)))
    right.setVelocity(max(-20, min(20, (v + w * AXLE / 2) / WHEEL_R)))


def front_clear():
    ranges = lidar.getRangeImage()
    n = len(ranges)
    front = ranges[n // 2 - 10:n // 2 + 10]  # +-20 degrees around the heading
    return min(front) > 0.25


goal, path, pause, next_snap = None, None, 0.0, 2.0
while robot.step(dt) != -1:
    t = robot.getTime()
    if snap_dir and t >= next_snap:
        robot.exportImage(os.path.join(snap_dir, f"webots_{int(t):03d}s.png"), 90)
        next_snap += 20
    if quit_at and t >= quit_at:
        robot.simulationQuit(0)
    if pause > 0:
        pause -= dt / 1000
        drive(0, 0)
        continue

    x, y, th = pose()
    here = xy_to_cell(x, y)
    if path is None:
        goal, path = plan(here)
        if path is None:
            drive(0, 0)  # nothing to do (or the arm, which waits in this phase)
            continue
        print(f"[{rid}] heading to {goal}, {len(path)} cells")
    if not path:
        if clean_around(goal):
            pause = CLEAN_S
        path = None
        continue

    tx, ty = cell_to_xy(path[0])
    dist = math.hypot(tx - x, ty - y)
    if dist < 0.12:
        path.pop(0)
        continue
    err = math.atan2(ty - y, tx - x) - th
    err = math.atan2(math.sin(err), math.cos(err))
    w = max(-MAX_W, min(MAX_W, 3.0 * err))
    v = 0.0 if abs(err) > 0.6 else MAX_V * min(1.0, dist / 0.3) * math.cos(err)
    if v > 0 and not front_clear():
        v = 0.0  # something (another robot) is right in front: hold until it moves
    drive(v, w)
