"""Console in Webots: the FLEET supervisor runs the same console, planner and robot agents as the 2D
simulator, and mirrors them into the 3D scene.

- Every tick of the console (default 0.25 s of Webots time, adjustable from the dashboard) the robots
  get their next grid cell; in between, the 3D robots glide smoothly from cell to cell.
- Dirt and obstacles in the scene follow the console's world: cleaned dirt disappears, a swept mixed
  spill turns into wet residue, and boxes ride on top of A1 while it carries them to storage.
- The web dashboard (voice commands, task board, log) is served at http://127.0.0.1:8781, so you talk
  to the robots there and watch them in Webots.

Environment variables (optional): CLEANSIM_PORT, CLEANSIM_SNAP_DIR (save the 3D view every 20 s),
CLEANSIM_QUIT_AT (quit after N simulated seconds).
"""
import atexit
import json
import math
import os
import subprocess
import sys
import threading
import time
from http.server import ThreadingHTTPServer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from controller import Supervisor  # noqa: E402
from layout import FRAMES_DIR, ROOT, OBSTACLE_LOOK, cell_to_xy, dirt_node, obstacle_node, parse_dirt_def, xy_to_cell  # noqa: E402
from run import load_config  # noqa: E402
from sim.asr import exit_on_sigterm, start_service  # noqa: E402
from sim.engine import Simulation  # noqa: E402
from sim.mapping import scan_from_ranges  # noqa: E402
from sim.perception import yolo_detections  # noqa: E402
from sim.server import make_handler  # noqa: E402

PORT = int(os.environ.get("CLEANSIM_PORT", "8781"))
SNAP_DIR = os.environ.get("CLEANSIM_SNAP_DIR")
QUIT_AT = float(os.environ.get("CLEANSIM_QUIT_AT", "0"))


class Bridge:
    """Same interface the dashboard server expects (lock, sim, paused, tps, reset)."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.lock = threading.Lock()
        self.paused = False
        self.tps = 4.0  # console ticks per second of Webots time
        self.sim = Simulation(cfg)
        self.frames_dir = os.environ.get("CLEANSIM_FRAMES", FRAMES_DIR)  # robots' camera_feed writes here
        os.makedirs(self.frames_dir, exist_ok=True)

    def reset(self, remap=False):
        with self.lock:
            self.sim = Simulation(self.cfg, remap=remap)


cfg = load_config(os.path.join(ROOT, "config.toml"))
start_service(cfg.get("asr", {}))  # local Whisper for the dashboard microphone, if configured
bridge = Bridge(cfg)
server = ThreadingHTTPServer(("127.0.0.1", PORT), make_handler(bridge))
threading.Thread(target=server.serve_forever, daemon=True).start()
print(f"[fleet] console running, dashboard at http://127.0.0.1:{PORT}")

# YOLO runs in its own Python environment (torch is too big for the Webots controller); start it if set up
DETECT_FPS = 2  # camera frames per second per robot for the dirt detector
yolo_py = os.path.join(ROOT, ".venv-yolo", "bin", "python")
yolo = None
if os.path.exists(yolo_py) and os.path.exists(os.path.join(ROOT, "weights", "dirt_yolov8n.pt")):
    yolo = subprocess.Popen([yolo_py, os.path.join(ROOT, "webots", "yolo_worker.py")],
                            env={**os.environ, "CLEANSIM_FRAMES": bridge.frames_dir})
    atexit.register(yolo.terminate)
    # YOLO now drives dispatch: keep every robot's camera on at DETECT_FPS even when nobody watches
    detect_flag = os.path.join(bridge.frames_dir, "_detect_fps")
    with open(detect_flag, "w") as f:
        f.write(str(DETECT_FPS))
    atexit.register(lambda: os.path.exists(detect_flag) and os.remove(detect_flag))
    exit_on_sigterm()
    print("[fleet] YOLO dirt detector started")
else:
    print("[fleet] no YOLO weights yet: run dataset_maker and webots/train_yolo.py to enable detection boxes")

sup = Supervisor()
dt = int(sup.getBasicTimeStep())
root = sup.getRoot().getField("children")

# Lidar sweeps radioed by the robots (camera_feed): the map is built from the real 3D lidar here
receiver = sup.getDevice("lidar receiver")
receiver.enable(dt)
latest_scan = {}  # robot id -> newest sweep not yet used
pose_log = {}     # robot id -> [(time, x, y, heading)] as this supervisor placed it, last ~2 s
LIDAR_DEBUG = os.environ.get("CLEANSIM_LIDAR_LOG")  # write raw sweeps + true poses here for analysis
debug_out = open(LIDAR_DEBUG, "w") if LIDAR_DEBUG else None


def receive_scans():
    while receiver.getQueueLength() > 0:
        msg = json.loads(receiver.getString())
        receiver.nextPacket()
        latest_scan[msg["id"]] = msg
        if debug_out:
            debug_out.write(json.dumps({"msg": msg, "poses": {k: v[-40:] for k, v in pose_log.items()}}) + "\n")


used_frame = {}  # robot id -> camera frame whose detections were already reported


def detect_source(robot):
    """YOLO boxes on this robot's newest camera frame, put on the map with the pose the frame was taken
    from. None when there is no new frame since last tick."""
    path = os.path.join(bridge.frames_dir, f"{robot.id}.det.json")
    try:
        with open(path) as f:
            det = json.load(f)
    except (OSError, ValueError):
        return None
    if det.get("frame") == used_frame.get(robot.id):
        return None
    used_frame[robot.id] = det.get("frame")
    return yolo_detections(det, xy_to_cell)


def scan_source(robot):
    """The robot's own sweep and GPS/compass pose -> map cells. Returns that land on another robot are
    not walls: the console knows where every robot is and filters them out."""
    m = latest_scan.pop(robot.id, None)
    if m is None:
        return None
    others = [log[-1] for rid, log in pose_log.items() if rid != robot.id and log]

    def on_robot(px, py):
        return any((px - o[1]) ** 2 + (py - o[2]) ** 2 < 0.3 ** 2 for o in others)

    return scan_from_ranges(m["x"], m["y"], m["yaw"], m["ranges"], m["fov"], m["max"], dynamic=on_robot)


def scene_dirt():
    out = {}
    for i in range(root.getCount()):
        node = root.getMFNode(i)
        name = node.getDef()
        if name.startswith("DIRT_"):
            cell, kind = parse_dirt_def(name)
            out[cell] = (kind, node)
    return out


def sync_scene(sim):
    """Make the 3D dirt and obstacles match the console's world."""
    want = {c: d.kind for c, d in sim.world.dirt.items()}
    for cell, (kind, node) in scene_dirt().items():
        if want.get(cell) != kind:
            node.remove()
    have = scene_dirt()
    for cell, kind in want.items():
        if cell not in have:
            root.importMFNodeFromString(-1, dirt_node(cell, kind))
    for o in sim.world.obstacles.values():
        if sup.getFromDef(f"OBS_{o.id}") is None:
            root.importMFNodeFromString(-1, obstacle_node(o.id, o.kind, o.cell or (0, 0)))
        if o.cell:
            x, y = cell_to_xy(o.cell)
            sup.getFromDef(f"OBS_{o.id}").getField("translation").setSFVec3f([x, y, OBSTACLE_LOOK[o.kind][2] / 2])
    for i in range(root.getCount() - 1, -1, -1):  # obstacles that no longer exist (after a reset)
        name = root.getMFNode(i).getDef()
        if name.startswith("OBS_") and int(name[4:]) not in sim.world.obstacles:
            root.getMFNode(i).remove()


# The dashboard's 3D view: export what the Webots window shows, but only while someone is watching
# (the dashboard touches _scene.want on every fetch). exportImage writes during the next step, so the
# picture goes to a temporary file first and replaces _scene.jpg one loop later (no half-written JPEG).
SCENE_PERIOD = 0.5  # seconds of wall time between pictures
scene_want = os.path.join(bridge.frames_dir, "_scene.want")
scene_tmp = os.path.join(bridge.frames_dir, "_scene.tmp.jpg")
scene_jpg = os.path.join(bridge.frames_dir, "_scene.jpg")
scene_pending, scene_last = False, 0.0


def export_scene():
    global scene_pending, scene_last
    if scene_pending and os.path.exists(scene_tmp):
        os.replace(scene_tmp, scene_jpg)
        scene_pending = False
    now = time.time()
    if scene_pending or now - scene_last < SCENE_PERIOD:
        return
    try:
        if now - os.path.getmtime(scene_want) > 3:
            return  # nobody has asked for a while
    except OSError:
        return
    sup.exportImage(scene_tmp, 80)
    scene_pending, scene_last = True, now


current, prev, heading, last_tick, last_sync = None, {}, {}, 0.0, 0.0
next_snap = 2.0
while sup.step(dt) != -1:
    t = sup.getTime()
    receive_scans()
    export_scene()
    with bridge.lock:
        sim = bridge.sim
        sim.scan_source = scan_source
        sim.detect_source = detect_source if yolo else None
        if sim is not current:  # first run or dashboard reset
            current, last_tick = sim, t
            prev = {r.id: r.pos for r in sim.robots}
            heading = {r.id: math.pi / 2 for r in sim.robots}
            sync_scene(sim)
        interval = 1.0 / bridge.tps
        if not bridge.paused and t - last_tick >= interval:
            prev = {r.id: r.pos for r in sim.robots}
            sim.step()
            last_tick = t
            sync_scene(sim)
        elif bridge.paused and t - last_sync >= 0.5:  # dirt placed or single steps taken while paused
            sync_scene(sim)
            last_sync = t
        alpha = min(1.0, (t - last_tick) / interval)
        for r in sim.robots:
            (ax, ay), (bx, by) = cell_to_xy(prev[r.id]), cell_to_xy(r.pos)
            if (ax, ay) != (bx, by):
                heading[r.id] = math.atan2(by - ay, bx - ax)
            x, y = ax + (bx - ax) * alpha, ay + (by - ay) * alpha
            node = sup.getFromDef(r.id)
            log = pose_log.setdefault(r.id, [])
            log.append((round(t, 3), round(x, 4), round(y, 4), round(heading[r.id], 4)))
            del log[:-64]
            node.getField("translation").setSFVec3f([x, y, 0.0])
            node.getField("rotation").setSFRotation([0, 0, 1, heading[r.id]])
            node.resetPhysics()
            if r.carrying:  # the box rides on top of the arm robot
                box = sup.getFromDef(f"OBS_{r.carrying.id}")
                if box:
                    box.getField("translation").setSFVec3f([x, y, 0.13 + OBSTACLE_LOOK[r.carrying.kind][2] / 2])
    if SNAP_DIR and t >= next_snap:
        sup.exportImage(os.path.join(SNAP_DIR, f"console_{int(t):03d}s.png"), 90)
        next_snap += 20
    if QUIT_AT and t >= QUIT_AT:
        server.shutdown()
        if yolo:
            yolo.terminate()
        sup.simulationQuit(0)
if yolo:  # Webots closed or the world was reloaded
    yolo.terminate()
