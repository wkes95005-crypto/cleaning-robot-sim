"""Per-robot camera controller for the console world: publishes this robot's camera to the dashboard.

The FLEET supervisor moves the robots, so this controller only handles the camera. To keep the cost
down it renders only while someone is watching: the dashboard server writes FRAMES/<id>.want with the
frame rate it wants every time it serves a frame, and the camera is switched off once that file has
not been touched for a few seconds. Frames are written as FRAMES/<id>.jpg (replaced atomically).

    controllerArgs [ "S1" ]   environment: CLEANSIM_FRAMES (default webots/.frames)
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from controller import Robot  # noqa: E402
from layout import FRAMES_DIR  # noqa: E402

IDLE_AFTER_S = 3.0  # wall-clock seconds without a viewer before the camera is switched off
CHECK_S = 0.5       # how often (simulated seconds) to look at the .want file

robot = Robot()
dt = int(robot.getBasicTimeStep())
rid = sys.argv[1] if len(sys.argv) > 1 else robot.getName()
frames = os.environ.get("CLEANSIM_FRAMES", FRAMES_DIR)
os.makedirs(frames, exist_ok=True)
camera = robot.getDevice("camera")
want_file = os.path.join(frames, f"{rid}.want")
out, tmp = os.path.join(frames, f"{rid}.jpg"), os.path.join(frames, f"{rid}.part.jpg")

period = 0          # current sampling period in ms; 0 = camera off
next_check = next_shot = 0.0


def wanted_period():
    """Sampling period the viewers ask for, or 0 when nobody has looked for a while."""
    try:
        if time.time() - os.path.getmtime(want_file) > IDLE_AFTER_S:
            return 0
        with open(want_file) as f:
            fps = max(0.5, min(15.0, float(f.read().strip() or 2)))
    except (OSError, ValueError):
        return 0
    return max(dt, int(round(1000 / fps / dt)) * dt)  # a multiple of the basic time step


while robot.step(dt) != -1:
    t = robot.getTime()
    if t >= next_check:
        next_check = t + CHECK_S
        p = wanted_period()
        if p != period:
            camera.disable() if p == 0 else camera.enable(p)
            period, next_shot = p, t + p / 1000  # first image is ready one period after enabling
    if period and t >= next_shot:
        next_shot = t + period / 1000
        if camera.saveImage(tmp, 75) == 0:
            os.replace(tmp, out)
