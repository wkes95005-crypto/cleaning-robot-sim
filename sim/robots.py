"""Robot agents. Each robot carries its own microphone + ASR and camera + dirt detector, plans its
own path with A* on the console's shared map, and talks to the console only through the bus."""
from __future__ import annotations

from .pathfinding import astar, chebyshev, neighbors4, neighbors8

SENSOR_RADIUS = 3
ROLE_LABEL = {"sweeper": "掃地機", "mopper": "拖地機", "arm": "障礙排除機"}
DRAIN = {"move": 0.12, "work": 0.2, "idle": 0.01}  # % per tick: a full charge lasts about 7 minutes of driving
HANDLE_TICKS = 4  # time for the arm to grasp or set down an obstacle


class Robot:
    def __init__(self, sim, rid, role, cell, battery, asr, detector, patience):
        self.sim, self.id, self.role = sim, rid, role
        self.pos = self.dock = cell
        self.battery = float(battery)
        self.water = 100.0 if role == "mopper" else None
        self.asr, self.detector = asr, detector
        self.patience = patience  # ticks to wait for another robot before routing around it
        self.state = "idle"
        self.mode = "park"        # None | "park" (go home and wait) | "charge" (go home, unavailable until full)
        self.task = None
        self.path = []
        self.local_blocked = set()
        self.reported_dirt = {}   # cell -> label last sent to the console
        self.reported_obs = set()  # (obstacle id, cell)
        self.action, self.busy = None, 0
        self.carrying = None
        self.wait = 0
        self.odometer = 0

    # ------------------------------------------------------------------ comms
    def send(self, topic, **payload):
        self.sim.bus.send(self.id, "console", topic, payload)

    def hear(self, audio):
        """Microphone -> on-board ASR -> text to the console, tagged with where the robot is."""
        text = self.asr.transcribe(audio)
        self.send("voice_text", text=text, cell=self.pos, zone=self.sim.world.zone_of(self.pos))
        return text

    def status(self):
        return {"pos": self.pos, "battery": round(self.battery, 1),
                "water": None if self.water is None else round(self.water, 1),
                "state": self.state, "mode": self.mode, "task": self.task["id"] if self.task else None,
                "carrying": self.carrying.id if self.carrying else None}

    def _on_message(self, m):
        if m.topic == "assign":
            self.task = dict(m.payload["task"])
            self.task["waypoints"] = [tuple(w) for w in self.task.get("waypoints", [])]
            self.task.setdefault("phase", "approach")
            self.mode, self.path, self.wait = None, [], 0
        elif m.topic == "cancel":
            if self.task and self.task["id"] == m.payload["task"] and not self.carrying:
                self.task, self.path = None, []
        elif m.topic == "dock":
            if not self.carrying:
                self.task, self.path, self.mode = None, [], "charge"
        elif m.topic == "hold":  # stop where you are (an arm still sets down what it is carrying)
            if not self.carrying:
                self.task, self.path, self.mode = None, [], None
        elif m.topic == "park":
            if not self.task and self.mode != "charge":
                self.mode = "park"

    # ------------------------------------------------------------------ main loop
    def step(self):
        for m in self.sim.bus.receive(self.id):
            self._on_message(m)
        self._perceive()
        if self.busy:
            self._work_tick()
        elif self.mode and self.pos == self.dock:
            self._at_dock()
        else:
            goal = self._goal()
            if goal is None:
                self.state = "idle"
                self._drain("idle")
            elif self.pos in goal:
                self._arrive()
            else:
                self._move_toward(goal)
        self.send("status", **self.status())

    def _drain(self, kind):
        self.battery = max(0.0, self.battery - DRAIN[kind])

    # ------------------------------------------------------------------ perception
    def _perceive(self):
        w = self.sim.world
        x0, y0 = self.pos
        in_view = []
        for dx in range(-SENSOR_RADIUS, SENSOR_RADIUS + 1):
            for dy in range(-SENSOR_RADIUS, SENSOR_RADIUS + 1):
                c = (x0 + dx, y0 + dy)
                d = w.dirt.get(c)
                if d:
                    in_view.append(d)
                else:
                    self.reported_dirt.pop(c, None)
                oid = w.obstacle_cells.get(c)
                if oid:
                    self.local_blocked.add(c)
                    if (oid, c) not in self.reported_obs:
                        self.reported_obs.add((oid, c))
                        o = w.obstacles[oid]
                        self.send("obstacle_detected", oid=oid, cell=c, kind=o.kind,
                                  movable=o.movable, stored=o.stored)
                else:
                    self.local_blocked.discard(c)
        for det in self.detector.detect(self.pos, in_view):
            if self.reported_dirt.get(det.cell) != det.label:
                self.reported_dirt[det.cell] = det.label
                self.send("dirt_detected", cell=det.cell, label=det.label,
                          conf=det.confidence, near=det.near)

    # ------------------------------------------------------------------ goals and motion
    def _goal(self):
        if self.mode:
            return {self.dock}
        t = self.task
        if not t:
            return None
        if t["kind"] == "clear":
            if t["phase"] == "approach":
                o = self.sim.world.obstacles.get(t["obstacle_id"])
                if not o or o.cell is None or o.cell != tuple(t["target"]):
                    self._finish(note="障礙物已不在原位")
                    return None
                goals = {n for n in neighbors4(o.cell) if self.sim.world.is_floor(n) and not self._known_blocked(n)}
                return goals or None
            return {tuple(t["drop"])}
        while t["waypoints"] and t["kind"] != "patrol" and self._covered(t["waypoints"][0]):
            t["waypoints"].pop(0)  # spot already cleaned while passing
        if not t["waypoints"]:
            self._finish()
            return None
        return {t["waypoints"][0]}

    def _covered(self, cell):
        # only spot tasks: their waypoints are the dirt cells themselves; zone waypoints are sweep centres
        return (chebyshev(cell, self.pos) <= 1 and self.task["kind"] in ("sweep", "mop")
                and not self.task.get("zone") and not self._dirty_for_me(cell))

    def _dirty_for_me(self, cell):
        d = self.sim.world.dirt.get(cell)
        if not d:
            return False
        return d.kind in ("dry", "mixed") if self.role == "sweeper" else d.kind == "wet"

    def _known_blocked(self, c):
        return c in self.local_blocked or c in self.sim.console.no_go_core

    def _passable(self, c):
        return self.sim.world.is_floor(c) and not self._known_blocked(c)

    def _hazard_cost(self, c):
        return 8 if self.role != "arm" and c in self.sim.console.no_go_buffer else 0

    def _replan(self, goal, avoid_robots=False):
        others = {r.pos for r in self.sim.robots if r is not self} if avoid_robots else set()
        return astar(self.pos, goal, lambda c: self._passable(c) and c not in others, self._hazard_cost)

    def _move_toward(self, goal):
        if not self.path or self.path[-1] not in goal or not self._passable(self.path[0]):
            self.path = self._replan(goal)
            if self.path is None:
                self.path = []
                self._unreachable()
                return
        nxt = self.path[0]
        if self.sim.try_move(self, nxt):
            self.path.pop(0)
            self.wait = 0
            self.odometer += 1
            self.state = "moving"
            self._drain("move")
            return
        self.wait += 1
        self.state = "waiting"
        self._drain("idle")
        if self.wait == self.patience:
            blocker = next((r.id for r in self.sim.robots if r.pos == nxt), None)
            detour = self._replan(goal, avoid_robots=True)
            if detour:
                self.path = detour
                self.send("yield", to=blocker, cell=nxt)
            self.wait = 0

    def _unreachable(self):
        t = self.task
        if self.mode or not t:
            return
        if t["kind"] == "clear":
            self.send("skipped", task=t["id"], cell=t["target"])
            self.task = None
        elif t["waypoints"]:
            cell = t["waypoints"].pop(0)
            self.send("skipped", task=t["id"], cell=cell)
            self.send("progress", task=t["id"], waypoints=t["waypoints"])

    # ------------------------------------------------------------------ work
    def _arrive(self):
        t = self.task
        if t["kind"] == "clear":
            self.action = "grasp" if t["phase"] == "approach" else "place"
            self.busy = HANDLE_TICKS
            self.state = "grasping" if self.action == "grasp" else "placing"
            return
        t["waypoints"].pop(0)
        cleaned = self._clean_here() if t["kind"] in ("sweep", "mop", "patrol") else []
        self.send("progress", task=t["id"], waypoints=t["waypoints"])
        if cleaned:
            self.busy, self.action, self.state = len(cleaned), "clean", "cleaning"

    def _clean_here(self):
        w = self.sim.world
        done, residue = [], []
        if self.role == "arm":
            return done
        for c in [self.pos] + neighbors8(self.pos):
            d = w.dirt.get(c)
            if not d:
                continue
            if self.role == "sweeper" and d.kind == "dry":
                del w.dirt[c]
                done.append(c)
            elif self.role == "sweeper" and d.kind == "mixed":
                d.kind = "wet"  # solids swept up, liquid left behind for the mopper
                residue.append(c)
            elif self.role == "mopper" and d.kind == "wet" and self.water >= 2:
                del w.dirt[c]
                self.water -= 2
                done.append(c)
        if done:
            self.send("dirt_cleaned", cells=done)
        if residue:
            self.send("residue", cells=residue)
        return done + residue

    def _work_tick(self):
        self.busy -= 1
        self._drain("work")
        if self.busy:
            return
        t, w = self.task, self.sim.world
        if t is None:  # task was withdrawn (sent home) mid-grasp
            self.action, self.state = None, "moving"
            return
        if self.action == "grasp":
            o = w.obstacles.get(t["obstacle_id"])
            if o and o.cell and chebyshev(o.cell, self.pos) <= 1:
                self.carrying = w.pick(o.id)
                self.local_blocked.discard(tuple(t["target"]))
                self.send("picked", task=t["id"], oid=o.id)
                t["phase"] = "carry"
                self.path = []
            else:
                self._finish(note="障礙物已不在原位")
        elif self.action == "place":
            drop = tuple(t["drop"])
            if w.blocked(drop):  # someone put something there meanwhile: use the nearest free drop cell
                free = [c for c in w.drop_cells if not w.blocked(c)]
                if free:
                    t["drop"] = min(free, key=lambda c: chebyshev(c, self.pos))
                self.action, self.path, self.state = None, [], "moving"
                return
            w.place(self.carrying.id, drop)
            self.send("placed", task=t["id"], oid=self.carrying.id, cell=drop)
            self.carrying = None
            self._finish()
        self.action = None
        if self.state in ("cleaning", "grasping", "placing"):
            self.state = "moving"

    def _finish(self, note=""):
        if self.task:
            self.send("done", task=self.task["id"], note=note)
        self.task, self.path = None, []

    def _at_dock(self):
        self.battery = min(100.0, self.battery + 2.0)
        if self.water is not None:
            self.water = min(100.0, self.water + 5.0)
        full = self.battery >= 95 and (self.water is None or self.water >= 95)
        if self.mode == "charge" and full:
            self.mode = "park"
            self.send("charged")
        self.state = "charging" if self.mode == "charge" or not full else "idle"
