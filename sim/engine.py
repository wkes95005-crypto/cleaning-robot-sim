"""Simulation clock: steps the console and every robot, and resolves who gets to move into which cell."""
from __future__ import annotations

import os

from .asr import make_asr
from .bus import Bus
from .console import Console
from .events import RandomEvents
from .perception import make_detector
from .planner import make_planner
from .robots import ROLE_LABEL
from .world import CELL_M, DIRT_KINDS, OBSTACLE_KINDS, W, H, ZONES, World, load_default_scenario

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TICK_S = 0.5  # simulated seconds per tick (robots move one 0.5 m cell per tick = 1 m/s)

DEFAULT_FLEET = [  # stepping order = right-of-way order
    {"id": "A1", "role": "arm", "battery": 90},
    {"id": "M1", "role": "mopper", "battery": 85},
    {"id": "S1", "role": "sweeper", "battery": 95},
    {"id": "S2", "role": "sweeper", "battery": 32},
]


class Simulation:
    def __init__(self, cfg=None, remap=False):
        from .robots import Robot

        self.cfg = cfg or {}
        simcfg = self.cfg.get("sim", {})
        self.seed = simcfg.get("seed", 7)
        self.world = World()
        if simcfg.get("scenario", "default") == "default":
            load_default_scenario(self.world)
        self.bus = Bus(simcfg.get("latency_ticks", 1))
        self.tick = 0
        self.console = Console(self, make_planner(self.cfg.get("planner", {})))
        asr = make_asr(self.cfg.get("asr", {}))
        detector = make_detector(self.cfg.get("detector", {}), self.seed)
        fleet = self.cfg.get("robots") or DEFAULT_FLEET
        self.robots = []
        for i, spec in enumerate(fleet):
            self.robots.append(Robot(self, spec["id"], spec["role"], self.world.docks[i], spec["battery"],
                                     asr, detector, patience=max(2, 2 * (len(fleet) - i))))
        self.events = RandomEvents(self, self.cfg.get("events", {}))  # litter and spills while open
        self.occupied = {r.pos for r in self.robots}
        self.reserved = set()
        self.collisions = 0
        # [mapping] mode: given = the map is known (no lidar); auto = load the saved map, or survey and
        # save one if there is none (or remap=True); explore = survey on every start
        mcfg = self.cfg.get("mapping", {})
        mode = mcfg.get("mode", "given")
        self.lidar = mode != "given"
        self.scan_source = None    # set by the Webots bridge: robot -> (free, hits, ends) from the 3D lidar
        self.detect_source = None  # set by the Webots bridge: robot -> YOLO detections on map cells, or None
        map_file = os.path.join(ROOT, mcfg.get("file", "maps/supermarket.json"))
        manual = mcfg.get("manual_zones", [])
        if mode == "auto" and os.path.exists(map_file) and not remap:
            self.console.load_map(map_file, manual)
        elif mode in ("auto", "explore"):
            self.console.start_mapping(map_file, manual)

    def robot(self, rid):
        return next((r for r in self.robots if r.id == rid), None)

    def step(self):
        self.tick += 1
        self.bus.tick = self.tick
        self.events.step()
        self.console.step()
        self.reserved = set()
        for r in self.robots:
            r.step()
        if len({r.pos for r in self.robots}) < len(self.robots):
            self.collisions += 1

    def try_move(self, robot, cell):
        """Cell reservation: one robot per cell, first come (= higher right of way) first served."""
        if cell in self.occupied or cell in self.reserved or self.world.blocked(cell):
            return False
        self.occupied.discard(robot.pos)
        robot.pos = cell
        self.occupied.add(cell)
        self.reserved.add(cell)
        return True

    # ------------------------------------------------------------------ operator actions
    def hear(self, text, rid="S1"):
        r = self.robot(rid) or self.robots[0]
        return r.id, r.hear(text)

    def spawn(self, kind, sub, cell):
        cell = tuple(cell)
        if cell in self.occupied:
            return False
        if kind == "dirt":
            return self.world.add_dirt(cell, sub)
        if kind == "obstacle":
            return self.world.add_obstacle(cell, sub) is not None
        return False

    # ------------------------------------------------------------------ views
    def map_info(self):
        w = self.world
        return {"w": W, "h": H, "cell_m": CELL_M, "walls": sorted(w.walls),
                "zones": [{"name": n, "rect": r} for n, (r, _) in ZONES.items()],
                "docks": w.docks, "drop_cells": w.drop_cells,
                "fleet": [{"id": r.id, "role": r.role, "label": ROLE_LABEL[r.role]} for r in self.robots],
                "dirt_kinds": DIRT_KINDS, "obstacle_kinds": OBSTACLE_KINDS,
                "floor_cells": sum(1 for x in range(W) for y in range(H) if w.is_floor((x, y)))}

    def snapshot(self):
        w, c = self.world, self.console
        open_tasks = sorted((t for t in c.tasks.values() if t.status in ("pending", "assigned")),
                            key=lambda t: (t.status != "assigned", -t.priority, t.created))
        recent = sorted((t for t in c.tasks.values() if t.status == "done" and t.kind != "patrol"),
                        key=lambda t: t.created)[-6:]
        return {
            "tick": self.tick,
            "time_s": round(self.tick * TICK_S, 1),
            "robots": [{
                "id": r.id, "role": r.role, "label": ROLE_LABEL[r.role], "x": r.pos[0], "y": r.pos[1],
                "battery": round(r.battery, 1), "water": None if r.water is None else round(r.water, 1),
                "state": r.state, "mode": r.mode, "task": r.task["id"] if r.task else None,
                "carrying": r.carrying.kind if r.carrying else None, "path": r.path[:40],
                "scan": [[round(x, 1), round(y, 1)] for x, y in r.scan_ends[::2]] if self.lidar else [],
                "dock": r.dock, "odometer_m": r.odometer * CELL_M, "held": r.id in c.held,
            } for r in self.robots],
            "dirt": [{"x": d.cell[0], "y": d.cell[1], "kind": d.kind,
                      "known": c.known_dirt.get(d.cell, {}).get("label")} for d in w.dirt.values()],
            "obstacles": [{"id": o.id, "x": o.cell[0], "y": o.cell[1], "kind": o.kind,
                           "known": o.id in c.known_obstacles, "stored": o.stored}
                          for o in w.obstacles.values() if o.cell],
            "no_go_core": sorted(c.no_go_core), "no_go_buffer": sorted(c.no_go_buffer),
            "tasks": [{"id": t.id, "kind": t.kind, "role": t.role, "status": t.status, "assignee": t.assignee,
                       "title": t.title, "priority": t.priority, "remaining": len(t.waypoints),
                       "depends_on": t.depends_on, "source": t.source, "pinned": t.pinned}
                      for t in open_tasks + recent],
            "log": list(c.log)[-120:],
            "chat": list(c.chat),
            "backends": {"planner": c.planner.name, "asr": self.robots[0].asr.name,
                         "detector": "yolo（Webots 鏡頭）" if self.detect_source else self.robots[0].detector.name},
            "held": sorted(c.held), "avoid": sorted(c.avoid_kinds), "thinking": len(c._plans),
            "map": c.map_view(),
            "stats": {**c.stats, "dirt_left": len(w.dirt), "dirt_known": len(c.known_dirt),
                      "collisions": self.collisions,
                      "events": {"on": self.events.enabled, "count": self.events.count}, "messages": self.bus.sent},
        }
