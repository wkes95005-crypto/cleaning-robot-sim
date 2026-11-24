"""Mapping: each robot's 2D lidar, the console's shared occupancy grid, and frontier detection.

How the map is built (the way a real fleet does it, minus localisation: robots know their pose):

1. every tick each robot sweeps its lidar (a ray every 3 degrees, 4 m range) and sends the cells the
   rays crossed (free) and the cells they hit (occupied) to the console;
2. the console fuses all robots' scans into one occupancy grid in log-odds, so a cell is only called
   a wall after several hits and becomes free again if later scans pass through it (a box that the
   arm has moved away disappears from the map);
3. a frontier is a known-free cell next to unknown space; the console sends each robot to a
   different frontier until none is left, then saves the map.

The lidar sits on top of the robot: it sees walls, shelves, boxes, carts and pallets, but not a cable
lying on the floor (only the camera finds those). Other robots are filtered out of the scans.
"""
from __future__ import annotations

import json
import math
import random

from .pathfinding import neighbors4, neighbors8
from .world import H, W

LIDAR_RANGE = 8           # cells (4 m): short enough that the robots have to work through the store
LIDAR_STEP_DEG = 3
LIDAR_DROPOUT = 0.03      # share of rays that return nothing (dark or shiny surfaces)
LIDAR_INVISIBLE = {"cable"}

L_HIT, L_FREE, L_MIN, L_MAX = 0.9, -0.4, -2.5, 2.5   # log-odds update and clamp
OCC, FREE = 0.6, -0.6                                 # thresholds for "wall" and "free"


def lidar_scan(world, pos, rng):
    """Cast rays from the centre of `pos`. Returns (free cells, hit cells, ray end points)."""
    x0, y0 = pos[0] + 0.5, pos[1] + 0.5
    free, hits, ends = set(), set(), []
    for k in range(0, 360, LIDAR_STEP_DEG):
        if rng.random() < LIDAR_DROPOUT:
            continue
        a = math.radians(k)
        dx, dy = math.cos(a) * 0.25, math.sin(a) * 0.25
        x, y, end = x0, y0, None
        for _ in range(LIDAR_RANGE * 4):
            x += dx
            y += dy
            c = (int(math.floor(x)), int(math.floor(y)))
            if not world.in_bounds(c):
                break
            oid = world.obstacle_cells.get(c)
            if c in world.walls or (oid and world.obstacles[oid].kind not in LIDAR_INVISIBLE):
                hits.add(c)
                end = (x, y)
                break
            free.add(c)
        ends.append(end or (x, y))
    free.discard(pos)
    return free - hits, hits, ends


class OccupancyMap:
    """The console's map, shared by every robot: log-odds per cell, unknown until scanned."""

    def __init__(self):
        self.l = [[0.0] * W for _ in range(H)]
        self.version = 0

    # ------------------------------------------------------------------ queries
    def odds(self, c):
        return self.l[c[1]][c[0]]

    def known(self, c):
        return 0 <= c[0] < W and 0 <= c[1] < H and abs(self.l[c[1]][c[0]]) >= OCC

    def is_free(self, c):
        return 0 <= c[0] < W and 0 <= c[1] < H and self.l[c[1]][c[0]] <= FREE

    def is_occupied(self, c):
        return 0 <= c[0] < W and 0 <= c[1] < H and self.l[c[1]][c[0]] >= OCC

    def free_cells(self):
        return [(x, y) for y in range(H) for x in range(W) if self.l[y][x] <= FREE]

    # ------------------------------------------------------------------ updates
    def update(self, free, hits):
        for x, y in free:
            self.l[y][x] = max(L_MIN, self.l[y][x] + L_FREE)
        for x, y in hits:
            self.l[y][x] = min(L_MAX, self.l[y][x] + L_HIT)
        self.version += 1

    def mark_free(self, cells):
        """Cells a robot stands on or has driven through are certainly floor."""
        for x, y in cells:
            self.l[y][x] = L_MIN

    @classmethod
    def from_world(cls, world):
        """The map a finished survey would give: used when mapping is turned off."""
        m = cls()
        for y in range(H):
            for x in range(W):
                m.l[y][x] = L_MAX if (x, y) in world.walls else L_MIN
        return m

    # ------------------------------------------------------------------ frontiers
    def frontiers(self):
        """Free cells next to unknown space, grouped 8-connected into clusters (largest first)."""
        cells = {(x, y) for y in range(H) for x in range(W)
                 if self.l[y][x] <= FREE and any(0 <= n[0] < W and 0 <= n[1] < H and not self.known(n)
                                                 for n in neighbors4((x, y)))}
        groups = []
        while cells:
            seed = cells.pop()
            group, todo = [seed], [seed]
            while todo:
                for n in neighbors8(todo.pop()):
                    if n in cells:
                        cells.remove(n)
                        group.append(n)
                        todo.append(n)
            groups.append(group)
        return sorted(groups, key=len, reverse=True)

    # ------------------------------------------------------------------ display / file
    def chars(self):
        """One character per cell, row by row: '#' wall, '.' free, '?' unknown."""
        return "".join("#" if v >= OCC else "." if v <= FREE else "?" for row in self.l for v in row)

    def to_json(self, zones, meta):
        rows = [self.chars()[y * W:(y + 1) * W] for y in range(H)]
        return json.dumps({"w": W, "h": H, "rows": rows, "zones": zones, **meta}, ensure_ascii=False, indent=1)

    @classmethod
    def from_json(cls, text):
        data = json.loads(text)
        m = cls()
        for y, row in enumerate(data["rows"]):
            for x, ch in enumerate(row):
                m.l[y][x] = L_MAX if ch == "#" else L_MIN if ch == "." else 0.0
        return m, {k: tuple(v) for k, v in data.get("zones", {}).items()}, data


def bfs_distances(start, passable, limit=4000):
    """Grid distance from `start` to every reachable passable cell."""
    dist, frontier = {start: 0}, [start]
    while frontier and len(dist) < limit:
        nxt = []
        for c in frontier:
            for n in neighbors4(c):
                if n not in dist and passable(n):
                    dist[n] = dist[c] + 1
                    nxt.append(n)
        frontier = nxt
    return dist


def make_rng(seed, rid):
    return random.Random(f"{seed}-{rid}-lidar")
