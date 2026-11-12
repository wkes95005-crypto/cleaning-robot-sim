"""Ground truth: a supermarket floor on a grid, plus the dirt and obstacles that are really there.

Robots never plan on this directly. They learn about dirt and obstacles only through their own
camera + detector, report it to the console, and plan on the console's shared map.
"""
from __future__ import annotations

from dataclasses import dataclass

W, H = 48, 32
CELL_M = 0.5  # one cell = 0.5 m, so the floor is 24 m x 16 m

# name -> ((x0, y0, x1, y1) inclusive, aliases). Overlaps resolve to the first zone listed.
ZONES = {
    "生鮮區": ((1, 1, 15, 7), ["生鮮", "蔬果"]),
    "冷凍區": ((16, 1, 32, 7), ["冷凍", "冷藏"]),
    "倉儲區": ((34, 1, 46, 7), ["倉儲", "倉庫"]),
    "主走道": ((1, 8, 46, 10), ["主走道"]),
    "貨架A區": ((1, 11, 15, 21), ["貨架A", "A區", "A 區"]),
    "中央走道": ((16, 11, 17, 22), ["中央走道"]),
    "貨架B區": ((18, 11, 31, 21), ["貨架B", "B區", "B 區"]),
    "促銷區": ((32, 11, 46, 21), ["促銷", "特賣"]),
    "收銀區": ((18, 23, 40, 29), ["收銀", "結帳"]),
    "入口": ((41, 23, 46, 30), ["入口", "門口", "大門"]),
    "充電站": ((1, 26, 10, 30), ["充電站"]),
}
PATROL_ZONES = ["貨架A區", "貨架B區", "促銷區", "生鮮區", "冷凍區", "收銀區", "主走道", "入口"]

DIRT_KINDS = {"dry": "乾垃圾", "wet": "液體", "mixed": "混合髒污"}
OBSTACLE_KINDS = {"box": "紙箱", "cart": "推車", "cable": "電線", "pallet": "棧板"}
FIXED_KINDS = {"cable"}  # plugged in / in use: never moved by the arm, only avoided

DOCKS = [(2, 30), (4, 30), (6, 30), (8, 30), (10, 30)]
DROP_CELLS = [(x, y) for y in (2, 4, 6) for x in (36, 38, 43, 45)]  # inside the storage room


@dataclass
class Dirt:
    cell: tuple
    kind: str


@dataclass
class Obstacle:
    id: int
    cell: tuple | None  # None while an arm is carrying it
    kind: str
    movable: bool = True
    stored: bool = False  # already put away in the storage room


def rect_cells(x0, y0, x1, y1):
    return [(x, y) for x in range(x0, x1 + 1) for y in range(y0, y1 + 1)]


def _walls():
    walls = set()
    for x in range(W):
        walls |= {(x, 0), (x, H - 1)}
    for y in range(H):
        walls |= {(0, y), (W - 1, y)}
    walls |= {(33, y) for y in range(1, 9)}  # storage room, door at x = 39..41
    walls |= {(x, 8) for x in range(33, W - 1) if x not in (39, 40, 41)}
    fixtures = [
        (4, 3, 6, 4), (10, 3, 12, 4),                          # fresh-food islands
        (19, 3, 30, 4),                                          # freezer row
        (35, 14, 37, 15), (41, 14, 43, 15), (38, 18, 40, 19),   # promo displays
        (21, 25, 22, 27), (26, 25, 27, 27), (31, 25, 32, 27), (36, 25, 37, 27),  # checkouts
    ]
    for r in fixtures:
        walls |= set(rect_cells(*r))
    for x in (3, 6, 9, 12, 20, 23, 26, 29):  # shelves
        walls |= {(x, y) for y in range(12, 21)}
    return walls


def resolve_zone(text):
    """Return the canonical zone name mentioned in `text` (longest alias wins), or None."""
    if not text:
        return None
    names = [(name, name) for name in ZONES] + [(a, name) for name, (_, al) in ZONES.items() for a in al]
    for alias, name in sorted(names, key=lambda p: -len(p[0])):
        if alias in text:
            return name
    return None


def find_zones(text):
    """All zones mentioned in `text`, in the order they appear ("促銷區和入口" -> both)."""
    names = [(name, name) for name in ZONES] + [(a, name) for name, (_, al) in ZONES.items() for a in al]
    taken, found = set(), []
    for alias, name in sorted(names, key=lambda p: -len(p[0])):
        start = text.find(alias)
        while start != -1:
            span = set(range(start, start + len(alias)))
            if not span & taken:
                taken |= span
                found.append((start, name))
            start = text.find(alias, start + 1)
    out = []
    for _, name in sorted(found):
        if name not in out:
            out.append(name)
    return out


def coverage_waypoints(cells, stride=3):
    """Boustrophedon sample of `cells` so a robot cleaning a 3x3 footprint at each point covers them."""
    cells = set(cells)
    if not cells:
        return []
    xs = [c[0] for c in cells]
    ys = [c[1] for c in cells]
    x0, y0 = min(xs), min(ys)
    off = stride // 2

    def lines(lo, hi):
        vals = list(range(lo + off, hi + 1, stride)) or [lo]
        if vals[-1] + 1 < hi:  # the 3x3 footprint must still reach the last row / column
            vals.append(hi)
        return vals

    out = []
    for i, y in enumerate(lines(y0, max(ys))):
        cols = lines(x0, max(xs))
        for x in (cols if i % 2 == 0 else reversed(cols)):
            # snap to a floor cell of the zone inside this 3x3 patch if the centre is a fixture
            patch = [(x + dx, y + dy) for dy in (0, -1, 1) for dx in (0, -1, 1)]
            hit = next((c for c in patch if c in cells), None)
            if hit and hit not in out:
                out.append(hit)
    return out


class World:
    def __init__(self):
        self.walls = _walls()
        self.zone_map = {}
        for name, (r, _) in ZONES.items():
            for c in rect_cells(*r):
                if c not in self.walls:
                    self.zone_map.setdefault(c, name)
        self.docks = list(DOCKS)
        self.drop_cells = list(DROP_CELLS)
        self.dirt = {}            # cell -> Dirt
        self.obstacles = {}       # id -> Obstacle
        self.obstacle_cells = {}  # cell -> id
        self._next_oid = 1

    def in_bounds(self, c):
        return 0 <= c[0] < W and 0 <= c[1] < H

    def is_floor(self, c):
        return self.in_bounds(c) and c not in self.walls

    def blocked(self, c):
        return not self.is_floor(c) or c in self.obstacle_cells

    def zone_of(self, c):
        return self.zone_map.get(c, "通道")

    def zone_cells(self, name):
        return sorted(c for c, z in self.zone_map.items() if z == name)

    def add_dirt(self, c, kind):
        if kind not in DIRT_KINDS or self.blocked(c):
            return False
        self.dirt[c] = Dirt(c, kind)
        return True

    def add_obstacle(self, c, kind, movable=None):
        if kind not in OBSTACLE_KINDS or self.blocked(c) or c in self.dirt:
            return None
        if movable is None:
            movable = kind not in FIXED_KINDS
        o = Obstacle(self._next_oid, c, kind, movable)
        self._next_oid += 1
        self.obstacles[o.id] = o
        self.obstacle_cells[c] = o.id
        return o

    def pick(self, oid):
        o = self.obstacles[oid]
        del self.obstacle_cells[o.cell]
        o.cell = None
        return o

    def place(self, oid, c):
        o = self.obstacles[oid]
        o.cell = c
        o.stored = c in self.drop_cells
        self.obstacle_cells[c] = oid
        self.dirt.pop(c, None)
        return o

    def summary(self):
        return {"zones": list(ZONES), "size_m": [W * CELL_M, H * CELL_M]}


def load_default_scenario(world):
    """A shift at the supermarket: some litter, a few spills, and things left in the aisles."""
    for c in [(4, 15), (5, 17), (8, 14), (24, 13), (25, 18), (39, 17), (44, 12), (13, 9)]:
        world.add_dirt(c, "dry")
    for c in [(8, 6), (9, 6), (8, 7)]:          # dropped juice bottle in fresh food: liquid + shards
        world.add_dirt(c, "mixed")
    for c in [(24, 28), (25, 28), (29, 23)]:    # wet floor at the checkouts
        world.add_dirt(c, "wet")
    for c in [(44, 26), (45, 27)]:              # rain tracked in at the entrance
        world.add_dirt(c, "wet")
    world.add_obstacle((21, 16), "box")         # boxes blocking a shelf aisle in zone B
    world.add_obstacle((22, 16), "box")
    world.add_obstacle((16, 18), "cart")        # abandoned trolley in the central aisle
    world.add_obstacle((34, 23), "cable")       # extension cable in use near the checkouts: avoid, don't move
