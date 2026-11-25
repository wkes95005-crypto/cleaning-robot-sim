"""Random events: litter, spills and things left in the aisles appear while the store is open.

This is ground truth (the world), not something the console knows: robots still have to see the new
dirt or obstacle with their own camera and lidar before anyone is sent to deal with it.

What turns up depends on the zone, the way it does in a real store: spills and broken jars in fresh
food, condensation in the freezers, rain at the entrance, fallen boxes in the shelf aisles, trolleys
left in the walkways. A spill covers one to three cells next to each other.
"""
from __future__ import annotations

import random

from .pathfinding import neighbors4
from .world import DIRT_KINDS, OBSTACLE_KINDS, ZONES

# zone -> (how busy it is, [(weight, "dirt" | "obstacle", kind, what happened)])
PROFILE = {
    "生鮮區": (3, [(3, "dirt", "wet", "蔬果水漬"), (2, "dirt", "mixed", "打破的果汁瓶"),
                (1, "dirt", "dry", "掉落的菜葉"), (1, "obstacle", "box", "補貨紙箱")]),
    "冷凍區": (2, [(3, "dirt", "wet", "冷凍櫃凝結水"), (1, "dirt", "dry", "包裝碎屑")]),
    "主走道": (3, [(3, "dirt", "dry", "紙屑"), (1, "dirt", "wet", "飲料滴落"), (1, "obstacle", "cart", "推車")]),
    "中央走道": (1, [(2, "dirt", "dry", "紙屑"), (1, "obstacle", "cart", "推車")]),
    "貨架A區": (2, [(3, "dirt", "dry", "零食碎屑"), (2, "obstacle", "box", "掉落的紙箱"),
                 (1, "dirt", "wet", "飲料滴落")]),
    "貨架B區": (2, [(3, "dirt", "dry", "零食碎屑"), (2, "obstacle", "box", "掉落的紙箱"),
                 (1, "dirt", "mixed", "打破的醬料瓶")]),
    "促銷區": (2, [(3, "dirt", "dry", "傳單"), (1, "obstacle", "box", "試吃紙箱"), (1, "dirt", "wet", "試飲灑出")]),
    "收銀區": (3, [(3, "dirt", "dry", "發票與塑膠袋"), (1, "dirt", "wet", "飲料打翻"), (1, "obstacle", "cart", "推車")]),
    "入口": (2, [(3, "dirt", "wet", "雨水腳印"), (1, "dirt", "dry", "落葉與泥沙"), (1, "obstacle", "cart", "推車")]),
}


class RandomEvents:
    def __init__(self, sim, cfg):
        self.sim = sim
        self.enabled = bool(cfg.get("enabled", False))
        self.mean_ticks = max(1.0, float(cfg.get("mean_interval_s", 20)) / 0.5)  # one tick = 0.5 s
        self.max_dirt = int(cfg.get("max_dirt", 30))
        self.max_obstacles = int(cfg.get("max_obstacles", 6))
        self.rng = random.Random(f"{sim.seed}-events")
        self.count = 0

    def step(self):
        if not self.enabled or self.sim.console.phase == "exploring":  # the survey runs in an empty store
            return
        if self.rng.random() < 1 / self.mean_ticks:
            self.fire()

    def fire(self):
        """Try a few times to make one thing happen somewhere it can. Returns a description or None."""
        for _ in range(20):
            zone = self.rng.choices(list(PROFILE), weights=[p[0] for p in PROFILE.values()])[0]
            options = PROFILE[zone][1]
            _, what, kind, story = self.rng.choices(options, weights=[o[0] for o in options])[0]
            cell = self._pick_cell(zone, what)
            if cell is None:
                continue
            if what == "dirt" and self._dirt_room():
                cells = self._spill(cell, kind)
                self.count += 1
                return self._note(f"{zone}出現{story}（{DIRT_KINDS[kind]}，{len(cells)} 格）")
            if what == "obstacle" and self._obstacle_room() and self._keeps_store_open(cell):
                self.sim.world.add_obstacle(cell, kind)
                self.count += 1
                return self._note(f"{zone}出現{story}（{OBSTACLE_KINDS[kind]}）")
        return None

    # ------------------------------------------------------------------ helpers
    def _note(self, text):
        self.sim.console._log("env", text)
        return text

    def _dirt_room(self):
        return len(self.sim.world.dirt) < self.max_dirt

    def _obstacle_room(self):
        w = self.sim.world
        return sum(1 for o in w.obstacles.values() if o.cell and not o.stored) < self.max_obstacles

    def _free(self, c):
        """Open floor with nothing on it, away from the robots, docks and the storage drop-off."""
        w, sim = self.sim.world, self.sim
        if w.blocked(c) or c in w.dirt or c in w.docks or c in w.drop_cells:
            return False
        return all(abs(c[0] - r.pos[0]) > 1 or abs(c[1] - r.pos[1]) > 1 for r in sim.robots)

    def _pick_cell(self, zone, what):
        x0, y0, x1, y1 = ZONES[zone][0]
        cells = [(x, y) for x in range(x0, x1 + 1) for y in range(y0, y1 + 1)
                 if self.sim.world.zone_of((x, y)) == zone and self._free((x, y))]
        if what == "obstacle":  # don't drop a box right next to another one
            w = self.sim.world
            cells = [c for c in cells if not any(n in w.obstacle_cells for n in neighbors4(c))]
        return self.rng.choice(cells) if cells else None

    def _spill(self, cell, kind):
        """Litter is one cell; a spill spreads to up to two more next to it."""
        cells = [cell]
        if kind in ("wet", "mixed"):
            for _ in range(self.rng.randint(0, 2)):
                around = [n for c in cells for n in neighbors4(c) if n not in cells and self._free(n)]
                if not around:
                    break
                cells.append(self.rng.choice(around))
        for c in cells:
            self.sim.world.add_dirt(c, kind)
        return cells

    def _keeps_store_open(self, cell):
        """An obstacle may narrow an aisle but must not cut any part of the store off from the docks."""
        w = self.sim.world
        start = w.docks[0]

        def reach(extra):
            seen, todo = {start}, [start]
            while todo:
                for n in neighbors4(todo.pop()):
                    if n not in seen and n != extra and w.is_floor(n) and n not in w.obstacle_cells:
                        seen.add(n)
                        todo.append(n)
            return len(seen)

        return reach(cell) >= reach(None) - 1
