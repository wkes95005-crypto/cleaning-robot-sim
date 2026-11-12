"""Central console (中控台), running on the laptop.

It owns the shared map (dirt and obstacles the robots have reported) and the task board, sends voice
commands to the LLM planner, and dispatches tasks to robots over the bus. Robots never receive
motor commands from here, only tasks.
"""
from __future__ import annotations

import itertools
import json
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from .pathfinding import chebyshev, neighbors8
from .robots import ROLE_LABEL
from .world import DIRT_KINDS, OBSTACLE_KINDS, PATROL_ZONES, ZONES, coverage_waypoints

LOW_BATTERY = 20     # send home to charge below this
READY_BATTERY = 30   # don't hand out new work below this
LOW_WATER = 10
MODE_LABEL = {"sweep": "掃地", "mop": "拖地", "sweep_then_mop": "先掃後拖"}


@dataclass
class Task:
    id: str
    kind: str        # sweep | mop | patrol | clear
    role: str        # which robot type can do it
    priority: int    # 3 requested obstacle removal, 2 voice command / found obstacle, 1 spot clean, 0 patrol
    title: str = ""
    waypoints: list = field(default_factory=list)
    zone: str | None = None          # set for zone tasks: they cover every dirt cell in the zone
    cells: set = field(default_factory=set)  # dirt cells a spot task was created for
    depends_on: list = field(default_factory=list)
    status: str = "pending"          # pending | assigned | done | cancelled
    assignee: str | None = None
    source: str = "auto"             # auto | voice
    obstacle_id: int | None = None
    target: tuple | None = None
    drop: tuple | None = None
    created: int = 0
    hold_until: int = 0              # don't retry before this tick (target was unreachable)
    pinned: str | None = None        # only this robot may take it (a command spoken to that robot)

    def payload(self):
        return {"id": self.id, "kind": self.kind, "waypoints": list(self.waypoints), "zone": self.zone,
                "obstacle_id": self.obstacle_id, "target": self.target, "drop": self.drop}

    def covers(self, cell, zone_of):
        if self.zone and self.kind in ("sweep", "mop"):
            return zone_of(cell) == self.zone
        return cell in self.cells


def clusters(cells, gap=3):
    """Group dirt cells that are within `gap` of each other into one spot task."""
    groups = []
    for c in sorted(cells):
        near = [g for g in groups if any(chebyshev(c, o) <= gap for o in g)]
        merged = [c] + [o for g in near for o in g]
        groups = [g for g in groups if g not in near] + [merged]
    return [sorted(g) for g in groups]


def nearest_first(start, cells):
    rest, out, cur = list(cells), [], start
    while rest:
        nxt = min(rest, key=lambda c: abs(c[0] - cur[0]) + abs(c[1] - cur[1]))
        rest.remove(nxt)
        out.append(nxt)
        cur = nxt
    return out


class Console:
    def __init__(self, sim, planner):
        self.sim = sim
        self.planner = planner
        self.tasks = {}
        self._ids = itertools.count(1)
        self.known_dirt = {}        # cell -> {"label", "conf"}
        self.known_obstacles = {}   # obstacle id -> {"cell", "kind", "movable", "stored"}
        self.no_go_core = set()     # obstacle cells: nobody drives through
        self.no_go_buffer = set()   # around hazards (e.g. cables): cleaning robots strongly avoid
        self.avoid_kinds = {"cable"}
        self.status = {}
        self.log = deque(maxlen=400)
        self.unreachable = {}       # cell -> tick before which it is not retried
        self.held = set()           # robots told to stop / go home; they get no work until told again
        self.chat = deque(maxlen=300)  # per-robot conversation shown on the dashboard
        self.auto_patrol = True
        self._patrol_next = 0
        self._sent = {}             # (robot, topic) -> tick, to avoid repeating dock / park orders
        self._last_yield = {}
        self._pool = ThreadPoolExecutor(max_workers=1)
        self._plans = []            # (future, robot id, zone) while the LLM is thinking
        self.drop_reserved = set()
        self.stats = {"cleaned": 0, "obstacles_cleared": 0, "commands": 0}

    # ------------------------------------------------------------------ helpers
    @property
    def tick(self):
        return self.sim.tick

    def _log(self, cat, text):
        self.log.append({"tick": self.tick, "cat": cat, "text": text})

    def _send(self, rid, topic, **payload):
        self.sim.bus.send("console", rid, topic, payload)

    def _new_task(self, **kw):
        t = Task(id=f"T{next(self._ids):03d}", created=self.tick, **kw)
        self.tasks[t.id] = t
        return t

    def _where(self, cell):
        return f"{self.sim.world.zone_of(cell)}{tuple(cell)}"

    def current_task(self, rid):
        return next((t for t in self.tasks.values() if t.status == "assigned" and t.assignee == rid), None)

    def _once(self, rid, topic, every=5):
        """True if `topic` wasn't sent to `rid` in the last `every` ticks (status reports lag by a tick)."""
        last = self._sent.get((rid, topic), -999)
        if self.tick - last < every:
            return False
        self._sent[(rid, topic)] = self.tick
        return True

    # ------------------------------------------------------------------ main loop
    def step(self):
        for m in self.sim.bus.receive("console"):
            getattr(self, "_on_" + m.topic)(m.sender, **m.payload)
        self._collect_plans()
        self._check_energy()
        if self.tick % 5 == 0:
            self._reconcile()
        self._dispatch()
        self._park_idle()

    # ------------------------------------------------------------------ messages from robots
    def _on_status(self, rid, **st):
        self.status[rid] = st

    def _on_voice_text(self, rid, text, cell, zone):
        self.stats["commands"] += 1
        self.chat.append({"tick": self.tick, "robot": rid, "from": "user", "text": text})
        self._log("voice", f"{rid} 麥克風收音，機上 ASR 轉文字：「{text}」（位置：{zone}）")
        role = self.sim.robot(rid).role
        context = {"robot_zone": zone, "heard_by": rid, "heard_by_role": ROLE_LABEL[role]}
        if self.planner.synchronous:
            self._apply_plan(self.planner.plan(text, context), rid, zone)
        else:
            self._log("llm", f"送交 {self.planner.name} 規劃中…")
            self._say(rid, "收到，讓我想一下…")
            self._plans.append((self._pool.submit(self.planner.plan, text, context), rid, zone))

    def _on_dirt_detected(self, rid, cell, label, conf, near):
        cell = tuple(cell)
        old = self.known_dirt.get(cell)
        if old and old["label"] == label:
            old["conf"] = max(old["conf"], conf)
            return
        if old and conf < old["conf"] and not (near and conf >= 0.8):
            return  # only a more confident or a close, confident look overrides what we have
        self.known_dirt[cell] = {"label": label, "conf": conf}
        if old:
            self._log("vision", f"{rid} 近距離複判 {self._where(cell)}：{DIRT_KINDS[old['label']]} → "
                                f"{DIRT_KINDS[label]}（信心 {conf:.2f}）")
        else:
            self._log("vision", f"{rid} 偵測到{DIRT_KINDS[label]} @ {self._where(cell)}（信心 {conf:.2f}）")

    def _on_residue(self, rid, cells):
        for c in map(tuple, cells):
            self.known_dirt[c] = {"label": "wet", "conf": 0.95}
        self._log("clean", f"{rid} 掃起碎片，留下 {len(cells)} 格液體殘留 → 轉給拖地機")

    def _on_dirt_cleaned(self, rid, cells):
        for c in map(tuple, cells):
            self.known_dirt.pop(c, None)
            self.stats["cleaned"] += 1
            for t in self.tasks.values():
                t.cells.discard(c)
        self._log("clean", f"{rid} 清除 {len(cells)} 格（{self._where(cells[0])}）")

    def _on_obstacle_detected(self, rid, oid, cell, kind, movable, stored):
        cell = tuple(cell)
        old = self.known_obstacles.get(oid)
        if old and old["cell"] == cell and old["stored"] == stored:
            return
        self.known_obstacles[oid] = {"cell": cell, "kind": kind, "movable": movable, "stored": stored}
        self._refresh_no_go()
        if stored:
            return
        extra = "，周圍加大避讓" if kind in self.avoid_kinds else ""
        fixed = "" if movable else "；不可移動，只能繞行"
        self._log("obstacle", f"{rid} 發現{OBSTACLE_KINDS[kind]} @ {self._where(cell)}，標記禁區{extra}{fixed}")
        if movable:
            self._ensure_clear_task(oid, priority=2)

    def _on_picked(self, rid, task, oid):
        o = self.known_obstacles.pop(oid, None)
        self._refresh_no_go()
        self.unreachable.clear()  # paths may be open again
        for t in self.tasks.values():
            t.hold_until = 0
        if o:
            self._log("obstacle", f"{rid} 夾起{OBSTACLE_KINDS[o['kind']]}，解除 {self._where(o['cell'])} 禁區")

    def _on_placed(self, rid, task, oid, cell):
        o = self.sim.world.obstacles[oid]
        self.known_obstacles[oid] = {"cell": tuple(cell), "kind": o.kind, "movable": o.movable, "stored": True}
        self._refresh_no_go()
        t = self.tasks.get(task)
        if t:
            self.drop_reserved.discard(t.drop)
        self.stats["obstacles_cleared"] += 1
        self._log("obstacle", f"{rid} 把{OBSTACLE_KINDS[o.kind]}放到倉儲區 {tuple(cell)}")

    def _on_progress(self, rid, task, waypoints):
        t = self.tasks.get(task)
        if t:
            t.waypoints = [tuple(w) for w in waypoints]

    def _on_done(self, rid, task, note=""):
        t = self.tasks.get(task)
        if not t or t.status != "assigned":
            return
        t.status = "done"
        if t.kind == "clear" and note:
            self.drop_reserved.discard(t.drop)
        if t.kind != "patrol":
            self._log("dispatch", f"{rid} 完成 {t.id}：{t.title}" + (f"（{note}）" if note else ""))

    def _on_skipped(self, rid, task, cell):
        cell = tuple(cell)
        self.unreachable[cell] = self.tick + 150
        t = self.tasks.get(task)
        if not t:
            return
        t.cells.discard(cell)
        if t.kind == "clear":
            t.status, t.assignee, t.hold_until = "pending", None, self.tick + 150
        self._log("dispatch", f"{rid} 到不了 {self._where(cell)}，先跳過，路線改變後再試")

    def _on_yield(self, rid, to, cell):
        if self.tick - self._last_yield.get(rid, -999) > 20:
            self._last_yield[rid] = self.tick
            self._log("conflict", f"{rid} 在 {tuple(cell)} 前遇到 {to}，依優先序讓行並改走替代路徑")

    def _on_charged(self, rid):
        st = self.status.get(rid, {})
        self._log("energy", f"{rid} 充電完成（電量 {st.get('battery', '?')}%），回到可派工狀態")

    # ------------------------------------------------------------------ shared map
    def _refresh_no_go(self):
        world = self.sim.world
        self.no_go_core = {tuple(o["cell"]) for o in self.known_obstacles.values()}
        buf = set()
        for o in self.known_obstacles.values():
            if o["kind"] in self.avoid_kinds and not o["stored"]:
                buf |= {n for n in neighbors8(tuple(o["cell"])) if world.is_floor(n)}
        self.no_go_buffer = buf - self.no_go_core

    # ------------------------------------------------------------------ planning
    def _collect_plans(self):
        still = []
        for fut, rid, zone in self._plans:
            if fut.done():
                try:
                    self._apply_plan(fut.result(), rid, zone)
                except Exception as e:
                    self._log("llm", f"規劃失敗：{e}")
            else:
                still.append((fut, rid, zone))
        self._plans = still

    def _apply_plan(self, plan, rid, heard_zone):
        """Turn a plan into tasks. Commands are addressed: work the hearing robot can do is pinned to it."""
        note = plan.get("note")
        self._log("llm", f"{self.planner.name} → {json.dumps(plan['steps'], ensure_ascii=False)}"
                         + (f"（who={plan.get('who', 'self')}）" if plan.get("who") == "all" else "")
                         + (f"（{note}）" if note else ""))
        if not plan["steps"]:
            self._log("llm", "聽不懂這句指令，沒有產生任務")
            self._say(rid, "抱歉，我聽不懂這句指令，可以換個說法嗎？")
            return
        pin = None if plan.get("who") == "all" else rid
        everyone = [r.id for r in self.sim.robots]
        replies, prev = [], []
        for s in plan["steps"]:
            a = s["action"]
            if a == "avoid":
                kind = s.get("object")
                if kind in OBSTACLE_KINDS:
                    self.avoid_kinds.add(kind)
                    self._refresh_no_go()
                    self._log("llm", f"新增避讓規則：所有機器人遠離{OBSTACLE_KINDS[kind]}")
                    replies.append(f"大家都會避開{OBSTACLE_KINDS[kind]}")
                else:
                    self._log("llm", f"不認得要避開的物體「{kind}」，略過")
                    replies.append("但我不認得要避開的東西")
            elif a == "dock":
                for r in ([pin] if pin else everyone):
                    self._hold(r, go_home=True)
                self._log("dispatch", f"{pin or '全體'}回充電站待命；未完成任務留在任務黑板")
                replies.append("我回充電站待命" if pin else "已通知全體回充電站待命")
            elif a == "stop":
                for r in ([pin] if pin else everyone):
                    self._hold(r)
                self._log("dispatch", f"{pin or '全體'}原地停下待命")
                replies.append("我原地停下" if pin else "已通知全體原地停下")
            elif a == "resume":
                self.held -= {pin} if pin else set(everyone)
                self._log("dispatch", f"{pin or '全體'}恢復工作")
                replies.append("我繼續工作" if pin else "全體恢復工作")
            else:
                zone = s.get("zone") or heard_zone
                if zone not in ZONES:
                    self._log("llm", f"指令沒有可用的區域（{zone}），略過這一步")
                    replies.append("但我不確定是哪個區域")
                    continue
                if pin:
                    self.held.discard(pin)
                else:
                    self.held.clear()
                if a == "clean_zone":
                    ids, text = self._zone_clean(zone, s["mode"], prev, pin)
                else:
                    ids, text = self._clear_request(zone, prev, pin)
                replies.append(text)
                if ids and not plan.get("parallel"):
                    prev = ids
        self._say(rid, "收到，" + "；".join(replies) + "。")

    def _say(self, rid, text):
        self.chat.append({"tick": self.tick, "robot": rid, "from": "robot", "text": text})

    def _role_held(self, role):
        ids = [r.id for r in self.sim.robots if r.role == role]
        return ids and all(i in self.held for i in ids)

    def _release(self, rid):
        """Take the robot's current task back onto the board (patrols are simply dropped)."""
        t = self.current_task(rid)
        if not t or (t.kind == "clear" and self.status.get(rid, {}).get("carrying")):
            return None
        t.status, t.assignee = ("cancelled" if t.kind == "patrol" else "pending"), None
        self._send(rid, "cancel", task=t.id)
        return t

    def _hold(self, rid, go_home=False):
        self.held.add(rid)
        self._release(rid)
        self._send(rid, "dock" if go_home else "hold")

    def _zone_clean(self, zone, mode, deps, pin=None):
        world = self.sim.world
        cells = [c for c in world.zone_cells(zone) if c not in self.no_go_core]
        role = self.sim.robot(pin).role if pin else None
        n_sweepers = sum(r.role == "sweeper" for r in self.sim.robots)
        sweep_ids, ids, said = [], [], []
        if role == "arm":
            said.append("我是障礙排除機，清潔交給其他機器人")
        elif role == "mopper" and mode == "sweep":
            said.append("我是拖地機")
        elif role == "sweeper" and mode == "mop":
            said.append("我是掃地機")
        if mode in ("sweep", "sweep_then_mop"):
            if role == "sweeper":  # the robot that was spoken to does the whole zone itself
                t = self._new_task(kind="sweep", role="sweeper", priority=3, zone=zone, source="voice",
                                   waypoints=coverage_waypoints(cells), depends_on=list(deps), pinned=pin,
                                   title=f"掃地：{zone}（指定 {pin}）")
                sweep_ids.append(t.id)
                said.append(f"我去掃{zone}")
            else:
                parts = [cells] if len(cells) < 60 or n_sweepers < 2 else self._split(cells, n_sweepers)
                for i, part in enumerate(parts):
                    suffix = f"（第 {i + 1}/{len(parts)} 段）" if len(parts) > 1 else ""
                    t = self._new_task(kind="sweep", role="sweeper", priority=2, zone=zone, source="voice",
                                       waypoints=coverage_waypoints(part), depends_on=list(deps),
                                       title=f"掃地：{zone}{suffix}")
                    sweep_ids.append(t.id)
                said.append(f"{zone}掃地交給掃地機" + ("分段同時作業" if len(parts) > 1 else ""))
        if mode in ("mop", "sweep_then_mop"):
            mine = role == "mopper"
            t = self._new_task(kind="mop", role="mopper", priority=3 if mine else 2, zone=zone, source="voice",
                               waypoints=coverage_waypoints(cells), depends_on=list(deps) + sweep_ids,
                               pinned=pin if mine else None,
                               title=f"拖地：{zone}" + (f"（指定 {pin}）" if mine else "")
                                     + ("（等掃地完成）" if sweep_ids else ""))
            ids.append(t.id)
            wait = "，等掃完再拖" if sweep_ids else ""
            said.append((f"我去拖{zone}" if mine else f"{zone}拖地交給拖地機") + wait)
        for r in ("sweeper", "mopper"):
            if r != role and self._role_held(r) and (r == "sweeper" and sweep_ids or r == "mopper" and ids):
                said.append(f"不過{'掃地機' if r == 'sweeper' else '拖地機'}目前待命中，恢復後才會執行")
        ids = sweep_ids + ids
        dep = f"，等 {', '.join(deps)} 完成後開始" if deps else ""
        self._log("dispatch", f"建立 {', '.join(ids)}：{zone} {MODE_LABEL[mode]}"
                              + (f"，指定 {pin} 執行" if role in ("sweeper", "mopper") else "")
                              + (f"，{len(sweep_ids)} 台掃地機分段同時作業" if len(sweep_ids) > 1 else "") + dep)
        return ids, "，".join(said)

    @staticmethod
    def _split(cells, n):
        cells = sorted(cells)
        k = -(-len(cells) // n)
        return [cells[i:i + k] for i in range(0, len(cells), k)]

    def _clear_request(self, zone, deps, pin=None):
        world = self.sim.world
        mine = pin if pin and self.sim.robot(pin).role == "arm" else None
        ids, fixed = [], set()
        for oid, o in self.known_obstacles.items():
            if o["stored"] or world.zone_of(o["cell"]) != zone:
                continue
            if not o["movable"]:
                fixed.add(OBSTACLE_KINDS[o["kind"]])
                continue
            tid = self._ensure_clear_task(oid, priority=3, deps=deps, pinned=mine)
            if tid:
                ids.append(tid)
        keep = f"；{'、'.join(sorted(fixed))}不能移動，大家會繞開" if fixed else ""
        if fixed:
            self._log("dispatch", f"{zone}的{'、'.join(sorted(fixed))}屬於不可移動障礙，不派搬移，只繞行")
        if ids:
            self._log("dispatch", f"{zone} 的搬移任務 {', '.join(ids)} 提高為最優先")
            return ids, (f"我去搬開{zone}的障礙物" if mine else f"已請障礙排除機搬開{zone}的障礙物") + keep
        if fixed:
            return [], f"{zone}只有{'、'.join(sorted(fixed))}，不能移動，大家會繞開"
        t = self._new_task(kind="patrol", role="arm", priority=1, zone=zone, source="voice", pinned=mine,
                           waypoints=coverage_waypoints([c for c in world.zone_cells(zone)
                                                         if c not in self.no_go_core], 4),
                           depends_on=list(deps), title=f"搜尋障礙物：{zone}")
        self._log("dispatch", f"{zone} 目前沒有已知障礙物，建立 {t.id} 讓障礙排除機先去巡查")
        return [t.id], (f"{zone}目前沒有已知障礙物，我先去巡查" if mine
                        else f"{zone}目前沒有已知障礙物，已請障礙排除機去巡查")

    def _ensure_clear_task(self, oid, priority, deps=(), pinned=None):
        for t in self.tasks.values():
            if t.kind == "clear" and t.obstacle_id == oid and t.status in ("pending", "assigned"):
                t.priority = max(t.priority, priority)
                t.pinned = t.pinned or pinned
                return t.id
        o = self.known_obstacles[oid]
        free = [c for c in self.sim.world.drop_cells
                if c not in self.drop_reserved and not self.sim.world.blocked(c)]
        if not free:
            self._log("obstacle", "倉儲區放置點已滿，無法再派搬移任務")
            return None
        drop = free[0]
        self.drop_reserved.add(drop)
        t = self._new_task(kind="clear", role="arm", priority=priority, obstacle_id=oid, pinned=pinned,
                           target=tuple(o["cell"]), drop=drop, depends_on=list(deps),
                           title=f"搬移{OBSTACLE_KINDS[o['kind']]} {tuple(o['cell'])} → 倉儲區")
        self._log("dispatch", f"建立 {t.id}：障礙排除機把{OBSTACLE_KINDS[o['kind']]}搬到倉儲區")
        return t.id

    # ------------------------------------------------------------------ task board
    def _reconcile(self):
        """Every known dirt cell must be covered by an open task of the right robot type."""
        zone_of = self.sim.world.zone_of
        open_tasks = [t for t in self.tasks.values()
                      if t.status in ("pending", "assigned") and t.kind in ("sweep", "mop")]
        loose = {"sweeper": [], "mopper": []}
        for cell, info in self.known_dirt.items():
            if self.unreachable.get(cell, 0) > self.tick or cell in self.no_go_core:
                continue
            role = "mopper" if info["label"] == "wet" else "sweeper"
            if not any(t.role == role and t.covers(cell, zone_of) for t in open_tasks):
                loose[role].append(cell)
        for role, cells in loose.items():
            for group in clusters(cells):
                kind = "sweep" if role == "sweeper" else "mop"
                zone = zone_of(group[0])
                t = self._new_task(kind=kind, role=role, priority=1, waypoints=group, cells=set(group),
                                   title=f"定點{MODE_LABEL[kind]}：{zone} {len(group)} 處")
                self._log("dispatch", f"建立 {t.id}：{t.title}")

    def _ready(self, rid):
        st = self.status.get(rid)
        if rid in self.held or not st or st["mode"] == "charge" or st["carrying"]:
            return False
        if st["battery"] < READY_BATTERY:
            return False
        return st["water"] is None or st["water"] >= 2 * LOW_WATER

    def _dispatch(self):
        taken = set()
        pending = [t for t in self.tasks.values()
                   if t.status == "pending" and t.hold_until <= self.tick
                   and all(self.tasks[d].status in ("done", "cancelled") for d in t.depends_on)]
        pending.sort(key=lambda t: (-t.priority, t.created))
        for t in pending:
            first = t.target if t.kind == "clear" else (t.waypoints[0] if t.waypoints else None)
            if first is None:
                t.status = "done"
                continue
            best = None
            for r in self.sim.robots:
                if r.role != t.role or r.id in taken or not self._ready(r.id):
                    continue
                if t.pinned and r.id != t.pinned:
                    continue
                cur = self.current_task(r.id)
                if cur and not self._can_preempt(cur, t):
                    continue
                st = self.status[r.id]
                pos = tuple(st["pos"])
                cost = abs(pos[0] - first[0]) + abs(pos[1] - first[1]) + (100 - st["battery"]) * 0.3
                if best is None or cost < best[0]:
                    best = (cost, r, cur)
            if best:
                self._assign(t, best[1], best[2], best[0])
                taken.add(best[1].id)
        if self.auto_patrol:
            self._assign_patrols(taken)

    @staticmethod
    def _can_preempt(cur, new):
        if cur.kind == "clear" or cur.priority >= new.priority:
            return False
        return cur.kind == "patrol" or new.pinned is not None  # a direct command outranks routine work

    def _assign(self, t, robot, preempted, cost):
        if preempted:
            preempted.status = "pending" if preempted.priority > 0 else "cancelled"
            preempted.assignee = None
            self._send(robot.id, "cancel", task=preempted.id)
            self._log("dispatch", f"{robot.id} 中斷「{preempted.title}」，優先處理 {t.id}")
        pos = tuple(self.status[robot.id]["pos"])
        if t.kind in ("sweep", "mop") and not t.zone:
            t.waypoints = nearest_first(pos, t.waypoints)
        elif t.waypoints and abs(t.waypoints[-1][0] - pos[0]) + abs(t.waypoints[-1][1] - pos[1]) < \
                abs(t.waypoints[0][0] - pos[0]) + abs(t.waypoints[0][1] - pos[1]):
            t.waypoints = t.waypoints[::-1]  # start the sweep from the nearer end
        t.status, t.assignee = "assigned", robot.id
        self._send(robot.id, "assign", task=t.payload())
        if t.kind != "patrol" or t.role == "arm":
            self._log("dispatch", f"派工 {t.id} → {robot.id}（{ROLE_LABEL[robot.role]}）：{t.title}，成本 {cost:.0f}")

    def _assign_patrols(self, taken):
        if any(t.status == "pending" and t.role == "sweeper" for t in self.tasks.values()):
            return
        busy_zones = {t.zone for t in self.tasks.values() if t.status == "assigned" and t.kind == "patrol"}
        for r in self.sim.robots:
            if r.role != "sweeper" or r.id in taken or not self._ready(r.id) or self.current_task(r.id):
                continue
            for _ in range(len(PATROL_ZONES)):
                zone = PATROL_ZONES[self._patrol_next % len(PATROL_ZONES)]
                self._patrol_next += 1
                if zone not in busy_zones:
                    break
            cells = [c for c in self.sim.world.zone_cells(zone) if c not in self.no_go_core]
            t = self._new_task(kind="patrol", role="sweeper", priority=0, zone=zone,
                               waypoints=coverage_waypoints(cells), title=f"巡邏清掃：{zone}")
            busy_zones.add(zone)
            self._assign(t, r, None, 0)

    # ------------------------------------------------------------------ energy and parking
    def _check_energy(self):
        for r in self.sim.robots:
            st = self.status.get(r.id)
            if not st or st["mode"] == "charge" or st["carrying"] or st["state"] in ("grasping", "placing"):
                continue  # let the arm finish what is in its gripper first
            low = st["battery"] < LOW_BATTERY
            dry = st["water"] is not None and st["water"] < LOW_WATER
            if not (low or dry) or not self._once(r.id, "dock"):
                continue
            self._send(r.id, "dock")
            why = f"電量 {st['battery']:.0f}%" if low else f"水箱剩 {st['water']:.0f}%"
            t = self.current_task(r.id)
            if t:
                t.status, t.assignee = ("cancelled" if t.kind == "patrol" else "pending"), None
                kept = "" if t.kind == "patrol" else f"，{t.id} 放回任務黑板由其他機器接手"
            else:
                kept = ""
            self._log("energy", f"{r.id} {why}，返回充電站{kept}")

    def _park_idle(self):
        for r in self.sim.robots:
            st = self.status.get(r.id)
            if st and st["state"] == "idle" and st["mode"] is None and not self.current_task(r.id) \
                    and r.id not in self.held \
                    and self._once(r.id, "park", every=10):
                self._send(r.id, "park")
