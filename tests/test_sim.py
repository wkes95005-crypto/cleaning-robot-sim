import unittest

from sim.engine import Simulation
from sim.pathfinding import astar
from sim.planner import RulePlanner, validate
from sim.world import World, coverage_waypoints


def run(sim, ticks, commands=()):
    """Step `sim`, issuing (tick, robot, text) voice commands on the way."""
    commands = sorted(commands)
    for _ in range(ticks):
        while commands and commands[0][0] <= sim.tick:
            _, rid, text = commands.pop(0)
            sim.hear(text, rid)
        sim.step()


class PathTest(unittest.TestCase):
    def test_astar_goes_around_shelves(self):
        w = World()
        path = astar((2, 15), {(5, 15)}, lambda c: not w.blocked(c))
        self.assertIsNotNone(path)
        self.assertEqual(path[-1], (5, 15))
        self.assertTrue(all(not w.blocked(c) for c in path))
        self.assertGreater(len(path), 3)  # shelf at x=3 forces a detour

    def test_coverage_reaches_every_cell(self):
        w = World()
        for zone in ("生鮮區", "貨架A區", "收銀區"):
            cells = w.zone_cells(zone)
            wps = coverage_waypoints(cells)
            missed = [c for c in cells if all(max(abs(c[0] - p[0]), abs(c[1] - p[1])) > 1 for p in wps)]
            self.assertLess(len(missed), len(cells) * 0.05, zone)


class PlannerTest(unittest.TestCase):
    def test_sequenced_command(self):
        plan = RulePlanner().plan("先清理生鮮區，避開地上的電線，最後再掃收銀區。")
        self.assertEqual(plan["steps"], [
            {"action": "clean_zone", "zone": "生鮮區", "mode": "sweep_then_mop"},
            {"action": "avoid", "object": "cable"},
            {"action": "clean_zone", "zone": "收銀區", "mode": "sweep"},
        ])
        self.assertFalse(plan["parallel"])

    def test_zone_carries_over_to_next_clause(self):
        plan = RulePlanner().plan("貨架B區有紙箱擋路，請搬開。")
        self.assertEqual(plan["steps"], [{"action": "clear_obstacles", "zone": "貨架B區"}])

    def test_two_zones_in_parallel(self):
        plan = RulePlanner().plan("促銷區和入口同時拖地")
        self.assertEqual([s["zone"] for s in plan["steps"]], ["促銷區", "入口"])
        self.assertTrue(plan["parallel"])

    def test_validate_drops_unknown_actions_and_maps_aliases(self):
        plan = validate({"steps": [{"action": "set_motor_speed", "v": 1},
                                   {"action": "clean_zone", "zone": "結帳", "mode": "mop"}]})
        self.assertEqual(plan["steps"], [{"action": "clean_zone", "zone": "收銀區", "mode": "mop"}])


class ScenarioTest(unittest.TestCase):
    def test_default_shift_gets_fully_cleaned(self):
        sim = Simulation()
        run(sim, 1500, [(60, "S1", "先清理生鮮區，避開地上的電線，最後再掃收銀區。"),
                        (200, "A1", "貨架B區有紙箱擋路，請搬開。")])
        self.assertEqual(sim.collisions, 0)
        self.assertEqual(len(sim.world.dirt), 0)
        movable = [o for o in sim.world.obstacles.values() if o.movable]
        cable = next(o for o in sim.world.obstacles.values() if o.kind == "cable")
        self.assertTrue(movable and all(o.stored for o in movable))
        self.assertEqual(cable.cell, (34, 23))  # never moved
        self.assertFalse(cable.stored)
        cats = {e["cat"] for e in sim.console.log}
        self.assertTrue({"voice", "llm", "dispatch", "obstacle", "energy"} <= cats)

    def test_mop_waits_for_sweep_in_same_zone(self):
        sim = Simulation()
        run(sim, 400, [(5, "S1", "大家先清理生鮮區")])
        tasks = sim.console.tasks.values()
        sweeps = [t for t in tasks if t.kind == "sweep" and t.zone == "生鮮區"]
        mop = next(t for t in tasks if t.kind == "mop" and t.zone == "生鮮區")
        self.assertTrue(set(t.id for t in sweeps) <= set(mop.depends_on))
        self.assertEqual(len(sweeps), 2)  # a fleet-wide command splits the zone between both sweepers


class CableTest(unittest.TestCase):
    def test_cable_is_avoided_not_moved(self):
        sim = Simulation()
        run(sim, 400, [(200, "A1", "收銀區的電線擋路，請搬開")])
        self.assertFalse(any(t.kind == "clear" and sim.world.obstacles[t.obstacle_id].kind == "cable"
                             for t in sim.console.tasks.values()))
        self.assertIn((34, 23), sim.console.no_go_core)
        self.assertTrue(sim.console.no_go_buffer)  # cleaning robots keep their distance
        self.assertIn("不能移動", sim.console.chat[-1]["text"])

    def test_cable_placed_from_dashboard_is_fixed(self):
        sim = Simulation()
        self.assertTrue(sim.spawn("obstacle", "cable", (30, 9)))
        self.assertFalse(sim.world.obstacles[max(sim.world.obstacles)].movable)


class AddressedCommandTest(unittest.TestCase):
    def test_command_goes_to_the_robot_spoken_to(self):
        sim = Simulation()
        run(sim, 60, [(5, "S2", "去掃貨架A區")])
        t = next(t for t in sim.console.tasks.values() if t.zone == "貨架A區" and t.kind == "sweep")
        self.assertEqual(t.pinned, "S2")
        self.assertEqual(t.assignee, "S2")
        self.assertEqual(sim.console.chat[-1]["text"], "收到，我去掃貨架A區。")

    def test_work_the_robot_cannot_do_is_handed_over(self):
        sim = Simulation()
        run(sim, 30, [(5, "M1", "去掃收銀區")])
        sweeps = [t for t in sim.console.tasks.values() if t.zone == "收銀區"]
        self.assertTrue(sweeps and all(t.role == "sweeper" and t.pinned is None for t in sweeps))
        self.assertIn("我是拖地機", sim.console.chat[-1]["text"])

    def test_stop_and_resume_only_affect_that_robot(self):
        sim = Simulation()
        run(sim, 120, [(5, "S1", "去掃貨架B區"), (60, "S1", "停下來")])
        s1 = sim.robot("S1")
        self.assertIn("S1", sim.console.held)
        self.assertIsNone(s1.task)
        pos = s1.pos
        run(sim, 20)
        self.assertEqual(sim.robot("S1").pos, pos)  # stays put
        self.assertNotIn("S2", sim.console.held)
        run(sim, 20, [(sim.tick, "S1", "繼續工作")])
        self.assertNotIn("S1", sim.console.held)
        t = next(t for t in sim.console.tasks.values() if t.zone == "貨架B區" and t.kind == "sweep")
        self.assertEqual(t.assignee, "S1")  # picks its own task back up

    def test_here_in_an_aisle_cleans_around_the_robot(self):
        sim = Simulation()
        w = sim.world
        aisle = next(c for c in sorted(w.zone_map.keys() | {(x, y) for x in range(48) for y in range(32)})
                     if w.is_floor(c) and w.zone_of(c) == "通道" and c not in w.obstacle_cells)
        plan = {"steps": [{"action": "clean_zone", "zone": None, "mode": "sweep"}], "parallel": False, "who": "self"}
        sim.console._apply_plan(plan, "S1", "通道", aisle)
        t = next(t for t in sim.console.tasks.values() if t.source == "voice")
        self.assertIsNone(t.zone)
        self.assertIn(aisle, t.cells)
        self.assertTrue(all(max(abs(c[0] - aisle[0]), abs(c[1] - aisle[1])) <= 2 for c in t.cells))
        self.assertEqual(sim.console.chat[-1]["text"], "收到，我去掃這附近。")

    def test_everyone_dock(self):
        sim = Simulation()
        run(sim, 80, [(40, "A1", "全部回充電站待命")])
        self.assertEqual(sim.console.held, {"A1", "M1", "S1", "S2"})

    def test_low_battery_hands_task_back(self):
        sim = Simulation()
        run(sim, 300)
        self.assertTrue(any("返回充電站" in e["text"] and e["text"].startswith("S2") for e in sim.console.log))

    def test_mixed_spill_needs_sweeper_then_mopper(self):
        sim = Simulation({"sim": {"scenario": "empty"}})
        sim.world.add_dirt((30, 9), "mixed")
        run(sim, 400)
        self.assertNotIn((30, 9), sim.world.dirt)
        texts = [e["text"] for e in sim.console.log]
        self.assertTrue(any("液體殘留" in t for t in texts))



class RandomEventsTest(unittest.TestCase):
    """[events]: litter, spills and obstacles appear while the store is open; robots must find them."""

    def test_events_appear_and_get_cleaned(self):
        sim = Simulation({"events": {"enabled": True, "mean_interval_s": 5, "max_obstacles": 3}})
        w = sim.world
        start_dirt = set(w.dirt)
        floor = sum(1 for x in range(48) for y in range(32) if w.is_floor((x, y)))
        spawned = set()
        for _ in range(600):
            before = set(w.dirt)
            sim.step()
            spawned |= set(w.dirt) - before
            loose = {c for c, oid in w.obstacle_cells.items() if not w.obstacles[oid].stored}
            self.assertFalse({r.pos for r in sim.robots} & loose, "spawned on a robot")
        self.assertGreater(sim.events.count, 5)
        self.assertLessEqual(sum(1 for o in w.obstacles.values() if o.cell and not o.stored), 3 + 4)
        self.assertTrue(any(e["cat"] == "env" for e in sim.console.log))
        cleaned = spawned - set(w.dirt) - start_dirt
        self.assertGreater(len(cleaned), len(spawned) // 3, "robots should find and clean new dirt")
        self.assertEqual(sim.collisions, 0)

    def test_off_by_default_and_paused_while_mapping(self):
        self.assertFalse(Simulation().events.enabled)
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            sim = Simulation({"events": {"enabled": True, "mean_interval_s": 1},
                              "mapping": {"mode": "explore", "file": f"{d}/m.json"}})
            for _ in range(20):
                sim.step()
            self.assertEqual(sim.events.count, 0)

if __name__ == "__main__":
    unittest.main()


class MappingTest(unittest.TestCase):
    """The robots survey the store with lidar before work starts (config [mapping])."""

    def setUp(self):
        import tempfile
        self.dir = tempfile.TemporaryDirectory()
        self.cfg = {"mapping": {"mode": "auto", "file": f"{self.dir.name}/map.json",
                                "manual_zones": ["促銷區", "入口"]}}

    def tearDown(self):
        self.dir.cleanup()

    def survey(self):
        sim = Simulation(self.cfg)
        while sim.console.phase == "exploring" and sim.tick < 1000:
            sim.step()
        return sim

    def test_progress_is_boundary_closure_from_the_map_alone(self):
        from sim.mapping import OccupancyMap
        m = OccupancyMap()
        self.assertEqual(m.closure(), 0.0)  # nothing known yet
        for x in range(10, 14):  # a 4x4 floor patch, walled on the west, open everywhere else
            for y in range(10, 14):
                m.l[y][x] = -4.0
        for y in range(10, 14):
            m.l[y][9] = 2.5
        self.assertAlmostEqual(m.closure(min_unknown=1), 4 / 16)
        sim = Simulation(self.cfg)
        seen = []
        while sim.console.phase == "exploring" and sim.tick < 1000:
            sim.step()
            seen.append(sim.console.closure)
        self.assertGreater(max(seen), 0.95)  # nearly closed when the last frontier goes
        self.assertEqual(sim.console.map_view()["closure"], round(sim.console.closure, 3))

    def test_survey_maps_the_floor_and_saves_it(self):
        sim = self.survey()
        c, w = sim.console, sim.world
        self.assertEqual(c.phase, "done")
        floor = {(x, y) for x in range(48) for y in range(32) if w.is_floor((x, y))}
        free = set(c.map.free_cells())
        self.assertFalse(free - floor, "a wall was mapped as floor")
        self.assertGreater(len(free & floor) / len(floor), 0.98)
        self.assertTrue(all(c.map.is_occupied(x) for x in w.walls))
        self.assertTrue(c.map.is_occupied((21, 16)))  # the box in aisle B is on the map...
        self.assertFalse(c.map.is_occupied((34, 23)))  # ...the cable on the floor is below the lidar
        self.assertEqual(sim.collisions, 0)
        import os
        self.assertTrue(os.path.exists(self.cfg["mapping"]["file"]))
        self.assertEqual(c.pending_zones, ["促銷區", "入口"])
        self.assertNotIn("入口", c.zones)

    def test_no_cleaning_while_mapping_then_cleaning_starts(self):
        sim = Simulation(self.cfg)
        run(sim, 40)
        self.assertEqual(sim.console.phase, "exploring")
        self.assertTrue(sim.console.known_dirt)  # cameras note dirt on the way...
        self.assertFalse([t for t in sim.console.tasks.values() if t.kind != "explore" and t.status == "assigned"])
        while sim.console.phase == "exploring":
            sim.step()
        run(sim, 30)
        self.assertTrue(any(t.kind in ("sweep", "mop") and t.status in ("assigned", "done")
                            for t in sim.console.tasks.values()))  # ...and it is cleaned once the map is done

    def test_saved_map_is_loaded_and_zones_are_labelled_by_hand(self):
        self.survey()
        sim = Simulation(self.cfg)  # second start: no survey
        c = sim.console
        self.assertEqual(c.phase, "done")
        self.assertTrue(c.map.is_free((44, 26)))
        run(sim, 10, [(1, "M1", "去拖入口")])
        self.assertIn("地圖上還沒有入口", c.chat[-1]["text"])
        ok, _ = c.label_zone("門口", [41, 23, 46, 30])  # alias of 入口
        self.assertTrue(ok)
        self.assertEqual(c.zone_of((44, 26)), "入口")
        bad, msg = c.label_zone("廁所", [1, 1, 2, 2])
        self.assertFalse(bad)
        run(sim, 10, [(sim.tick, "M1", "去拖入口")])
        self.assertTrue(any(t.zone == "入口" and t.kind == "mop" for t in c.tasks.values()))
        reloaded = Simulation(self.cfg).console  # the label was saved with the map
        self.assertIn("入口", reloaded.zones)
        self.assertEqual(reloaded.pending_zones, ["促銷區"])

    def test_map_forgets_a_box_after_the_arm_moves_it(self):
        sim = self.survey()
        self.assertTrue(sim.console.map.is_occupied((21, 16)))
        run(sim, 400)
        self.assertNotIn((21, 16), sim.world.obstacle_cells)
        self.assertFalse(sim.console.map.is_occupied((21, 16)))


class RealLidarTest(unittest.TestCase):
    """scan_from_ranges turns a Webots range image + pose into map cells."""

    def sweep(self, cell, yaw, noise=0.0, seed=0):
        import math
        import random
        from sim.world import CELL_M
        w = World()
        rng = random.Random(seed)
        gx, gy = cell[0] + 0.5, cell[1] + 0.5
        n, fov, ranges = 180, 2 * math.pi, []
        for i in range(n):  # ray-march the true walls, beams at slice centres like Webots
            a = yaw + fov / 2 - (i + 0.5) * fov / n
            dx, dy, r = math.cos(a), -math.sin(a), -1
            for k in range(1, 8 * 64):
                if (math.floor(gx + dx * k / 64), math.floor(gy + dy * k / 64)) in w.walls:
                    r = k / 64 * CELL_M + rng.gauss(0, noise)
                    break
            ranges.append(r if 0 < r < 4 else -1)
        x, y = (gx - 24) * CELL_M, (16 - gy) * CELL_M
        return w, (x, y, yaw, ranges, fov, 4.0)

    def test_walls_and_floor_land_in_the_right_cells(self):
        from sim.mapping import OccupancyMap, scan_from_ranges
        m = OccupancyMap()
        for k, (cell, yaw) in enumerate([((16, 15), 0.3), ((10, 22), 2.0), ((30, 9), -1.2), ((40, 26), 0.9)]):
            w, args = self.sweep(cell, yaw, noise=0.02, seed=k)
            free, hits, _ = scan_from_ranges(*args)
            self.assertFalse(free & w.walls)
            m.update(free, hits)
        self.assertFalse([c for c in w.walls if m.is_free(c)])
        self.assertFalse([(x, y) for x in range(48) for y in range(32) if m.is_occupied((x, y)) and (x, y) not in w.walls])

    def test_returns_on_other_robots_are_not_walls(self):
        from sim.mapping import scan_from_ranges
        w, args = self.sweep((16, 15), 0.0)
        ranges = list(args[3])
        ranges[0] = 0.6  # something 0.6 m away along the first beam
        x, y, yaw = args[0], args[1], args[2]
        free, hits, _ = scan_from_ranges(x, y, yaw, ranges, args[4], args[5], dynamic=lambda px, py: True)
        self.assertFalse(hits)  # every return was "on a robot": free space only


class FalseAlarmTest(unittest.TestCase):
    """Camera detections are not trusted blindly (YOLO in Webots reports some dirt that isn't there)."""

    def test_close_look_withdraws_and_far_repeats_are_ignored(self):
        sim = Simulation()
        c = sim.console
        cell = (30, 15)  # clean floor in the promo zone
        c._on_dirt_detected("S1", cell, "wet", 0.62, False)
        self.assertIn(cell, c.known_dirt)
        c._on_no_dirt("M1", [cell])
        self.assertNotIn(cell, c.known_dirt)
        c._on_dirt_detected("S2", cell, "wet", 0.65, False)   # same false alarm from far away
        self.assertNotIn(cell, c.known_dirt)
        c._on_dirt_detected("S2", cell, "wet", 0.9, True)     # a close look still counts
        self.assertIn(cell, c.known_dirt)

    def test_detection_on_a_wall_is_dropped(self):
        sim = Simulation()
        sim.console._on_dirt_detected("S1", (3, 15), "dry", 0.8, False)  # (3, 15) is a shelf
        self.assertNotIn((3, 15), sim.console.known_dirt)

    def test_robot_reports_missing_dirt_next_to_it(self):
        sim = Simulation()
        r = sim.robot("S1")
        ghost = (r.pos[0] + 1, r.pos[1] - 1)
        sim.console.known_dirt[ghost] = {"label": "dry", "conf": 0.6}
        run(sim, 3)
        self.assertNotIn(ghost, sim.console.known_dirt)


class AutoZoneTest(unittest.TestCase):
    def test_one_click_labels_every_pending_zone(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            sim = Simulation({"mapping": {"mode": "explore", "file": f"{d}/m.json", "manual_zones": ["促銷區", "入口"]}})
            while sim.console.phase == "exploring":
                sim.step()
            self.assertEqual(sim.console.label_pending_zones(), ["促銷區", "入口"])
            self.assertEqual(sim.console.pending_zones, [])
            self.assertEqual(sim.console.zone_of((44, 26)), "入口")
