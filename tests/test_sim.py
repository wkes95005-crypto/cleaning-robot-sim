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


if __name__ == "__main__":
    unittest.main()
