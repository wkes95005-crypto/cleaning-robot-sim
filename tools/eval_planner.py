"""Compare task planners on spoken-style commands, including ones with no obvious keywords.

    python3 tools/eval_planner.py                 # rule planner + ollama gemma3:4b
    python3 tools/eval_planner.py gemma3:12b      # other Ollama models

A case passes when the plan has exactly the expected steps (in order), and the expected who/parallel.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sim.planner import OllamaPlanner, RulePlanner  # noqa: E402


def clean(zone, mode):
    return {"action": "clean_zone", "zone": zone, "mode": mode}


CASES = [
    # the voice test sentences (as Whisper transcribed them)
    ("去掃貨架A區。", [clean("貨架A區", "sweep")]),
    ("入口那邊地板濕濕的,幫我拖一下。", [clean("入口", "mop")]),
    ("收銀台前面好像有人打翻飲料,還有玻璃碎片。", [clean("收銀區", "sweep_then_mop")]),
    ("先清理生鮮區、避開地上的電線,最後再掃收銀區。",
     [clean("生鮮區", "sweep_then_mop"), {"action": "avoid", "object": "cable"}, clean("收銀區", "sweep")]),
    ("貨架B區有紙箱擋道路了,叫手臂去搬。", [{"action": "clear_obstacles", "zone": "貨架B區"}]),
    ("大家先停下來。", [{"action": "stop"}], {"who": "all"}),
    ("好了,繼續工作吧。", [{"action": "resume"}]),
    ("電快沒了就回去充電。", [{"action": "dock"}]),
    # meaning without the usual keywords
    ("這邊黏黏的", [clean(None, "sweep_then_mop")]),
    ("冷凍區那邊冰融化了一灘水", [clean("冷凍區", "mop")]),
    ("促銷區地上都是紙屑", [clean("促銷區", "sweep")]),
    ("先不要動", [{"action": "stop"}]),
    ("今天收工了，所有人都回去", [{"action": "dock"}], {"who": "all"}),
    ("主走道有台推車卡在中間", [{"action": "clear_obstacles", "zone": "主走道"}]),
    ("生鮮區跟入口一起拖，兩邊同時", [clean("生鮮區", "mop"), clean("入口", "mop")], {"parallel": True}),
    ("那條延長線不要碰到，繞過去", [{"action": "avoid", "object": "cable"}]),
]


def check(plan, steps, extra):
    if plan["steps"] != steps:
        return False
    return plan["who"] == extra.get("who", "self") and plan["parallel"] == extra.get("parallel", False)


def run(planner, context):
    ok, times = 0, []
    print(f"\n== {planner.name}")
    for case in CASES:
        text, steps, extra = case[0], case[1], (case[2] if len(case) > 2 else {})
        t0 = time.perf_counter()
        plan = planner.plan(text, context)
        times.append(time.perf_counter() - t0)
        good = check(plan, steps, extra)
        ok += good
        shown = "; ".join(" ".join(str(v) for v in s.values()) for s in plan["steps"]) or "（空）"
        flags = ("" if plan["who"] == "self" else " who=all") + (" 同時" if plan["parallel"] else "")
        note = f"  [{plan['note']}]" if plan.get("note") else ""
        print(f"  {'✓' if good else '✗'} {times[-1] * 1000:5.0f} ms  {text}  →  {shown}{flags}{note}")
    times.sort()
    print(f"  {ok}/{len(CASES)} correct, median {times[len(times) // 2] * 1000:.0f} ms, max {times[-1] * 1000:.0f} ms")
    return ok


if __name__ == "__main__":
    context = {"heard_by": "S1", "heard_by_role": "sweeper"}
    run(RulePlanner(), context)
    for model in sys.argv[1:] or ["gemma3:4b"]:
        p = OllamaPlanner(model)
        p.plan("去掃貨架A區", context)  # load the model before timing
        run(p, context)
