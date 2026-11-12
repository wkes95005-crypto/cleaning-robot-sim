"""Task planners for the console: turn a spoken command into a structured plan.

Every planner returns the same JSON shape, so the backend is chosen in config.toml and can be
swapped without touching the dispatcher:

    {"steps": [{"action": "clean_zone", "zone": "生鮮區", "mode": "sweep_then_mop"},
               {"action": "avoid", "object": "cable"},
               {"action": "clear_obstacles", "zone": "貨架B區"},
               {"action": "dock"}],
     "parallel": false,
     "who": "self"}          # self = only the robot that heard it; all = the whole fleet

The LLM only plans. It never drives motors; A* and the robots' controllers do the moving.
"""
from __future__ import annotations

import json
import re
import urllib.request

from .world import OBSTACLE_KINDS, ZONES, find_zones, resolve_zone

ACTIONS = {"clean_zone", "avoid", "clear_obstacles", "dock", "stop", "resume"}
MODES = {"sweep", "mop", "sweep_then_mop"}

OBJECT_WORDS = {"電線": "cable", "延長線": "cable", "線材": "cable", "紙箱": "box", "箱子": "box",
                "推車": "cart", "購物車": "cart", "棧板": "pallet"}


def validate(data):
    """Keep only steps the dispatcher understands; map zone aliases to canonical names."""
    steps = []
    for s in (data or {}).get("steps", []):
        a = s.get("action")
        if a not in ACTIONS:
            continue
        step = {"action": a}
        if a in ("clean_zone", "clear_obstacles"):
            zone = s.get("zone")
            step["zone"] = zone if zone in ZONES else resolve_zone(zone or "")
        if a == "clean_zone":
            step["mode"] = s.get("mode") if s.get("mode") in MODES else "sweep_then_mop"
        if a == "avoid":
            obj = s.get("object")
            step["object"] = obj if obj in OBSTACLE_KINDS else OBJECT_WORDS.get(obj, obj)
        steps.append(step)
    who = "all" if (data or {}).get("who") == "all" else "self"
    return {"steps": steps, "parallel": bool((data or {}).get("parallel")), "who": who}


class RulePlanner:
    """Keyword planner. No model needed; also the fallback when the LLM is unreachable."""

    name = "rule"
    synchronous = True

    def plan(self, text, context=None):
        steps, last_zone = [], None
        for clause in re.split(r"[，,。；;！!？?\n]|然後|接著|之後|最後|再來", text):
            clause = clause.strip()
            if not clause:
                continue
            # "貨架B區有紙箱擋路，請搬開" -> the second clause still means zone B
            zones = find_zones(clause) or ([last_zone] if last_zone else [None])
            last_zone = zones[-1]
            if any(k in clause for k in ("避開", "小心", "注意", "不要碰", "別碰", "繞過", "繞開")):
                objs = [v for k, v in OBJECT_WORDS.items() if k in clause]
                steps.append({"action": "avoid", "object": objs[0] if objs else "unknown"})
                continue
            if any(k in clause for k in ("搬開", "移開", "搬走", "挪開", "擋路", "清除障礙", "障礙物")):
                steps += [{"action": "clear_obstacles", "zone": z} for z in zones]
                continue
            if any(k in clause for k in ("回充", "充電", "收工", "待命", "回去")):
                steps.append({"action": "dock"})
                continue
            if any(k in clause for k in ("停下", "停止", "暫停", "別動", "不要動", "等一下")):
                steps.append({"action": "stop"})
                continue
            if any(k in clause for k in ("繼續", "恢復", "開始工作", "回去工作")):
                steps.append({"action": "resume"})
                continue
            sweep = any(k in clause for k in ("掃", "吸"))
            mop = "拖" in clause
            general = any(k in clause for k in ("清理", "清潔", "打掃", "處理", "弄乾淨", "打翻", "灑", "潑"))
            if not (sweep or mop or general):
                continue
            mode = "sweep_then_mop" if general or (sweep and mop) else ("mop" if mop else "sweep")
            steps += [{"action": "clean_zone", "zone": z, "mode": mode} for z in zones]
        deduped = [s for i, s in enumerate(steps) if i == 0 or s != steps[i - 1]]
        who = "all" if any(k in text for k in ("全部", "大家", "所有", "全體", "每台")) else "self"
        return validate({"steps": deduped, "parallel": "同時" in text, "who": who})


SYSTEM_PROMPT = """你是賣場清潔機器人中控台的任務規劃器。使用者是對其中一台機器人說話（heard_by，角色 heard_by_role），
把指令轉成 JSON，不要輸出其他文字。
格式：{"steps":[...],"parallel":false,"who":"self"}
who：指令是對這台機器人說的用 "self"；說「全部、大家、所有機器人」時用 "all"。
可用動作：
- {"action":"clean_zone","zone":<區域>,"mode":"sweep"|"mop"|"sweep_then_mop"}  掃地、拖地或先掃後拖
- {"action":"avoid","object":"cable"|"box"|"cart"|"pallet"}  之後都要避開這類物體
- {"action":"clear_obstacles","zone":<區域或 null>}  派機械手臂機器人把紙箱、推車等障礙物搬到倉儲區（電線不能搬，只會繞開）
- {"action":"dock"}  回充電站待命
- {"action":"stop"}  停下來，原地待命
- {"action":"resume"}  恢復工作
區域只能用：%s。指令沒說區域時 zone 用 null，系統會改用聽到指令那台機器人所在的區域。
steps 依指令的先後順序排列；只有使用者說「同時」時 parallel 才是 true。"""


class OllamaPlanner:
    """Any model served by a local Ollama (Gemma 3, Llama 3.x, Phi-4, Granite ...); the model name is config."""

    synchronous = False

    def __init__(self, model, url="http://localhost:11434", timeout=60):
        self.model, self.url, self.timeout = model, url.rstrip("/"), timeout
        self.name = f"ollama:{model}"
        self.fallback = RulePlanner()

    def plan(self, text, context=None):
        body = {
            "model": self.model,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0},
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT % "、".join(ZONES)},
                {"role": "user", "content": json.dumps({"command": text, **(context or {})}, ensure_ascii=False)},
            ],
        }
        req = urllib.request.Request(f"{self.url}/api/chat", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                content = json.loads(resp.read())["message"]["content"]
            plan = validate(json.loads(content))
            if plan["steps"]:
                return plan
            note = "LLM 回傳空計畫"
        except Exception as e:  # unreachable server, model missing, malformed JSON
            note = f"LLM 失敗（{type(e).__name__}）"
        plan = self.fallback.plan(text, context)
        plan["note"] = f"{note}，改用規則規劃器"
        return plan


def make_planner(cfg):
    if cfg.get("backend") == "ollama":
        return OllamaPlanner(cfg.get("model", "gemma3:4b"), cfg.get("url", "http://localhost:11434"),
                             cfg.get("timeout", 60))
    return RulePlanner()
