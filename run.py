"""Entry point.

    python3 run.py                 # dashboard at http://127.0.0.1:8780
    python3 run.py --headless 1500 # run without UI and print a summary
"""
import argparse
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def load_config(path):
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    with open(p, "rb") as f:
        return tomllib.load(f)


def headless(cfg, ticks):
    from sim.engine import Simulation

    sim = Simulation(cfg)
    for _ in range(ticks):
        sim.step()
    snap = sim.snapshot()
    for entry in snap["log"][-25:]:
        print(f"[{entry['tick']:5d}] {entry['cat']:<8} {entry['text']}")
    print(snap["stats"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.toml")
    ap.add_argument("--port", type=int, default=8780)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--headless", type=int, metavar="TICKS")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.headless:
        headless(cfg, args.headless)
    else:
        from sim.server import serve
        serve(cfg, args.host, args.port)


if __name__ == "__main__":
    main()
