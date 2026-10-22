#!/usr/bin/env python
"""Summarise protocol-simulator runs (src/paie_exp/sim/results/sim_*.json) and make PAIRED comparisons between runs that used the same
episodes (same seed / regime / held-out benchmark / episode index => identical pairs, splits and label streams).
  .venv/bin/python scripts/exp_sim_report.py                      # table of every run
  .venv/bin/python scripts/exp_sim_report.py --pair A B [regime]  # per-benchmark paired difference B - A (tags without sim_ / .json)
"""
import argparse, json, sys
from pathlib import Path
import numpy as np

RES = Path(__file__).resolve().parents[1] / "src" / "paie_exp" / "sim" / "results"


def load(tag):
    return json.loads((RES / f"sim_{tag}.json").read_text())


def table():
    for f in sorted(RES.glob("sim_*.json")):
        j = json.loads(f.read_text())
        for reg, v in j["regimes"].items():
            s = v["summary"]
            pb = " ".join(f"{b[:6]}={x['alc']:.4f}" for b, x in s["per_bench"].items())
            print(f"{f.stem[4:]:32s} {reg:9s} macro {s['macro']:.4f} micro {s['micro']:.4f} | {pb}")


def paired(a, b, regime):
    A, B = load(a)["regimes"][regime]["pairs"], load(b)["regimes"][regime]["pairs"]
    assert len(A) == len(B), (len(A), len(B))
    by = {}
    for x, y in zip(A, B):
        assert (x["bench"], x["sid"]) == (y["bench"], y["sid"]), "runs do not share episodes"
        by.setdefault(x["bench"], []).append(y["alc"] - x["alc"])
    print(f"paired {b} - {a} [{regime}] (negative = {b} better)")
    means = []
    for bench, d in sorted(by.items()):
        d = np.array(d); means.append(d.mean())
        se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else 0.0
        print(f"   {bench:18s} n {len(d):4d}  mean diff {d.mean():+.4f} +- {se:.4f}")
    print(f"   macro mean diff {np.mean(means):+.4f}; benchmarks where {b} better: {sum(m < 0 for m in means)}/{len(means)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--pair", nargs="+"); a = ap.parse_args()
    if a.pair:
        paired(a.pair[0], a.pair[1], a.pair[2] if len(a.pair) > 2 else "practice")
    else:
        table()
