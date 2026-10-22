"""Paired comparison of two faithful-simulator runs stored in src/paie_exp/synth/sim/results (same episodes => identical pairs, splits,
label streams). Prints per-benchmark ALC of both, paired ALC difference (+- SE), and per-budget mean Brier of both.
  .venv/bin/python scripts/exp_synth_simreport.py A B regime"""
import json, sys
from pathlib import Path
import numpy as np
RES = Path(__file__).resolve().parents[1] / "src" / "paie_exp" / "synth" / "sim" / "results"
BUD = ["0", "1", "3", "7", "15", "31"]


def main():
    a, b, reg = sys.argv[1], sys.argv[2], sys.argv[3]
    A = json.loads((RES / f"sim_{a}.json").read_text())["regimes"][reg]["pairs"]
    B = json.loads((RES / f"sim_{b}.json").read_text())["regimes"][reg]["pairs"]
    assert len(A) == len(B)
    by = {}
    for x, y in zip(A, B):
        assert (x["bench"], x["sid"]) == (y["bench"], y["sid"])
        by.setdefault(x["bench"], []).append((x, y))
    print(f"[{reg}] {a} -> {b}   (negative diff = {b} better)")
    ma, mb, md = [], [], []
    for bench, lst in sorted(by.items()):
        xa = np.array([x["alc"] for x, _ in lst]); xb = np.array([y["alc"] for _, y in lst]); d = xb - xa
        se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else 0.0
        ma.append(xa.mean()); mb.append(xb.mean()); md.append(d.mean())
        print(f"   {bench:18s} n {len(d):4d}  {xa.mean():.4f} -> {xb.mean():.4f}  diff {d.mean():+.4f} +- {se:.4f}")
    print(f"   MACRO {np.mean(ma):.4f} -> {np.mean(mb):.4f}  diff {np.mean(md):+.4f}; {b} better on {sum(m < 0 for m in md)}/{len(md)} benchmarks")
    allx = [x for lst in by.values() for x, _ in lst]; ally = [y for lst in by.values() for _, y in lst]
    ka = [np.mean([p["B"][k] if k in p["B"] else p["B"][int(k)] for p in allx]) for k in BUD]
    kb = [np.mean([p["B"][k] if k in p["B"] else p["B"][int(k)] for p in ally]) for k in BUD]
    print("   per-budget Brier (pair-weighted) " + " ".join(f"B{k}:{u:.4f}->{v:.4f}" for k, u, v in zip(BUD, ka, kb)))


if __name__ == "__main__":
    main()
