"""Tables for docs/V6_NOTES.md from the synth logs: 3-seed harness means per regime and benchmark (v5 shipped vs v6 final), and
paired faithful-simulator summaries (macro = mean over held-out benchmarks, micro = mean over pairs, as the official metric)."""
import json, re, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; L = ROOT / "src/paie_exp/synth"
BN = ["matharena", "mmdocrag", "multi_swebench", "real_webagents", "researchcodebench", "swe_rebench"]


def parse_cv(path, name_prefix):
    out = {}
    for line in open(path):
        if not line.startswith(name_prefix) or "seed=" not in line or "ALC" not in line:
            continue
        m = re.search(r"seed=(\d+)\s+(\w+)\s+ALC ([0-9.]+)", line)
        seed, reg, alc = int(m.group(1)), m.group(2), float(m.group(3))
        pb = {k[:6]: float(v) for k, v in re.findall(r"(\w+):([0-9.]+)", line.split("ALC")[1])}
        out[(reg, seed)] = (alc, pb)
    return out


def cv_table(v5, v6):
    regs = ["single", "mix2", "mix8", "pooled"]
    print("| regime | v5 shipped (3 seeds) | v6 (3 seeds) | diff | seeds v6 better | " + " | ".join(BN) + " |")
    print("|" + "---|" * (5 + len(BN)))
    for r in regs:
        seeds = sorted(s for (rr, s) in v6 if rr == r)
        a5 = np.mean([v5[(r, s)][0] for s in seeds]); a6 = np.mean([v6[(r, s)][0] for s in seeds])
        nb = sum(v6[(r, s)][0] < v5[(r, s)][0] for s in seeds)
        cells = []
        for b in BN:
            k = b[:6]
            x5 = np.mean([v5[(r, s)][1][k] for s in seeds]); x6 = np.mean([v6[(r, s)][1][k] for s in seeds])
            cells.append(f"{x5:.4f} → {x6:.4f}")
        print(f"| {r} | {a5:.4f} | {a6:.4f} | {a6 - a5:+.4f} | {nb}/{len(seeds)} | " + " | ".join(cells) + " |")
    # without swe_rebench (single-system benchmark that is a lone pair in every regime)
    for r in regs:
        seeds = sorted(s for (rr, s) in v6 if rr == r)
        f = lambda d, s: np.mean([d[(r, s)][1][b[:6]] for b in BN[:5]])
        print(f"   {r:7s} 5 multi-system benchmarks (no swe_rebench): v5 {np.mean([f(v5, s) for s in seeds]):.4f} -> v6 {np.mean([f(v6, s) for s in seeds]):.4f}"
              f"  per seed diff " + " ".join(f"{f(v6, s) - f(v5, s):+.4f}" for s in seeds))


def sim_summary(a, b, reg):
    A = json.loads((L / "sim/results" / f"sim_{a}.json").read_text())["regimes"][reg]
    B = json.loads((L / "sim/results" / f"sim_{b}.json").read_text())["regimes"][reg]
    sa, sb = A["summary"], B["summary"]
    d = {}
    for x, y in zip(A["pairs"], B["pairs"]):
        d.setdefault(x["bench"], []).append(y["alc"] - x["alc"])
    better = sum(np.mean(v) < 0 for v in d.values())
    return (f"| {reg} | {sa['macro']:.4f} | {sb['macro']:.4f} | {sb['macro'] - sa['macro']:+.4f} | {sa['micro']:.4f} | {sb['micro']:.4f} | "
            f"{sb['micro'] - sa['micro']:+.4f} | {better}/{len(d)} | " + " | ".join(f"{np.mean(d[bn]):+.4f}" for bn in BN) + " |")


if __name__ == "__main__":
    v5 = parse_cv(L / "log_cv_v5_seeds012.txt", "v5shipped")
    v6 = parse_cv(L / sys.argv[1] if len(sys.argv) > 1 else L / "log_cv_v6final_seeds012.txt", sys.argv[2] if len(sys.argv) > 2 else "v6final")
    cv_table(v5, v6)
    tag = sys.argv[3] if len(sys.argv) > 3 else "v6final"
    print("\n| regime | v5 macro | v6 macro | diff | v5 micro | v6 micro | diff | benchmarks v6 better | " + " | ".join(BN) + " |")
    print("|" + "---|" * (8 + len(BN)))
    print(sim_summary("v5_practice", f"{tag}_practice", "practice"))
    for r in ("k1", "k2", "k4", "k8"):
        print(sim_summary("v5_k", f"{tag}_k", r))
