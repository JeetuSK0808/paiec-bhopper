"""exp_synth_cv (tag 'synth'): run a list of model configs through the pooled agent's exact-reproduction harness
(scripts/exp_pooled_cv.py: item_prior_cv.py splits, fractional labels, leave-one-benchmark-out, per-fold subject prior).
  .venv/bin/python scripts/exp_synth_cv.py --regimes single,pooled,mix2 --seeds 0 --cfgs '[["name", {...}], ...]'
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "src"))
import exp_pooled_cv as H  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfgs", required=True); ap.add_argument("--regimes", default="single,pooled"); ap.add_argument("--seeds", default="0")
    ap.add_argument("--perbench", action="store_true")
    a = ap.parse_args()
    cfgs = json.loads(a.cfgs); regimes = a.regimes.split(","); seeds = [int(s) for s in a.seeds.split(",")]
    tab = {}
    for seed in seeds:
        benches = H.prepare(seed, 0.5)
        for name, cfg in cfgs:
            for reg in regimes:
                t0 = time.time()
                mean, pairw, res = H.evaluate(benches, cfg, reg, seed, "random", None, False)
                tab[(name, reg, seed)] = (mean, {h: r["alc"] for h, r in res.items()})
                pb = " ".join(f"{h[:6]}:{r['alc']:.4f}" for h, r in res.items()) if a.perbench else ""
                print(f"{name:28s} seed={seed} {reg:7s} ALC {mean:.4f}  {pb} ({time.time()-t0:.0f}s)", flush=True)
    if len(seeds) > 1:
        print("--- mean over seeds")
        for name, _ in cfgs:
            print(f"{name:28s} " + "  ".join(f"{reg} {np.mean([tab[(name, reg, s)][0] for s in seeds]):.4f}" for reg in regimes))


if __name__ == "__main__":
    main()
