"""exp_pooled_sweep (tag 'pooled'): run many model configs x regimes x seeds in one process with exp_pooled_cv.evaluate.
  .venv/bin/python scripts/exp_pooled_sweep.py --grid NAME --regimes pooled,single --seeds 0,1,2
Grids are defined below; every line printed as RESULT is one actual run."""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import exp_pooled_cv as H  # noqa: E402

GRIDS = {
    "first": [("v5", {}, "random"), ("v5_betaprior", {"beta_prior": True}, "random"),
              ("eb_s2", {"eb_s": 2.0}, "random"), ("eb_s5", {"eb_s": 5.0}, "random"), ("eb_s10", {"eb_s": 10.0}, "random"),
              ("eb_i5", {"eb_i": 5.0}, "random"), ("eb_i20", {"eb_i": 20.0}, "random"),
              ("probit0.25", {"probit": 0.25}, "random"), ("probit0.5", {"probit": 0.5}, "random"), ("probit1", {"probit": 1.0}, "random"),
              ("cg0.5", {"c_g": 0.5}, "random"), ("cg1", {"c_g": 1.0}, "random"), ("cg2", {"c_g": 2.0}, "random"),
              ("sim0.5", {"sim": 0.5}, "random"), ("sim1", {"sim": 1.0}, "random"), ("sim2", {"sim": 2.0}, "random"),
              ("cover", {}, "cover"), ("cover2", {}, "cover2"), ("anchor", {}, "anchor")],
}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--grid", default="first"); ap.add_argument("--regimes", default="pooled,single")
    ap.add_argument("--seeds", default="0"); ap.add_argument("--only", default=""); ap.add_argument("--extra", default="")
    ap.add_argument("--pair-order", default="roundrobin")
    a = ap.parse_args()
    grid = list(GRIDS.get(a.grid, []))
    if a.extra:  # JSON list of [name, cfg, acq]
        grid += [tuple(x) for x in json.loads(a.extra)]
    only = [x for x in a.only.split(",") if x] or None
    seeds = [int(s) for s in a.seeds.split(",")]
    table = {}
    for seed in seeds:
        benches = H.prepare(seed, 0.5)
        for name, cfg, acq in grid:
            for regime in a.regimes.split(","):
                if acq != "random" and regime == "single":
                    continue
                t0 = time.time()
                mean, pairw, res = H.evaluate(benches, cfg, regime, seed, acq, only, verbose=False, pair_order=a.pair_order)
                table.setdefault((name, regime), []).append((mean, {h: r["alc"] for h, r in res.items()}))
                print(f"RESULT {name:14s} seed={seed} {regime:7s} acq={acq:7s} ALC={mean:.4f} pairw={pairw:.4f} | " +
                      " ".join(f"{h[:8]}={r['alc']:.4f}" for h, r in res.items()) + f"  cfg={json.dumps(cfg, sort_keys=True)} ({time.time()-t0:.0f}s)", flush=True)
    print("\nSUMMARY (mean over seeds; per-bench mean over seeds)")
    for (name, regime), v in table.items():
        m = np.mean([x[0] for x in v]); hs = v[0][1].keys()
        print(f"{name:14s} {regime:7s} ALC={m:.4f} (n_seeds={len(v)}) | " + " ".join(f"{h[:8]}={np.mean([x[1][h] for x in v]):.4f}" for h in hs))


if __name__ == "__main__":
    main()
