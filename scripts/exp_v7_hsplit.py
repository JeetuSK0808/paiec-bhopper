"""exp_v7_hsplit (tag 'v7'): hidden-calibrated simulator results split by pair type, plus the per-upload score distribution.

Uses the episodes of scripts/exp_synth_hiddensim.py (the three practice samples we were scored on: 9 pairs over 7 hidden benchmarks,
two benchmarks with 2 systems). Reports, for v6 and the level-only baselines, ALC and Brier by budget separately for
  * lone pairs   (the only system of its benchmark in the sample; 5 of 9 pairs), and
  * shared pairs (2 systems on the benchmark; 4 of 9 pairs).
Also reports the distribution of a whole-sample ALC (mean over the sample's pairs, as the board scores one upload) and the expected
best-of-N over independent uploads.
  nice -n 10 env VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -B scripts/exp_v7_hsplit.py --episodes 60 --seeds 0,1,2
"""
import argparse, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "src"))
sys.dont_write_bytecode = True
import exp_synth_hiddensim as hs  # noqa: E402

W = np.array([.1, .2, .2, .2, .2, .1])


def boot_rows(rng, tag):
    """A new 9-pair sample in the board's structure: every hidden benchmark once plus 2 benchmarks twice; each pair's (accuracy, eval
    size) drawn from that benchmark's observed pairs in our dev logs."""
    by = {}
    for h in hs.H:
        by.setdefault(h[2], []).append(h)
    names = sorted(by)
    extra = rng.choice(len(names), size=2, replace=False)
    rows = []
    for j, b in enumerate(names):
        for r in range(2 if j in extra else 1):
            h = by[b][int(rng.integers(len(by[b])))]
            rows.append((tag, f"{b}_{r}", b, h[3], h[4], h[5]))
    return rows


def run_emp(ep):
    """The organisers' empirical-mean baseline: own-label mean, 0.5 without labels (unsmoothed)."""
    out = []
    for bd in ep:
        for p in bd["pairs"]:
            b = []
            for k in hs.BUDGETS:
                n = min(k, len(p["acq_y"])); q = 0.5 if n == 0 else float(p["acq_y"][:n].mean())
                b.append(np.mean((q - p["ev_y"]) ** 2))
            out.append(np.array(b))
    return out


def pair_kinds(ep):
    return [len(bd["pairs"]) > 1 for bd in ep for _ in bd["pairs"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=60); ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--tau-true", type=float, default=1.5); ap.add_argument("--sub", default="submission_v6")
    ap.add_argument("--boot", action="store_true", help="fresh bootstrap samples in the board structure instead of the 3 dev samples")
    a = ap.parse_args()
    M = hs.load_sub({"sub": a.sub})
    cands = {"oracle": lambda ep: hs.run_oracle(ep), "beta1": lambda ep: hs.run_beta(ep, 1.0),
             "beta0.7": lambda ep: hs.run_beta(ep, 0.7), "emp_mean": run_emp, a.sub: lambda ep: hs.run_sub(ep, M)}
    res = {n: {"lone": [], "shared": [], "sample": []} for n in cands}
    for seed in map(int, a.seeds.split(",")):
        for sample in (("boot",) if a.boot else ("v3", "v4", "v5")):
            for e in range(a.episodes * (3 if a.boot else 1)):
                rng = np.random.default_rng([seed, {"v3": 3, "v4": 4, "v5": 5, "boot": 9}[sample], e, int(a.tau_true * 100)])
                if sample == "boot":
                    tag = f"boot_{seed}_{e}"
                    hs.H[:] = [h for h in hs.H if not str(h[0]).startswith("boot")] + boot_rows(rng, tag)
                    ep = hs.make_episode(tag, rng, a.tau_true)
                else:
                    ep = hs.make_episode(sample, rng, a.tau_true)
                kinds = pair_kinds(ep)
                for n, f in cands.items():
                    r = f(ep)
                    for b, sh in zip(r, kinds):
                        res[n]["shared" if sh else "lone"].append(np.asarray(b))
                    res[n]["sample"].append(float(np.mean([np.asarray(b) @ W for b in r])))
    print(f"hidden-calibrated simulator, tau_true {a.tau_true}, {a.episodes} episodes x 3 samples x seeds {a.seeds}")
    for n in cands:
        for k in ("lone", "shared"):
            B = np.array(res[n][k])
            print(f"  {n:16s} {k:6s} pairs {len(B):5d}  ALC {float(np.mean(B @ W)):.4f}   B0..B31 " + " ".join(f"{x:.4f}" for x in B.mean(0)))
        s = np.array(res[n]["sample"])
        rng = np.random.default_rng(123)
        bon = {N: float(np.mean(np.min(rng.choice(s, size=(4000, N)), axis=1))) for N in (1, 6, 20, 40, 100)}
        print(f"  {n:16s} sample ALC mean {s.mean():.4f} sd {s.std():.4f}  expected best-of-N: " +
              " ".join(f"N={N}:{v:.4f}" for N, v in bon.items()))
    ref = np.array(res[a.sub]["sample"])
    for n in cands:
        d = np.array(res[n]["sample"]) - ref
        print(f"  paired per-sample difference {n} - {a.sub}: mean {d.mean():+.4f} sd {d.std():.4f}  P(diff<0) {np.mean(d < 0):.2f}")


if __name__ == "__main__":
    main()
