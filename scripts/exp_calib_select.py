"""exp_calib: nested leave-one-benchmark-out selection of (budget-dependent) shrinkage settings from a grid json
(scripts/exp_calib_grid.py). Reports: baseline config, best single global config (nested), per-budget config (nested), and the
per-benchmark ORACLE choice (cheating upper bound on what per-benchmark empirical-Bayes tuning could buy)."""
import sys, json, numpy as np
W = {0: 0.1, 1: 0.2, 3: 0.2, 7: 0.2, 15: 0.2, 31: 0.1}; KS = [0, 1, 3, 7, 15, 31]
R = json.load(open(sys.argv[1])); base = sys.argv[2] if len(sys.argv) > 2 else None
cfgs = list(R); benches = sorted(R[cfgs[0]])
B = np.array([[[R[c][b][str(k)] for k in KS] for b in benches] for c in cfgs])  # cfg x bench x budget
w = np.array([W[k] for k in KS]); alc = B @ w  # cfg x bench
def show(lbl, per):
    print(f"{lbl:48s} mean {np.mean(per):.4f} | " + " ".join(f"{b[:8]} {v:.4f}" for b, v in zip(benches, per)))
if base:
    show(f"baseline [{base}]", alc[cfgs.index(base)])
glob = [alc[np.argmin(np.delete(alc, h, 1).mean(1)), h] for h in range(len(benches))]
show("nested: one global config", glob)
print("   chosen:", [cfgs[np.argmin(np.delete(alc, h, 1).mean(1))] for h in range(len(benches))])
per_budget = []
for h in range(len(benches)):
    tot = 0.0; ch = []
    for kk in range(len(KS)):
        c = np.argmin(np.delete(B[:, :, kk], h, 1).mean(1)); tot += w[kk] * B[c, h, kk]; ch.append(cfgs[c])
    per_budget.append(tot)
    if h == 0:
        print("   per-budget choice (held-out 0):", ch)
show("nested: per-budget config", per_budget)
show("ORACLE per-benchmark config (cheating)", alc.min(0))
show("ORACLE per-benchmark per-budget (cheating)", (B.min(0) * w).sum(1))
print("   best global (in-sample):", cfgs[np.argmin(alc.mean(1))], f"{alc.mean(1).min():.4f}")
