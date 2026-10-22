"""exp_calib: nested leave-one-benchmark-out post-hoc calibration on a dump from scripts/exp_calib_cv.py.
For each held-out benchmark H and budget k > 0, calibrator parameters are fitted on the OTHER benchmarks' evaluation rows at budget k
(objective: benchmark-equal mean Brier, or pair-equal with --weight pair) and applied to H. B0 stays 0.5 unless stated.
  .venv/bin/python scripts/exp_calib_offline.py src/paie_exp/calib/dump_single.npz [--weight bench|pair]"""
import sys, argparse, numpy as np
from scipy.optimize import minimize
sys.path.insert(0, "scripts")
from exp_calib_lib import load, fmt, sigmoid, logit, eb_posterior, TH, BUDGETS, WEIGHTS

ap = argparse.ArgumentParser(); ap.add_argument("dump"); ap.add_argument("--weight", default="bench"); ap.add_argument("--only", default="")
a = ap.parse_args()
D = load(a.dump); y = D["y"]; z = D["z"]; K = D["k"]; Bn = D["bench"]
V = D["vb"] + D["vs"] + D["vi"]; t = D["beta"] + D["delta"]; e = D["e"]
names = D["bench_names"]; benches = np.unique(Bn)
if a.weight == "pair":  # every pair equal regardless of benchmark (rows of a pair at one budget sum to 1/npairs_total)
    w = np.zeros(len(y))
    for k in BUDGETS:
        mk = K == k; cnt = np.bincount(D["pair"][mk], minlength=len(D["pair_names"]))
        w[mk] = 1.0 / cnt[D["pair"][mk]]
    W = w
else:
    W = D["w"]
IDX = {(bi, k): np.nonzero((Bn == bi) & (K == k))[0] for bi in benches for k in BUDGETS}


def nested(pred, x0, label, fix_b0=True, show=True, return_params=False):
    """pred(params, idx) -> probabilities for rows idx. Fits params per (held-out, k) on the other benchmarks."""
    p = sigmoid(z).copy(); P = {}
    for H in benches:
        for k in BUDGETS:
            if k == 0 and fix_b0:
                p[IDX[(H, 0)]] = 0.5; continue
            tr = np.concatenate([IDX[(bi, k)] for bi in benches if bi != H])
            wt = W[tr] / W[tr].sum(); yt = y[tr]
            f = lambda th: float(np.sum(wt * (pred(th, tr) - yt) ** 2))
            r = minimize(f, np.array(x0, float), method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-8, "maxiter": 2000})
            P[(H, k)] = r.x
            p[IDX[(H, k)]] = pred(r.x, IDX[(H, k)])
    m, s, alcs, pb = fmt(D, p, label)
    if show:
        print(s, flush=True)
    return (m, p, P) if return_params else (m, p)


def per_budget(p, label):
    m, s, alcs, pb = fmt(D, p, label)
    print(f"   {label:40s} " + " ".join(f"B{k}={np.mean([pb[bi][k] for bi in pb]):.4f}" for k in BUDGETS))


base = sigmoid(z)
print(fmt(D, base, "baseline (identity)")[1])
per_budget(base, "baseline")
# C1 Platt per budget; C2 temperature only
m1, p1, P1 = nested(lambda th, i: sigmoid(th[0] * z[i] + th[1]), [1.0, 0.0], "C1 Platt a_k z + b_k", return_params=True)
m2, p2, P2 = nested(lambda th, i: sigmoid(th[0] * z[i]), [1.0], "C2 temperature a_k z", return_params=True)
per_budget(p2, "C2")
print("   C2 fitted a_k (per held-out):", {names[H][:6]: [round(float(P2[(H, k)][0]), 3) for k in BUDGETS[1:]] for H in benches})
# C3 Laplace moderation (probit approximation of the posterior predictive)
m3, p3, P3 = nested(lambda th, i: sigmoid(z[i] / np.sqrt(1 + abs(th[0]) * V[i])), [np.pi / 8], "C3 moderation z/sqrt(1+lam V)", return_params=True)
print("   C3 fitted lam_k:", {names[H][:6]: [round(float(abs(P3[(H, k)][0])), 3) for k in BUDGETS[1:]] for H in benches})
p3f = np.where(K == 0, 0.5, sigmoid(z / np.sqrt(1 + np.pi / 8 * V)))
print(fmt(D, p3f, "C3' fixed moderation lam = pi/8 (no fit)")[1])
# C4 component-wise scales (pair-level part vs item part)
m4, p4, P4 = nested(lambda th, i: sigmoid(th[0] * t[i] + th[1] * e[i] + th[2]), [1.0, 1.0, 0.0], "C4 a_k t + c_k e + b_k", return_params=True)
print("   C4 fitted (a,c,b):", {names[H][:6]: [tuple(np.round(P4[(H, k)], 2)) for k in BUDGETS[1:]] for H in benches})
m4b, p4b = nested(lambda th, i: sigmoid(th[0] * t[i] + th[1] * e[i]), [1.0, 1.0], "C4b a_k t + c_k e (no bias)")
# C5 clipping
m5, p5 = nested(lambda th, i: np.clip(sigmoid(z[i]), abs(th[0]), 1 - abs(th[0])), [0.02], "C5 clip [c_k, 1-c_k]")
# C6 moderation + temperature
m6, p6 = nested(lambda th, i: sigmoid(th[0] * z[i] / np.sqrt(1 + abs(th[1]) * V[i])), [1.0, 0.3], "C6 a_k z/sqrt(1+lam_k V)")
# B0 from the subject prior (m_s = 0.5 x name/meta offset): p0 = sigmoid(c m_s)
ms = D["ms"]
def b0test(th, i):
    return sigmoid(th[0] * ms[i])
pz = base.copy()
for H in benches:
    tr = np.concatenate([IDX[(bi, 0)] for bi in benches if bi != H]); wt = W[tr] / W[tr].sum()
    r = minimize(lambda th: float(np.sum(wt * (b0test(th, tr) - y[tr]) ** 2)), [0.5], method="Nelder-Mead")
    pz[IDX[(H, 0)]] = b0test(r.x, IDX[(H, 0)])
    print(f"   B0 subject-prior scale fitted without {names[H][:8]}: {r.x[0]:.3f}")
pz[K > 0] = base[K > 0]
print(fmt(D, pz, "C7 B0 = sigmoid(c m_s), rest baseline")[1])
