"""exp_calib: empirical-Bayes subject-level posteriors combined with the model's item offsets, evaluated on a SINGLE-regime dump with
nested leave-one-benchmark-out for every fitted weight. Priors over theta = logit(pair accuracy):
  N(0, s^2); symmetrised KDE of the other benchmarks' pair logits (benchmarks weighted equally; symmetric so B0 stays 0.5).
Predictions: EB-only E[sigmoid(theta)]; EB + centred item offsets E[sigmoid(theta + c o_i)]; convex blend with the model."""
import sys, numpy as np
from scipy.optimize import minimize
sys.path.insert(0, "scripts")
from exp_calib_lib import load, fmt, sigmoid, logit, eb_posterior, TH, BUDGETS, WEIGHTS
D = load(sys.argv[1]); y = D["y"]; z = D["z"]; K = D["k"]; Bn = D["bench"]; P = D["pair"]; W = D["w"]
names = D["bench_names"]; benches = np.unique(Bn); npair = len(D["pair_names"])
IDX = {(bi, k): np.nonzero((Bn == bi) & (K == k))[0] for bi in benches for k in BUDGETS}
base = sigmoid(z); print(fmt(D, base, "baseline")[1])
m0 = K == 0
acc = np.bincount(P[m0], y[m0], npair) / np.maximum(np.bincount(P[m0], None, npair), 1)
bop = np.zeros(npair, int); bop[P] = Bn
# centred item offsets per (pair, budget)
o = np.zeros(len(y)); key = P * 100 + K
sums = {}; 
for kk_ in np.unique(key):
    pass
uk, inv = np.unique(key, return_inverse=True)
mean_e = np.bincount(inv, D["e"]) / np.bincount(inv); o = D["e"] - mean_e[inv]
def kde_sym(b_excl, bw):
    d = np.zeros_like(TH)
    for bb in benches:
        if bb == b_excl: continue
        v = logit(acc[bop == bb], 0.01); v = np.concatenate([v, -v])
        d += np.exp(-0.5 * ((TH[:, None] - v[None, :]) / bw) ** 2).sum(1) / len(v)   # each benchmark weight 1
    d /= d.sum(); return np.log(d + 1e-300)
def eb_rows(prior_fn, c=0.0, rows=None):
    """E[sigmoid(theta + c*o)] for each row, posterior from (k_own, sum_own) of its pair; prior_fn(bench)->logpdf over TH."""
    out = np.zeros(len(y))
    for H in benches:
        lp = prior_fn(H)
        for k in BUDGETS:
            idx = IDX[(H, k)]
            if k == 0: out[idx] = 0.5; continue
            pr = P[idx]; up, fi = np.unique(pr, return_index=True); rep = idx[fi]
            wpost = eb_posterior(D["k_own"][rep], D["sum_own"][rep], np.broadcast_to(lp, (len(rep), len(TH))))
            row_w = wpost[np.searchsorted(up, pr)]  # rows x TH
            if c == 0.0:
                out[idx] = (row_w * sigmoid(TH)[None, :]).sum(1)
            else:
                out[idx] = (row_w * sigmoid(TH[None, :] + c * o[idx][:, None])).sum(1)
    return out
priors = {"N(0,1.5^2)": lambda H: -0.5 * (TH / 1.5) ** 2}
for bw in (0.3, 0.6):
    priors[f"symKDE bw{bw}"] = (lambda bw_: (lambda H: kde_sym(H, bw_)))(bw)
EB = {}
for nm, pf in priors.items():
    EB[nm] = eb_rows(pf); print(fmt(D, EB[nm], f"EB-only {nm}")[1])
    EB[nm + " +o"] = eb_rows(pf, c=1.0); print(fmt(D, EB[nm + " +o"], f"EB {nm} + item offsets (c=1)")[1])
# nested blend weights per budget: p = (1-w) model + w EB
def nested_blend(pe, label):
    p = base.copy(); ws = []
    for H in benches:
        for k in BUDGETS:
            if k == 0: p[IDX[(H, 0)]] = 0.5; continue
            tr = np.concatenate([IDX[(bi, k)] for bi in benches if bi != H]); wt = W[tr] / W[tr].sum()
            f = lambda th: float(np.sum(wt * ((1 - th[0]) * base[tr] + th[0] * pe[tr] - y[tr]) ** 2))
            r = minimize(f, [0.5], method="Nelder-Mead"); wv = float(np.clip(r.x[0], 0, 1)); ws.append(round(wv, 2))
            p[IDX[(H, k)]] = (1 - wv) * base[IDX[(H, k)]] + wv * pe[IDX[(H, k)]]
    print(fmt(D, p, label)[1]); print("     weights (held-out x budget):", ws)
    return p
for nm in list(EB):
    nested_blend(EB[nm], f"nested blend model + EB[{nm}]")
# per-budget view of the best-looking ones
for nm in ("N(0,1.5^2) +o", "symKDE bw0.3 +o"):
    m, s, alcs, pb = fmt(D, EB[nm], nm)
    print(f"   {nm:24s} " + " ".join(f"B{k}={np.mean([pb[bi][k] for bi in pb]):.4f}" for k in BUDGETS))
m, s, alcs, pb = fmt(D, base, "base"); print(f"   {'model':24s} " + " ".join(f"B{k}={np.mean([pb[bi][k] for bi in pb]):.4f}" for k in BUDGETS))
