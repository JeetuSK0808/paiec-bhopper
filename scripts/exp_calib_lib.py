"""exp_calib: shared helpers for offline calibration experiments on the dumps written by scripts/exp_calib_cv.py.
ALC aggregation exactly as the harness: per pair mean Brier over its evaluation items, per benchmark mean over pairs, ALC over budgets,
mean over benchmarks (each held-out benchmark weighted equally)."""
import numpy as np

BUDGETS = [0, 1, 3, 7, 15, 31]; WEIGHTS = {0: 0.1, 1: 0.2, 3: 0.2, 7: 0.2, 15: 0.2, 31: 0.1}
TH = np.linspace(-8, 8, 321)  # logit grid for EB posteriors


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def logit(p, eps=1e-4):
    p = np.clip(p, eps, 1 - eps); return np.log(p / (1 - p))


def load(path):
    z = np.load(path, allow_pickle=True)
    D = {k: z[k] for k in z.files}
    D["bench_names"] = list(D["bench_names"]); D["pair_names"] = list(D["pair_names"])
    # row weight so that sum over a benchmark's rows at one budget = 1 (pairs equal within benchmark)
    w = np.zeros(len(D["y"]))
    for k in BUDGETS:
        mk = D["k"] == k
        pr = D["pair"][mk]; cnt = np.bincount(pr, minlength=len(D["pair_names"]))
        b = D["bench"][mk]
        npairs = {bb: len(np.unique(pr[b == bb])) for bb in np.unique(b)}
        w[mk] = 1.0 / (cnt[pr] * np.array([npairs[bb] for bb in b]))
    D["w"] = w
    return D


def per_bench(D, p, mask=None):
    """Returns {bench: {k: Brier}} for predictions p (same length as D rows)."""
    out = {}
    se = (p - D["y"]) ** 2 * D["w"]
    for bi in np.unique(D["bench"]):
        out[bi] = {}
        for k in BUDGETS:
            m = (D["bench"] == bi) & (D["k"] == k)
            out[bi][k] = float(se[m].sum())
    return out


def alc_table(D, p, label=""):
    pb = per_bench(D, p)
    alcs = {bi: sum(WEIGHTS[k] * pb[bi][k] for k in BUDGETS) for bi in pb}
    return float(np.mean(list(alcs.values()))), alcs, pb


def fmt(D, p, label):
    m, alcs, pb = alc_table(D, p)
    names = D["bench_names"]
    s = f"{label:44s} mean {m:.4f} | " + " ".join(f"{names[bi][:8]} {alcs[bi]:.4f}" for bi in sorted(alcs))
    return m, s, alcs, pb


def eb_posterior(k_own, s_own, prior_logpdf):
    """Grid posterior over theta (logit of pair accuracy) given k own labels with sum s (fractional labels ok).
    prior_logpdf: array over TH (or 2D pairs x TH). Returns weights (pairs x TH)."""
    ll = s_own[:, None] * np.log(sigmoid(TH))[None, :] + (k_own - s_own)[:, None] * np.log(sigmoid(-TH))[None, :]
    lp = ll + prior_logpdf
    lp = lp - lp.max(1, keepdims=True); w = np.exp(lp); w /= w.sum(1, keepdims=True)
    return w
