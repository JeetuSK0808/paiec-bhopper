"""Report Table 1: oracle Brier scores per public benchmark (full data, in-sample).
Trial-level Brier of a prediction p on a pair with mean response ybar: (p - ybar)^2 + ybar (1 - ybar), averaged over pairs.
  mean    : p = the benchmark's mean response
  subject : p = the system's mean response on the benchmark
  item    : p = the item's mean response over systems
  rasch   : p = sigmoid(delta_s + eps_i), an additive logistic fit on all pairs (light ridge)
  .venv/bin/python -B scripts/oracle_table.py [--ridge 0.01] [--iters 60]
"""
import argparse, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from paie.data import load_all  # noqa: E402


def brier(p, y):
    return float(np.mean((p - y) ** 2 + y * (1 - y)))


def rasch(si, ii, y, ns, ni, ridge, iters):
    d, e = np.zeros(ns), np.zeros(ni)
    for _ in range(iters):
        for which in (0, 1):
            p = 1 / (1 + np.exp(-(d[si] + e[ii])))
            g, h = y - p, p * (1 - p)
            idx, v, n = (si, d, ns) if which == 0 else (ii, e, ni)
            G = np.bincount(idx, g, n) - ridge * v
            H = np.bincount(idx, h, n) + ridge
            v += np.clip(G / H, -2, 2)
    return 1 / (1 + np.exp(-(d[si] + e[ii])))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ridge", type=float, default=0.01); ap.add_argument("--iters", type=int, default=60)
    a = ap.parse_args()
    pairs, _, _ = load_all()
    print(f"{'benchmark':20s} {'mean':>6s} {'subject':>8s} {'item':>6s} {'rasch':>6s}  (systems, items, pairs)")
    for b, g in pairs.groupby("benchmark"):
        y = g["y"].to_numpy(float)
        s_codes, s_idx = np.unique(g["subject_id"].astype(str), return_inverse=True)
        i_codes, i_idx = np.unique(g["item_id"].astype(str), return_inverse=True)
        p_mean = np.full_like(y, y.mean())
        p_subj = (np.bincount(s_idx, y) / np.bincount(s_idx))[s_idx]
        p_item = (np.bincount(i_idx, y) / np.bincount(i_idx))[i_idx]
        p_rasch = rasch(s_idx, i_idx, y, len(s_codes), len(i_codes), a.ridge, a.iters)
        print(f"{b:20s} {brier(p_mean, y):6.3f} {brier(p_subj, y):8.3f} {brier(p_item, y):6.3f} {brier(p_rasch, y):6.3f}"
              f"  ({len(s_codes)}, {len(i_codes)}, {len(y)})", flush=True)


if __name__ == "__main__":
    main()
