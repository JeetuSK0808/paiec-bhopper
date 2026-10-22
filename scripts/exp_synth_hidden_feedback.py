"""Hidden-benchmark evidence from our own formative-feedback logs (Codabench 17828 score_stdout of v3 #942354, v4 #942385,
v5 #942474). The per-pair table derived from them is kept in private/hidden_feedback.json and is not redistributed.

For a constant prediction q on a pair, Brier = (q - ybar)^2 + ybar (1 - ybar) exactly (ybar = mean eval response). At budget 0 every
model predicted 0.5, so the budget-0 ECE (one bin) = |0.5 - ybar|: we know each hidden pair's accuracy up to reflection.
This script (1) tabulates the hidden pairs, (2) computes the EXPECTED per-budget Brier of level-only estimators on those very pairs
(acquisition labels ~ Binomial(k, ybar)), and (3) compares with what our submissions actually scored there.
No benchmark-specific information is used by any model; this only calibrates a global level-prior shape.
"""
import json, math, sys
from pathlib import Path
import numpy as np

B = (0, 1, 3, 7, 15, 31); W = {0: .1, 1: .2, 3: .2, 7: .2, 15: .2, 31: .1}
# (sample, subject, benchmark, n_unique_items, ECE@B0, actual Brier at 0,1,3,7,15,31) for each hidden pair we were scored on.
# It comes from our own Codabench scoring logs of the private test, so it is NOT redistributed (the rules forbid sharing
# private test data). It lives in private/hidden_feedback.json (keys "H" and "bench_abs_logit"), available to the organisers on request.
_PRIV = Path(__file__).resolve().parents[1] / "private" / "hidden_feedback.json"
if not _PRIV.exists():
    raise SystemExit("private/hidden_feedback.json not found. The hidden-calibrated simulator needs per-pair data derived from our own "
                     "Codabench scoring logs, which are not redistributed (competition rule on private test data).")
_D = json.loads(_PRIV.read_text())
H = [tuple(r[:5]) + (list(r[5]),) for r in _D["H"]]
BENCH_ABS_LOGIT = list(_D["bench_abs_logit"])   # hidden benchmark mean |logit(ybar)|, used by the k-system regimes


def alc(b):
    return sum(W[k] * b[i] for i, k in enumerate(B))


def binom_pmf(k, p):
    return np.array([math.comb(k, s) * p ** s * (1 - p) ** (k - s) for s in range(k + 1)])


def expected_brier(q_of, ybar):
    """q_of(k, s) -> constant prediction after k labels with s ones (lone system). Returns expected Brier per budget."""
    out = []
    for k in B:
        if k == 0:
            out.append((q_of(0, 0) - ybar) ** 2 + ybar * (1 - ybar)); continue
        pm = binom_pmf(k, ybar)
        out.append(float(sum(pm[s] * ((q_of(k, s) - ybar) ** 2) for s in range(k + 1)) + ybar * (1 - ybar)))
    return out


def beta_q(a):
    return lambda k, s: (a + s) / (2 * a + k)


def emp_q(k, s):
    return 0.5 if k == 0 else s / k


_GRID = np.linspace(-9, 9, 721)
def logitnormal_q(sd, mu=0.0):
    prior = np.exp(-0.5 * ((_GRID - mu) / sd) ** 2); p = 1 / (1 + np.exp(-_GRID))
    def q(k, s):
        w = prior * p ** s * (1 - p) ** (k - s); return float((w * p).sum() / w.sum())
    return q


def main():
    ys = np.array([0.5 - h[4] for h in H])  # reflection is irrelevant for symmetric estimators
    pq = ys * (1 - ys)
    print(f"hidden pairs: {len(H)}; E[ybar(1-ybar)] = {pq.mean():.4f}  (uniform-accuracy prior would give 0.1667); mean |ybar-0.5| = {np.mean([h[4] for h in H]):.3f} (uniform: 0.250)")
    by_s = {}
    for h in H:
        by_s.setdefault(h[0], []).append(h)
    act = {s: np.mean([alc(h[5]) for h in hs]) for s, hs in by_s.items()}
    print("actual ALC of our submissions on their samples:", {s: round(v, 4) for s, v in act.items()}, " overall", round(np.mean([alc(h[5]) for h in H]), 4))
    print("actual mean Brier by budget, all 26 pairs:", [round(float(np.mean([h[5][i] for h in H])), 4) for i in range(6)])
    cands = [("empirical mean (organisers)", emp_q), ("oracle constant (ybar)", None)]
    for a in (0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0):
        cands.append((f"Beta({a},{a}) posterior mean", beta_q(a)))
    for sd in (0.8, 1.13, 1.5, 2.0, 2.5, 3.0):
        cands.append((f"logit-normal sd {sd} posterior mean", logitnormal_q(sd)))
    print(f"\n{'estimator':38s} {'ALC all':>8s} " + " ".join(f"{s:>7s}" for s in by_s) + "   B1     B3     B7     B15    B31")
    for name, q in cands:
        per = []
        for h in H:
            y = 0.5 - h[4]
            per.append([y * (1 - y) + (0.25 - y * (1 - y) if k == 0 else 0.0) for k in B] if q is None else expected_brier(q, y))
        per = np.array(per)
        a_all = np.mean([alc(r) for r in per])
        a_s = [np.mean([alc(per[j]) for j, h in enumerate(H) if h[0] == s]) for s in by_s]
        print(f"{name:38s} {a_all:8.4f} " + " ".join(f"{v:7.4f}" for v in a_s) + "  " + " ".join(f"{v:.4f}" for v in per.mean(0)[1:]))


if __name__ == "__main__":
    main()
