"""exp_calib: nested leave-one-benchmark-out test of 'tau depends on the number of systems n sharing the benchmark'.
For every regime G (1 = single, 2, 4, 8, 16, all) the tau for that regime is chosen on the OTHER benchmarks and scored on the held-out one.
swe_rebench has one system, so in every regime it is scored with the n = 1 (single-regime) tau, chosen without it."""
import json, numpy as np
W = {0: 0.1, 1: 0.2, 3: 0.2, 7: 0.2, 15: 0.2, 31: 0.1}; KS = [0, 1, 3, 7, 15, 31]
gg = json.load(open("src/paie_exp/calib/grid_groups.json")); gs = json.load(open("src/paie_exp/calib/grid_single.json"))
alc = lambda R, c, b: sum(W[k] * R[c][b][str(k)] for k in KS)
benches = sorted(gs[list(gs)[0]]); multi = [b for b in benches if b != "swe_rebench"]
TAUS1 = ["0.6", "0.7", "0.8", "0.9", "1.0"]
S = {t: {b: alc(gs, f"--tau {t} --tau-i 1.5 --text-scale 1 --single", b) for b in benches} for t in TAUS1}
def tau1_for(H):  # single-regime tau chosen without H
    return min(TAUS1, key=lambda t: np.mean([S[t][b] for b in benches if b != H]))
print("single-regime tau chosen without each benchmark:", {b[:8]: tau1_for(b) for b in benches})
for G in ["1", "2", "4", "8", "16", "0"]:
    if G == "1":
        taus = TAUS1; A = S
    else:
        taus = sorted({c.split()[3] for c in gg if c.startswith(f"--group-size {G} ")}, key=float)
        A = {t: {b: alc(gg, f"--group-size {G} --tau {t} --tau-i 1.5", b) for b in benches} for t in taus}
    v5 = {b: (A["0.8"][b] if b != "swe_rebench" else S["0.8"][b]) for b in benches}
    nested = {}; chosen = {}
    for H in benches:
        if H == "swe_rebench":
            t = tau1_for(H); nested[H] = S[t][H]; chosen[H] = t; continue
        t = min(taus, key=lambda t: np.mean([A[t][b] for b in multi if b != H])); nested[H] = A[t][H]; chosen[H] = t
    lab = {"1": "single", "0": "all (pooled)"}.get(G, f"groups of {G}")
    print(f"{lab:14s} v5 tau0.8: {np.mean(list(v5.values())):.4f}   nested tau(n): {np.mean(list(nested.values())):.4f}   "
          f"wins {sum(nested[b] < v5[b] - 1e-6 for b in benches)}/6  | chosen " + " ".join(f"{b[:5]}={chosen[b]}" for b in benches))
    print("      per-bench v5 -> nested: " + " ".join(f"{b[:8]} {v5[b]:.4f}->{nested[b]:.4f}" for b in benches))
