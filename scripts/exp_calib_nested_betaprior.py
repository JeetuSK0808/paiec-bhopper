"""exp_calib: nested leave-one-benchmark-out tau selection per regime under the SHIPPED beta update (--beta-prior grids)."""
import json, numpy as np
W = {0: 0.1, 1: 0.2, 3: 0.2, 7: 0.2, 15: 0.2, 31: 0.1}; KS = [0, 1, 3, 7, 15, 31]
gs = json.load(open("src/paie_exp/calib/grid_betaprior_single.json")); gg = json.load(open("src/paie_exp/calib/grid_betaprior_groups.json"))
alc = lambda R, c, b: sum(W[k] * R[c][b][str(k)] for k in KS)
benches = sorted(gs[list(gs)[0]]); multi = [b for b in benches if b != "swe_rebench"]
T1 = ["0.8", "1.0", "1.25", "1.5", "2.0"]; S = {t: {b: alc(gs, f"--tau {t} --tau-i 1.5 --single --beta-prior", b) for b in benches} for t in T1}
t1 = lambda H: min(T1, key=lambda t: np.mean([S[t][b] for b in benches if b != H]))
for G in ["1", "2", "4", "16", "0"]:
    if G == "1": A, taus = S, T1
    else:
        taus = ["0.8", "1.0", "1.25", "1.5"]; A = {t: {b: alc(gg, f"--group-size {G} --tau {t} --tau-i 1.5 --beta-prior", b) for b in benches} for t in taus}
    v5 = {b: (A["0.8"][b] if b != "swe_rebench" else S["0.8"][b]) for b in benches}; nest = {}; ch = {}
    for H in benches:
        if H == "swe_rebench": t = t1(H); nest[H] = S[t][H]; ch[H] = t; continue
        pool = multi if G != "1" else benches
        t = min(taus, key=lambda t: np.mean([A[t][b] for b in pool if b != H])); nest[H] = A[t][H]; ch[H] = t
    print(f"G={G:3s} shipped tau0.8 {np.mean(list(v5.values())):.4f}  nested tau {np.mean(list(nest.values())):.4f}  wins {sum(nest[b] < v5[b] - 1e-6 for b in benches)}/6 ties {sum(abs(nest[b]-v5[b])<1e-6 for b in benches)}  chosen " + " ".join(f"{b[:5]}={ch[b]}" for b in benches))
