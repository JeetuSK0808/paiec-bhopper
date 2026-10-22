"""exp_verify_paired (tag 'verify'): paired per-pair comparison of shipped-v5 math (scripts/exp_pooled_core.py cfg beta_prior+dup, as
scripts/exp_synth_cv.py) against the EXACT code of one or more submission dirs (fit_core / batch predict_core, as scripts/exp_synth_v6cv.py)
on the item_prior_cv splits. Reports macro and pair-weighted ALC, paired differences with SE, per-benchmark differences.
  .venv/bin/python -B scripts/exp_verify_paired.py --subs submission_v6,src/paie_exp/verify/subs/v6fix --regimes single,pooled --seeds 0,1,2
Also summarises two stored simulator runs pairwise: --sim A,B (tags under src/paie_exp/<dir>/sim/results, --simdir synth|verify).
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "src"))
sys.dont_write_bytecode = True
import exp_pooled_cv as H  # noqa: E402
import exp_synth_v6cv as V  # noqa: E402

WV = np.array([.1, .2, .2, .2, .2, .1]); BUD = [0, 1, 3, 7, 15, 31]


def sub_pairs(M, benches, regime, seed):
    """Per-pair ALC of submission module M (same math as exp_synth_v6cv.run)."""
    out = {}
    for held, B in benches.items():
        pools = B["pools"]; n = len(pools); texts, feats_raw = V.raw_items(held)
        C_all = np.stack([M.hash_counts(t) for t in texts]).astype(np.float32)
        F_all = [M._parse_features(f if isinstance(f, str) else None) for f in feats_raw]
        canon = B["canon"]; briers = np.zeros((n, 6))
        for j in range(n):
            briers[j, 0] = np.mean((0.5 - pools[j]["ev_y"]) ** 2)
        for grp in H.groups_for(n, regime, seed, held):
            m_s_all = B["m_subj"][grp] * (M.SP_SCALE / 0.5)
            for bi, k in enumerate(BUD[1:], start=1):
                si = np.concatenate([[a] * min(k, len(pools[j]["acq_i"])) for a, j in enumerate(grp)]).astype(int)
                ii_g = canon[np.concatenate([pools[j]["acq_i"][:k] for j in grp]).astype(int)]
                y = np.concatenate([pools[j]["acq_y"][:k] for j in grp]).astype(float)
                sx, ix = {}, {}
                for s_, i_ in zip(si.tolist(), ii_g.tolist()):
                    sx.setdefault(s_, len(sx)); ix.setdefault(i_, len(ix))
                items = list(ix)
                fit = M.fit_core(np.array([sx[s_] for s_ in si.tolist()]), np.array([ix[i_] for i_ in ii_g.tolist()]), y, len(sx), len(ix),
                                 np.array([m_s_all[s_] for s_ in sx], float), C_all[items], [F_all[i_] for i_ in items])
                ev_s = np.concatenate([[a] * len(pools[j]["ev_i"]) for a, j in enumerate(grp)]).astype(int)
                ev_i = canon[np.concatenate([pools[j]["ev_i"] for j in grp]).astype(int)]
                e = np.zeros(len(ev_i)); lab = np.array([i_ in ix for i_ in ev_i.tolist()])
                if lab.any():
                    e[lab] = fit["eps"][[ix[i_] for i_ in ev_i[lab].tolist()]]
                un = ~lab
                if un.any():
                    if fit["prior"] is not None:
                        e[un] += fit["prior"].predict(C_all[ev_i[un]])
                    if fit["grp"] is not None:
                        e[un] += np.stack([fit["grp"].row(F_all[i_]) for i_ in ev_i[un].tolist()]) @ fit["g"]
                d = np.array([fit["delta"][sx[s_]] if s_ in sx else m_s_all[s_] for s_ in ev_s.tolist()])
                pr = np.clip(1 / (1 + np.exp(-(fit["beta"] + d + e))), M.P_MIN, M.P_MAX)
                off = 0
                for a, j in enumerate(grp):
                    m = len(pools[j]["ev_i"]); briers[j, bi] = np.mean((pr[off:off + m] - pools[j]["ev_y"]) ** 2); off += m
        out[held] = briers @ WV
    return out


def report(name, A, Bd):
    """A, Bd: {held: per-pair ALC arrays} (baseline, candidate)."""
    heads = list(A)
    ma = np.mean([A[h].mean() for h in heads]); mb = np.mean([Bd[h].mean() for h in heads])
    da = np.concatenate([Bd[h] - A[h] for h in heads])
    pa = np.concatenate([A[h] for h in heads]).mean(); pb = np.concatenate([Bd[h] for h in heads]).mean()
    per = " ".join(f"{h[:6]}:{(Bd[h] - A[h]).mean():+.4f}" for h in heads)
    better = sum((Bd[h] - A[h]).mean() < 0 for h in heads)
    ex = [h for h in heads if h != "swe_rebench"]
    ma_x = np.mean([A[h].mean() for h in ex]); mb_x = np.mean([Bd[h].mean() for h in ex])
    print(f"  {name:44s} macro {ma:.4f}->{mb:.4f} ({mb - ma:+.4f}) | pair-wtd {pa:.4f}->{pb:.4f} ({pb - pa:+.4f} +- {da.std(ddof=1) / np.sqrt(len(da)):.4f}, "
          f"n={len(da)}) | better on {better}/{len(heads)} | excl. swe_rebench {mb_x - ma_x:+.4f} | {per}", flush=True)


def sim_report(simdir, a, b):
    res = ROOT / "src" / "paie_exp" / simdir / "sim" / "results"
    A = json.loads((res / f"sim_{a}.json").read_text())["regimes"]; Bj = json.loads((res / f"sim_{b}.json").read_text())["regimes"]
    for reg in A:
        if reg not in Bj:
            continue
        pa, pb = A[reg]["pairs"], Bj[reg]["pairs"]
        assert len(pa) == len(pb) and all((x["bench"], x["sid"]) == (y["bench"], y["sid"]) for x, y in zip(pa, pb))
        by = {}
        for x, y in zip(pa, pb):
            by.setdefault(x["bench"], []).append((x["alc"], y["alc"]))
        d = np.array([y - x for lst in by.values() for x, y in lst])
        macro_d = np.mean([np.mean([y - x for x, y in lst]) for lst in by.values()])
        per = " ".join(f"{h[:6]}:{np.mean([y - x for x, y in lst]):+.4f}" for h, lst in sorted(by.items()))
        print(f"  sim {a} -> {b} [{reg:8s}] micro {np.mean([x['alc'] for x in pa]):.4f}->{np.mean([y['alc'] for y in pb]):.4f} "
              f"({d.mean():+.4f} +- {d.std(ddof=1) / np.sqrt(len(d)):.4f}, n={len(d)}) macro diff {macro_d:+.4f} | {per}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subs", default="submission_v6"); ap.add_argument("--regimes", default="single,pooled"); ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--sim", default=""); ap.add_argument("--simdir", default="synth")
    a = ap.parse_args()
    if a.sim:
        x, y = a.sim.split(","); sim_report(a.simdir, x, y); return
    mods = {s: V.load_model(s) for s in a.subs.split(",")}
    for reg in a.regimes.split(","):
        allA, allB = {}, {s: {} for s in mods}
        for seed in [int(s) for s in a.seeds.split(",")]:
            benches = H.prepare(seed, 0.5)
            _, _, res = H.evaluate(benches, {"beta_prior": True, "dup": 1}, reg, seed, "random", None, False)
            A = {h: np.asarray(r["alc_pairs"]) for h, r in res.items()}
            for s, M in mods.items():
                Bd = sub_pairs(M, benches, reg, seed)
                report(f"{reg} seed={seed} v5shipped -> {s.split('/')[-1]}", A, Bd)
                for h in A:
                    allB[s].setdefault(h, []).append(Bd[h])
            for h in A:
                allA.setdefault(h, []).append(A[h])
        cat = lambda D: {h: np.concatenate(v) for h, v in D.items()}  # noqa: E731
        for s in mods:
            report(f"{reg} ALL SEEDS v5shipped -> {s.split('/')[-1]}", cat(allA), cat(allB[s]))


if __name__ == "__main__":
    main()
