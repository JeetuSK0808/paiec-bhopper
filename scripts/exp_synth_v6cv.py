"""exp_synth_v6cv (tag 'synth'): item_prior_cv-style leave-one-benchmark-out CV of the EXACT submission code
(<sub>/model.py fit_core / predict_core) on the splits of scripts/exp_pooled_cv.py (= item_prior_cv.py RNG stream, fractional labels,
per-fold subject prior x0.5, identical-item merge as the submission's dict keys do). Regimes: single, mixN, pooled.
  .venv/bin/python scripts/exp_synth_v6cv.py --sub submission_v6 --regimes single,mix2,mix8,pooled --seeds 0,1,2 [--set TAU_B=1.4,...]
"""
import argparse, importlib.util, os, sys, time
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "src"))
sys.dont_write_bytecode = True
import exp_pooled_cv as H  # noqa: E402
from paie.data import load_all  # noqa: E402

W = np.array([.1, .2, .2, .2, .2, .1]); BUD = [0, 1, 3, 7, 15, 31]


def load_model(sub):
    d = str((ROOT / sub).resolve()); sys.path.insert(0, d)
    try:
        spec = importlib.util.spec_from_file_location("_v6m_" + Path(sub).name, os.path.join(d, "model.py"))
        m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    finally:
        sys.path.remove(d)
    return m


_RAW = {}
def raw_items(held):
    if held not in _RAW:
        pairs, _, items = load_all(); items = items.drop_duplicates("item_id")
        test = pairs[pairs["benchmark"] == held]; all_items = list(test["item_id"].unique())
        it = items[items["benchmark"] == held].set_index("item_id")
        _RAW[held] = ([str(it.loc[i, "content"]) if pd.notna(it.loc[i, "content"]) else "" for i in all_items],
                      [it.loc[i, "item_features"] for i in all_items])
    return _RAW[held]


def run(M, benches, regime, seed, verbose=False):
    res = {}
    for held, B in benches.items():
        t0 = time.time(); pools = B["pools"]; n = len(pools)
        texts, feats_raw = raw_items(held)
        from textprior import hash_counts  # noqa: E402  (the submission's own copy is on sys.path via the module)
        C_all = np.stack([M.hash_counts(t) for t in texts]).astype(np.float32)
        F_all = [M._parse_features(f if isinstance(f, str) else None) for f in feats_raw]
        canon = B["canon"]
        briers = np.zeros((n, 6))
        for j in range(n):
            briers[j, 0] = np.mean((0.5 - pools[j]["ev_y"]) ** 2)
        for grp in H.groups_for(n, regime, seed, held):
            m_s_all = B["m_subj"][grp] * (M.SP_SCALE / 0.5)  # harness prior is 0.5 x offset
            for bi, k in enumerate(BUD[1:], start=1):
                si = np.concatenate([[a] * min(k, len(pools[j]["acq_i"])) for a, j in enumerate(grp)]).astype(int)
                ii_g = canon[np.concatenate([pools[j]["acq_i"][:k] for j in grp]).astype(int)]
                y = np.concatenate([pools[j]["acq_y"][:k] for j in grp]).astype(float)
                # local indices in first-appearance order (as the submission builds them)
                sx, ix = {}, {}
                for s_, i_ in zip(si.tolist(), ii_g.tolist()):
                    sx.setdefault(s_, len(sx)); ix.setdefault(i_, len(ix))
                items = list(ix)
                fit = M.fit_core(np.array([sx[s_] for s_ in si.tolist()]), np.array([ix[i_] for i_ in ii_g.tolist()]), y, len(sx), len(ix),
                                 np.array([m_s_all[s_] for s_ in sx], float), C_all[items], [F_all[i_] for i_ in items])
                # batch prediction (same math as predict_core)
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
        res[held] = float(np.mean(briers @ W))
        if verbose:
            print(f"   {regime} {held:18s} {res[held]:.4f} ({time.time()-t0:.0f}s)", flush=True)
    return float(np.mean(list(res.values()))), res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", default="submission_v6"); ap.add_argument("--regimes", default="single,pooled"); ap.add_argument("--seeds", default="0")
    ap.add_argument("--set", default="", help="override module constants, e.g. TAU_B=1.4,TAU_G=0"); ap.add_argument("--tag", default="")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    M = load_model(a.sub)
    for kv in filter(None, a.set.split(",")):
        k, v = kv.split("="); setattr(M, k, type(getattr(M, k))(v))
    tag = a.tag or (a.sub + (" " + a.set if a.set else ""))
    tab = {}
    for seed in [int(s) for s in a.seeds.split(",")]:
        benches = H.prepare(seed, 0.5)
        for reg in a.regimes.split(","):
            t0 = time.time(); mean, res = run(M, benches, reg, seed, a.verbose); tab[(reg, seed)] = mean
            print(f"{tag:34s} seed={seed} {reg:7s} ALC {mean:.4f}  " + " ".join(f"{h[:6]}:{v:.4f}" for h, v in res.items()) + f"  ({time.time()-t0:.0f}s)", flush=True)
    seeds = [int(s) for s in a.seeds.split(",")]
    if len(seeds) > 1:
        print(f"{tag:34s} MEAN over seeds " + "  ".join(f"{reg} {np.mean([tab[(reg, s)] for s in seeds]):.4f}" for reg in a.regimes.split(",")))


if __name__ == "__main__":
    main()
