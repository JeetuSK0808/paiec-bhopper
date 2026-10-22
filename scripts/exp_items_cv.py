"""Tag 'items' CV harness: leave-one-benchmark-out, same protocol, pools and RNG stream as scripts/item_prior_cv.py
(--feat hash --alpha A --rounds R --tau T --tau-i TI --sp SP [--single]), plus item_features structure (scripts/exp_items_core.py).
With all new options off it reproduces the v5 numbers (pooled 0.1616 / single 0.1856).
  nice -n 10 env VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python scripts/exp_items_cv.py [--single] \
      [--groups --tau-g 0.8] [--identity --tau-id 1.0] [--numeric z --tau-n 0.3] [--sg --tau-sg 0.5] [--kernel rbf --gamma 5]
"""
import argparse, json, sys, time, zlib
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from paie.data import load_all  # noqa: E402
from paie.textprior import hash_counts  # noqa: E402
from paie_cv import BUDGETS, WEIGHTS  # noqa: E402
from exp_items_core import ItemStructure, fit_hier, item_offset, subj_group_offset, tfidf_matrix, kernel, sigmoid  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=1.0); ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--tau", type=float, default=0.8); ap.add_argument("--tau-i", type=float, default=1.5)
    ap.add_argument("--sp", type=float, default=0.5); ap.add_argument("--sp-kw", type=float, default=0.6); ap.add_argument("--sp-mw", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--single", action="store_true")
    ap.add_argument("--only", default="", help="comma list of benchmarks to score (RNG stream is still consumed for all)")
    # new structure options
    ap.add_argument("--groups", action="store_true", help="categorical key=value group effects")
    ap.add_argument("--tau-g", type=float, default=0.8)
    ap.add_argument("--identity", action="store_true", help="full feature string as an extra level")
    ap.add_argument("--tau-id", type=float, default=1.0)
    ap.add_argument("--numeric", default="off", help="off|z|rank|log  numeric keys as covariates")
    ap.add_argument("--tau-n", type=float, default=0.3); ap.add_argument("--mu-n", type=float, default=0.0)
    ap.add_argument("--num-cat", action="store_true", help="numeric keys also as categorical levels")
    ap.add_argument("--sg", action="store_true", help="subject x group interactions (ordinary keys, not identity)")
    ap.add_argument("--tau-sg", type=float, default=0.5)
    ap.add_argument("--kernel", default="linear"); ap.add_argument("--gamma", type=float, default=5.0)
    ap.add_argument("--beta-map", action="store_true"); ap.add_argument("--drop-keys", default="")
    ap.add_argument("--min-lab", type=int, default=10, help="min labelled items before the text ridge is used")
    ap.add_argument("--center", action="store_true", help="centre group effects on the labelled rows (mean goes to beta)")
    ap.add_argument("--min-rows-group", type=int, default=0, help="fit group effects only once this many labels are in the fit")
    ap.add_argument("--nsys", type=int, default=0, help="regime: random groups of this many systems per benchmark (0 = all pooled)")
    ap.add_argument("--ridge-w", default="counts", help="counts|sqrt|none sample weights of the text ridge")
    ap.add_argument("--tau-b", type=float, default=0.0, help="separate prior sd of the benchmark intercept (0 = --tau)")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    only = set(x for x in a.only.split(",") if x)
    pairs, subjects, items = load_all(); items = items.drop_duplicates("item_id")
    rng = np.random.default_rng(a.seed); results = {}; allB = {}
    t_all = time.time()
    for held in sorted(pairs["benchmark"].unique()):
        t0 = time.time(); test = pairs[pairs["benchmark"] == held]
        pools = []
        for sid, g in test.groupby("subject_id"):
            if len(g) < 80:
                continue
            g = g.sample(frac=1.0, random_state=int(rng.integers(1e9))); half = len(g) // 2
            pools.append((sid, g.iloc[:half], g.iloc[half:]))
        if only and held not in only:
            continue
        if not pools:
            print(f"{held:18s} skipped (no subject with >= 80 items)"); continue
        all_items = list(test["item_id"].unique()); N = len(all_items); ix = {i_: n for n, i_ in enumerate(all_items)}
        it_h = items[items["benchmark"] == held].set_index("item_id")
        C_all = np.stack([hash_counts(str(it_h.at[i_, "content"]) if pd.notna(it_h.at[i_, "content"]) else "", dim=4096) for i_ in all_items])
        T = tfidf_matrix(C_all); del C_all
        Gram = T @ T.T if N <= 2500 else None
        feats = [it_h.at[i_, "item_features"] for i_ in all_items]
        drop = tuple(x.strip().lower() for x in a.drop_keys.split(",") if x.strip())
        struct = ItemStructure(feats, identity=a.identity, numeric_mode=(a.numeric if a.numeric != "off" else "z"), numeric_as_cat=a.num_cat, drop_keys=drop)
        groups, tau_g = [], []
        for (name, g, l) in struct.groups:
            if name == "__identity__":
                if a.identity:
                    groups.append((name, g, l)); tau_g.append(a.tau_id)
            elif a.groups:
                groups.append((name, g, l)); tau_g.append(a.tau_g)
        X = struct.X if (a.numeric != "off" and struct.X is not None) else None
        sg_keys = tuple(k for k, (name, _, _) in enumerate(groups) if a.sg and name != "__identity__")
        # subject prior (exactly as item_prior_cv --sp)
        m_subj = {}
        if a.sp > 0:
            from paie.subjprior import SubjectPrior
            sd = subjects.drop_duplicates("subject_id").set_index("subject_id")
            def sdict(sid):
                r = sd.loc[sid]; return {k: ("" if pd.isna(r.get(k)) else str(r.get(k))) for k in ("normalized_name", "provider", "release_date", "reasoning_effort", "subject_features_extra")}
            trp = pairs[pairs["benchmark"] != held].groupby(["benchmark", "subject_id"]).y.agg(["mean", "size"]).reset_index()
            spm = SubjectPrior(known_weight=a.sp_kw, meta_weight=a.sp_mw).fit([(sdict(r.subject_id), r.benchmark, r["mean"], r["size"]) for _, r in trp.iterrows()])
            m_subj = {sid: a.sp * spm.offset(sdict(sid)) for sid, _, _ in pools}

        def gram(q, lab):
            G_ = Gram[np.ix_(q, lab)] if Gram is not None else T[q] @ T[lab].T
            if a.kernel == "linear":
                return G_
            if a.kernel == "rbf":
                return np.exp(a.gamma * (G_ - 1.0))
            return G_ + np.exp(a.gamma * (G_ - 1.0))

        def run_fit(sids, s_idx, i_idx, y, need):
            """sids: local subject ids; need: universe items whose prior mean is needed (eval + labelled)."""
            m_s = np.array([m_subj.get(s_, 0.0) for s_ in sids])
            m_i = np.zeros(N)
            kw = dict(groups=groups, tau_g=tau_g, X=X, tau_n=a.tau_n, mu_n=a.mu_n, sg_keys=sg_keys, tau_sg=a.tau_sg,
                      tau_b=(a.tau_b if a.tau_b > 0 else a.tau), tau_s=a.tau, tau_i=a.tau_i, beta_map=a.beta_map,
                      center=a.center, min_rows_group=a.min_rows_group)
            fit = fit_hier(s_idx, i_idx, y, m_s, m_i, **kw)
            lab, cnt = np.unique(i_idx, return_counts=True)
            if a.rounds > 0 and len(lab) >= a.min_lab:
                wr = cnt.astype(float) if a.ridge_w == "counts" else (np.sqrt(cnt.astype(float)) if a.ridge_w == "sqrt" else np.ones(len(cnt)))
                K = gram(lab, lab) + a.alpha * np.diag(1.0 / wr)
                Kq = gram(need, lab)
                for _ in range(a.rounds):
                    tgt = fit["eps"][lab]  # residual item effect (after group / numeric structure)
                    mean_y = float(np.average(tgt, weights=wr)); coef = np.linalg.solve(K, tgt - mean_y)
                    m_i = np.zeros(N); m_i[need] = Kq @ coef + mean_y
                    fit = fit_hier(s_idx, i_idx, y, m_s, m_i, **kw)
            fit["m_i"] = m_i
            return fit

        B = {}
        for k in BUDGETS:
            if k == 0:
                B[k] = float(np.mean([np.mean((0.5 - ev["y"].values) ** 2) for _, _, ev in pools])); continue
            bs = []
            if a.single:
                teams = [[n] for n in range(len(pools))]
            elif a.nsys > 0:  # independent groups of nsys systems sharing the benchmark (separate RNG: pools unchanged)
                perm = np.random.default_rng([a.seed, 7919, zlib.crc32(held.encode())]).permutation(len(pools))
                teams = [list(perm[j:j + a.nsys]) for j in range(0, len(pools), a.nsys)]
            else:
                teams = [list(range(len(pools)))]
            for team in teams:
                sids = [pools[n][0] for n in team]
                s_idx = np.concatenate([np.full(min(k, len(pools[n][1])), j, dtype=np.int64) for j, n in enumerate(team)])
                i_idx = np.concatenate([np.array([ix[i_] for i_ in pools[n][1]["item_id"].values[:k]], dtype=np.int64) for n in team])
                y = np.concatenate([pools[n][1]["y"].values[:k] for n in team]).astype(float)
                evs = [np.array([ix[i_] for i_ in pools[n][2]["item_id"].values], dtype=np.int64) for n in team]
                need = np.arange(N) if len(team) == len(pools) and not a.single else np.unique(np.concatenate([i_idx] + evs))
                fit = run_fit(sids, s_idx, i_idx, y, need)
                for j, n in enumerate(team):
                    ei = evs[j]
                    z = fit["beta"] + fit["delta"][j] + item_offset(fit, groups, X, ei) + subj_group_offset(fit, groups, j, ei)
                    bs.append(np.mean((sigmoid(z) - pools[n][2]["y"].values) ** 2))
            B[k] = float(np.mean(bs))
        alc = sum(WEIGHTS[k] * B[k] for k in BUDGETS); results[held] = alc; allB[held] = B
        gdesc = ",".join(f"{nm}:{l}" for nm, _, l in groups) + (f" num:{struct.numeric_names}" if X is not None else "")
        print(f"{held:18s} subjects {len(pools):3d}  ALC {alc:.4f}   " + " ".join(f"B{k}={B[k]:.3f}" for k in BUDGETS) + f"   ({time.time()-t0:.0f}s) [{gdesc}]", flush=True)
    mean = float(np.mean(list(results.values())))
    regime = "single" if a.single else (f"nsys{a.nsys}" if a.nsys > 0 else "pooled")
    print(f"\n{regime} mean ALC over {len(results)} benchmarks: {mean:.4f}  ({time.time()-t_all:.0f}s)  args={vars(a)}")
    print("JSON " + json.dumps({"tag": a.tag, "regime": regime, "mean": round(mean, 5), "per": {k: round(v, 5) for k, v in results.items()},
                                "B": {h: {str(k): round(v, 4) for k, v in b.items()} for h, b in allB.items()}, "args": vars(a)}))


if __name__ == "__main__":
    main()
