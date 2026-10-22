"""CV harness variant: pooled hierarchical adaptation WITH an item-difficulty prior regressed from item text.
At each budget: pooled fit (beta, delta_s, eps_i with eps shrunk toward a prior mean m_i) -> ridge from item text features to
the fitted eps of labelled items -> m_i for every item (labelled or not) -> refit. Evaluation items are never labelled for the
target subject and often not at all, so m_i is where the text helps.
  .venv/bin/python scripts/item_prior_cv.py --feat tfidf|emb --emb results/item_emb.npz --alpha 10 --rounds 2 --shrink 1.0
"""
import argparse, sys, time
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from paie.data import load_all, subject_features  # noqa: E402
from paie_cv import BUDGETS, WEIGHTS, logit, sigmoid  # noqa: E402


feat_with_features = False


def item_matrix(items_df, feat, emb_path, dim):
    ids = items_df["item_id"].values
    if feat == "hash":  # raw hashed counts; the ridge is done by TextPrior (numpy-only, same code as the submission)
        from paie.textprior import hash_counts
        txt = (items_df["content"].fillna("").astype(str) + " " + items_df["item_features"].fillna("").astype(str).str.replace("[=;]", " ", regex=True)).values if feat_with_features else items_df["content"].fillna("").astype(str).values
        return dict(zip(ids, [hash_counts(t, dim=4096) for t in txt]))
    if feat == "emb":
        z = np.load(emb_path, allow_pickle=True); lookup = dict(zip(z["item_id"], z["emb"]))
        X = np.stack([lookup[i] for i in ids])
    else:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD
        txt = items_df["content"].fillna("").astype(str).str.slice(0, 20000).values
        T = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=60000, sublinear_tf=True).fit_transform(txt)
        k = min(dim, T.shape[1] - 1, T.shape[0] - 1)
        X = TruncatedSVD(n_components=k, random_state=0).fit_transform(T)
    X = X - X.mean(0); X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    return dict(zip(ids, X))


def fit_pooled(rs, ri, ry, m_item, tau_b, tau_s, tau_i, sweeps=3):
    """Returns beta, delta{s}, eps{i}; eps shrunk toward m_item[i] (prior mean) instead of 0."""
    beta, delta, eps = 0.0, {}, {}
    subs, items = np.unique(rs), np.unique(ri)
    for i_ in items:
        eps[i_] = m_item.get(i_, 0.0)
    def z():
        return beta + np.array([delta.get(s_, 0.0) for s_ in rs]) + np.array([eps[i_] for i_ in ri])
    for _ in range(sweeps):
        p = sigmoid(z()); beta += np.sum(ry - p) / (np.sum(p * (1 - p)) + 1.0 / tau_b ** 2)
        p = sigmoid(z())
        for s_ in subs:
            m = rs == s_; delta[s_] = delta.get(s_, 0.0) + np.sum(ry[m] - p[m]) / (np.sum(p[m] * (1 - p[m])) + 1.0 / tau_s ** 2)
        p = sigmoid(z())
        for i_ in items:
            m = ri == i_; mi = m_item.get(i_, 0.0)
            eps[i_] += (np.sum(ry[m] - p[m]) - (eps[i_] - mi) / tau_i ** 2) / (np.sum(p[m] * (1 - p[m])) + 1.0 / tau_i ** 2)
    return beta, delta, eps


def fit_pooled_s(rs, ri, ry, m_item, m_subj, tau_b, tau_s, tau_i, sweeps=3):
    beta, delta, eps = 0.0, {}, {}
    subs, items = np.unique(rs), np.unique(ri)
    for i_ in items:
        eps[i_] = m_item.get(i_, 0.0)
    for s_ in subs:
        delta[s_] = m_subj.get(s_, 0.0)
    def z():
        return beta + np.array([delta[s_] for s_ in rs]) + np.array([eps[i_] for i_ in ri])
    for _ in range(sweeps):
        p = sigmoid(z()); beta += np.sum(ry - p) / (np.sum(p * (1 - p)) + 1.0 / tau_b ** 2)
        p = sigmoid(z())
        for s_ in subs:
            m = rs == s_; ms = m_subj.get(s_, 0.0)
            delta[s_] += (np.sum(ry[m] - p[m]) - (delta[s_] - ms) / tau_s ** 2) / (np.sum(p[m] * (1 - p[m])) + 1.0 / tau_s ** 2)
        p = sigmoid(z())
        for i_ in items:
            m = ri == i_; mi = m_item.get(i_, 0.0)
            eps[i_] += (np.sum(ry[m] - p[m]) - (eps[i_] - mi) / tau_i ** 2) / (np.sum(p[m] * (1 - p[m])) + 1.0 / tau_i ** 2)
    return beta, delta, eps


def ridge_prior(eps, counts, Xmap, alpha, all_items, shrink, feat="tfidf"):
    from sklearn.linear_model import Ridge
    ids = [i for i in eps if i in Xmap]
    if len(ids) < 10:
        return {}
    if feat == "hash":
        from paie.textprior import TextPrior
        C_all = np.stack([Xmap[i] for i in all_items]); C_lab = np.stack([Xmap[i] for i in ids])
        tp = TextPrior(alpha=alpha).fit(C_all, C_lab, np.array([eps[i] for i in ids]), np.array([counts[i] for i in ids], float))
        return dict(zip(all_items, tp.predict(C_all) * shrink))
    X = np.stack([Xmap[i] for i in ids]); y = np.array([eps[i] for i in ids]); w = np.array([counts[i] for i in ids], float)
    r = Ridge(alpha=alpha).fit(X, y, sample_weight=w)
    Xa = np.stack([Xmap[i] for i in all_items]); pred = r.predict(Xa) * shrink
    return dict(zip(all_items, pred))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--feat", default="tfidf"); ap.add_argument("--emb", default=str(ROOT / "results/item_emb.npz"))
    ap.add_argument("--dim", type=int, default=128); ap.add_argument("--alpha", type=float, default=10.0); ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--shrink", type=float, default=1.0); ap.add_argument("--tau", type=float, default=0.5); ap.add_argument("--tau-i", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--only", default="")
    ap.add_argument("--cross", type=float, default=0.0, help="weight of the cross-benchmark text->difficulty prior (0 = off)")
    ap.add_argument("--cross-alpha", type=float, default=10.0); ap.add_argument("--subj-prior", type=float, default=0.0, help="ridge alpha for subject metadata prior (0 = off)")
    ap.add_argument("--with-features", action="store_true"); ap.add_argument("--acq", default="random"); ap.add_argument("--sp", type=float, default=0.0, help="subject-prior scale (0=off)"); ap.add_argument("--sp-kw", type=float, default=0.6); ap.add_argument("--sp-mw", type=float, default=0.5); ap.add_argument("--sp-b0", action="store_true"); ap.add_argument("--force-s", action="store_true"); ap.add_argument("--single", action="store_true", help="each subject alone (no other subjects' labels), like the dev test"); ap.add_argument("--cross-rel", type=float, default=0.0, help="weight of a RELATIVE cross-benchmark text->difficulty prior")
    a = ap.parse_args()
    global feat_with_features; feat_with_features = a.with_features
    pairs, subjects, items = load_all(); items = items.drop_duplicates("item_id")
    rng = np.random.default_rng(a.seed); results = []
    for held in sorted(pairs["benchmark"].unique()):
        if a.only and held != a.only:
            continue
        t0 = time.time(); test = pairs[pairs["benchmark"] == held]
        Xmap = item_matrix(items[items["benchmark"] == held], a.feat, a.emb, a.dim) if a.rounds > 0 else {}
        pools = []
        for sid, g in test.groupby("subject_id"):
            if len(g) < 80:
                continue
            g = g.sample(frac=1.0, random_state=int(rng.integers(1e9))); half = len(g) // 2
            acq = g.iloc[:half]
            if a.acq == "fps" and Xmap:
                X = np.stack([Xmap[i] for i in acq["item_id"].values]); X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
                order = [0]; dmin = 1 - X @ X[0]
                for _ in range(min(31, len(acq)) - 1):
                    j = int(np.argmax(dmin)); order.append(j); dmin = np.minimum(dmin, 1 - X @ X[j])
                rest = [i for i in range(len(acq)) if i not in set(order)]
                acq = acq.iloc[order + rest]
            elif a.acq == "uncertain_len":  # longest items first (proxy for hard/informative)
                acq = acq.iloc[np.argsort(-acq["item_id"].map(lambda i: Xmap[i].sum() if i in Xmap else 0).values, kind="stable")]
            pools.append((sid, acq, g.iloc[half:]))
        if not pools:
            print(f"{held:18s} skipped (no subject with >= 80 items)"); continue
        all_items = list(test["item_id"].unique())
        # cross-benchmark prior: ridge from text features (fit on ALL items so held-out items share the space) to the item's
        # logit mean accuracy on the training benchmarks; the intercept carries the base rate, so it is only a rough B0 guess.
        m_cross = {}
        if a.cross > 0:
            from sklearn.linear_model import Ridge
            Xall = item_matrix(items, a.feat, a.emb, a.dim)
            tr = pairs[pairs["benchmark"] != held].groupby("item_id")["y"].agg(["mean", "size"]).reset_index()
            tr = tr[tr["item_id"].isin(Xall)]
            r = Ridge(alpha=a.cross_alpha).fit(np.stack([Xall[i] for i in tr["item_id"]]), logit(tr["mean"].values), sample_weight=np.sqrt(tr["size"].values))
            pc = r.predict(np.stack([Xall[i] for i in all_items]))
            m_cross = dict(zip(all_items, a.cross * pc))
        m_subj = {}
        if a.sp > 0:
            from paie.subjprior import SubjectPrior
            sd = subjects.drop_duplicates("subject_id").set_index("subject_id")
            def sdict(sid):
                r = sd.loc[sid]; return {k: ("" if pd.isna(r.get(k)) else str(r.get(k))) for k in ("normalized_name", "provider", "release_date", "reasoning_effort", "subject_features_extra")}
            trp = pairs[pairs["benchmark"] != held].groupby(["benchmark", "subject_id"]).y.agg(["mean", "size"]).reset_index()
            spm = SubjectPrior(known_weight=a.sp_kw, meta_weight=a.sp_mw).fit([(sdict(r.subject_id), r.benchmark, r["mean"], r["size"]) for _, r in trp.iterrows()])
            m_subj = {sid: a.sp * spm.offset(sdict(sid)) for sid, _, _ in pools}
        if a.force_s and not m_subj:
            m_subj = {sid: 0.0 for sid, _, _ in pools}
        if a.cross_rel > 0:
            from paie.textprior import hash_counts, TextPrior
            it_all = items; C = {i_: hash_counts(t_) for i_, t_ in zip(it_all["item_id"], it_all["content"].fillna("").astype(str))}
            trp = pairs[pairs["benchmark"] != held].copy()
            lg = lambda p_: np.log(np.clip(p_, .02, .98) / (1 - np.clip(p_, .02, .98)))
            ia = trp.groupby(["benchmark", "item_id"]).y.agg(["mean", "size"]).reset_index(); ia["rel"] = lg(ia["mean"]) - ia.groupby("benchmark")["mean"].transform(lambda v: lg(v).mean())
            ia = ia[ia["item_id"].isin(C)]
            C_all = np.stack([C[i_] for i_ in list(ia["item_id"]) + all_items]); C_lab = np.stack([C[i_] for i_ in ia["item_id"]])
            tpx = TextPrior(alpha=3.0).fit(C_all, C_lab, ia["rel"].values, np.sqrt(ia["size"].values.astype(float)))
            pr = tpx.predict(np.stack([C[i_] for i_ in all_items])); pr = pr - pr.mean()
            m_cross = dict(zip(all_items, a.cross_rel * pr))
        sf_map = {}
        if a.subj_prior > 0:
            sf = subject_features(subjects.drop_duplicates("subject_id")); cols = [c for c in sf.columns if c != "subject_id"]
            sf_map = dict(zip(sf["subject_id"], sf[cols].values.astype(float)))
        B = {}
        for k in BUDGETS:
            if k == 0:
                if m_cross:
                    B[k] = float(np.mean([np.mean((sigmoid(np.array([m_cross[i_] for i_ in ev["item_id"].values])) - ev["y"].values) ** 2) for _, _, ev in pools]))
                else:
                    B[k] = float(np.mean([np.mean((0.5 - ev["y"].values) ** 2) for _, _, ev in pools]))
                continue
            if a.single:
                bs = []
                for sid, acq, ev in pools:
                    rs1 = np.array([sid] * min(k, len(acq))); ri1 = acq["item_id"].values[:k]; ry1 = acq["y"].values[:k].astype(float)
                    cnt1 = pd.Series(ri1).value_counts().to_dict(); m1 = dict(m_cross); ms1 = {sid: m_subj.get(sid, 0.0)} if (m_subj or a.force_s) else {}
                    fp1 = (lambda r_, i_, y_, mi_: fit_pooled_s(r_, i_, y_, mi_, ms1, a.tau, a.tau, a.tau_i)) if ms1 else (lambda r_, i_, y_, mi_: fit_pooled(r_, i_, y_, mi_, a.tau, a.tau, a.tau_i))
                    b1, d1, e1 = fp1(rs1, ri1, ry1, m1)
                    for _ in range(a.rounds):
                        m1 = ridge_prior(e1, cnt1, Xmap, a.alpha, all_items, a.shrink, a.feat) or m1; b1, d1, e1 = fp1(rs1, ri1, ry1, m1)
                    zev = b1 + d1.get(sid, ms1.get(sid, 0.0)) + np.array([e1.get(i_, m1.get(i_, 0.0)) for i_ in ev["item_id"].values])
                    bs.append(np.mean((sigmoid(zev) - ev["y"].values) ** 2))
                B[k] = float(np.mean(bs)); continue
            rs = np.concatenate([[sid] * min(k, len(acq)) for sid, acq, _ in pools]); ri = np.concatenate([acq["item_id"].values[:k] for _, acq, _ in pools])
            ry = np.concatenate([acq["y"].values[:k] for _, acq, _ in pools])
            counts = pd.Series(ri).value_counts().to_dict()
            m_item = dict(m_cross)
            fp = (lambda rs_, ri_, ry_, mi_: fit_pooled_s(rs_, ri_, ry_, mi_, m_subj, a.tau, a.tau, a.tau_i)) if (m_subj or a.force_s) else (lambda rs_, ri_, ry_, mi_: fit_pooled(rs_, ri_, ry_, mi_, a.tau, a.tau, a.tau_i))
            beta, delta, eps = fp(rs, ri, ry, m_item)
            for _ in range(a.rounds):
                m_item = ridge_prior(eps, counts, Xmap, a.alpha, all_items, a.shrink, a.feat)
                beta, delta, eps = fp(rs, ri, ry, m_item)
            if sf_map and k <= 7:
                # subject metadata prior: ridge from subject features to the fitted delta, then refit with delta shrunk toward it
                from sklearn.linear_model import Ridge
                sids = [s_ for s_ in delta if s_ in sf_map]
                if len(sids) >= 8:
                    r = Ridge(alpha=a.subj_prior).fit(np.stack([sf_map[s_] for s_ in sids]), np.array([delta[s_] for s_ in sids]))
                    m_s = dict(zip(sids, r.predict(np.stack([sf_map[s_] for s_ in sids]))))
                    beta, delta, eps = fit_pooled_s(rs, ri, ry, m_item, m_s, a.tau, a.tau, a.tau_i)
            bs = []
            for sid, acq, ev in pools:
                zev = beta + delta.get(sid, m_subj.get(sid, 0.0)) + np.array([eps.get(i_, m_item.get(i_, 0.0)) for i_ in ev["item_id"].values])
                bs.append(np.mean((sigmoid(zev) - ev["y"].values) ** 2))
            B[k] = float(np.mean(bs))
        alc = sum(WEIGHTS[k] * B[k] for k in BUDGETS); results.append(alc)
        print(f"{held:18s} subjects {len(pools):3d}  ALC {alc:.4f}   " + " ".join(f"B{k}={B[k]:.3f}" for k in BUDGETS) + f"   ({time.time()-t0:.0f}s)", flush=True)
    print(f"\nmean ALC over {len(results)} benchmarks: {np.mean(results):.4f}  (feat={a.feat}, dim={a.dim}, alpha={a.alpha}, rounds={a.rounds}, shrink={a.shrink}, tau={a.tau}, tau_i={a.tau_i}, cross={a.cross}/{a.cross_alpha}, subj_prior={a.subj_prior})")


if __name__ == "__main__":
    main()
