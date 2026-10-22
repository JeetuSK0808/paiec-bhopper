"""Leave-one-benchmark-out CV with the competition's protocol: cold-start benchmark; per (subject, benchmark) with >= 80 items the
items split 50/50 into acquisition / evaluation pools; labels revealed at budgets 0,1,3,7,15,31 from the acquisition pool;
Brier on the evaluation pool; Brier ALC = 0.1 B0 + 0.2 (B1+B3+B7+B15) + 0.1 B31.
  .venv/bin/python scripts/paie_cv.py [--model gbm|mean|half|official|subj] [--acq random|uncertain|common|spread] [--tau 1.0] [--tau-i 1.0] [--pooled]"""
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from paie.data import item_features, load_all, subject_features  # noqa: E402

BUDGETS = [0, 1, 3, 7, 15, 31]; WEIGHTS = {0: 0.1, 1: 0.2, 3: 0.2, 7: 0.2, 15: 0.2, 31: 0.1}
SUBJ_SCALE = 1.0  # damping of the carried-over subject offset for --model subj (set from --subj-scale)


def logit(p): p = np.clip(p, 1e-4, 1 - 1e-4); return np.log(p / (1 - p))
def sigmoid(z): return 1 / (1 + np.exp(-z))


def fit_p0(train, sf, itf, model):
    from sklearn.ensemble import HistGradientBoostingRegressor
    X = train.merge(sf, on="subject_id").merge(itf, on="item_id")
    feats = [c for c in X.columns if c not in ("subject_id", "item_id", "benchmark", "y", "n_trials")]
    if model in ("mean", "official"):
        mu = float(train["y"].mean()); return (lambda df: np.full(len(df), mu)), feats
    if model == "half":  # flat 0.5 cold-start prior (benchmarks differ too much for a cross-benchmark mean)
        return (lambda df: np.full(len(df), 0.5)), feats
    if model == "subj":
        # cold-start prior = the subject's RELATIVE strength carried over from the training benchmarks: per benchmark, the
        # subject's accuracy logit minus that benchmark's mean logit (so benchmark difficulty cancels), averaged over the
        # benchmarks the subject appears in and shrunk by n/(n+1); unknown subjects get 0.5. The pooled adaptation then adds
        # the benchmark level, subject and item corrections from revealed labels as before.
        acc = train.groupby(["benchmark", "subject_id"])["y"].mean().reset_index()
        acc["lz"] = logit(acc["y"].values)
        acc["rel"] = acc["lz"] - acc.groupby("benchmark")["lz"].transform("mean")
        g = acc.groupby("subject_id")["rel"].agg(["mean", "count"])
        off = (g["mean"] * g["count"] / (g["count"] + 1.0) * SUBJ_SCALE).to_dict()
        return (lambda df: sigmoid(np.array([off.get(s_, 0.0) for s_ in df["subject_id"].values]))), feats
    reg = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06, max_leaf_nodes=31, l2_regularization=1.0, loss="squared_error").fit(X[feats], X["y"])
    def predict(df):
        Z = df.merge(sf, on="subject_id", how="left").merge(itf, on="item_id", how="left")
        return np.clip(reg.predict(Z[feats].fillna(0)), 0.02, 0.98)
    return predict, feats


def adapt(p0_eval, p0_acq, y_acq, k, tau):
    """Own-labels-only shift (kept for reference): one Newton step for delta_s ~ N(0, tau^2)."""
    if k == 0:
        return p0_eval
    y, p = y_acq[:k], p0_acq[:k]
    delta = np.sum(y - p) / (np.sum(p * (1 - p)) + 1.0 / tau ** 2)
    return sigmoid(logit(p0_eval) + delta)


def adapt_pooled(subj_eval, item_eval, p0_eval, revealed, tau_b=1.0, tau_s=1.0, tau_i=1.0, sweeps=3):
    """Pooled adaptation from ALL revealed labels of the benchmark: z = logit(p0) + beta_b + delta_s + eps_i, each a shrunk
    Newton step (normal priors with std tau_*), refined by a few alternating sweeps. `revealed` = arrays (subj, item, p0, y)."""
    if len(revealed[3]) == 0:
        return p0_eval
    rs, ri, rp0, ry = revealed
    z0 = logit(rp0)
    beta = 0.0; delta = {}; eps = {}
    subs, items = np.unique(rs), np.unique(ri)
    for _ in range(sweeps):
        z = z0 + beta + np.array([delta.get(s_, 0.0) for s_ in rs]) + np.array([eps.get(i_, 0.0) for i_ in ri])
        p = sigmoid(z)
        beta += np.sum(ry - p) / (np.sum(p * (1 - p)) + 1.0 / tau_b ** 2)
        z = z0 + beta + np.array([delta.get(s_, 0.0) for s_ in rs]) + np.array([eps.get(i_, 0.0) for i_ in ri]); p = sigmoid(z)
        for s_ in subs:
            m = rs == s_
            delta[s_] = delta.get(s_, 0.0) + np.sum(ry[m] - p[m]) / (np.sum(p[m] * (1 - p[m])) + 1.0 / tau_s ** 2)
        z = z0 + beta + np.array([delta.get(s_, 0.0) for s_ in rs]) + np.array([eps.get(i_, 0.0) for i_ in ri]); p = sigmoid(z)
        for i_ in items:
            m = ri == i_
            eps[i_] = eps.get(i_, 0.0) + np.sum(ry[m] - p[m]) / (np.sum(p[m] * (1 - p[m])) + 1.0 / tau_i ** 2)
    z_eval = logit(p0_eval) + beta + np.array([delta.get(s_, 0.0) for s_ in subj_eval]) + np.array([eps.get(i_, 0.0) for i_ in item_eval])
    return sigmoid(z_eval)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", default="gbm"); ap.add_argument("--acq", default="random"); ap.add_argument("--tau", type=float, default=1.0); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--pooled", action="store_true"); ap.add_argument("--tau-i", type=float, default=1.0); ap.add_argument("--subj-scale", type=float, default=1.0)
    a = ap.parse_args()
    global SUBJ_SCALE
    SUBJ_SCALE = a.subj_scale
    pairs, subjects, items = load_all()
    sf, itf = subject_features(subjects.drop_duplicates("subject_id")), item_features(items.drop_duplicates("item_id"))
    rng = np.random.default_rng(a.seed)
    print(f"{len(pairs)} pairs, {pairs['subject_id'].nunique()} subjects, {pairs['item_id'].nunique()} items, benchmarks {sorted(pairs['benchmark'].unique())}")
    results = []
    for held in sorted(pairs["benchmark"].unique()):
        train, test = pairs[pairs["benchmark"] != held], pairs[pairs["benchmark"] == held]
        predict, feats = fit_p0(train, sf, itf, a.model)
        briers = {k: [] for k in BUDGETS}
        pools = []
        for sid, g in test.groupby("subject_id"):
            if len(g) < 80:
                continue
            g = g.sample(frac=1.0, random_state=int(rng.integers(1e9)))
            half = len(g) // 2
            acq, ev = g.iloc[:half].copy(), g.iloc[half:].copy()
            p_acq, p_ev = predict(acq), predict(ev)
            order = np.argsort(np.abs(p_acq - 0.5)) if a.acq == "uncertain" else np.arange(len(acq))
            pools.append((sid, acq.iloc[order], p_acq[order], ev, p_ev))
        if a.acq in ("common", "spread"):
            # common: reveal the items most other subjects also reveal (clean subject effects, few distinct items -> was WORSE:
            # 0.180 vs 0.165). spread: the opposite -- rarest items first, so revealed labels cover as many distinct items
            # (and hence as many evaluation items) as possible.
            # reveal the items most other subjects also reveal: item effects become well estimated, so each revealed label
            # identifies the subject effect instead of being confounded with an unknown item difficulty
            from collections import Counter
            cnt = Counter()
            for _, acq, *_ in pools:
                cnt.update(acq["item_id"].tolist())
            sign = -1 if a.acq == "common" else 1
            pools = [(sid, acq.iloc[o], p_acq[o], ev, p_ev) for sid, acq, p_acq, ev, p_ev in pools
                     for o in [np.argsort([sign * cnt[i] for i in acq["item_id"].values], kind="stable")]]
        n_eval_subjects = len(pools)
        for k in BUDGETS:
            if a.pooled:
                rs = np.concatenate([[sid] * min(k, len(acq)) for sid, acq, *_ in pools]) if k else np.array([])
                ri = np.concatenate([acq["item_id"].values[:k] for _, acq, *_ in pools]) if k else np.array([])
                rp = np.concatenate([p_acq[:k] for _, _, p_acq, *_ in pools]) if k else np.array([])
                ry = np.concatenate([acq["y"].values[:k] for _, acq, *_ in pools]) if k else np.array([])
                revealed = (rs, ri, rp, ry)
            for sid, acq, p_acq, ev, p_ev in pools:
                if a.model == "official":  # organizers' empirical-mean baseline: 0.5 without own labels, else their unsmoothed mean
                    kk = min(k, len(acq))
                    pk = np.full(len(ev), 0.5 if kk == 0 else float(np.mean(acq["y"].values[:kk])))
                elif a.pooled:
                    pk = adapt_pooled(np.array([sid] * len(ev)), ev["item_id"].values, p_ev, revealed, a.tau, a.tau, a.tau_i)
                else:
                    pk = adapt(p_ev, p_acq, acq["y"].values, min(k, len(acq)), a.tau)
                briers[k].append(np.mean((pk - ev["y"].values) ** 2))
        if n_eval_subjects == 0:
            print(f"{held:18s} no subject with >= 80 items — skipped"); continue
        B = {k: float(np.mean(v)) for k, v in briers.items()}
        alc = sum(WEIGHTS[k] * B[k] for k in BUDGETS)
        results.append((held, n_eval_subjects, alc, B))
        print(f"{held:18s} subjects {n_eval_subjects:3d}  ALC {alc:.4f}   " + " ".join(f"B{k}={B[k]:.3f}" for k in BUDGETS))
    if results:
        print(f"\nmean ALC over {len(results)} held-out benchmarks: {np.mean([r[2] for r in results]):.4f}   (model={a.model}, acq={a.acq}, tau={a.tau}, pooled={a.pooled}, tau_i={a.tau_i}, subj_scale={a.subj_scale})")


if __name__ == "__main__":
    main()
