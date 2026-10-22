"""exp_calib: leave-one-benchmark-out CV harness for calibration / empirical-Bayes experiments (tag 'calib').

Reproduces scripts/item_prior_cv.py (--feat hash --alpha A --rounds R --tau T --tau-i TI --sp SP [--single]) with vectorised
block-Jacobi Newton sweeps (np.bincount instead of per-item Python loops; same update equations and order) and a cached TF-IDF matrix
per held-out benchmark, and DUMPS every evaluation prediction with the fit internals needed for offline calibration:
  y, z (model logit), beta, delta_s, e_i (eval-item effect used), n_i (labels on the eval item), Laplace variances
  (v_beta, v_s, v_i), the subject prior m_s, the pair's own revealed labels (k_own, sum_own), benchmark/pair/budget ids.
Extra online variants (all default off = baseline):
  --beta-prior      include the -beta/tau_b^2 prior gradient in the beta step (as shipped in submission_v5/model.py)
  --tau-sched       budget-dependent tau for beta/delta, e.g. "1:0.6,3:0.7,7:0.8,15:0.8,31:0.8"
  --taui-sched      budget-dependent tau_i, same format
Run (CPU etiquette):  nice -n 10 env VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
   .venv/bin/python scripts/exp_calib_cv.py --tau 0.8 --tau-i 1.5 --sp 0.5 [--single] --dump src/paie_exp/calib/dump_single.npz
"""
import argparse, sys, time
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from paie.data import load_all  # noqa: E402
from paie.textprior import hash_counts, TextPrior  # noqa: E402
from paie_cv import BUDGETS, WEIGHTS  # noqa: E402


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def parse_sched(s, default):
    d = {k: default for k in BUDGETS}
    if s:
        for tok in s.split(","):
            k, v = tok.split(":"); d[int(k)] = float(v)
    return d


class TfidfCache:
    """TF-IDF rows for all items of one benchmark (idf from all items, exactly as TextPrior.fit computes it every call)."""

    def __init__(self, C_all, alpha):
        tp = TextPrior(alpha=alpha)
        df = (C_all > 0).sum(0).astype(np.float64)
        tp.idf = (np.log((1 + C_all.shape[0]) / (1 + df)) + 1.0).astype(np.float32)
        self.X = tp._tfidf(C_all).astype(np.float64); self.alpha = alpha

    def prior(self, lab_idx, y, w):
        """Weighted dual ridge on rows lab_idx -> prediction for all items (TextPrior.fit + predict)."""
        X = self.X[lab_idx]; mean_y = float(np.average(y, weights=w)); yc = y - mean_y
        K = X @ X.T + self.alpha * np.diag(1.0 / w)
        a = np.linalg.solve(K, yc)
        return self.X @ (X.T @ a) + mean_y


def fit_vec(rs, ri, ry, n_s, n_i, m_item, m_subj, tau_b, tau_s, tau_i, sweeps=3, beta_prior=False):
    """Same as item_prior_cv.fit_pooled_s (block-Jacobi MAP-ish Newton sweeps), vectorised. rs/ri integer codes.
    Returns beta, delta[n_s], eps[n_i] and Laplace variances (v_beta, v_s[n_s], v_i[n_i])."""
    beta = 0.0; delta = m_subj.copy(); eps = m_item.copy()
    for _ in range(sweeps):
        p = sigmoid(beta + delta[rs] + eps[ri])
        beta += (np.sum(ry - p) - (beta / tau_b ** 2 if beta_prior else 0.0)) / (np.sum(p * (1 - p)) + 1.0 / tau_b ** 2)
        p = sigmoid(beta + delta[rs] + eps[ri]); w = p * (1 - p)
        delta = delta + (np.bincount(rs, ry - p, n_s) - (delta - m_subj) / tau_s ** 2) / (np.bincount(rs, w, n_s) + 1.0 / tau_s ** 2)
        p = sigmoid(beta + delta[rs] + eps[ri]); w = p * (1 - p)
        eps = eps + (np.bincount(ri, ry - p, n_i) - (eps - m_item) / tau_i ** 2) / (np.bincount(ri, w, n_i) + 1.0 / tau_i ** 2)
    p = sigmoid(beta + delta[rs] + eps[ri]); w = p * (1 - p)
    vb = 1.0 / (np.sum(w) + 1.0 / tau_b ** 2)
    vs = 1.0 / (np.bincount(rs, w, n_s) + 1.0 / tau_s ** 2)
    vi = 1.0 / (np.bincount(ri, w, n_i) + 1.0 / tau_i ** 2)
    return beta, delta, eps, vb, vs, vi


def fit_with_text(rs, ri, ry, n_s, n_i, m_subj, tc, rounds, tau_b, tau_s, tau_i, beta_prior, single, tscale=1.0):
    """Pooled fit + `rounds` text-prior refits (ridge from TF-IDF to fitted eps of labelled items, weights = label counts)."""
    m_item = np.zeros(n_i)
    beta, delta, eps, vb, vs, vi = fit_vec(rs, ri, ry, n_s, n_i, m_item, m_subj, tau_b, tau_s, tau_i, beta_prior=beta_prior)
    cnt = np.bincount(ri, minlength=n_i); lab = np.nonzero(cnt)[0]  # sorted codes == sorted order of np.unique(ri)
    for _ in range(rounds):
        if len(lab) < 10:
            if single:  # `m1 = ridge_prior(...) or m1`: keep the previous (empty) prior
                pass
            else:       # pooled: m_item = {} -> zeros
                m_item = np.zeros(n_i)
        else:
            m_item = tc.prior(lab, eps[lab], cnt[lab].astype(float)) * tscale
        beta, delta, eps, vb, vs, vi = fit_vec(rs, ri, ry, n_s, n_i, m_item, m_subj, tau_b, tau_s, tau_i, beta_prior=beta_prior)
    return beta, delta, eps, vb, vs, vi, m_item, cnt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=1.0); ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--tau", type=float, default=0.8); ap.add_argument("--tau-i", type=float, default=1.5)
    ap.add_argument("--sp", type=float, default=0.5); ap.add_argument("--sp-kw", type=float, default=0.6); ap.add_argument("--sp-mw", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--only", default=""); ap.add_argument("--single", action="store_true")
    ap.add_argument("--beta-prior", action="store_true"); ap.add_argument("--tau-sched", default=""); ap.add_argument("--taui-sched", default="")
    ap.add_argument("--binary-labels", action="store_true", help="reveal Bernoulli(y) labels (official labels are binary); eval targets stay y")
    ap.add_argument("--dump", default=""); ap.add_argument("--json", default="", help="write per-benchmark B_k to this json")
    ap.add_argument("--em-tau-s", type=int, default=0, help="EM-estimate tau_s when >= this many systems share the benchmark (0 = off)")
    ap.add_argument("--em-iters", type=int, default=3); ap.add_argument("--em-lo", type=float, default=0.5); ap.add_argument("--em-hi", type=float, default=2.5); ap.add_argument("--em-refit", action="store_true")
    ap.add_argument("--group-size", type=int, default=0, help="pooled mode: fit systems in random groups of this size (0 = all together)")
    ap.add_argument("--tau-rule", default="", help="tau from #systems n in the fit: '1:0.7,2:0.9,8:1.25' = tau of the largest threshold <= n (overrides --tau)")
    ap.add_argument("--text-scale", type=float, default=1.0, help="multiply the text prior m_i (= item_prior_cv --shrink)"); ap.add_argument("--tscale-sched", default="")
    a = ap.parse_args()
    tau_k = parse_sched(a.tau_sched, a.tau); taui_k = parse_sched(a.taui_sched, a.tau_i); ts_k = parse_sched(a.tscale_sched, a.text_scale)
    rule = sorted((int(x.split(":")[0]), float(x.split(":")[1])) for x in a.tau_rule.split(",")) if a.tau_rule else []
    def rule_tau(n, default):
        t = default
        for th, v in rule:
            if n >= th:
                t = v
        return t
    pairs, subjects, items = load_all(); items = items.drop_duplicates("item_id")
    rng = np.random.default_rng(a.seed); results = []; brng = np.random.default_rng(12345); JS = {}; TS_LOG = []
    D = {k: [] for k in ("bench", "pair", "k", "y", "z", "beta", "delta", "e", "n_i", "vb", "vs", "vi", "ms", "k_own", "sum_own")}
    bench_names = sorted(pairs["benchmark"].unique()); pair_names = []
    for bi, held in enumerate(bench_names):
        if a.only and held != a.only:
            continue
        t0 = time.time(); test = pairs[pairs["benchmark"] == held]
        pools = []
        for sid, g in test.groupby("subject_id"):
            if len(g) < 80:
                continue
            g = g.sample(frac=1.0, random_state=int(rng.integers(1e9))); half = len(g) // 2
            pools.append((sid, g.iloc[:half], g.iloc[half:]))
        if not pools:
            print(f"{held:18s} skipped (no subject with >= 80 items)"); continue
        all_items = list(test["item_id"].unique()); icode = {i_: j for j, i_ in enumerate(all_items)}; n_i = len(all_items)
        it = items[items["benchmark"] == held].set_index("item_id")
        C_all = np.stack([hash_counts(str(it.loc[i_, "content"]) if pd.notna(it.loc[i_, "content"]) else "", dim=4096) for i_ in all_items])
        tc = TfidfCache(C_all, a.alpha)
        m_subj_d = {}
        if a.sp > 0:
            from paie.subjprior import SubjectPrior
            sd = subjects.drop_duplicates("subject_id").set_index("subject_id")
            def sdict(sid):
                r = sd.loc[sid]; return {k: ("" if pd.isna(r.get(k)) else str(r.get(k))) for k in ("normalized_name", "provider", "release_date", "reasoning_effort", "subject_features_extra")}
            trp = pairs[pairs["benchmark"] != held].groupby(["benchmark", "subject_id"]).y.agg(["mean", "size"]).reset_index()
            spm = SubjectPrior(known_weight=a.sp_kw, meta_weight=a.sp_mw).fit([(sdict(r.subject_id), r.benchmark, r["mean"], r["size"]) for _, r in trp.iterrows()])
            m_subj_d = {sid: a.sp * spm.offset(sdict(sid)) for sid, _, _ in pools}
        scode = {sid: j for j, (sid, _, _) in enumerate(pools)}; n_s = len(pools)
        m_subj = np.array([m_subj_d.get(sid, 0.0) for sid, _, _ in pools])
        pid0 = len(pair_names); pair_names += [f"{held}|{sid}" for sid, _, _ in pools]
        acq_codes = [np.array([icode[i_] for i_ in acq["item_id"].values]) for _, acq, _ in pools]
        acq_y = [acq["y"].values.astype(float) for _, acq, _ in pools]
        if a.binary_labels:
            acq_y = [(brng.random(len(v)) < v).astype(float) for v in acq_y]
        ev_codes = [np.array([icode[i_] for i_ in ev["item_id"].values]) for _, _, ev in pools]
        ev_y = [ev["y"].values.astype(float) for _, _, ev in pools]
        B = {}

        def record(j, k, z, beta, delta_j, e, n_ie, vb, vs_j, vi_e, kk):
            n = len(z)
            D["bench"].append(np.full(n, bi)); D["pair"].append(np.full(n, pid0 + j)); D["k"].append(np.full(n, k)); D["y"].append(ev_y[j])
            D["z"].append(z); D["beta"].append(np.full(n, beta)); D["delta"].append(np.full(n, delta_j)); D["e"].append(e); D["n_i"].append(n_ie)
            D["vb"].append(np.full(n, vb)); D["vs"].append(np.full(n, vs_j)); D["vi"].append(vi_e); D["ms"].append(np.full(n, m_subj[j]))
            D["k_own"].append(np.full(n, kk)); D["sum_own"].append(np.full(n, float(acq_y[j][:kk].sum())))

        for k in BUDGETS:
            bs = []
            if k == 0:
                for j in range(n_s):
                    n = len(ev_y[j]); z = np.zeros(n)
                    record(j, 0, z, 0.0, m_subj[j], np.zeros(n), np.zeros(n), tau_k[1] ** 2, tau_k[1] ** 2, np.full(n, taui_k[1] ** 2), 0)
                    bs.append(np.mean((0.5 - ev_y[j]) ** 2))
                B[k] = float(np.mean(bs)); continue
            tb = ts = tau_k[k]; ti = taui_k[k]
            if a.single:
                for j in range(n_s):
                    kk = min(k, len(acq_codes[j])); rs = np.zeros(kk, dtype=int); ri = acq_codes[j][:kk]; ry = acq_y[j][:kk]
                    t1 = rule_tau(1, tb)
                    beta, delta, eps, vb, vs, vi, m_item, cnt = fit_with_text(rs, ri, ry, 1, n_i, m_subj[j:j + 1], tc, a.rounds, t1, t1, ti, a.beta_prior, True, ts_k[k])
                    ec = ev_codes[j]; z = beta + delta[0] + eps[ec]
                    record(j, k, z, beta, delta[0], eps[ec], cnt[ec].astype(float), vb, vs[0], vi[ec], kk)
                    bs.append(np.mean((sigmoid(z) - ev_y[j]) ** 2))
                B[k] = float(np.mean(bs)); continue
            kks_all = [min(k, len(c)) for c in acq_codes]
            G = a.group_size if a.group_size > 0 else n_s
            perm = np.random.default_rng(1000 + bi).permutation(n_s) if a.group_size > 0 else np.arange(n_s)
            for g0 in range(0, n_s, G):
                grp = perm[g0:g0 + G]; ng = len(grp); kks = [kks_all[j] for j in grp]
                tbg = tsg = rule_tau(ng, tb)
                rs = np.concatenate([np.full(kk, jj) for jj, kk in enumerate(kks)]); ri = np.concatenate([acq_codes[j][:kk] for j, kk in zip(grp, kks)])
                ry = np.concatenate([acq_y[j][:kk] for j, kk in zip(grp, kks)])
                beta, delta, eps, vb, vs, vi, m_item, cnt = fit_with_text(rs, ri, ry, ng, n_i, m_subj[grp], tc, a.rounds, tbg, tsg, ti, a.beta_prior, False, ts_k[k])
                if a.em_tau_s > 0 and ng >= a.em_tau_s:
                    ts_hat = tsg
                    for _ in range(a.em_iters):  # EM for the between-system variance: tau_s^2 <- mean((delta - m)^2 + v_s)
                        ts_hat = float(np.clip(np.sqrt(np.mean((delta - m_subj[grp]) ** 2 + vs)), a.em_lo, a.em_hi))
                        beta, delta, eps, vb, vs, vi = fit_vec(rs, ri, ry, ng, n_i, m_item, m_subj[grp], tbg, ts_hat, ti, beta_prior=a.beta_prior)
                    TS_LOG.append((held, k, ng, ts_hat))
                    if a.em_refit:
                        beta, delta, eps, vb, vs, vi, m_item, cnt = fit_with_text(rs, ri, ry, ng, n_i, m_subj[grp], tc, a.rounds, tbg, ts_hat, ti, a.beta_prior, False, ts_k[k])
                for jj, j in enumerate(grp):
                    ec = ev_codes[j]; z = beta + delta[jj] + eps[ec]
                    record(j, k, z, beta, delta[jj], eps[ec], cnt[ec].astype(float), vb, vs[jj], vi[ec], kks[jj])
                    bs.append(np.mean((sigmoid(z) - ev_y[j]) ** 2))
            B[k] = float(np.mean(bs))
        alc = sum(WEIGHTS[k] * B[k] for k in BUDGETS); results.append(alc); JS[held] = {str(k): B[k] for k in BUDGETS}
        print(f"{held:18s} subjects {len(pools):3d}  ALC {alc:.4f}   " + " ".join(f"B{k}={B[k]:.3f}" for k in BUDGETS) + f"   ({time.time()-t0:.0f}s)", flush=True)
    print(f"\nmean ALC over {len(results)} benchmarks: {np.mean(results):.4f}  (single={a.single}, alpha={a.alpha}, rounds={a.rounds}, tau={a.tau_sched or a.tau}, tau_i={a.taui_sched or a.tau_i}, sp={a.sp}, beta_prior={a.beta_prior}, binary_labels={a.binary_labels}, tscale={a.tscale_sched or a.text_scale}, group={a.group_size}, rule={a.tau_rule}, em_tau_s={a.em_tau_s}/{a.em_iters}/{a.em_refit})")
    if TS_LOG:
        import collections; agg = collections.defaultdict(list)
        for h_, k_, n_, t_ in TS_LOG: agg[(h_, k_)].append(t_)
        print('EM tau_s (mean over groups):', ' '.join(f'{h_[:6]}/k{k_}={np.mean(v):.2f}' for (h_, k_), v in sorted(agg.items())))
    if a.json:
        import json; json.dump(JS, open(a.json, "w"), indent=1)
    if a.dump:
        out = {k: np.concatenate(v) for k, v in D.items()}
        np.savez_compressed(a.dump, bench_names=np.array(bench_names), pair_names=np.array(pair_names), **out)
        print("dumped", a.dump, len(out["y"]), "rows")


if __name__ == "__main__":
    main()
