"""'subjects' experiments: leave-one-benchmark-out CV of subject-side priors, in the pooled / single / group-of-G regimes.

A vectorised copy of scripts/item_prior_cv.py's logic (--feat hash, text-prior rounds, fit_pooled_s updates), with pluggable
subject priors and extra subject-side model terms:
  --prior none|v5|canon|stack     static subject prior m_s (v5 = submission_v5's SubjectPrior refit without the held-out benchmark)
  --sp 0.5                        scale on m_s
  --map                           submission-style MAP beta step (item_prior_cv.py omits -beta/tau_b^2 in the beta step)
  --adapt L                       per-benchmark regression of delta_s on subject features [m_s, release, effort] refit on the fly
                                  (prior mean w0 = [1,0,0], precision L*n_labels_scale); 0 = off
  --grp-key T / --grp-har T       within-benchmark random effects for the canonical model key / the harness (std T); 0 = off
  --b0 c                          budget-0 prediction sigmoid(c * m_s) instead of 0.5
  --group G                       0 = pooled (all systems of the benchmark), 1 = single (each system alone), G>=2 random groups of G
Reproduces the v5 baseline with: --prior v5 --sp 0.5 --tau 0.8 --tau-i 1.5 (pooled 0.1616, --group 1 -> 0.1856).
"""
import argparse, sys, time
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "src/paie_exp/subjects"))
from paie.data import load_all  # noqa: E402
from paie.textprior import hash_counts  # noqa: E402
from paie.subjprior import SubjectPrior, norm_name  # noqa: E402
from paie_cv import BUDGETS, WEIGHTS  # noqa: E402
import subjpriors as SPX  # noqa: E402

KEYS = ("normalized_name", "provider", "release_date", "access_date", "harness", "harness_version", "reasoning_effort", "subject_features_extra")


FROZEN: dict = {}


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def tfidf_all(C_all):
    """Exactly TextPrior._tfidf with idf from C_all (float32, L2 rows), returned as float64."""
    df = (C_all > 0).sum(0).astype(np.float64)
    idf = (np.log((1 + C_all.shape[0]) / (1 + df)) + 1.0).astype(np.float32)
    X = np.log1p(C_all) * idf
    n = np.linalg.norm(X, axis=1, keepdims=True); n[n == 0] = 1.0
    return (X / n).astype(np.float64)


def text_prior(X_all, lab_idx, y, w, alpha, out_idx=None):
    """TextPrior.fit/predict in dual form on precomputed tf-idf rows; returns predictions for out_idx (default: all rows)."""
    Xl = X_all[lab_idx]; mean_y = float(np.average(y, weights=w))
    K = Xl @ Xl.T + alpha * np.diag(1.0 / w)
    a = np.linalg.solve(K, y - mean_y); beta = Xl.T @ a
    Xo = X_all if out_idx is None else X_all[out_idx]
    return Xo @ beta + mean_y


def fit(s, i, y, n_s, n_i, m_s, m_i, a, key=None, har=None, F=None, w0=None):
    """MAP-ish coordinate Newton sweeps for logit p = beta + delta_s + g_key(s) + h_har(s) + eps_i.
    delta_s ~ N(m_s (+ F_s.w if adaptive), tau^2); eps_i ~ N(m_i, tau_i^2); beta ~ N(0, tau^2) (prior pull only with --map)."""
    tb = a.tau_b if a.tau_b > 0 else a.tau; ts = a.tau_s if a.tau_s > 0 else a.tau; ti = a.tau_i
    beta = 0.0; delta = m_s.copy(); eps = m_i.copy()
    gk = np.zeros(key.max() + 1) if key is not None else None
    hh = np.zeros(har.max() + 1) if har is not None else None
    # the last slot holds singletons (value not shared within the group) and is frozen at 0 when such systems exist
    key_frozen = FROZEN.get("key", False); har_frozen = FROZEN.get("har", False)
    wv = w0.copy() if w0 is not None else None
    ms_eff = m_s.copy()
    active = np.zeros(n_s, bool); active[np.unique(s)] = True
    def zf():
        z = beta + delta[s] + eps[i]
        if gk is not None:
            z = z + gk[key[s]]
        if hh is not None:
            z = z + hh[har[s]]
        return z
    for _ in range(a.sweeps):
        p = sigmoid(zf())
        g = np.sum(y - p) - (beta / tb ** 2 if a.map else 0.0); beta += g / (np.sum(p * (1 - p)) + 1.0 / tb ** 2)
        if gk is not None:
            p = sigmoid(zf()); ks = key[s]
            gk += (np.bincount(ks, y - p, len(gk)) - gk / a.grp_key ** 2) / (np.bincount(ks, p * (1 - p), len(gk)) + 1.0 / a.grp_key ** 2)
            if key_frozen:
                gk[-1] = 0.0
        if hh is not None:
            p = sigmoid(zf()); hs = har[s]
            hh += (np.bincount(hs, y - p, len(hh)) - hh / a.grp_har ** 2) / (np.bincount(hs, p * (1 - p), len(hh)) + 1.0 / a.grp_har ** 2)
            if har_frozen:
                hh[-1] = 0.0
        if wv is not None:
            # ridge of the current subject effects (active subjects) on F, prior mean w0: m_s = F_s.w
            Fa = F[active]; da = delta[active]
            lam = a.adapt * np.ones(len(wv))
            A = Fa.T @ Fa / ts ** 2 + np.diag(lam); bvec = Fa.T @ da / ts ** 2 + lam * w0
            wv = np.linalg.solve(A, bvec); ms_eff = F @ wv
        p = sigmoid(zf())
        delta += (np.bincount(s, y - p, n_s) - (delta - ms_eff) / ts ** 2) / (np.bincount(s, p * (1 - p), n_s) + 1.0 / ts ** 2)
        if wv is not None:
            delta[~active] = ms_eff[~active]
        p = sigmoid(zf())
        eps += (np.bincount(i, y - p, n_i) - (eps - m_i) / ti ** 2) / (np.bincount(i, p * (1 - p), n_i) + 1.0 / ti ** 2)
    extra = np.zeros(n_s)
    if gk is not None:
        extra += gk[key]
    if hh is not None:
        extra += hh[har]
    return beta, delta + extra, eps


def run_group(members, k, X_all, n_i, m_s_all, a, key_all, har_all, F_all, w0):
    """members: list of (subject index, acq item idx array, acq y, eval item idx array, eval y). Returns Brier per member."""
    sidx = np.array([m[0] for m in members])
    loc = {g: j for j, g in enumerate(sidx)}; n_s = len(sidx)
    s = np.concatenate([[j] * min(k, len(m[1])) for j, m in enumerate(members)]).astype(int)
    i = np.concatenate([m[1][:k] for m in members]).astype(int); y = np.concatenate([m[2][:k] for m in members]).astype(float)
    m_s = m_s_all[sidx]
    key = har = F = None

    def shared_codes(vals):
        # codes 0..m-1 for values shared by >= 2 systems of this group; singletons -> slot m, frozen at 0 inside fit()
        u, inv, cnt = np.unique(vals, return_inverse=True, return_counts=True)
        shared = np.where(cnt >= 2)[0]
        if len(shared) == 0:
            return None, False
        remap = np.full(len(u), len(shared)); remap[shared] = np.arange(len(shared))
        code = remap[inv]
        return code, bool((code == len(shared)).any())
    FROZEN.clear()
    if key_all is not None:
        key, FROZEN["key"] = shared_codes(key_all[sidx])
    if har_all is not None:
        har, FROZEN["har"] = shared_codes(har_all[sidx])
    if w0 is not None and n_s >= 5:
        F = F_all[sidx]
    else:
        w0 = None
    m_i = np.zeros(n_i)
    b, d, e = fit(s, i, y, n_s, n_i, m_s, m_i, a, key, har, F, w0)
    lab = np.unique(i)
    if len(lab) >= 10 and a.rounds > 0:
        cnt = np.bincount(i, minlength=n_i)[lab].astype(float)
        for _ in range(a.rounds):
            m_i = text_prior(X_all, lab, e[lab], cnt, a.alpha)
            b, d, e = fit(s, i, y, n_s, n_i, m_s, m_i, a, key, har, F, w0)
    out = []
    for j, m in enumerate(members):
        pe = sigmoid(b + d[j] + e[m[3]])
        out.append(float(np.mean((pe - m[4]) ** 2)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prior", default="v5"); ap.add_argument("--sp", type=float, default=0.5)
    ap.add_argument("--tau", type=float, default=0.8); ap.add_argument("--tau-i", type=float, default=1.5); ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--rounds", type=int, default=3); ap.add_argument("--sweeps", type=int, default=3); ap.add_argument("--map", action="store_true")
    ap.add_argument("--adapt", type=float, default=0.0); ap.add_argument("--grp-key", type=float, default=0.0); ap.add_argument("--grp-har", type=float, default=0.0)
    ap.add_argument("--b0", type=float, default=0.0); ap.add_argument("--group", type=int, default=0); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", default=""); ap.add_argument("--kw", type=float, default=0.6); ap.add_argument("--mw", type=float, default=0.5)
    ap.add_argument("--stack-alpha", type=float, default=3.0); ap.add_argument("--tag", default="")
    ap.add_argument("--tau-b", type=float, default=0.0); ap.add_argument("--tau-s", type=float, default=0.0); ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    pairs, subjects, items = load_all(); items = items.drop_duplicates("item_id")
    sd_first = subjects.drop_duplicates("subject_id").set_index("subject_id")
    sdict = {sid: {k: ("" if pd.isna(r.get(k)) else str(r.get(k))) for k in KEYS} for sid, r in sd_first.iterrows()}
    rng = np.random.default_rng(a.seed); rng_grp = np.random.default_rng(a.seed + 1000); results = {}; t_all = time.time()
    for held in sorted(pairs["benchmark"].unique()):
        if a.only and held != a.only:
            continue
        t0 = time.time(); test = pairs[pairs["benchmark"] == held]
        it = items[items["benchmark"] == held]; cmap = dict(zip(it["item_id"], it["content"].fillna("").astype(str)))
        pools = []
        for sid, g in test.groupby("subject_id"):
            if len(g) < 80:
                continue
            g = g.sample(frac=1.0, random_state=int(rng.integers(1e9))); half = len(g) // 2
            pools.append((sid, g.iloc[:half], g.iloc[half:]))
        if not pools:
            print(f"{held:18s} skipped (no subject with >= 80 items)"); continue
        all_items = list(test["item_id"].unique()); iidx = {x: j for j, x in enumerate(all_items)}
        X_all = tfidf_all(np.stack([hash_counts(cmap[x], dim=4096) for x in all_items])) if a.rounds > 0 else None
        sids = [p[0] for p in pools]
        train = pairs[pairs["benchmark"] != held]
        if a.prior.startswith("oracle"):  # DIAGNOSTIC ONLY (leaks the held-out benchmark): true relative / absolute strength
            accs = test.groupby("subject_id").y.mean(); lz = np.log(np.clip(accs, .02, .98) / (1 - np.clip(accs, .02, .98)))
            ref = lz[sids].mean() if a.prior == "oracle_rel" else 0.0
            m_s_all = a.sp * np.array([lz[x] - ref for x in sids])
        else:
            m_s_all = a.sp * SPX.build_prior(a.prior, train, sdict, sids, a)
        key_all = np.array([SPX.canon_key(sdict[x]) or f"__nokey_{x}" for x in sids]) if a.grp_key > 0 else None
        har_all = np.array([(sdict[x].get("harness") or "").strip().lower() or f"__none_{x}" for x in sids]) if a.grp_har > 0 else None  # missing harness = singleton
        F_all = w0 = None
        if a.adapt > 0:
            F_all = SPX.adapt_features(sids, sdict, m_s_all); w0 = np.zeros(F_all.shape[1]); w0[0] = 1.0
        members = [(j, np.array([iidx[x] for x in acq["item_id"].values]), acq["y"].values.astype(float),
                    np.array([iidx[x] for x in ev["item_id"].values]), ev["y"].values.astype(float)) for j, (sid, acq, ev) in enumerate(pools)]
        if a.group == 0:
            groups = [members]
        else:
            order = rng_grp.permutation(len(members)); groups = [[members[j] for j in order[q:q + a.group]] for q in range(0, len(order), a.group)]
        B = {}
        for k in BUDGETS:
            if k == 0:
                bs = []
                for m in members:
                    p0 = sigmoid(a.b0 * m_s_all[m[0]]) if a.b0 else 0.5
                    bs.append(np.mean((p0 - m[4]) ** 2))
                B[k] = float(np.mean(bs)); continue
            bs = []
            for grp in groups:
                bs += run_group(grp, k, X_all, len(all_items), m_s_all, a, key_all, har_all, F_all, w0)
            B[k] = float(np.mean(bs))
        alc = sum(WEIGHTS[k] * B[k] for k in BUDGETS); results[held] = alc
        print(f"{held:18s} subjects {len(pools):3d}  ALC {alc:.4f}   " + " ".join(f"B{k}={B[k]:.3f}" for k in BUDGETS) + f"   ({time.time()-t0:.0f}s)", flush=True)
    mean = float(np.mean(list(results.values())))
    print(f"\nmean ALC over {len(results)} benchmarks: {mean:.4f}  [{a.tag}] prior={a.prior} sp={a.sp} tau={a.tau} tau_b={a.tau_b} tau_s={a.tau_s} tau_i={a.tau_i} map={a.map} adapt={a.adapt} "
          f"grp_key={a.grp_key} grp_har={a.grp_har} b0={a.b0} group={a.group} seed={a.seed}  ({time.time()-t_all:.0f}s)")
    print("RESULT", a.tag, f"{mean:.4f}", " ".join(f"{b}={v:.4f}" for b, v in results.items()))


if __name__ == "__main__":
    main()
