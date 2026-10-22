"""Tag 'items' diagnostic 2: after the additive pooled fit at budget k, do residuals on the SAME item transfer between systems, and more so
between similar systems? For each evaluation pair (s, i) whose item has revealed labels from other systems s', correlate the eval residual
y_si - p_si with the residuals r_s'i = y_s'i - p_s'i, split by a metadata similarity of (s, s')."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from paie.data import load_all  # noqa: E402
from paie.textprior import hash_counts  # noqa: E402
from paie.subjprior import SubjectPrior, norm_name  # noqa: E402
from exp_items_core import ItemStructure, fit_hier, item_offset, tfidf_matrix, sigmoid  # noqa: E402


def family(name):
    n = norm_name(name)
    for t in ("(high)", "(medium)", "(low)", "(xhigh)", "(minimal)", "high", "medium", "low", "xhigh", "minimal", "(no-think)", "(think)"):
        n = n.replace(t, " ")
    return " ".join(n.split())


def main():
    pairs, subjects, items = load_all(); items = items.drop_duplicates("item_id")
    rng = np.random.default_rng(0)
    sd = subjects.drop_duplicates("subject_id").set_index("subject_id")
    def sdict(sid):
        r = sd.loc[sid]; return {k: ("" if pd.isna(r.get(k)) else str(r.get(k))) for k in ("normalized_name", "provider", "release_date", "reasoning_effort", "subject_features_extra")}
    for held in sorted(pairs["benchmark"].unique()):
        test = pairs[pairs["benchmark"] == held]
        pools = []
        for sid, g in test.groupby("subject_id"):
            if len(g) < 80:
                continue
            g = g.sample(frac=1.0, random_state=int(rng.integers(1e9))); half = len(g) // 2
            pools.append((sid, g.iloc[:half], g.iloc[half:]))
        if len(pools) < 2:
            continue
        all_items = list(test["item_id"].unique()); N = len(all_items); ix = {i_: n for n, i_ in enumerate(all_items)}
        it_h = items[items["benchmark"] == held].set_index("item_id")
        T = tfidf_matrix(np.stack([hash_counts(str(it_h.at[i_, "content"]) if pd.notna(it_h.at[i_, "content"]) else "") for i_ in all_items])); Gram = T @ T.T
        struct = ItemStructure([it_h.at[i_, "item_features"] for i_ in all_items], identity=False); groups = struct.groups; tau_g = [0.8] * len(groups)
        trp = pairs[pairs["benchmark"] != held].groupby(["benchmark", "subject_id"]).y.agg(["mean", "size"]).reset_index()
        spm = SubjectPrior(known_weight=0.6, meta_weight=0.5).fit([(sdict(r.subject_id), r.benchmark, r["mean"], r["size"]) for _, r in trp.iterrows()])
        m_subj = np.array([0.5 * spm.offset(sdict(sid)) for sid, _, _ in pools])
        names = [str(sd.loc[sid].get("normalized_name")) for sid, _, _ in pools]; fam = [family(n) for n in names]
        prov = [str(sd.loc[sid].get("provider")).lower() for sid, _, _ in pools]
        rd = pd.to_datetime(pd.Series([sd.loc[sid].get("release_date") for sid, _, _ in pools]), errors="coerce")
        print(f"== {held}: {len(pools)} systems, {len(set(fam))} families. e.g. {names[:6]}")
        for k in (7, 31):
            s_idx = np.concatenate([np.full(min(k, len(acq)), n, dtype=np.int64) for n, (_, acq, _) in enumerate(pools)])
            i_idx = np.concatenate([np.array([ix[i_] for i_ in acq["item_id"].values[:k]], dtype=np.int64) for _, acq, _ in pools])
            y = np.concatenate([acq["y"].values[:k] for _, acq, _ in pools]).astype(float)
            kw = dict(groups=groups, tau_g=tau_g, tau_b=0.8, tau_s=0.8, tau_i=1.5)
            fit = fit_hier(s_idx, i_idx, y, m_subj, np.zeros(N), **kw)
            lab, cnt = np.unique(i_idx, return_counts=True)
            if len(lab) >= 10:
                K = Gram[np.ix_(lab, lab)] + np.diag(1.0 / cnt); Kq = Gram[:, lab]
                for _ in range(3):
                    t = fit["eps"][lab]; mu = float(np.average(t, weights=cnt)); m_i = Kq @ np.linalg.solve(K, t - mu) + mu
                    fit = fit_hier(s_idx, i_idx, y, m_subj, m_i, **kw)
            p_rev = sigmoid(fit["beta"] + fit["delta"][s_idx] + item_offset(fit, groups, None, i_idx)); r_rev = y - p_rev
            by_item = {}
            for r_, s_, i_ in zip(r_rev, s_idx, i_idx):
                by_item.setdefault(i_, []).append((s_, r_))
            rows = []
            for n, (sid, acq, ev) in enumerate(pools):
                ei = np.array([ix[i_] for i_ in ev["item_id"].values])
                pe = sigmoid(fit["beta"] + fit["delta"][n] + item_offset(fit, groups, None, ei)); re_ = ev["y"].values - pe
                for j, i_ in enumerate(ei):
                    for s2, r2 in by_item.get(i_, []):
                        if s2 == n:
                            continue
                        dd = abs((rd[n] - rd[s2]).days) if pd.notna(rd[n]) and pd.notna(rd[s2]) else 9999
                        rows.append((re_[j], r2, fam[n] == fam[s2], prov[n] == prov[s2], dd))
            r = pd.DataFrame(rows, columns=["re", "r2", "fam", "prov", "dd"])
            def c(df):
                return f"{np.corrcoef(df.re, df.r2)[0,1]:+.3f} (n={len(df)})" if len(df) > 20 else "n/a"
            print(f"  k={k:2d} corr(eval resid, other-system resid same item): all {c(r)} | same family {c(r[r.fam])} | same provider, diff family {c(r[r.prov & ~r.fam])} "
                  f"| diff provider {c(r[~r.prov])} | release <120d {c(r[r.dd < 120])} | >365d {c(r[r.dd > 365])}", flush=True)


if __name__ == "__main__":
    main()
