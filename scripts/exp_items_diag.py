"""Tag 'items' diagnostic: where does the pooled-regime error sit? For each held-out benchmark and budget, split evaluation pairs by
whether their item carries >= 1 revealed label (from other systems) and report model Brier vs an in-sample Rasch oracle
(beta + delta_s + eps_i fitted on ALL pairs of the benchmark, weak prior) on the same pairs. Same pools/RNG as the CV harness."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from paie.data import load_all  # noqa: E402
from paie.textprior import hash_counts  # noqa: E402
from paie.subjprior import SubjectPrior  # noqa: E402
from exp_items_core import ItemStructure, fit_hier, item_offset, tfidf_matrix, sigmoid  # noqa: E402

GROUPS = "--groups" in sys.argv


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
        T = tfidf_matrix(np.stack([hash_counts(str(it_h.at[i_, "content"]) if pd.notna(it_h.at[i_, "content"]) else "") for i_ in all_items]))
        Gram = T @ T.T
        struct = ItemStructure([it_h.at[i_, "item_features"] for i_ in all_items], identity=False)
        groups = [(n, g, l) for n, g, l in struct.groups] if GROUPS else []
        tau_g = [0.8] * len(groups)
        trp = pairs[pairs["benchmark"] != held].groupby(["benchmark", "subject_id"]).y.agg(["mean", "size"]).reset_index()
        spm = SubjectPrior(known_weight=0.6, meta_weight=0.5).fit([(sdict(r.subject_id), r.benchmark, r["mean"], r["size"]) for _, r in trp.iterrows()])
        m_subj = np.array([0.5 * spm.offset(sdict(sid)) for sid, _, _ in pools])
        # in-sample oracle on all pairs of the pooled subjects
        allp = pd.concat([pd.concat([acq, ev]) for _, acq, ev in pools])
        sidx = {sid: n for n, (sid, _, _) in enumerate(pools)}
        os_ = allp["subject_id"].map(sidx).values.astype(np.int64); oi = allp["item_id"].map(ix).values.astype(np.int64); oy = allp["y"].values.astype(float)
        of = fit_hier(os_, oi, oy, np.zeros(len(pools)), np.zeros(N), tau_b=3.0, tau_s=3.0, tau_i=3.0, sweeps=30, beta_map=True)
        print(f"== {held}  subjects {len(pools)}  items {N}")
        for k in (1, 3, 7, 15, 31):
            s_idx = np.concatenate([np.full(min(k, len(acq)), n, dtype=np.int64) for n, (_, acq, _) in enumerate(pools)])
            i_idx = np.concatenate([np.array([ix[i_] for i_ in acq["item_id"].values[:k]], dtype=np.int64) for _, acq, _ in pools])
            y = np.concatenate([acq["y"].values[:k] for _, acq, _ in pools]).astype(float)
            kw = dict(groups=groups, tau_g=tau_g, tau_b=0.8, tau_s=0.8, tau_i=1.5)
            fit = fit_hier(s_idx, i_idx, y, m_subj, np.zeros(N), **kw)
            lab, cnt = np.unique(i_idx, return_counts=True)
            m_i = np.zeros(N)
            if len(lab) >= 10:
                K = Gram[np.ix_(lab, lab)] + np.diag(1.0 / cnt); Kq = Gram[:, lab]
                for _ in range(3):
                    t = fit["eps"][lab]; mu = float(np.average(t, weights=cnt)); m_i = Kq @ np.linalg.solve(K, t - mu) + mu
                    fit = fit_hier(s_idx, i_idx, y, m_subj, m_i, **kw)
            labset = np.zeros(N, bool); labset[lab] = True
            rows = []
            for n, (sid, acq, ev) in enumerate(pools):
                ei = np.array([ix[i_] for i_ in ev["item_id"].values])
                z = fit["beta"] + fit["delta"][n] + item_offset(fit, groups, None, ei)
                zo = of["beta"] + of["delta"][n] + of["eps"][ei]
                # hybrids: oracle item part + model subject part (item part centred on the labelled mean), and vice versa
                zio = fit["beta"] + fit["delta"][n] + (of["eps"][ei] - of["eps"][lab].mean() + item_offset(fit, groups, None, lab).mean())
                zso = of["beta"] + of["delta"][n] + (item_offset(fit, groups, None, ei) - item_offset(fit, groups, None, lab).mean() + of["eps"][lab].mean())
                for j in range(len(ei)):
                    rows.append((labset[ei[j]], (sigmoid(z[j]) - ev["y"].values[j]) ** 2, (sigmoid(zo[j]) - ev["y"].values[j]) ** 2,
                                 (sigmoid(zio[j]) - ev["y"].values[j]) ** 2, (sigmoid(zso[j]) - ev["y"].values[j]) ** 2, n))
            r = pd.DataFrame(rows, columns=["lab", "b", "bo", "bio", "bso", "s"])
            # pair-weighted like the metric: mean over subjects of per-subject mean
            def pm(df, c):
                return df.groupby("s")[c].mean().mean()
            print(f"  k={k:2d}  frac eval pairs w/ labelled item {r.lab.mean():.2f} | Brier all {pm(r,'b'):.4f} oracle {pm(r,'bo'):.4f} "
                  f"item-oracle {pm(r,'bio'):.4f} subj-oracle {pm(r,'bso'):.4f} | labelled {r[r.lab].b.mean():.4f} (orc {r[r.lab].bo.mean():.4f}) "
                  f"unlabelled {r[~r.lab].b.mean():.4f} (orc {r[~r.lab].bo.mean():.4f})", flush=True)


if __name__ == "__main__":
    main()
