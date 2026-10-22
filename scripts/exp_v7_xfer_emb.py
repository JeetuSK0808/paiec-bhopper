"""exp_v7_xfer_emb (tag 'v7'): does item difficulty transfer ACROSS benchmarks from text embeddings? Leave-one-benchmark-out ridge
from item embeddings (bge-small 384-d, bge-base 768-d; already computed for all 12,549 public items) and generic text traits to the
within-benchmark item difficulty (centred smoothed logit of the item's solve rate over all subjects). Reports per-held-out-benchmark
Spearman and Pearson correlations. A cross-benchmark item prior only helps if these are positive on (nearly) every benchmark.
  .venv/bin/python -B scripts/exp_v7_xfer_emb.py
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from paie.data import load_all, item_features  # noqa: E402


def spearman(a, b):
    ra = pd.Series(a).rank().to_numpy(); rb = pd.Series(b).rank().to_numpy()
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    pairs, _, items = load_all()
    it = items.drop_duplicates("item_id").set_index("item_id")
    g = pairs.groupby(["benchmark", "item_id"]).agg(y=("y", "mean"), n=("y", "size")).reset_index()
    g = g[g["n"] >= 3]
    p = (g["y"] * g["n"] + 0.5) / (g["n"] + 1.0)
    g["d"] = np.log(p / (1 - p)); g["d"] -= g.groupby("benchmark")["d"].transform("mean")
    feats = {}
    for tag, f in (("bge-small", "results/item_emb.npz"), ("bge-base", "results/item_emb_bge-base-en-v1.5.npz")):
        z = np.load(ROOT / f, allow_pickle=True); feats[tag] = dict(zip(z["item_id"].tolist(), z["emb"]))
    import glob as _g
    for f in sorted(_g.glob(str(ROOT / "results" / "item_lm_*.npz"))):
        z = np.load(f, allow_pickle=True); base = Path(f).stem.replace("item_lm_", "")
        for part in ("mid", "last", "tok"):
            feats[f"{base}:{part}"] = dict(zip(z["item_id"].tolist(), z[part]))
        feats[f"{base}:nll"] = dict(zip(z["item_id"].tolist(), np.stack([z["nll"], np.log1p(z["ntok"])], 1)))
    tf = item_features(it.reset_index()).set_index("item_id")
    trait_cols = [c for c in tf.columns]
    have = set.intersection(*[set(v) for v in feats.values()])
    g = g[g["item_id"].isin(have)]
    benches = sorted(g["benchmark"].unique())
    print("items with >=3 subjects per benchmark:", g.groupby("benchmark").size().to_dict())
    tags = sys.argv[1].split(",") if len(sys.argv) > 1 else ["bge-small", "bge-base", "traits", "bge-base+traits"]
    for tag in tags:
        def X_of(ids):
            parts = []
            if tag.split("+")[0] in feats:
                E = feats[tag.split("+")[0]]; parts.append(np.stack([E[i] for i in ids]).astype(np.float64))
            if "traits" in tag:
                T = tf.loc[ids, trait_cols].to_numpy(float); parts.append(T)
            return np.concatenate(parts, 1)
        for lam in (1.0, 10.0, 100.0):
            rows = []
            for hb in benches:
                tr = g[g["benchmark"] != hb]; te = g[g["benchmark"] == hb]
                Xtr = X_of(tr["item_id"].tolist()); Xte = X_of(te["item_id"].tolist())
                # standardise each training benchmark's features (within-benchmark centring), same for the held-out one
                def cen(X, bser):
                    X = X.copy()
                    for b in np.unique(bser):
                        m = bser == b; X[m] -= X[m].mean(0); X[m] /= (X[m].std(0) + 1e-6)
                    return X
                Xtr = cen(Xtr, tr["benchmark"].to_numpy()); Xte = cen(Xte, te["benchmark"].to_numpy())
                # equal weight per training benchmark
                w = 1.0 / tr.groupby("benchmark")["item_id"].transform("size").to_numpy()
                A = Xtr.T @ (Xtr * w[:, None]) + lam * np.eye(Xtr.shape[1]) * w.mean()
                beta = np.linalg.solve(A, Xtr.T @ (w * tr["d"].to_numpy()))
                pr = Xte @ beta
                rows.append((hb, spearman(pr, te["d"].to_numpy()), float(np.corrcoef(pr, te["d"].to_numpy())[0, 1])))
            print(f"{tag:16s} lam {lam:6.1f}  " + "  ".join(f"{b[:10]} {s:+.2f}" for b, s, _ in rows) +
                  f"   mean rho {np.mean([s for _, s, _ in rows]):+.3f}  min {min(s for _, s, _ in rows):+.2f}")


if __name__ == "__main__":
    main()
