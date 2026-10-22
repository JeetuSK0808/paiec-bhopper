"""Single-system CV with an empirical-Bayes posterior predictive: theta = logit accuracy of the system on the benchmark, prior = histogram of
pair-level logit accuracies on the OTHER public benchmarks (smoothed), item offsets m_i from the within-benchmark text ridge (own labels,
>=10) or 0; prediction = sum_theta w(theta | labels) * sigma((theta + m_j) / sqrt(1 + pi s_i^2 / 8)).  Compared with the MAP model."""
import sys, argparse, numpy as np, pandas as pd
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
from paie.data import load_all
from paie.textprior import hash_counts, TextPrior
from item_prior_cv import BUDGETS, WEIGHTS, sigmoid
ap = argparse.ArgumentParser(); ap.add_argument("--s-item", type=float, default=1.0); ap.add_argument("--bw", type=float, default=0.4); ap.add_argument("--text", type=int, default=1)
ap.add_argument("--prior", default="empirical"); a = ap.parse_args()
pairs, subs, items = load_all(); items = items.drop_duplicates("item_id")
C = {i_: hash_counts(t_) for i_, t_ in zip(items["item_id"], items["content"].fillna("").astype(str))}
lg = lambda p: np.log(np.clip(p, .01, .99) / (1 - np.clip(p, .01, .99)))
TH = np.linspace(-6, 6, 241)
rng = np.random.default_rng(0); res = []
for held in sorted(pairs.benchmark.unique()):
    other = pairs[pairs.benchmark != held]; pa = other.groupby(["benchmark", "subject_id"]).y.agg(["mean", "size"]); pa = pa[pa["size"] >= 30]
    th0 = lg(pa["mean"].values)
    if a.prior == "empirical":
        prior = np.exp(-0.5 * ((TH[:, None] - th0[None, :]) / a.bw) ** 2).sum(1)
    else:
        prior = np.exp(-0.5 * (TH / float(a.prior)) ** 2)
    prior /= prior.sum()
    test = pairs[pairs.benchmark == held]; pools = []
    for sid, g in test.groupby("subject_id"):
        if len(g) < 80: continue
        g = g.sample(frac=1.0, random_state=int(rng.integers(1e9))); h = len(g) // 2; pools.append((sid, g.iloc[:h], g.iloc[h:]))
    if not pools: continue
    all_items = list(test.item_id.unique()); C_all = np.stack([C[i_] for i_ in all_items]); idx = {i_: j for j, i_ in enumerate(all_items)}
    B = {}
    for k in BUDGETS:
        bs = []
        for sid, acq, ev in pools:
            li, ly = acq.item_id.values[:k], acq.y.values[:k].astype(float)
            m = np.zeros(len(all_items))
            if a.text and k >= 10:
                # within-benchmark text ridge on this system's own labels: target = centred label logit proxy
                t = (ly - ly.mean()) * 2.0
                tp = TextPrior(alpha=1.0).fit(C_all, np.stack([C[i_] for i_ in li]), t); m = tp.predict(C_all); m = m - m.mean()
            s2 = 1 + np.pi * a.s_item ** 2 / 8
            ll = np.zeros_like(TH)
            for i_, y_ in zip(li, ly):
                p = sigmoid((TH + m[idx[i_]]) / np.sqrt(s2)); ll += y_ * np.log(p + 1e-12) + (1 - y_) * np.log(1 - p + 1e-12)
            w = prior * np.exp(ll - ll.max()); w /= w.sum()
            me = m[[idx[i_] for i_ in ev.item_id.values]]
            pe = (w[:, None] * sigmoid((TH[:, None] + me[None, :]) / np.sqrt(s2))).sum(0)
            bs.append(np.mean((pe - ev.y.values) ** 2))
        B[k] = float(np.mean(bs))
    alc = sum(WEIGHTS[k] * B[k] for k in BUDGETS); res.append(alc)
    print(f"{held:18s} ALC {alc:.4f}  " + " ".join(f"B{k}={B[k]:.3f}" for k in BUDGETS), flush=True)
print(f"mean ALC {np.mean(res):.4f}  (prior={a.prior}, bw={a.bw}, s_item={a.s_item}, text={a.text})")
