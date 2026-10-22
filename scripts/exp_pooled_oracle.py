"""exp_pooled_oracle (tag 'pooled'): where is the pooled-regime headroom? Full-data Rasch fit per benchmark (all pool subjects, all
their items incl. evaluation pools, light ridge) gives oracle item effects eps* and subject effects delta*. Then at each budget:
  model      : v5 (pooled)
  item-oracle: eps fixed to eps*, beta/delta learned from the k revealed labels per system (same priors)
  subj-oracle: beta+delta fixed to beta*+delta*, eps learned from revealed labels (text prior as v5)
  both       : full-data Rasch on the evaluation pools (in-sample floor)
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import exp_pooled_cv as H  # noqa: E402
import exp_pooled_core as core  # noqa: E402

W = H.WEIGHTS


def full_rasch(si, ii, y, nS, nI, tau=3.0, iters=60):
    beta, d, e = 0.0, np.zeros(nS), np.zeros(nI)
    for _ in range(iters):
        p = core.sig(beta + d[si] + e[ii]); beta += np.sum(y - p) / np.sum(p * (1 - p))
        p = core.sig(beta + d[si] + e[ii]); r = y - p; w = p * (1 - p)
        d += (np.bincount(si, r, nS) - d / tau ** 2) / (np.bincount(si, w, nS) + 1 / tau ** 2)
        p = core.sig(beta + d[si] + e[ii]); r = y - p; w = p * (1 - p)
        e += (np.bincount(ii, r, nI) - e / tau ** 2) / (np.bincount(ii, w, nI) + 1 / tau ** 2)
    return beta, d, e


def main():
    benches = H.prepare(0, 0.5)
    for held, B in benches.items():
        P = B["pools"]; n = len(P); nI = B["nI"]
        si = np.concatenate([[j] * (len(p["acq_i"]) + len(p["ev_i"])) for j, p in enumerate(P)])
        ii = np.concatenate([np.concatenate([p["acq_i"], p["ev_i"]]) for p in P]); y = np.concatenate([np.concatenate([p["acq_y"], p["ev_y"]]) for p in P])
        bs, ds, es = full_rasch(si, ii, y, n, nI)
        prior = core.ItemPrior(B["XA"], B["GA"], 1.0, 0.0)
        out = {name: {0: np.mean([np.mean((0.5 - p["ev_y"]) ** 2) for p in P])} for name in ("model", "item-oracle", "subj-oracle", "both")}
        for k in H.BUDGETS[1:]:
            rs = np.concatenate([[j] * min(k, len(p["acq_i"])) for j, p in enumerate(P)]); ri = np.concatenate([p["acq_i"][:k] for p in P]); ry = np.concatenate([p["acq_y"][:k] for p in P])
            ts = np.concatenate([[j] * len(p["ev_i"]) for j, p in enumerate(P)]); ti = np.concatenate([p["ev_i"] for p in P]); ty = np.concatenate([p["ev_y"] for p in P])
            idx = np.cumsum([0] + [len(p["ev_i"]) for p in P])
            per = lambda pr: float(np.mean([np.mean((pr[idx[j]:idx[j + 1]] - ty[idx[j]:idx[j + 1]]) ** 2) for j in range(n)]))
            fit = core.fit_snapshot(rs, ri, ry, n, nI, B["m_subj"], prior, {})
            out["model"][k] = per(core.predict_targets(fit, ts, ti, {}))
            # item oracle: offsets eps*, learn beta + delta (prior as v5: tau_s 0.8 around m_subj, beta free)
            b, d = 0.0, B["m_subj"].copy()
            for _ in range(10):
                p = core.sig(b + d[rs] + es[ri]); b += np.sum(ry - p) / (np.sum(p * (1 - p)) + 1 / 0.64)
                p = core.sig(b + d[rs] + es[ri]); d += (np.bincount(rs, ry - p, n) - (d - B["m_subj"]) / 0.64) / (np.bincount(rs, p * (1 - p), n) + 1 / 0.64)
            out["item-oracle"][k] = per(core.sig(b + d[ts] + es[ti]))
            # subject oracle: offsets beta*+delta*, learn eps (tau_i 1.5) with text-prior rounds
            off = bs + ds
            e = np.zeros(nI); m_i = np.zeros(nI)
            for rnd in range(4):
                e = m_i.copy()
                for _ in range(10):
                    p = core.sig(off[rs] + e[ri]); e += (np.bincount(ri, ry - p, nI) - (e - m_i) / 2.25) / (np.bincount(ri, p * (1 - p), nI) + 1 / 2.25)
                lab = np.unique(ri)
                if len(lab) >= 10 and rnd < 3:
                    m_i = prior.fit_predict(lab, e[lab], np.bincount(ri, minlength=nI)[lab].astype(float))
            out["subj-oracle"][k] = per(core.sig(off[ts] + e[ti]))
            out["both"][k] = per(core.sig(bs + ds[ts] + es[ti]))
        for name, Bk in out.items():
            alc = sum(W[k] * Bk[k] for k in H.BUDGETS)
            print(f"{held:18s} {name:12s} ALC {alc:.4f}  " + " ".join(f"B{k}={Bk[k]:.3f}" for k in H.BUDGETS), flush=True)


if __name__ == "__main__":
    main()
