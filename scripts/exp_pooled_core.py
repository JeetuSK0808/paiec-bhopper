"""exp_pooled_core (tag 'pooled'): vectorised re-implementation of the v5 pooled hierarchical logistic model plus pooled-regime
extensions. numpy only, deterministic.

logit p = beta + delta_s + eps_i ; delta_s ~ N(m_s, tau_s^2), eps_i ~ N(m_i, tau_i^2); m_i from a dual ridge on item text TF-IDF
(optionally + item_features group one-hots); MAP by block-Newton sweeps (same update order as scripts/item_prior_cv.fit_pooled_s).

Extensions (all default off => exact v5 harness behaviour):
  eb_s  : empirical-Bayes (MAP-EM) estimate of tau_s from all revealed labels, inverse-gamma-like pseudo-count eb_s at tau_s0
  eb_i  : same for tau_i (estimated on the fit without text prior, then held for the text rounds)
  probit: predictive mean E[sigmoid(z)] with Laplace posterior variances (probit approximation), variance scale `probit`
  c_g   : weight of the item_features group kernel added to the text kernel in the ridge prior
  sim   : subject-similarity residual sharing (kernel over subject name tokens) for items other systems revealed
"""
from __future__ import annotations
import re
import numpy as np


def sig(z):
    return 1.0 / (1.0 + np.exp(-z))


# ----------------------------------------------------------------------------------------------------------------------------
# item text / feature kernels
# ----------------------------------------------------------------------------------------------------------------------------
def tfidf_all(C_all: np.ndarray) -> np.ndarray:
    """Exactly TextPrior._tfidf with idf from C_all (float32 math, cast to float64)."""
    df = (C_all > 0).sum(0).astype(np.float64)
    idf = (np.log((1 + C_all.shape[0]) / (1 + df)) + 1.0).astype(np.float32)
    X = np.log1p(C_all) * idf
    n = np.linalg.norm(X, axis=1, keepdims=True); n[n == 0] = 1.0
    return (X / n).astype(np.float64)


def group_matrix(feats: list, normalise: bool = True) -> np.ndarray | None:
    """item_features strings 'k=v;k=v' -> row-normalised design: one-hot per k=v token (tokens shared by >=2 items and not by all)
    plus a z-scored numeric column per key whose values are numeric (>=3 distinct values). None if nothing usable."""
    parsed = []
    for f in feats:
        d = {}
        if isinstance(f, str) and f:
            for part in f.split(";"):
                if "=" in part:
                    k, v = part.split("=", 1); d[k.strip()] = v.strip()
        parsed.append(d)
    n = len(parsed)
    if n == 0:
        return None
    tok_count = {}
    for d in parsed:
        for k, v in d.items():
            tok_count[(k, v)] = tok_count.get((k, v), 0) + 1
    toks = sorted(t for t, c in tok_count.items() if 2 <= c < n)
    cols = []
    if toks:
        ix = {t: j for j, t in enumerate(toks)}
        OH = np.zeros((n, len(toks)))
        for r, d in enumerate(parsed):
            for k, v in d.items():
                j = ix.get((k, v))
                if j is not None:
                    OH[r, j] = 1.0
        cols.append(OH)
    keys = sorted({k for d in parsed for k in d})
    for k in keys:
        vals = []
        for d in parsed:
            try:
                vals.append(float(d[k]))
            except Exception:
                vals.append(np.nan)
        vals = np.array(vals)
        ok = ~np.isnan(vals)
        if ok.mean() >= 0.9 and len(np.unique(vals[ok])) >= 3:
            z = np.zeros(n); z[ok] = (vals[ok] - vals[ok].mean()) / (vals[ok].std() + 1e-9)
            cols.append(z[:, None])
    if not cols:
        return None
    G = np.concatenate(cols, axis=1)
    if not normalise:
        return G
    nr = np.linalg.norm(G, axis=1, keepdims=True); nr[nr == 0] = 1.0
    return G / nr


class ItemPrior:
    """Dual ridge (sample-weighted) from item features to targets; kernel = text TF-IDF linear kernel + c_g^2 * group kernel."""

    def __init__(self, XA: np.ndarray, GA: np.ndarray | None = None, alpha: float = 1.0, c_g: float = 0.0):
        self.XA, self.GA, self.alpha, self.c_g = XA, GA, alpha, c_g

    def fit_predict(self, lab: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
        X = self.XA[lab]
        mean_y = float(np.average(y, weights=w)); yc = y - mean_y
        K = X @ X.T
        use_g = self.GA is not None and self.c_g > 0
        if use_g:
            Gl = self.GA[lab]; K = K + self.c_g ** 2 * (Gl @ Gl.T)
        K[np.diag_indices_from(K)] += self.alpha / w
        a = np.linalg.solve(K, yc)
        m = self.XA @ (X.T @ a) + mean_y
        if use_g:
            m = m + self.c_g ** 2 * (self.GA @ (Gl.T @ a))
        return m


# ----------------------------------------------------------------------------------------------------------------------------
# subject similarity
# ----------------------------------------------------------------------------------------------------------------------------
_TOK = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")


def subject_tokens(sd: dict) -> set:
    name = str(sd.get("normalized_name", "") or "").lower()
    extra = str(sd.get("subject_features_extra", "") or "").lower()
    t = set(_TOK.findall(name))
    for part in extra.split(";"):
        if "=" in part:
            k, v = part.split("=", 1)
            if k.strip() in ("model_config", "source_model_name", "model", "model_name"):
                t |= set(_TOK.findall(v.replace("/", " ")))
    for key in ("provider", "harness", "reasoning_effort"):
        v = str(sd.get(key, "") or "").strip().lower()
        if v and v not in ("nan", "none"):
            t.add(key[:4] + ":" + v)
    return t


def similarity_matrix(sdicts: list, power: float = 2.0) -> np.ndarray:
    toks = [subject_tokens(d) for d in sdicts]
    n = len(toks); K = np.zeros((n, n))
    for a in range(n):
        for b in range(a + 1, n):
            u = len(toks[a] | toks[b])
            j = len(toks[a] & toks[b]) / u if u else 0.0
            K[a, b] = K[b, a] = j ** power
    return K


# ----------------------------------------------------------------------------------------------------------------------------
# the model
# ----------------------------------------------------------------------------------------------------------------------------
DEFAULT = dict(tau_b=0.8, tau_s=0.8, tau_i=1.5, sweeps=3, rounds=3, min_lab=10, beta_prior=False,
               eb_s=0.0, eb_i=0.0, probit=0.0, sim=0.0, sim_min=0.0, eb_s_max=3.0, eb_i_max=3.0, tau_g=0.0, tau_sg=0.0)


def newton(si, ii, y, nS, nI, m_s, m_i, tau_b, ts2, ti2, sweeps, beta_prior, eb_s=0.0, tau_s0=0.8, eb_s_max=3.0, Gr=None, tau_g=0.0, g0=None,
           tau_sg=0.0, gam0=None):
    """Block-Newton MAP sweeps (beta, then all delta_s, then all eps_i), as in item_prior_cv.fit_pooled_s.
    If eb_s > 0, ts2 is re-estimated after each sweep by MAP-EM over subjects that have labels.
    If Gr (items x groups, raw one-hot/numeric) and tau_g > 0: item offset Gr @ g with g ~ N(0, tau_g^2) (damped diagonal Newton).
    If tau_sg > 0: subject x group interaction gamma[s, c] ~ N(0, tau_sg^2) on the one-hot group columns."""
    beta = 0.0; delta = m_s.copy(); eps = m_i.copy()
    has_s = np.bincount(si, minlength=nS) > 0
    use_g = Gr is not None and tau_g > 0; use_sg = Gr is not None and tau_sg > 0
    nG = Gr.shape[1] if Gr is not None else 0
    g = (np.zeros(nG) if g0 is None else g0.copy()) if use_g else None
    gam = (np.zeros((nS, nG)) if gam0 is None else gam0.copy()) if use_sg else None
    if use_g or use_sg:
        damp = float(max(1.0, (Gr != 0).sum(1).max()))
        Gl = Gr[ii]  # labels x groups
    def zf():
        z = beta + delta[si] + eps[ii]
        if use_g:
            z = z + (Gr @ g)[ii]
        if use_sg:
            z = z + np.einsum("lc,lc->l", Gl, gam[si])
        return z
    for _ in range(sweeps):
        p = sig(zf())
        gb = float(np.sum(y - p)) - (beta / tau_b ** 2 if beta_prior else 0.0)
        beta += gb / (float(np.sum(p * (1 - p))) + 1.0 / tau_b ** 2)
        p = sig(zf()); r = y - p; w = p * (1 - p)
        hs = np.bincount(si, w, nS) + 1.0 / ts2
        delta = delta + (np.bincount(si, r, nS) - (delta - m_s) / ts2) / hs
        if use_g:
            p = sig(zf()); r = y - p; w = p * (1 - p)
            ri_ = np.bincount(ii, r, nI); wi_ = np.bincount(ii, w, nI)
            g = g + (Gr.T @ ri_ - g / tau_g ** 2) / ((Gr ** 2).T @ wi_ + 1.0 / tau_g ** 2) / damp
        if use_sg:
            p = sig(zf()); r = y - p; w = p * (1 - p)
            # per (subject, group) sums over labels
            gr_ = np.zeros((nS, nG)); gw_ = np.zeros((nS, nG))
            np.add.at(gr_, si, Gl * r[:, None]); np.add.at(gw_, si, (Gl ** 2) * w[:, None])
            gam = gam + (gr_ - gam / tau_sg ** 2) / (gw_ + 1.0 / tau_sg ** 2) / damp
        p = sig(zf()); r = y - p; w = p * (1 - p)
        hi = np.bincount(ii, w, nI) + 1.0 / ti2
        eps = eps + (np.bincount(ii, r, nI) - (eps - m_i) / ti2) / hi
        if eb_s > 0 and has_s.sum() >= 2:
            p = sig(zf()); w = p * (1 - p)
            hs = np.bincount(si, w, nS) + 1.0 / ts2
            ss = float(np.sum(((delta - m_s) ** 2 + 1.0 / hs)[has_s]))
            ts2 = min(eb_s_max ** 2, (eb_s * tau_s0 ** 2 + ss) / (eb_s + has_s.sum()))
    return beta, delta, eps, ts2, g, gam


def fit_snapshot(si, ii, y, nS, nI, m_s, prior: ItemPrior | None, cfg: dict, Gr: np.ndarray | None = None):
    """Returns dict(beta, delta, eps(+group offset), m_i, ts2, ti2, hs, hi) for one snapshot of revealed labels."""
    c = dict(DEFAULT); c.update(cfg)
    ts2, ti2 = c["tau_s"] ** 2, c["tau_i"] ** 2
    m_i = np.zeros(nI)
    kw = dict(Gr=Gr, tau_g=c["tau_g"], tau_sg=c["tau_sg"])
    beta, delta, eps, ts2, g, gam = newton(si, ii, y, nS, nI, m_s, m_i, c["tau_b"], ts2, ti2, c["sweeps"], c["beta_prior"], c["eb_s"], c["tau_s"], c["eb_s_max"], **kw)
    if c["eb_i"] > 0:
        cnt = np.bincount(ii, minlength=nI); has_i = cnt > 0
        for _ in range(3):
            p = sig(beta + delta[si] + eps[ii]); w = p * (1 - p)
            hi = np.bincount(ii, w, nI) + 1.0 / ti2
            ss = float(np.sum(((eps - m_i) ** 2 + 1.0 / hi)[has_i]))
            ti2 = min(c["eb_i_max"] ** 2, (c["eb_i"] * c["tau_i"] ** 2 + ss) / (c["eb_i"] + has_i.sum()))
            beta, delta, eps, ts2, g, gam = newton(si, ii, y, nS, nI, m_s, m_i, c["tau_b"], ts2, ti2, c["sweeps"], c["beta_prior"], c["eb_s"], c["tau_s"], c["eb_s_max"], **kw)
    lab = np.unique(ii)
    if prior is not None and c["rounds"] > 0 and len(lab) >= c["min_lab"]:
        counts = np.bincount(ii, minlength=nI)[lab].astype(float)
        for _ in range(c["rounds"]):
            m_i = prior.fit_predict(lab, eps[lab], counts)
            beta, delta, eps, ts2, g, gam = newton(si, ii, y, nS, nI, m_s, m_i, c["tau_b"], ts2, ti2, c["sweeps"], c["beta_prior"], c["eb_s"], c["tau_s"], c["eb_s_max"],
                                                   g0=g, gam0=gam, **kw)
    off = np.zeros(nI)
    if g is not None:
        off = Gr @ g
    eps_tot = eps + off
    z = beta + delta[si] + eps_tot[ii]
    if gam is not None:
        z = z + np.einsum("lc,lc->l", Gr[ii], gam[si])
    p = sig(z); w = p * (1 - p)
    hs = np.bincount(si, w, nS) + 1.0 / ts2; hi = np.bincount(ii, w, nI) + 1.0 / ti2
    return dict(beta=beta, delta=delta, eps=eps_tot, m_i=m_i, ts2=ts2, ti2=ti2, hs=hs, hi=hi, si=si, ii=ii, y=y, gam=gam, Gr=Gr)


def predict_targets(fit: dict, ts, ti, cfg: dict, K: np.ndarray | None = None) -> np.ndarray:
    c = dict(DEFAULT); c.update(cfg)
    z = fit["beta"] + fit["delta"][ts] + fit["eps"][ti]
    if fit.get("gam") is not None:
        z = z + np.einsum("lc,lc->l", fit["Gr"][ti], fit["gam"][ts])
    if c["sim"] > 0 and K is not None:
        # similarity-weighted residuals of OTHER subjects on the same item (GP-like one-step posterior of a subject x item term)
        si, ii, y = fit["si"], fit["ii"], fit["y"]
        p = sig(fit["beta"] + fit["delta"][si] + fit["eps"][ii]); r = y - p; w = p * (1 - p)
        order = np.argsort(ii, kind="stable"); ii_s = ii[order]
        lo = np.searchsorted(ii_s, ti, "left"); hi_ = np.searchsorted(ii_s, ti, "right")
        add = np.zeros(len(ts))
        for t in np.nonzero(hi_ > lo)[0]:
            ls = order[lo[t]:hi_[t]]
            k = K[ts[t], si[ls]]
            k = np.where(si[ls] == ts[t], 0.0, k)
            if c["sim_min"] > 0:
                k = np.where(k >= c["sim_min"], k, 0.0)
            if k.sum() > 0:
                add[t] = c["sim"] * float(np.sum(k * r[ls])) / (1.0 + c["sim"] * float(np.sum(k * w[ls])))
        z = z + add
    if c["probit"] > 0:
        v = 1.0 / fit["hs"][ts] + 1.0 / fit["hi"][ti]
        z = z / np.sqrt(1.0 + np.pi * c["probit"] * v / 8.0)
    return sig(z)
