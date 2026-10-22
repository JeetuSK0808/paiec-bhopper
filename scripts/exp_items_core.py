"""Tag 'items': item_features + within-benchmark structure for the PAIE hierarchical model (numpy + stdlib only).

logit p(s, i) = beta + delta_s + sum_k gamma_{k, level_k(i)} [+ u_{s, k, level_k(i)}] + theta . x_i + eps_i

* item_features 'k1=v1;k2=v2' (or dict / JSON) are parsed generically. Keys whose values are mostly numeric with >= 3 distinct values
  become numeric (ordinal) covariates x_i; all other keys become categorical group effects gamma_k ~ N(0, tau_g^2). The whole
  feature string is an extra 'identity' level (items that share every feature, e.g. the same competition problem stored under several
  item ids) with its own tau. Levels seen on a single known item and keys with one level are dropped (they cannot transfer).
* eps_i ~ N(m_i, tau_i^2), m_i from a kernel ridge on hashed TF-IDF of the item text (linear kernel = the v5 TextPrior).
* Everything degrades to the v5 model when item_features is None/empty.
"""
from __future__ import annotations
import json, math, re
import numpy as np

_NUM = re.compile(r"^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$")


def parse_features(f) -> list:
    """-> list of (key, value) string pairs. Accepts None, '', 'k=v;k2=v2', a dict or a JSON object string."""
    if f is None:
        return []
    if isinstance(f, float) and math.isnan(f):
        return []
    pairs = None
    if isinstance(f, dict):
        pairs = list(f.items())
    else:
        s = str(f).strip()
        if not s or s.lower() in ("none", "nan", "null", "{}", "[]"):
            return []
        if s[0] == "{":
            try:
                j = json.loads(s)
                if isinstance(j, dict):
                    pairs = list(j.items())
            except Exception:
                pairs = None
        if pairs is None:
            pairs = []
            for tok in re.split(r"[;\n]", s):
                tok = tok.strip()
                if not tok:
                    continue
                if "=" in tok:
                    k, v = tok.split("=", 1)
                else:
                    k, v = tok, "1"
                pairs.append((k, v))
    out = []
    for k, v in pairs:
        if isinstance(v, (dict, list)):
            v = json.dumps(v, sort_keys=True)
        k = str(k).strip().lower()
        if k:
            out.append((k, str(v).strip()))
    return out


def _isnum(v: str) -> bool:
    return bool(_NUM.match(v.strip()))


class ItemStructure:
    """Per-benchmark item structure from the feature strings of the KNOWN item universe (labelled + target items).
    groups: list of (name, level index per item (-1 = none), n_levels); numeric: (N, P) matrix (0 = missing / mean)."""

    def __init__(self, feats: list, identity: bool = True, numeric_mode: str = "z", numeric_as_cat: bool = False,
                 min_level_items: int = 2, drop_keys: tuple = ()):
        N = len(feats)
        parsed = [parse_features(f) for f in feats]
        keyvals: dict = {}
        for n, pr in enumerate(parsed):
            for k, v in pr:
                if k in drop_keys:
                    continue
                keyvals.setdefault(k, []).append((n, v))
        self.groups, self.numeric_names, cols = [], [], []
        for k in sorted(keyvals):
            lst = keyvals[k]
            vals = [v for _, v in lst]
            numv = [float(v) for v in vals if _isnum(v)]
            is_num = len(numv) >= 0.8 * len(vals) and len(set(numv)) >= 3
            if is_num:
                x = np.full(N, np.nan)
                for n, v in lst:
                    if _isnum(v):
                        x[n] = float(v)
                ok = ~np.isnan(x)
                if numeric_mode == "rank":
                    r = np.zeros(N)
                    order = np.argsort(np.argsort(x[ok], kind="stable"), kind="stable")
                    r[ok] = (order + 0.5) / ok.sum() - 0.5
                    col = r * math.sqrt(12.0)
                elif numeric_mode == "log":
                    lx = np.sign(x[ok]) * np.log1p(np.abs(x[ok]))
                    col = np.zeros(N); sd = lx.std() or 1.0; col[ok] = (lx - lx.mean()) / sd
                else:  # z-score
                    col = np.zeros(N); sd = x[ok].std() or 1.0; col[ok] = (x[ok] - x[ok].mean()) / sd
                cols.append(col); self.numeric_names.append(k)
                if not numeric_as_cat:
                    continue
            self._add_group(k, lst, N, min_level_items)
        if identity:
            idl = []
            for n, pr in enumerate(parsed):
                pr2 = [(k, v) for k, v in pr if k not in drop_keys]
                if len(pr2) >= 2:
                    idl.append((n, ";".join(f"{k}={v}" for k, v in sorted(pr2))))
            if idl:
                self._add_group("__identity__", idl, N, min_level_items)
        self.X = np.stack(cols, 1) if cols else None

    def _add_group(self, name, lst, N, min_level_items):
        cnt: dict = {}
        for _, v in lst:
            cnt[v] = cnt.get(v, 0) + 1
        levels = sorted(v for v, c in cnt.items() if c >= min_level_items)
        if len(levels) < 2 and name != "__identity__":
            return
        if not levels:
            return
        idx = {v: j for j, v in enumerate(levels)}
        g = np.full(N, -1, dtype=np.int64)
        for n, v in lst:
            if v in idx:
                g[n] = idx[v]
        self.groups.append((name, g, len(levels)))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def fit_hier(s_idx, i_idx, y, m_s, m_i, groups=(), tau_g=(), X=None, tau_n=1.0, mu_n=0.0, sg_keys=(), tau_sg=0.5,
             tau_b=0.8, tau_s=0.8, tau_i=1.5, sweeps=3, beta_map=False, subj_map=True, center=False, min_rows_group=0):
    """MAP-style Newton sweeps (same update order/form as scripts/item_prior_cv.fit_pooled_s: beta damped step, delta & eps MAP).
    s_idx: (R,) local subject index; i_idx: (R,) universe item index; y: (R,) 0/1; m_s: (S,) subject prior means;
    m_i: (N,) item prior means. groups: list of (name, g (N,) level idx or -1, L); tau_g: per group prior sd.
    X: (N, P) numeric covariates (theta ~ N(mu_n, tau_n^2)). sg_keys: indices into groups that get subject x level interactions.
    Returns dict(beta, delta, eps, gam (list of (L+1,) arrays, last = 0), theta, u (dict k -> (S, L+1))."""
    S, N = len(m_s), len(m_i)
    beta = 0.0; delta = np.asarray(m_s, float).copy(); eps = np.asarray(m_i, float).copy()
    rg, gam, L = [], [], []
    for (_, g, l) in groups:
        gi = g[i_idx].copy(); gi[gi < 0] = l; rg.append(gi); gam.append(np.zeros(l + 1)); L.append(l)
    P = 0 if X is None else X.shape[1]
    theta = np.full(P, mu_n, float) if P else np.zeros(0)
    Xr = X[i_idx] if P else None
    u = {k: np.zeros((S, L[k] + 1)) for k in sg_keys}
    tb2, ts2, ti2 = tau_b ** 2, tau_s ** 2, tau_i ** 2

    def zfun():
        z = beta + delta[s_idx] + eps[i_idx]
        for k in range(len(rg)):
            z = z + gam[k][rg[k]]
        for k in u:
            z = z + u[k][s_idx, rg[k]]
        if P:
            z = z + Xr @ theta
        return z

    active = len(y) >= min_rows_group
    for _ in range(sweeps):
        p = sigmoid(zfun()); w = p * (1 - p)
        if beta_map:
            beta += (np.sum(y - p) - beta / tb2) / (np.sum(w) + 1.0 / tb2)
        else:
            beta += np.sum(y - p) / (np.sum(w) + 1.0 / tb2)
        p = sigmoid(zfun()); w = p * (1 - p)
        gs = np.bincount(s_idx, y - p, S); hs = np.bincount(s_idx, w, S)
        if subj_map:
            delta += (gs - (delta - m_s) / ts2) / (hs + 1.0 / ts2)
        else:
            delta += gs / (hs + 1.0 / ts2)
        for k in (range(len(rg)) if active else ()):
            p = sigmoid(zfun()); w = p * (1 - p); t2 = tau_g[k] ** 2; l = L[k]
            gg = np.bincount(rg[k], y - p, l + 1)[:l]; hh = np.bincount(rg[k], w, l + 1)[:l]
            gam[k][:l] += (gg - gam[k][:l] / t2) / (hh + 1.0 / t2)
            if center:  # group effects model deviations only: move their row-weighted mean into beta
                on = rg[k] < l
                if on.any():
                    mu = float(gam[k][rg[k][on]].mean()); gam[k][:l] -= mu; beta += mu * on.mean()
        for k in u:
            p = sigmoid(zfun()); w = p * (1 - p); l = L[k]; t2 = tau_sg ** 2
            flat = s_idx * (l + 1) + rg[k]
            gg = np.bincount(flat, y - p, S * (l + 1)).reshape(S, l + 1); hh = np.bincount(flat, w, S * (l + 1)).reshape(S, l + 1)
            u[k] += (gg - u[k] / t2) / (hh + 1.0 / t2); u[k][:, l] = 0.0
        if P:
            p = sigmoid(zfun()); w = p * (1 - p)
            H = Xr.T @ (Xr * w[:, None]) + np.eye(P) / tau_n ** 2
            gr = Xr.T @ (y - p) - (theta - mu_n) / tau_n ** 2
            theta = theta + np.linalg.solve(H, gr)
        p = sigmoid(zfun()); w = p * (1 - p)
        gi_ = np.bincount(i_idx, y - p, N); hi_ = np.bincount(i_idx, w, N)
        eps += (gi_ - (eps - m_i) / ti2) / (hi_ + 1.0 / ti2)
    return dict(beta=beta, delta=delta, eps=eps, gam=gam, theta=theta, u=u, L=L)


def item_offset(fit, groups, X, items, include_eps=True):
    """Item part of the logit (groups + numeric + eps) for universe item indices `items`."""
    z = fit["eps"][items].copy() if include_eps else np.zeros(len(items))
    for k, (_, g, l) in enumerate(groups):
        gi = g[items].copy(); gi[gi < 0] = l; z += fit["gam"][k][gi]
    if X is not None and len(fit["theta"]):
        z += X[items] @ fit["theta"]
    return z


def subj_group_offset(fit, groups, s_local, items):
    z = np.zeros(len(items))
    for k, uk in fit["u"].items():
        _, g, l = groups[k]; gi = g[items].copy(); gi[gi < 0] = l; z += uk[s_local, gi]
    return z


def tfidf_matrix(C: np.ndarray) -> np.ndarray:
    """Same transform as TextPrior (idf from the rows of C, log1p tf, L2 rows), float64."""
    df = (C > 0).sum(0).astype(np.float64)
    idf = (np.log((1 + C.shape[0]) / (1 + df)) + 1.0).astype(np.float32)
    X = np.log1p(C) * idf
    n = np.linalg.norm(X, axis=1, keepdims=True); n[n == 0] = 1.0
    return (X / n).astype(np.float64)


def kernel(A, B, kind="linear", gamma=5.0):
    c = A @ B.T
    if kind == "linear":
        return c
    if kind == "rbf":  # rows are unit vectors: exp(-gamma/2 |a-b|^2) = exp(gamma (c - 1))
        return np.exp(gamma * (c - 1.0))
    if kind == "mix":  # linear + sharp local term
        return c + np.exp(gamma * (c - 1.0))
    raise ValueError(kind)


def ridge_prior(Xlab, y, w, Xq, alpha=1.0, kind="linear", gamma=5.0):
    """Weighted dual (kernel) ridge from labelled items' effects y to query rows; linear kernel == v5 TextPrior."""
    mean_y = float(np.average(y, weights=w)); yc = y - mean_y
    K = kernel(Xlab, Xlab, kind, gamma) + alpha * np.diag(1.0 / w)
    a = np.linalg.solve(K, yc)
    return kernel(Xq, Xlab, kind, gamma) @ a + mean_y
