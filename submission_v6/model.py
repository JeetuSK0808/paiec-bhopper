"""PAIEC submission v6 (team BHopper): hierarchical pooled adaptation, re-calibrated for the hidden benchmarks.

logit p = beta_benchmark + delta_subject + eps_item + sum_c G[item, c] * g_c, fitted per benchmark by regularised block-Newton
sweeps (MAP) on ALL revealed labels of the item's benchmark (every system that shares the benchmark pools its labels).
Priors: beta ~ N(0, TAU_B^2); delta_s ~ N(m_s, TAU_S^2) with m_s = SP_SCALE x the system's name-matched relative strength on the six
public Measurement-DB benchmarks (metadata ridge fallback; subject_prior.json); eps_i ~ N(m_i, TAU_I^2) with m_i from a hashed word
uni+bigram TF-IDF dual ridge fitted on the labelled items' effects; g_c ~ N(0, TAU_G^2) for item_features tokens 'key=value' shared by
>= 2 (not all) labelled items of the benchmark, plus one z-scored slope per numeric key. Budget 0 (no label on the benchmark): 0.5.

Changes from v5 (docs/V6_NOTES.md has every number):
  * TAU_B 0.8 -> 1.4. v5's MAP beta update with TAU_B = 0.8 made the benchmark level adapt too slowly for a system that is alone on
    its benchmark; the hidden benchmarks' accuracies (from our own formative-feedback logs) are spread like Uniform(0, 1) across pairs,
    with a large between-benchmark and a small within-benchmark spread.
  * item_features group effects (TAU_G 0.5).
  * robustness: soft (fractional) labels are used as-is instead of raising; malformed labeled entries are skipped; missing or non-dict
    subject / item / benchmark_id fall back gracefully; any internal error falls back to a smoothed per-pair mean, then 0.5.
  * speed: items and subjects are keyed by tuples of their field strings (no per-call json.dumps of long item texts); the parsed
    labeled list is cached while the evaluator passes the same list object.
  * verification fixes (28 Sep 2026): numeric item_features z-scores are clipped to +-Z_CLIP = 3 (fit and prediction rows alike), so
    an unseen item whose value lies far outside the labelled items' range cannot get an unbounded group offset (a heavy-tailed
    numeric feature otherwise drove predictions to 0.005 / 0.995); a non-finite linear predictor now raises inside predict_core
    (predict() then falls back) instead of being mapped to ~1.0 by the min/max clamp.
numpy + standard library only; deterministic.
"""
from __future__ import annotations
import json, math, os
import numpy as np
from textprior import TextPrior, hash_counts
from subjprior import SubjectPrior

TAU_B, TAU_S, TAU_I, TAU_G = 1.4, 0.8, 1.5, 0.5
TAU_S_MANY, N_MANY = 0.0, 25          # optional TAU_S when >= N_MANY systems share a benchmark; OFF (hurt on hidden-like data)
SWEEPS, ROUNDS, ALPHA, MIN_LABELLED_ITEMS, SP_SCALE = 3, 3, 1.0, 10, 0.5
P_MIN, P_MAX = 0.005, 0.995
Z_CLIP = 3.0                          # numeric item_features z-scores are clipped to +-Z_CLIP (fit and prediction rows alike)
_HERE = os.path.dirname(os.path.abspath(__file__))


def _load_subject_prior() -> SubjectPrior | None:
    try:
        with open(os.path.join(_HERE, "subject_prior.json")) as fh:
            j = json.load(fh)
        sp = SubjectPrior()
        sp.table = {k: tuple(v) for k, v in j["table"].items()}
        sp.mu = np.array(j["mu"], float); sp.sd = np.array(j["sd"], float); sp.w = np.array(j["w"], float)
        sp.kw = float(j["kw"]); sp.mw = float(j["mw"])
        return sp
    except Exception:
        return None


_SP = _load_subject_prior()
_ITEMS: dict = {}      # item key -> (benchmark, hashed counts or None, parsed features)
_SUBJ: dict = {}       # subject key -> prior mean m_s
_FITS: dict = {}       # (benchmark, rows hash) -> fit
_LAB = {"obj": None, "n": -1, "rows": None}   # parsed `labeled` (kept while the evaluator reuses the same list object)


# ------------------------------------------------------------------------------------------------------------------ keys
def _s(v) -> str:
    if isinstance(v, str):
        return v
    if v is None:
        return ""
    try:
        return json.dumps(v, sort_keys=True, default=str)
    except Exception:
        return str(v)


def _dict_key(d) -> tuple:
    if isinstance(d, dict):
        return tuple(sorted((str(k), _s(v)) for k, v in d.items()))
    return (("__raw__", _s(d)),)


def _bench(item) -> str:
    b = item.get("benchmark_id") if isinstance(item, dict) else None
    return b if isinstance(b, str) and b else ("__nobench__" if b is None or b == "" else _s(b))


def _parse_features(f) -> dict:
    """'k=v;k=v' string, dict or JSON object -> {key: value-string}; anything else -> {}."""
    try:
        if f is None:
            return {}
        pairs = None
        if isinstance(f, dict):
            pairs = list(f.items())
        else:
            s = str(f).strip()
            if not s or s.lower() in ("none", "nan", "null", "{}", "[]"):
                return {}
            if s[0] == "{":
                try:
                    j = json.loads(s)
                    pairs = list(j.items()) if isinstance(j, dict) else None
                except Exception:
                    pairs = None
            if pairs is None:
                pairs = []
                for tok in s.replace("\n", ";").split(";"):
                    tok = tok.strip()
                    if tok and "=" in tok:
                        k, v = tok.split("=", 1); pairs.append((k, v))
        out = {}
        for k, v in pairs:
            k = str(k).strip()
            if k:
                out[k] = _s(v).strip()
        return out
    except Exception:
        return {}


def _register_item(item) -> tuple:
    k = _dict_key(item)
    if k not in _ITEMS:
        content = item.get("item_content", "") if isinstance(item, dict) else ""
        try:
            counts = hash_counts(_s(content))
        except Exception:
            counts = None
        _ITEMS[k] = (_bench(item), counts, _parse_features(item.get("item_features") if isinstance(item, dict) else None))
    return k


def _register_subject(subj) -> tuple:
    k = _dict_key(subj)
    if k not in _SUBJ:
        m = 0.0
        if _SP is not None and isinstance(subj, dict):
            try:
                m = SP_SCALE * float(_SP.offset(subj))
                if not math.isfinite(m):
                    m = 0.0
            except Exception:
                m = 0.0
        _SUBJ[k] = m
    return k


def _label(y):
    if isinstance(y, bool):
        return float(y)
    try:
        v = float(y)
    except Exception:
        return None
    if not math.isfinite(v):
        return None
    return min(1.0, max(0.0, v))


def _rows_by_bench(labeled) -> dict:
    """{benchmark: (rows tuple, hash)} with rows = ((subject key, item key, label), ...) in acquisition order."""
    if _LAB["obj"] is labeled and _LAB["n"] == len(labeled):
        return _LAB["rows"]
    by: dict = {}
    for entry in labeled:
        try:
            (subj, item), y = entry
        except Exception:
            continue
        v = _label(y)
        if v is None or not isinstance(item, dict):
            continue
        ik = _register_item(item)
        by.setdefault(_ITEMS[ik][0], []).append((_register_subject(subj), ik, v))
    rows = {b: (tuple(r), hash(tuple(r))) for b, r in by.items()}
    _LAB.update(obj=labeled, n=len(labeled), rows=rows)
    return rows


# ------------------------------------------------------------------------------------------------------------ the model
def _sig(z):
    return 1.0 / (1.0 + np.exp(-z))


class Groups:
    """item_features design from the labelled items of one benchmark: one-hot 'key=value' tokens carried by >= 2 (and not all) of them,
    plus a z-scored column per numeric key (>= 90% numeric, >= 3 distinct values)."""

    def __init__(self, feats: list):
        n = len(feats); cnt: dict = {}
        for d in feats:
            for kv in d.items():
                cnt[kv] = cnt.get(kv, 0) + 1
        self.toks = {t: j for j, t in enumerate(sorted(t for t, c in cnt.items() if 2 <= c < n))}
        self.num = []
        for key in sorted({k for d in feats for k in d}):
            vals = []
            for d in feats:
                try:
                    vals.append(float(d[key]))
                except Exception:
                    vals.append(float("nan"))
            v = np.array(vals, float); ok = np.isfinite(v)
            if n and ok.mean() >= 0.9 and len(np.unique(v[ok])) >= 3:
                self.num.append((key, float(v[ok].mean()), float(v[ok].std()) + 1e-9))
        self.dim = len(self.toks) + len(self.num)

    def row(self, d: dict) -> np.ndarray:
        r = np.zeros(self.dim)
        for kv in d.items():
            j = self.toks.get(kv)
            if j is not None:
                r[j] = 1.0
        for a, (key, mu, sd) in enumerate(self.num):
            try:
                x = (float(d[key]) - mu) / sd
                if math.isfinite(x):
                    r[len(self.toks) + a] = min(Z_CLIP, max(-Z_CLIP, x))   # no unbounded extrapolation to unseen values
            except Exception:
                pass
        return r


def _newton(si, ii, y, nS, nI, m_s, m_i, G, g0):
    ts = TAU_S_MANY if (TAU_S_MANY > 0 and nS >= N_MANY) else TAU_S
    beta = 0.0; delta = m_s.copy(); eps = m_i.copy()
    use_g = G is not None
    g = (np.zeros(G.shape[1]) if g0 is None else g0.copy()) if use_g else None
    damp = float(max(1.0, (G != 0).sum(1).max())) if use_g else 1.0
    G2 = G ** 2 if use_g else None

    def z():
        return beta + delta[si] + eps[ii] + ((G @ g)[ii] if use_g else 0.0)
    for _ in range(SWEEPS):
        p = _sig(z())
        beta += float((np.sum(y - p) - beta / TAU_B ** 2) / (np.sum(p * (1 - p)) + 1.0 / TAU_B ** 2))
        p = _sig(z()); r = y - p; w = p * (1 - p)
        delta = delta + (np.bincount(si, r, nS) - (delta - m_s) / ts ** 2) / (np.bincount(si, w, nS) + 1.0 / ts ** 2)
        if use_g:
            p = _sig(z()); r = y - p; w = p * (1 - p)
            g = g + (G.T @ np.bincount(ii, r, nI) - g / TAU_G ** 2) / (G2.T @ np.bincount(ii, w, nI) + 1.0 / TAU_G ** 2) / damp
        p = _sig(z()); r = y - p; w = p * (1 - p)
        eps = eps + (np.bincount(ii, r, nI) - (eps - m_i) / TAU_I ** 2) / (np.bincount(ii, w, nI) + 1.0 / TAU_I ** 2)
    return beta, delta, eps, g


def fit_core(si, ii, y, nS, nI, m_s, C_lab, feats_lab):
    """One benchmark snapshot. si/ii: integer subject / item index per label; C_lab: (nI x dim) hashed counts or None; feats_lab: parsed
    item_features per labelled item. Returns a dict used by predict_core. Pure function of its inputs (deterministic)."""
    grp = Groups(feats_lab) if TAU_G > 0 else None
    G = np.stack([grp.row(d) for d in feats_lab]) if grp is not None and grp.dim > 0 else None
    m_i = np.zeros(nI); prior = None
    beta, delta, eps, g = _newton(si, ii, y, nS, nI, m_s, m_i, G, None)
    if ROUNDS > 0 and C_lab is not None and nI >= MIN_LABELLED_ITEMS:
        counts = np.bincount(ii, minlength=nI).astype(float)
        for _ in range(ROUNDS):
            prior = TextPrior(alpha=ALPHA).fit(C_lab, C_lab, eps, counts)
            m_i = prior.predict(C_lab)
            beta, delta, eps, g = _newton(si, ii, y, nS, nI, m_s, m_i, G, g)
    off = G @ g if G is not None else np.zeros(nI)
    return dict(beta=float(beta), delta=delta, eps=eps + off, prior=prior, grp=grp if G is not None else None, g=g)


def predict_core(fit, s_idx, m_s_target, i_idx, counts_target, feats_target) -> float:
    e = None
    if i_idx is not None:
        e = float(fit["eps"][i_idx])
    else:
        e = 0.0
        if fit["prior"] is not None and counts_target is not None:
            e += float(fit["prior"].predict(counts_target[None, :])[0])
        if fit["grp"] is not None:
            e += float(fit["grp"].row(feats_target) @ fit["g"])
    d = float(fit["delta"][s_idx]) if s_idx is not None else m_s_target
    z = fit["beta"] + d + e
    if not math.isfinite(z):
        raise ValueError("non-finite linear predictor")   # predict() then falls back (min/max would map NaN to ~1.0)
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))


def _fit(bench: str, rows: tuple, h: int):
    ck = (bench, len(rows), h)
    hit = _FITS.get(ck)
    if hit is not None:
        return hit
    sx: dict = {}; ix: dict = {}
    for s_, i_, _ in rows:
        sx.setdefault(s_, len(sx)); ix.setdefault(i_, len(ix))
    si = np.array([sx[r[0]] for r in rows], int); ii = np.array([ix[r[1]] for r in rows], int)
    y = np.array([r[2] for r in rows], float)
    items = list(ix)
    cl = [_ITEMS[i_][1] for i_ in items]
    C_lab = np.stack(cl) if all(c is not None for c in cl) else None
    m_s = np.array([_SUBJ.get(s_, 0.0) for s_ in sx], float)
    fit = fit_core(si, ii, y, len(sx), len(ix), m_s, C_lab, [_ITEMS[i_][2] for i_ in items])
    fit["sx"] = sx; fit["ix"] = ix
    _FITS[ck] = fit
    if len(_FITS) > 32:
        _FITS.pop(next(iter(_FITS)))
    return fit


def _fallback(sk, bench, rows_by) -> float:
    """Smoothed per-pair mean (then benchmark mean), used only if the model raises."""
    try:
        rows = rows_by.get(bench, ((), 0))[0]
        own = [r[2] for r in rows if r[0] == sk]
        use = own if own else [r[2] for r in rows]
        return (1.0 + sum(use)) / (2.0 + len(use)) if use else 0.5
    except Exception:
        return 0.5


def predict(input: list, labeled: list | None = None) -> float:
    try:
        subject, item = input
    except Exception:
        return 0.5
    rows_by = {}
    try:
        ik = _register_item(item)
        sk = _register_subject(subject)
        if not labeled:
            return 0.5
        rows_by = _rows_by_bench(labeled)
        bench = _ITEMS[ik][0]
        if bench not in rows_by:
            return 0.5
        rows, h = rows_by[bench]
        fit = _fit(bench, rows, h)
        _, counts, feats = _ITEMS[ik]
        p = predict_core(fit, fit["sx"].get(sk), _SUBJ.get(sk, 0.0), fit["ix"].get(ik), counts, feats)
        if not math.isfinite(p):
            raise ValueError("non-finite prediction")
    except Exception:
        try:
            p = _fallback(_dict_key(subject), _bench(item), rows_by)
        except Exception:
            p = 0.5
    return float(min(P_MAX, max(P_MIN, p)))
