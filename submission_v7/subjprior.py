"""Subject-strength prior for PAIE: a system's relative strength (logit accuracy minus the benchmark's mean logit), carried across
benchmarks by NORMALISED MODEL NAME, plus a ridge from subject metadata (release date, provider, size/variant tokens, reasoning effort)
for systems never seen before. Returns a logit offset used as the prior mean of delta_s in the pooled adaptation."""
from __future__ import annotations
import re
import numpy as np

_PROVIDERS = ["openai", "anthropic", "google", "meta", "moonshot", "deepseek", "alibaba", "qwen", "mistral", "xai", "zhipu", "microsoft", "bytedance", "minimax", "amazon", "nvidia"]
_TOKENS = ["opus", "sonnet", "haiku", "pro", "flash", "mini", "nano", "lite", "max", "turbo", "think", "reason", "preview", "instruct", "coder", "gpt-5", "gpt-4", "o3", "o4", "gemini 3", "gemini 2", "claude 4", "claude 3", "llama", "qwen3", "kimi", "grok 4", "grok 3", "deepseek v3", "r1"]


def norm_name(s: str) -> str:
    s = str(s or "").lower()
    for t in ("(think)", "(no-think)", "thinking"):
        s = s.replace(t, " ")
    return " ".join(s.replace("_", "-").split())


def meta_features(subj: dict) -> np.ndarray:
    name = norm_name(subj.get("normalized_name", "")) + " " + str(subj.get("subject_features_extra", "")).lower()
    prov = str(subj.get("provider", "")).lower()
    rd = str(subj.get("release_date", ""))[:10]
    try:
        y, m, d = (int(x) for x in rd.split("-")); days = (y - 2023) * 365 + (m - 1) * 30.4 + d
    except Exception:
        days = np.nan
    size = re.findall(r"(\d+(?:\.\d+)?)\s*b\b", name); size = float(size[0]) if size else np.nan
    eff = str(subj.get("reasoning_effort", "")).lower()
    f = [0.0 if np.isnan(days) else days / 365.0, float(np.isnan(days)), 0.0 if np.isnan(size) else np.log1p(size), float(np.isnan(size)),
         float(eff in ("high", "xhigh")), float(eff == "low"), float("think" in name or "reason" in name)]
    f += [float(p in prov or p in name) for p in _PROVIDERS] + [float(t in name) for t in _TOKENS]
    return np.array(f, dtype=float)


class SubjectPrior:
    def __init__(self, alpha: float = 3.0, known_weight: float = 0.6, meta_weight: float = 0.5):
        self.alpha, self.kw, self.mw = alpha, known_weight, meta_weight
        self.table = {}; self.w = None; self.mu = None; self.sd = None

    def fit(self, rows):
        """rows: iterable of (subject_dict, benchmark, accuracy, n_items)."""
        import collections
        by_b = collections.defaultdict(list)
        for subj, b, acc, n in rows:
            by_b[b].append((subj, acc, n))
        rel = collections.defaultdict(list); X, Y, W = [], [], []
        for b, lst in by_b.items():
            lg = np.array([np.log(np.clip(a, .02, .98) / (1 - np.clip(a, .02, .98))) for _, a, _ in lst]); c = lg - lg.mean()
            for (subj, a, n), v in zip(lst, c):
                rel[norm_name(subj.get("normalized_name", ""))].append(v); X.append(meta_features(subj)); Y.append(v); W.append(min(n, 200))
        self.table = {k: (float(np.mean(v)), len(v)) for k, v in rel.items()}
        X = np.array(X); Y = np.array(Y); W = np.array(W, float)
        self.mu = X.mean(0); self.sd = X.std(0) + 1e-6; Xs = (X - self.mu) / self.sd
        A = Xs.T @ (Xs * W[:, None]) + self.alpha * np.eye(X.shape[1]) * W.mean(); self.w = np.linalg.solve(A, Xs.T @ (W * Y)); self.b0 = 0.0
        return self

    def offset(self, subj: dict) -> float:
        meta = float(((meta_features(subj) - self.mu) / self.sd) @ self.w) * self.mw
        k = self.table.get(norm_name(subj.get("normalized_name", "")))
        if k is None:
            return meta
        v, n = k; wk = self.kw * n / (n + 1.0)
        return wk * v + (1 - wk) * meta
