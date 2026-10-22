"""numpy-only item-text -> difficulty prior: hashed word uni+bigram TF-IDF (L2-normalised) + ridge in dual form.
Shared by the CV harness (--feat hash) and the submission model so the shipped code is exactly what was validated."""
from __future__ import annotations
import re, zlib
import numpy as np

_TOKEN = re.compile(r"[a-z0-9_]+")


def hash_counts(text: str, dim: int = 4096, max_tokens: int = 3000) -> np.ndarray:
    toks = _TOKEN.findall(text.lower())[:max_tokens]
    v = np.zeros(dim, dtype=np.float32)
    for i, t in enumerate(toks):
        v[zlib.crc32(t.encode()) % dim] += 1.0
        if i + 1 < len(toks):
            v[zlib.crc32((t + " " + toks[i + 1]).encode()) % dim] += 1.0
    return v


class TextPrior:
    """Fit: X (n_items x dim) hashed counts for ALL known items (labelled or not); idf from them; ridge from the labelled
    rows' TF-IDF vectors to their targets (dual form, sample weights). Predict: any TF-IDF row -> prior mean."""

    def __init__(self, dim: int = 4096, alpha: float = 1.0):
        self.dim, self.alpha = dim, alpha
        self.idf = None; self.w = None; self.mean_y = 0.0

    def _tfidf(self, C: np.ndarray) -> np.ndarray:
        X = np.log1p(C) * self.idf
        n = np.linalg.norm(X, axis=1, keepdims=True); n[n == 0] = 1.0
        return X / n

    def fit(self, C_all: np.ndarray, C_lab: np.ndarray, y: np.ndarray, weights: np.ndarray | None = None):
        df = (C_all > 0).sum(0).astype(np.float64)
        self.idf = (np.log((1 + C_all.shape[0]) / (1 + df)) + 1.0).astype(np.float32)
        X = self._tfidf(C_lab).astype(np.float64); self.mean_y = float(np.average(y, weights=weights))
        yc = y - self.mean_y
        w = np.ones(len(y)) if weights is None else np.asarray(weights, float)
        # weighted dual ridge: minimise sum w_i (y_i - x_i.beta)^2 + alpha |beta|^2  ->  beta = X^T (X X^T + alpha W^-1)^-1 y
        K = X @ X.T + self.alpha * np.diag(1.0 / w)
        a = np.linalg.solve(K, yc)
        self.w = X.T @ a
        return self

    def predict(self, C: np.ndarray) -> np.ndarray:
        return self._tfidf(C).astype(np.float64) @ self.w + self.mean_y
