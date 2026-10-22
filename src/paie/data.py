"""Load the Measurement Data Bank into one pair table: (benchmark, subject, item, y) with y = mean binarized response over
trials, plus subject and item feature frames. Cold-start protocol: held-out BENCHMARKS."""
from __future__ import annotations

import glob
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/hf"


def load_all():
    pairs, subjects, items = [], [], []
    for b in sorted(glob.glob(str(DATA) + "/*/")):  # keep the trailing slash: pathlib would strip it and glob would return files
        name = b.rstrip("/").split("/")[-1]
        rfs = glob.glob(f"{b}response*.parquet")
        if not rfs:  # e.g. the .cache directory snapshot_download leaves behind
            continue
        rf = rfs[0]
        r = pd.read_parquet(rf, columns=["subject_id", "item_id", "trial", "response"])
        r = r.dropna(subset=["response"])
        r["y"] = (r["response"] >= 0.5).astype(float)
        g = r.groupby(["subject_id", "item_id"]).agg(y=("y", "mean"), n_trials=("y", "size")).reset_index()
        g["benchmark"] = name
        pairs.append(g)
        s = pd.read_parquet(f"{b}subjects.parquet"); s["benchmark"] = name; subjects.append(s)
        it = pd.read_parquet(f"{b}items.parquet"); it["benchmark"] = name; items.append(it)
    return pd.concat(pairs, ignore_index=True), pd.concat(subjects, ignore_index=True), pd.concat(items, ignore_index=True)


def subject_features(s: pd.DataFrame) -> pd.DataFrame:
    """Benchmark-agnostic subject attributes (the competition passes exactly these keys)."""
    f = pd.DataFrame(index=s.index)
    rd = pd.to_datetime(s["release_date"], errors="coerce")
    f["release_days"] = (rd - pd.Timestamp("2023-01-01")).dt.days.fillna(-1)
    f["release_missing"] = rd.isna().astype(int)
    prov = s["provider"].fillna("").str.lower()
    for p in ["openai", "anthropic", "google", "meta", "moonshot", "deepseek", "alibaba", "qwen", "mistral", "xai", "zhipu", "microsoft"]:
        f[f"prov_{p}"] = prov.str.contains(p).astype(int)
    f["prov_missing"] = (prov == "").astype(int)
    name = (s["normalized_name"].fillna("") + " " + s["display_name"].fillna("") + " " + s["subject_features_extra"].fillna("")).str.lower()
    for k in ["think", "reason", "pro", "mini", "flash", "opus", "sonnet", "haiku", "gpt-5", "gpt-4", "o3", "o4", "claude", "gemini", "llama", "qwen", "deepseek", "kimi", "grok", "high", "low", "medium", "70b", "8b", "instruct"]:
        f[f"name_{k}"] = name.str.contains(k, regex=False).astype(int)
    f["reasoning_effort_set"] = s["reasoning_effort"].notna().astype(int)
    f["harness_set"] = s["harness"].notna().astype(int)
    f["subject_id"] = s["subject_id"].values
    return f


def item_features(it: pd.DataFrame) -> pd.DataFrame:
    """Cheap, benchmark-agnostic text statistics of the item content (a text embedding can be added later)."""
    c = it["content"].fillna("").astype(str)
    f = pd.DataFrame(index=it.index)
    f["len_chars"] = np.log1p(c.str.len()); f["len_words"] = np.log1p(c.str.split().str.len().fillna(0))
    f["frac_digits"] = c.str.count(r"\d") / (c.str.len() + 1)
    f["frac_symbols"] = c.str.count(r"[\$\\{}\[\]<>=+*/^_|]") / (c.str.len() + 1)
    f["n_lines"] = np.log1p(c.str.count("\n"))
    f["has_code"] = c.str.contains(r"def |class |import |\{|\}|;", regex=True).astype(int)
    f["has_math"] = c.str.contains(r"\\\(|\\\[|\$|\\frac|\\sqrt|\\boxed", regex=True).astype(int)
    f["has_url"] = c.str.contains("http", regex=False).astype(int)
    f["question_marks"] = np.log1p(c.str.count(r"\?"))
    f["item_id"] = it["item_id"].values
    return f
