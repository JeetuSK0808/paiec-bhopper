"""Build the bundled stacked subject prior (canonical-name tables + stacked ridge weights) for src/paie_exp/subjects/model.py.
  .venv/bin/python scripts/exp_subjects_build_prior.py [--exclude BENCH] [--out path.json]
--exclude drops one public benchmark (for functional tests of model.py against the leave-one-benchmark-out CV)."""
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "src/paie_exp/subjects"))
from paie.data import load_all  # noqa: E402
from subjpriors import StackSubjectPrior  # noqa: E402

KEYS = ("normalized_name", "provider", "release_date", "access_date", "harness", "harness_version", "reasoning_effort", "subject_features_extra")
ap = argparse.ArgumentParser(); ap.add_argument("--exclude", default=""); ap.add_argument("--alpha", type=float, default=3.0)
ap.add_argument("--out", default=str(ROOT / "src/paie_exp/subjects/subject_prior_stack.json")); a = ap.parse_args()
pairs, subjects, _ = load_all()
if a.exclude:
    pairs = pairs[pairs.benchmark != a.exclude]
sd = subjects.drop_duplicates("subject_id").set_index("subject_id")
sdict = {sid: {k: ("" if pd.isna(r.get(k)) else str(r.get(k))) for k in KEYS} for sid, r in sd.iterrows()}
t = pairs.groupby(["benchmark", "subject_id"]).y.agg(["mean", "size"]).reset_index()
spm = StackSubjectPrior(alpha=a.alpha).fit([(sdict[r.subject_id], r.benchmark, r["mean"], r["size"]) for _, r in t.iterrows()])
json.dump({"w": spm.w.tolist(), "kt": {k: list(v) for k, v in spm.kt.items()}, "ft": {k: list(v) for k, v in spm.ft.items()},
           "excluded": a.exclude, "benchmarks": sorted(pairs.benchmark.unique())}, open(a.out, "w"))
print("stack weights [key val, key hit, fam val, fam hit, release-2025, release missing, effort, log size-3, size missing]:", np.round(spm.w, 3))
print(f"{len(spm.kt)} keys, {len(spm.ft)} families -> {a.out}")
