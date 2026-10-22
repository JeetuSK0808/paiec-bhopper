"""Analysis for the 'subjects' direction: how well do subject priors (built WITHOUT the held-out benchmark) predict each system's
relative strength (logit accuracy minus the benchmark's mean logit) on the held-out benchmark?  Compares the v5 SubjectPrior
(norm_name table + metadata ridge) with a canonical-name table (canon_key) and simple metadata-only predictors.
  nice -n 10 env VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python scripts/exp_subjects_analysis.py
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "src/paie_exp/subjects"))
from paie.data import load_all  # noqa: E402
from paie.subjprior import SubjectPrior, norm_name  # noqa: E402
from canon import canon_key, family_key, effort_level, release_years  # noqa: E402

KEYS = ("normalized_name", "provider", "release_date", "access_date", "harness", "harness_version", "reasoning_effort", "subject_features_extra")


def lg(p):
    p = np.clip(p, .02, .98); return np.log(p / (1 - p))


pairs, subjects, items = load_all()
sd = subjects.drop_duplicates(["benchmark", "subject_id"]).set_index(["benchmark", "subject_id"])
def sdict(b, sid):
    r = sd.loc[(b, sid)]; return {k: ("" if pd.isna(r.get(k)) else str(r.get(k))) for k in KEYS}

acc = pairs.groupby(["benchmark", "subject_id"]).y.agg(["mean", "size"]).reset_index()
acc["lz"] = lg(acc["mean"].values)
acc["rel"] = acc["lz"] - acc.groupby("benchmark")["lz"].transform("mean")
acc["subj"] = [sdict(b, s) for b, s in zip(acc.benchmark, acc.subject_id)]
acc["old"] = [norm_name(d["normalized_name"]) for d in acc.subj]
acc["key"] = [canon_key(d) for d in acc.subj]
acc["fam"] = [family_key(d) for d in acc.subj]
acc["eff"] = [effort_level(d) for d in acc.subj]
acc["ry"] = [release_years(d) for d in acc.subj]

print("== canonical keys per benchmark (first 12) ==")
for b, g in acc.groupby("benchmark"):
    print(b, "|", "; ".join(sorted(set(g.key))[:12]))

print("\n== name coverage: fraction of held-out systems (>=80 items) with a match on another benchmark ==")
for b in sorted(acc.benchmark.unique()):
    g = acc[(acc.benchmark == b) & (acc["size"] >= 80)]; o = acc[acc.benchmark != b]
    print(f"{b:18s} n={len(g):3d}  old-name {np.mean([k in set(o.old) and k != '' for k in g.old]):.2f}   canon {np.mean([k in set(o.key) and k != '' for k in g.key]):.2f}   family {np.mean([k in set(o.fam) and k != '' for k in g.fam]):.2f}")


def table_from(o, keycol, per_bench_mean=True):
    """canonical key -> (mean relative strength over benchmarks, #benchmarks). Within a benchmark, variants (harness, effort)
    of one key are averaged first so multi_swebench's 7 harnesses do not count 7 times."""
    t = o.groupby([keycol, "benchmark"]).rel.mean().reset_index() if per_bench_mean else o[[keycol, "benchmark", "rel"]]
    g = t.groupby(keycol).rel.agg(["mean", "count"])
    return {k: (v["mean"], v["count"]) for k, v in g.iterrows() if k}


print("\n== LOBO: corr(prior, held-out relative strength) and OLS slope, systems with >= 80 items ==")
rows = []
for b in sorted(acc.benchmark.unique()):
    g = acc[(acc.benchmark == b) & (acc["size"] >= 80)].copy(); o = acc[acc.benchmark != b]
    if len(g) < 5:
        continue
    spm = SubjectPrior().fit([(d, bb, m, n) for d, bb, m, n in zip(o.subj, o.benchmark, o["mean"], o["size"])])
    g["v5"] = [spm.offset(d) for d in g.subj]
    t_old = table_from(o, "old", per_bench_mean=False); t_key = table_from(o, "key"); t_fam = table_from(o, "fam")
    g["tab_old"] = [t_old.get(k, (np.nan, 0))[0] for k in g.old]
    g["tab_key"] = [t_key.get(k, (np.nan, 0))[0] for k in g.key]
    g["tab_fam"] = [t_fam.get(k, (np.nan, 0))[0] for k in g.fam]
    out = {"bench": b, "n": len(g)}
    for c in ("v5", "tab_old", "tab_key", "tab_fam", "ry", "eff"):
        m = g[c].notna() & np.isfinite(g[c])
        if m.sum() >= 4 and g.loc[m, c].std() > 0:
            x, y = g.loc[m, c].values, g.loc[m, "rel"].values
            out[c] = f"{np.corrcoef(x, y)[0,1]:+.2f}/{np.polyfit(x, y, 1)[0]:+.2f}/{m.sum()}"
        else:
            out[c] = "-"
    rows.append(out)
print(pd.DataFrame(rows).to_string(index=False))
print("(cells: correlation / OLS slope / #systems with a value)")
print("\nwithin-benchmark sd of relative strength (>=80 items):", acc[acc["size"] >= 80].groupby("benchmark").rel.std().round(2).to_dict())
print("benchmark mean logit (pairs>=80):", acc[acc["size"] >= 80].groupby("benchmark").lz.mean().round(2).to_dict())
