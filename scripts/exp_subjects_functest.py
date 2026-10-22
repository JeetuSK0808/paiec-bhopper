"""Functional test: drive a submission-style model.py through predict(input, labeled) on the CV protocol's pools for ONE held-out
benchmark (same RNG sequence as scripts/exp_subjects_cv.py, so the acquisition / evaluation splits are identical) and report ALC.
Labels handed to predict() are binarised (y >= 0.5), as the real protocol only reveals 0/1; Brier uses the same targets as the CV.
  PAIE_SUBJECT_PRIOR=<json without the bench> .venv/bin/python scripts/exp_subjects_functest.py --model src/paie_exp/subjects/model.py --bench real_webagents --group 1
"""
import argparse, importlib.util, sys, time
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from paie.data import load_all  # noqa: E402
from paie_cv import BUDGETS, WEIGHTS  # noqa: E402

KEYS = ("normalized_name", "provider", "release_date", "access_date", "harness", "harness_version", "reasoning_effort", "subject_features_extra")
ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--bench", required=True)
ap.add_argument("--group", type=int, default=1); ap.add_argument("--seed", type=int, default=0); a = ap.parse_args()
mdir = str(Path(a.model).resolve().parent)
sys.path.insert(0, mdir); sys.path.insert(1, str(ROOT / "submission_v5"))  # textprior.py from submission_v5 when not bundled
spec = importlib.util.spec_from_file_location("cand_model", a.model); M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)

pairs, subjects, items = load_all(); items = items.drop_duplicates("item_id")
sd = subjects.drop_duplicates("subject_id").set_index("subject_id")
sdict = {sid: {k: (None if pd.isna(r.get(k)) else str(r.get(k))) for k in KEYS} for sid, r in sd.iterrows()}
rng = np.random.default_rng(a.seed); rng_grp = np.random.default_rng(a.seed + 1000)
for held in sorted(pairs["benchmark"].unique()):
    test = pairs[pairs["benchmark"] == held]; pools = []
    for sid, g in test.groupby("subject_id"):
        if len(g) < 80:
            continue
        g = g.sample(frac=1.0, random_state=int(rng.integers(1e9))); half = len(g) // 2
        pools.append((sid, g.iloc[:half], g.iloc[half:]))
    if held != a.bench:
        if a.group >= 1 and pools:
            rng_grp.permutation(len(pools))
        continue
    it = items[items["benchmark"] == held].set_index("item_id")
    idict = {x: {"item_content": str(it.loc[x, "content"] or ""), "item_features": it.loc[x, "item_features"], "interactors": None,
                 "benchmark_id": "anon_" + held} for x in test["item_id"].unique()}
    for x in test["item_id"].unique():  # register every item first (the CV's text prior sees all items of the benchmark)
        M.predict([sdict[pools[0][0]], idict[x]], None)
    if a.group == 0:
        groups = [pools]
    else:
        order = rng_grp.permutation(len(pools)); groups = [[pools[j] for j in order[q:q + a.group]] for q in range(0, len(order), a.group)]
    t0 = time.time(); B = {}
    for k in BUDGETS:
        bs = []
        for grp in groups:
            labeled = [[[sdict[sid], idict[x]], int(yv >= 0.5)] for sid, acq, _ in grp for x, yv in zip(acq["item_id"].values[:k], acq["y"].values[:k])]
            for sid, _, ev in grp:
                p = np.array([M.predict([sdict[sid], idict[x]], labeled) for x in ev["item_id"].values])
                bs.append(np.mean((p - ev["y"].values) ** 2))
        B[k] = float(np.mean(bs))
    alc = sum(WEIGHTS[k] * B[k] for k in BUDGETS)
    print(f"{held:18s} group={a.group} ALC {alc:.4f}   " + " ".join(f"B{k}={B[k]:.3f}" for k in BUDGETS) + f"   ({time.time()-t0:.0f}s)")
