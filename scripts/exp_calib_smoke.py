"""exp_calib: end-to-end smoke test of a submission directory's predict() in the SINGLE-system protocol on small public benchmarks
(labels binarised y >= 0.5 as the official stream requires; eval targets = y). Not leakage-free: the bundled subject prior was fit on all
public benchmarks. Compares two submission dirs on the same splits.
  .venv/bin/python scripts/exp_calib_smoke.py submission_v5 src/paie_exp/calib/submission_v5_taurule"""
import sys, importlib.util, numpy as np, pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from paie.data import load_all
W = {0: 0.1, 1: 0.2, 3: 0.2, 7: 0.2, 15: 0.2, 31: 0.1}
def load_model(d, name):
    sys.path.insert(0, str(ROOT / d))
    for m in ("textprior", "subjprior"):
        sys.modules.pop(m, None)
    spec = importlib.util.spec_from_file_location(name, str(ROOT / d / "model.py")); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    sys.path.pop(0); return mod
pairs, subjects, items = load_all(); items = items.drop_duplicates("item_id").set_index("item_id"); sd = subjects.drop_duplicates("subject_id").set_index("subject_id")
def sdict(s):
    r = sd.loc[s]; return {k: (None if k not in r or pd.isna(r.get(k)) else str(r.get(k))) for k in ("normalized_name", "provider", "release_date", "access_date", "harness", "harness_version", "reasoning_effort", "subject_features_extra")}
def idict(i, b):
    r = items.loc[i]; return {"item_content": str(r["content"]) if pd.notna(r["content"]) else "", "item_features": None if pd.isna(r.get("item_features")) else str(r.get("item_features")), "interactors": None, "benchmark_id": "bench_" + str(abs(hash(b)) % 997)}
for d in sys.argv[1:]:
    mod = load_model(d, "m_" + str(abs(hash(d))))
    rng = np.random.default_rng(0); out = []
    for b in ("real_webagents", "researchcodebench"):
        test = pairs[pairs.benchmark == b]; its = {i: idict(i, b) for i in test.item_id.unique()}
        for i, it in its.items():
            mod.predict([sdict(test.subject_id.iloc[0]), it], [])  # register every item (idf over the benchmark, as in the harness)
        alcs = []
        for s, g in test.groupby("subject_id"):
            if len(g) < 80: continue
            g = g.sample(frac=1.0, random_state=int(rng.integers(1e9))); h = len(g) // 2; acq, ev = g.iloc[:h], g.iloc[h:]
            S = sdict(s); Bk = {}
            for k in W:
                lab = [[[S, its[i]], int(y >= 0.5)] for i, y in zip(acq.item_id.values[:k], acq.y.values[:k])]
                p = np.array([mod.predict([S, its[i]], lab) for i in ev.item_id.values]); Bk[k] = float(np.mean((p - ev.y.values) ** 2))
            alcs.append(sum(W[k] * Bk[k] for k in W))
        out.append((b, np.mean(alcs), len(alcs)))
    print(f"{d:45s} " + "  ".join(f"{b} ALC {a:.4f} ({n} pairs)" for b, a, n in out) + f"  mean {np.mean([a for _, a, _ in out]):.4f}", flush=True)
