"""exp_pooled_e2e (tag 'pooled'): end-to-end check through the real predict(input, labeled) of submission modules, on the harness splits
(seed 0 RNG stream of item_prior_cv.py), pooled and single regimes. Revealed labels are binarised (round(y)); Brier against the harness's y.
  .venv/bin/python scripts/exp_pooled_e2e.py real_webagents,researchcodebench [max_eval_per_pair]
Compares: v5 (submission_v5/model.py) vs candidate (src/paie_exp/pooled/model.py) with TAU_G / BETA_PRIOR settings."""
from __future__ import annotations
import importlib.util, sys, time
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "submission_v5"))
from paie.data import load_all  # noqa: E402

FIELDS = ["normalized_name", "provider", "release_date", "access_date", "harness", "harness_version", "reasoning_effort", "subject_features_extra"]
W = {0: .1, 1: .2, 3: .2, 7: .2, 15: .2, 31: .1}


def load_module(path, name, **overrides):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    for k, v in overrides.items():
        setattr(m, k, v)
    return m


def main():
    benches = sys.argv[1].split(",") if len(sys.argv) > 1 else ["real_webagents", "researchcodebench"]
    max_ev = int(sys.argv[2]) if len(sys.argv) > 2 else 10 ** 9
    pairs, subs, items = load_all(); items = items.drop_duplicates("item_id"); subs = subs.drop_duplicates("subject_id")
    itxt = dict(zip(items.item_id, items.content.fillna("").astype(str))); ifeat = dict(zip(items.item_id, items.item_features.fillna("").astype(str)))
    sdict = {r["subject_id"]: {f: (str(r.get(f)) if pd.notna(r.get(f)) else "") for f in FIELDS} for _, r in subs.iterrows()}
    rng = np.random.default_rng(0); pools_by = {}
    for held in sorted(pairs.benchmark.unique()):
        test = pairs[pairs.benchmark == held]; pools = []
        for sid, g in test.groupby("subject_id"):
            if len(g) < 80:
                continue
            g = g.sample(frac=1.0, random_state=int(rng.integers(1e9))); h = len(g) // 2; pools.append((sid, g.iloc[:h], g.iloc[h:]))
        pools_by[held] = pools
    variants = [("v5", ROOT / "submission_v5/model.py", {}),
                ("cand_as_v5", ROOT / "src/paie_exp/pooled/model.py", dict(TAU_G=0.0, TAU_B=0.8)),
                ("cand_tg.5_tb.8", ROOT / "src/paie_exp/pooled/model.py", dict(TAU_G=0.5, TAU_B=0.8)),
                ("cand_default", ROOT / "src/paie_exp/pooled/model.py", {})]
    for held in benches:
        pools = pools_by[held]
        item_dict = lambda i: {"item_content": itxt[i], "benchmark_id": "bench_" + str(abs(hash(held)) % 997), "item_features": ifeat[i], "interactors": None}
        for regime in ("pooled", "single"):
            groups = [list(range(len(pools)))] if regime == "pooled" else [[j] for j in range(len(pools))]
            for name, path, ov in variants:
                t0 = time.time(); B = {k: [] for k in W}
                for grp in groups:
                    M = load_module(path, f"m_{name}_{held}_{regime}_{grp[0]}", **ov)
                    for k in W:
                        labeled = [[[sdict[pools[j][0]], item_dict(i)], int(round(y))] for j in grp for i, y in zip(pools[j][1].item_id.values[:k], pools[j][1].y.values[:k])]
                        for j in grp:
                            sid, _, ev = pools[j]; ev = ev.iloc[:max_ev]
                            p = np.array([M.predict([sdict[sid], item_dict(i)], labeled) for i in ev.item_id.values])
                            B[k].append(np.mean((p - ev.y.values) ** 2))
                Bk = {k: float(np.mean(v)) for k, v in B.items()}
                print(f"E2E {held:18s} {regime:6s} {name:14s} ALC {sum(W[k] * Bk[k] for k in W):.4f}  " + " ".join(f"B{k}={Bk[k]:.3f}" for k in W) + f"  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
