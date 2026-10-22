"""Tag 'items': run a submission directory's real predict(input, labeled) under the CV protocol (same pools/RNG as the harness) to check
that shipped code matches the harness numbers. Regimes: --single (each system alone) or --nsys M (random groups of M systems) or pooled.
  nice -n 10 env VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python scripts/exp_items_subcheck.py \
      --model-dir submission_v5 --only mmdocrag,multi_swebench --single
"""
import argparse, importlib, sys, time, zlib
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from paie.data import load_all  # noqa: E402
from paie_cv import BUDGETS, WEIGHTS  # noqa: E402

FIELDS = ["normalized_name", "provider", "release_date", "access_date", "harness", "harness_version", "reasoning_effort", "subject_features_extra"]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model-dir", required=True); ap.add_argument("--only", default="")
    ap.add_argument("--single", action="store_true"); ap.add_argument("--nsys", type=int, default=0); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--set", nargs="*", default=[], help="override module constants, e.g. TAU_B=1.5")
    ap.add_argument("--loo-sp", action="store_true", help="refit the subject prior without the held-out benchmark")
    ap.add_argument("--max-eval", type=int, default=0, help="score at most this many eval items per system (0 = all)")
    a = ap.parse_args()
    sys.path.insert(0, str((ROOT / a.model_dir).resolve()))
    M = importlib.import_module("model")
    for kv in a.set:  # e.g. --set TAU_B=1.5 overrides a module constant of the submission
        k, v = kv.split("="); setattr(M, k, type(getattr(M, k))(eval(v)))
    only = set(x for x in a.only.split(",") if x)
    pairs, subs, items = load_all(); items = items.drop_duplicates("item_id"); subs = subs.drop_duplicates("subject_id").set_index("subject_id")
    rng = np.random.default_rng(a.seed); res = {}
    for held in sorted(pairs["benchmark"].unique()):
        test = pairs[pairs["benchmark"] == held]; pools = []
        for sid, g in test.groupby("subject_id"):
            if len(g) < 80:
                continue
            g = g.sample(frac=1.0, random_state=int(rng.integers(1e9))); h = len(g) // 2; pools.append((sid, g.iloc[:h], g.iloc[h:]))
        if (only and held not in only) or not pools:
            continue
        it = items[items["benchmark"] == held].set_index("item_id")
        def sdict(sid):
            r = subs.loc[sid]; d = {f: ("" if pd.isna(r.get(f)) else str(r.get(f))) for f in FIELDS}; d["harness_version"] = (d["harness_version"] + f" [{sid}]").strip()  # unique subjects; field unused by the priors
            return d
        def idict(i_):
            c = it.at[i_, "content"]; f = it.at[i_, "item_features"]
            return {"item_content": "" if pd.isna(c) else str(c), "item_features": "" if (f is None or (isinstance(f, float) and np.isnan(f))) else str(f), "interactors": "", "benchmark_id": f"bench_{zlib.crc32(held.encode())}"}
        if a.single:
            teams = [[n] for n in range(len(pools))]
        elif a.nsys > 0:
            perm = np.random.default_rng([a.seed, 7919, zlib.crc32(held.encode())]).permutation(len(pools)); teams = [list(perm[j:j + a.nsys]) for j in range(0, len(pools), a.nsys)]
        else:
            teams = [list(range(len(pools)))]
        if a.loo_sp:  # leave-one-benchmark-out subject prior (as the harness), fresh caches per benchmark
            from paie.subjprior import SubjectPrior
            def sd5(sid):
                r = subs.loc[sid]; return {f: ("" if pd.isna(r.get(f)) else str(r.get(f))) for f in ("normalized_name", "provider", "release_date", "reasoning_effort", "subject_features_extra")}
            trp = pairs[pairs["benchmark"] != held].groupby(["benchmark", "subject_id"]).y.agg(["mean", "size"]).reset_index()
            spm = SubjectPrior(known_weight=0.6, meta_weight=0.5).fit([(sd5(r.subject_id), r.benchmark, r["mean"], r["size"]) for _, r in trp.iterrows()])
            M._SP.table, M._SP.mu, M._SP.sd, M._SP.w, M._SP.kw, M._SP.mw = spm.table, spm.mu, spm.sd, spm.w, spm.kw, spm.mw
            for c in ("_SUBJ", "_CACHE", "_ITEM_COUNTS"):
                if hasattr(M, c):
                    getattr(M, c).clear()
            if hasattr(M, "reset_caches"):
                M.reset_caches()
        t0 = time.time(); B = {k: [] for k in BUDGETS}
        SD = {sid: sdict(sid) for sid, _, _ in pools}; ID = {}
        def gi(i_):
            if i_ not in ID:
                ID[i_] = idict(i_)
            return ID[i_]
        for k in BUDGETS:
            for team in teams:
                labeled = [[[SD[pools[n][0]], gi(i_)], int(y >= 0.5)] for n in team for i_, y in zip(pools[n][1]["item_id"].values[:k], pools[n][1]["y"].values[:k])]
                for n in team:
                    sid, _, ev = pools[n]; evi = ev["item_id"].values; evy = ev["y"].values
                    if a.max_eval:
                        evi, evy = evi[:a.max_eval], evy[:a.max_eval]
                    p = np.array([M.predict([SD[sid], gi(i_)], labeled) for i_ in evi])
                    B[k].append(np.mean((p - evy) ** 2))
        Bm = {k: float(np.mean(v)) for k, v in B.items()}; alc = sum(WEIGHTS[k] * Bm[k] for k in BUDGETS); res[held] = alc
        print(f"{held:18s} ALC {alc:.4f}  " + " ".join(f"B{k}={Bm[k]:.3f}" for k in BUDGETS) + f"  ({time.time()-t0:.0f}s)", flush=True)
    print(f"mean over {len(res)}: {np.mean(list(res.values())):.4f}  model={a.model_dir} single={a.single} nsys={a.nsys}")


if __name__ == "__main__":
    main()
