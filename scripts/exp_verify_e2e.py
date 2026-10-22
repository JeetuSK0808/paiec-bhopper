"""exp_verify_e2e (tag 'verify'): end-to-end leave-one-benchmark-out CV through the PUBLIC predict(input, labeled) of a submission
directory, on the exact splits of scripts/exp_pooled_cv.py (= scripts/item_prior_cv.py RNG stream), with a per-fold subject_prior.json
(rebuilt WITHOUT the held-out benchmark by scripts/exp_sim_protocol.build_prior_json) in a fold copy of the submission under
src/paie_exp/verify/folds/. A fresh module is imported per (fold, regime, budget), as the evaluator recreates workers per checkpoint.

  .venv/bin/python -B scripts/exp_verify_e2e.py --sub submission_v6 --regimes single,pooled --seed 0 [--binary] [--max-eval N] [--memo]

--binary      acquisition labels rounded (y >= 0.5) so v5 (which rejects fractional labels) can run; evaluation targets stay fractional.
--max-eval N  score only the first N evaluation items of each pair (the pools are already randomly ordered) to bound runtime.
--memo        memoise v5's pure _key(dict) by object identity (bit-identical outputs; runtime only).
Prints per-benchmark ALC and the macro mean; writes per-pair ALC to src/paie_exp/verify/e2e_<tag>.json.
"""
import argparse, hashlib, json, shutil, sys, time
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "src"))
sys.dont_write_bytecode = True
import exp_pooled_cv as H  # noqa: E402
import exp_sim_protocol as S  # noqa: E402
from paie.data import load_all  # noqa: E402

OUT = ROOT / "src" / "paie_exp" / "verify"
W = {0: .1, 1: .2, 3: .2, 7: .2, 15: .2, 31: .1}; BUD = [0, 1, 3, 7, 15, 31]


def fold_copy(sub_dir: Path, held: str, benches_sim, subj_first) -> Path:
    dst = OUT / "folds" / f"{sub_dir.name}__{held}"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(sub_dir, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shipped = json.loads((sub_dir / "subject_prior.json").read_text())
    j = S.build_prior_json(benches_sim, subj_first, exclude={held}, kw=shipped.get("kw", 0.6), mw=shipped.get("mw", 0.5))
    (dst / "subject_prior.json").write_text(json.dumps(j))
    return dst


def memoise(mod):
    if callable(getattr(mod, "_key", None)):
        orig, memo = mod._key, {}

        def _key(d, _o=orig, _m=memo):
            hit = _m.get(id(d))
            if hit is not None and hit[0] is d:
                return hit[1]
            v = _o(d); _m[id(d)] = (d, v); return v
        mod._key = _key
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", default="submission_v6"); ap.add_argument("--regimes", default="single,pooled"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--binary", action="store_true"); ap.add_argument("--max-eval", type=int, default=0); ap.add_argument("--memo", action="store_true")
    ap.add_argument("--only", default=""); ap.add_argument("--tag", default="")
    a = ap.parse_args()
    sub_dir = (ROOT / a.sub).resolve()
    tag = a.tag or f"{sub_dir.name}_s{a.seed}{'_bin' if a.binary else ''}{'_me%d' % a.max_eval if a.max_eval else ''}"
    benches = H.prepare(a.seed, 0.5)
    pairs, _, items = load_all(); items = items.drop_duplicates("item_id")
    benches_sim, subj_first = S.load_public()
    only = set(filter(None, a.only.split(",")))
    out = {}
    for reg in a.regimes.split(","):
        res = {}
        for held, B in benches.items():
            if only and held not in only:
                continue
            t0 = time.time()
            test = pairs[pairs["benchmark"] == held]; all_items = list(test["item_id"].unique())
            assert len(all_items) == B["nI"]
            it = items[items["benchmark"] == held].set_index("item_id")
            anon = "benchmark_%06d" % (int(hashlib.sha256(f"verify:{held}".encode()).hexdigest(), 16) % 10 ** 6)
            idicts = []
            for i_ in all_items:
                c = it.loc[i_, "content"]; f = it.loc[i_, "item_features"]
                idicts.append({"item_content": str(c) if pd.notna(c) else "", "item_features": f if isinstance(f, str) else None,
                               "interactors": None, "benchmark_id": anon})
            sd = B["sdicts"]; pools = B["pools"]; n = len(pools)
            fold = fold_copy(sub_dir, held, benches_sim, subj_first)
            ev_idx = [p["ev_i"][:a.max_eval] if a.max_eval else p["ev_i"] for p in pools]
            ev_y = [p["ev_y"][:a.max_eval] if a.max_eval else p["ev_y"] for p in pools]
            briers = np.zeros((n, 6))
            for j in range(n):
                briers[j, 0] = np.mean((0.5 - ev_y[j]) ** 2)
            groups = H.groups_for(n, reg, a.seed, held)
            for bi, k in enumerate(BUD[1:], start=1):
                M = S.load_module(fold)
                if a.memo:
                    memoise(M)
                for grp in groups:
                    labeled = []
                    for j in grp:
                        P = pools[j]
                        for t in range(min(k, len(P["acq_i"]))):
                            y = float(P["acq_y"][t]); y = int(y >= 0.5) if a.binary else y
                            labeled.append([[sd[j], idicts[int(P["acq_i"][t])]], y])
                    for j in grp:
                        pr = np.array([S.validate_score(M.predict([sd[j], idicts[int(i_)]], labeled)) for i_ in ev_idx[j]])
                        briers[j, bi] = np.mean((pr - ev_y[j]) ** 2)
            alc = briers @ np.array([W[k] for k in BUD])
            res[held] = dict(alc=float(alc.mean()), alc_pairs=alc.tolist(), B=briers.mean(0).tolist())
            print(f"{tag:28s} {reg:7s} {held:18s} n={n:3d} ALC {alc.mean():.4f}  " + " ".join(f"B{k}={v:.3f}" for k, v in zip(BUD, briers.mean(0))) +
                  f"  ({time.time() - t0:.0f}s)", flush=True)
        mean = float(np.mean([r["alc"] for r in res.values()]))
        pw = float(np.mean(np.concatenate([r["alc_pairs"] for r in res.values()])))
        print(f"{tag:28s} {reg:7s} MACRO ALC {mean:.4f}  pair-weighted {pw:.4f}", flush=True)
        out[reg] = res
    (OUT / f"e2e_{tag}.json").write_text(json.dumps(out))


if __name__ == "__main__":
    main()
