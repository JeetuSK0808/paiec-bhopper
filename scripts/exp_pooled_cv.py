"""exp_pooled_cv (tag 'pooled'): fast leave-one-benchmark-out harness, same splits as scripts/item_prior_cv.py (same RNG stream per
seed), vectorised model from exp_pooled_core. Regimes: pooled (all systems of the benchmark share labels), single (each system alone),
mixN (systems randomly partitioned into groups of N that share labels). Acquisition: random (= harness) or stream-simulated policies.

  nice -n 10 env VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python scripts/exp_pooled_cv.py \
      --regimes pooled,single --cfg '{}'            # v5 reproduction (tau 0.8, tau_i 1.5, sp 0.5, alpha 1, rounds 3)
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from paie.data import load_all  # noqa: E402
from paie.textprior import hash_counts  # noqa: E402
from paie.subjprior import SubjectPrior  # noqa: E402
import exp_pooled_core as core  # noqa: E402

BUDGETS = [0, 1, 3, 7, 15, 31]; WEIGHTS = {0: 0.1, 1: 0.2, 3: 0.2, 7: 0.2, 15: 0.2, 31: 0.1}
SKEYS = ("normalized_name", "provider", "release_date", "reasoning_effort", "subject_features_extra")
SKEYS_ALL = SKEYS + ("harness", "harness_version", "access_date")


def prepare(seed: int = 0, sp: float = 0.5, cache={}):
    key = (seed, sp)
    if key in cache:
        return cache[key]
    pairs, subjects, items = load_all(); items = items.drop_duplicates("item_id")
    sd = subjects.drop_duplicates("subject_id").set_index("subject_id")

    def sdict(sid, keys=SKEYS):
        r = sd.loc[sid]; return {k: ("" if pd.isna(r.get(k)) else str(r.get(k))) for k in keys}

    rng = np.random.default_rng(seed); benches = {}
    for held in sorted(pairs["benchmark"].unique()):
        test = pairs[pairs["benchmark"] == held]
        pools = []
        for sid, g in test.groupby("subject_id"):
            if len(g) < 80:
                continue
            g = g.sample(frac=1.0, random_state=int(rng.integers(1e9))); half = len(g) // 2
            pools.append((sid, g.iloc[:half], g.iloc[half:]))
        if not pools:
            continue
        all_items = list(test["item_id"].unique()); ix = {i_: j for j, i_ in enumerate(all_items)}
        it = items[items["benchmark"] == held].set_index("item_id")
        C_all = np.stack([hash_counts(str(it.loc[i_, "content"]) if pd.notna(it.loc[i_, "content"]) else "", dim=4096) for i_ in all_items])
        XA = core.tfidf_all(C_all)
        feats = [it.loc[i_, "item_features"] for i_ in all_items]
        GA = core.group_matrix(feats); GR = core.group_matrix(feats, normalise=False)
        # duplicate clusters: identical (item_content, item_features) -> canonical index (first occurrence)
        keys = [(str(it.loc[i_, "content"]) if pd.notna(it.loc[i_, "content"]) else "", str(f)) for i_, f in zip(all_items, feats)]
        first = {}; canon = np.array([first.setdefault(k_, j) for j, k_ in enumerate(keys)])
        m_subj = np.zeros(len(pools))
        if sp > 0:
            trp = pairs[pairs["benchmark"] != held].groupby(["benchmark", "subject_id"]).y.agg(["mean", "size"]).reset_index()
            spm = SubjectPrior(known_weight=0.6, meta_weight=0.5).fit([(sdict(r.subject_id), r.benchmark, r["mean"], r["size"]) for _, r in trp.iterrows()])
            m_subj = np.array([sp * spm.offset(sdict(sid)) for sid, _, _ in pools])
        P = []
        for sid, acq, ev in pools:
            P.append(dict(sid=sid, acq_i=np.array([ix[i_] for i_ in acq["item_id"].values]), acq_y=acq["y"].values.astype(float),
                          ev_i=np.array([ix[i_] for i_ in ev["item_id"].values]), ev_y=ev["y"].values.astype(float)))
        sdl = [sdict(sid, SKEYS_ALL) for sid, _, _ in pools]
        benches[held] = dict(pools=P, nI=len(all_items), XA=XA, GA=GA, GR=GR, canon=canon, m_subj=m_subj, sdicts=sdl, K=core.similarity_matrix(sdl))
    cache[key] = benches
    return benches


def groups_for(n_pairs: int, regime: str, seed: int, held: str):
    if regime == "pooled":
        return [list(range(n_pairs))]
    if regime == "single":
        return [[j] for j in range(n_pairs)]
    if regime.startswith("mix"):
        N = int(regime[3:]); r = np.random.default_rng([seed, 12345, sum(map(ord, held))]); perm = r.permutation(n_pairs)
        return [sorted(perm[a:a + N].tolist()) for a in range(0, n_pairs, N)]
    raise ValueError(regime)


def acquisition_orders(B, grp, policy: str, seed: int, pair_order: str = "roundrobin"):
    """Per pair in grp: array of acquired acquisition-pool positions in acquisition order (<= 31). Streams each pair's pool in its
    (random) order; a policy may skip an item permanently unless items_remaining <= labels_remaining (as the official hook)."""
    pools = B["pools"]
    if policy == "random":
        return {j: np.arange(min(31, len(pools[j]["acq_i"]))) for j in grp}
    cnt = np.zeros(B["nI"], int); got = {j: [] for j in grp}; cur = {j: 0 for j in grp}

    def step(j, target):
        pool = pools[j]["acq_i"]
        while len(got[j]) < target and cur[j] < len(pool):
            item = pool[cur[j]]; items_remaining = len(pool) - cur[j]; labels_remaining = 31 - len(got[j])
            must = items_remaining <= labels_remaining
            if policy == "cover":        # skip items another system already revealed
                take = must or cnt[item] == 0
            elif policy == "cover2":     # skip items already revealed >= 2 times
                take = must or cnt[item] < 2
            elif policy == "anchor":     # after the first label, prefer items other systems already revealed
                take = must or cnt[item] > 0 or len(got[j]) == 0
            else:
                raise ValueError(policy)
            cur[j] += 1
            if take:
                got[j].append(cur[j] - 1); cnt[item] += 1

    for b in [1, 3, 7, 15, 31]:
        if pair_order == "sequential":
            for j in grp:
                step(j, b)
        else:  # round robin, one label per pair per turn
            while True:
                moved = False
                for j in grp:
                    if len(got[j]) < b and cur[j] < len(pools[j]["acq_i"]):
                        step(j, len(got[j]) + 1); moved = True
                if not moved:
                    break
    return {j: np.array(got[j], int) for j in grp}


def evaluate(benches, cfg: dict, regime: str = "pooled", seed: int = 0, acq: str = "random", only=None, verbose=True, pair_order="roundrobin"):
    res = {}
    for held, B in benches.items():
        if only and held not in only:
            continue
        t0 = time.time(); pools = B["pools"]; n = len(pools)
        prior = core.ItemPrior(B["XA"], B["GA"], alpha=cfg.get("alpha", 1.0), c_g=cfg.get("c_g", 0.0))
        briers = {k: np.zeros(n) for k in BUDGETS}
        for j in range(n):
            briers[0][j] = np.mean((0.5 - pools[j]["ev_y"]) ** 2)
        for grp in groups_for(n, regime, seed, held):
            orders = acquisition_orders(B, grp, acq, seed, pair_order)
            loc = {j: a for a, j in enumerate(grp)}; nS = len(grp)
            m_s = B["m_subj"][grp] * (cfg.get("sp", 0.5) / 0.5)
            Kg = B["K"][np.ix_(grp, grp)]
            for k in BUDGETS[1:]:
                si = np.concatenate([[loc[j]] * min(k, len(orders[j])) for j in grp]).astype(int)
                ii = np.concatenate([pools[j]["acq_i"][orders[j][:k]] for j in grp]).astype(int)
                y = np.concatenate([pools[j]["acq_y"][orders[j][:k]] for j in grp]).astype(float)
                ts = np.concatenate([[loc[j]] * len(pools[j]["ev_i"]) for j in grp]).astype(int)
                ti = np.concatenate([pools[j]["ev_i"] for j in grp]).astype(int)
                if cfg.get("dup", 0):
                    ii = B["canon"][ii]; ti = B["canon"][ti]
                fit = core.fit_snapshot(si, ii, y, nS, B["nI"], m_s, prior, cfg, B["GR"])
                pr = core.predict_targets(fit, ts, ti, cfg, Kg)
                off = 0
                for j in grp:
                    m = len(pools[j]["ev_i"]); briers[k][j] = np.mean((pr[off:off + m] - pools[j]["ev_y"]) ** 2); off += m
        Bk = {k: float(np.mean(v)) for k, v in briers.items()}
        alc_pairs = sum(WEIGHTS[k] * briers[k] for k in BUDGETS)
        alc = float(np.mean(alc_pairs))
        res[held] = dict(alc=alc, B=Bk, n=n, alc_pairs=alc_pairs)
        if verbose:
            print(f"  {regime:7s} {held:18s} n={n:3d} ALC {alc:.4f}  " + " ".join(f"B{k}={Bk[k]:.3f}" for k in BUDGETS) + f"  ({time.time()-t0:.0f}s)", flush=True)
    mean = float(np.mean([r["alc"] for r in res.values()]))
    pairw = float(np.mean(np.concatenate([r["alc_pairs"] for r in res.values()])))
    return mean, pairw, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="{}"); ap.add_argument("--regimes", default="pooled,single"); ap.add_argument("--seeds", default="0")
    ap.add_argument("--acq", default="random"); ap.add_argument("--only", default=""); ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--pair-order", default="roundrobin"); ap.add_argument("--tag", default="")
    a = ap.parse_args()
    cfg = json.loads(a.cfg); only = [x for x in a.only.split(",") if x] or None
    out = {}
    for seed in [int(s) for s in a.seeds.split(",")]:
        benches = prepare(seed, 0.5)
        for regime in a.regimes.split(","):
            t0 = time.time()
            mean, pairw, res = evaluate(benches, cfg, regime, seed, a.acq, only, not a.quiet, a.pair_order)
            out[(seed, regime)] = (mean, pairw, {h: r["alc"] for h, r in res.items()})
            print(f"RESULT tag={a.tag} seed={seed} regime={regime} acq={a.acq} mean_ALC={mean:.4f} pairweighted={pairw:.4f} per-bench=" +
                  ",".join(f"{h}:{r['alc']:.4f}" for h, r in res.items()) + f" cfg={json.dumps(cfg, sort_keys=True)} ({time.time()-t0:.0f}s)", flush=True)
    if len(out) > 1:
        for regime in a.regimes.split(","):
            v = [out[(s, regime)][0] for s in [int(s) for s in a.seeds.split(",")]]
            print(f"SUMMARY tag={a.tag} regime={regime} mean over seeds {np.mean(v):.4f} (+-{np.std(v):.4f}) cfg={json.dumps(cfg, sort_keys=True)}")


if __name__ == "__main__":
    main()
