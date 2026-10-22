"""exp_v7_acq (tag 'v7'): coordinated acquisition for benchmarks shared by several systems, in the hidden-calibrated simulator.

The official interface passes `labeled` (every label revealed so far, all pairs) to labeling.py. A label that system B reveals on item X
helps system A only if X is in A's EVALUATION pool. Splits are independent per pair, so an item that A has already acquired is certainly
in A's acquisition pool and is worth nothing to A's scored predictions. Policy 'skipco': B skips a candidate that another system of the
same benchmark has already acquired (unless every remaining candidate is needed to fill the 31-label allowance); otherwise the
evaluator's random rule. Labels stay a random sample of the pair's own acquisition pool (the skip rule does not look at outcomes).
Policy 'skipseen' (needs state kept by labeling.py across calls): skip any item already OFFERED to another system of the benchmark
(acquired or skipped), i.e. any known member of another system's acquisition pool. 'skipco_greedy': skipco, but take every other
candidate (labels come from the front of the stream).

Episodes follow scripts/exp_synth_hiddensim.py (Rasch items N(0, tau_true^2) shared by the systems of a benchmark; each pair's accuracy
from our dev logs; random texts) with the coordinator's order: for each budget 1, 3, 7, 15, 31, pairs in a fixed random order, each
streaming its own candidates (fixed random order) until it holds `budget` labels. Predictions: the EXACT submission core (fit_core).
Regimes: practice (the three dev samples), boot (fresh samples in the board structure) and kN (N systems on each of the 7 hidden-like
benchmarks, within-benchmark sd 0.4).
  nice -n 10 env VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -B scripts/exp_v7_acq.py --regime practice --seeds 0,1,2
"""
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "src"))
sys.dont_write_bytecode = True
import exp_synth_hiddensim as hs  # noqa: E402
import exp_pooled_core as core  # noqa: E402
from exp_v7_hsplit import boot_rows  # noqa: E402

BUDGETS = [0, 1, 3, 7, 15, 31]; W = np.array([.1, .2, .2, .2, .2, .1]); MAXL = 31


def episode(rows, rng, tau_true):
    """rows: [(sample, subject, bench, n_eval, ece, ...)] -> list of benchmarks with pairs (full acquisition streams)."""
    benches = {}
    for h in rows:
        benches.setdefault(h[2], []).append(h)
    ep = []
    for b, hs_ in benches.items():
        sign = 1.0 if rng.random() < 0.5 else -1.0
        n_pool = max(2 * h[3] for h in hs_) + int(rng.integers(0, 12))
        eps = rng.normal(0, tau_true, n_pool)
        C = ((rng.random((n_pool, 4096)) < 0.012) * rng.integers(1, 4, (n_pool, 4096))).astype(np.float32)
        pairs = []
        for h in hs_:
            ybar = float(np.clip(0.5 + sign * h[4], 0.004, 0.996))
            its = rng.choice(n_pool, size=min(n_pool, 2 * h[3]), replace=False)
            theta = hs.solve_theta(eps[its], ybar)
            y = (rng.random(len(its)) < 1 / (1 + np.exp(-(theta + eps[its])))).astype(float)
            order = rng.permutation(len(its)); n_acq = len(its) // 2
            acq, ev = order[:n_acq], order[n_acq:]
            pairs.append(dict(stream=its[acq], stream_y=y[acq], ev_i=its[ev], ev_y=y[ev]))
        ep.append(dict(nI=n_pool, C=C, pairs=pairs))
    return ep


def episode_k(n_sys, rng, tau_true, sd_within, n_items=120):
    ep = []
    for mu_abs in hs.BENCH_ABS_LOGIT:
        sign = 1.0 if rng.random() < 0.5 else -1.0
        eps = rng.normal(0, tau_true, n_items)
        C = ((rng.random((n_items, 4096)) < 0.012) * rng.integers(1, 4, (n_items, 4096))).astype(np.float32)
        pairs = []
        for _ in range(n_sys):
            ybar = float(1 / (1 + np.exp(-(sign * mu_abs + rng.normal(0, sd_within)))))
            theta = hs.solve_theta(eps, ybar)
            y = (rng.random(n_items) < 1 / (1 + np.exp(-(theta + eps)))).astype(float)
            order = rng.permutation(n_items); acq, ev = order[:n_items // 2], order[n_items // 2:]
            pairs.append(dict(stream=acq, stream_y=y[acq], ev_i=ev, ev_y=y[ev]))
        ep.append(dict(nI=n_items, C=C, pairs=pairs))
    return ep


def u01(*key):
    return int.from_bytes(hashlib.sha256(json.dumps(key).encode()).digest()[:8], "big") / 2 ** 64


def acquire(ep, policy, rng):
    """Coordinator: returns per benchmark, per pair, the list of (item, y) acquired, with the checkpoint lengths per budget."""
    flat = [(bi, pj) for bi, bd in enumerate(ep) for pj in range(len(bd["pairs"]))]
    order = [flat[j] for j in rng.permutation(len(flat))]
    got = {f: [] for f in flat}; cur = {f: 0 for f in flat}; snaps = {}
    seen = {f: set() for f in flat}   # every candidate offered to a pair so far (acquired or skipped): its acquisition pool
    salt = int(rng.integers(1 << 30))
    for k in BUDGETS[1:]:
        for f in order:
            bi, pj = f; p = ep[bi]["pairs"][pj]; n = len(p["stream"])
            while len(got[f]) < k and cur[f] < n:
                c = cur[f]; cur[f] += 1
                lab_rem, it_rem = MAXL - len(got[f]), n - c
                must = it_rem <= lab_rem
                item = int(p["stream"][c])
                if policy == "random":
                    q = must or u01(salt, bi, pj, c) < min(1.0, lab_rem / it_rem)
                elif policy in ("skipco", "skipco_greedy", "skipseen"):
                    if policy == "skipseen":
                        others = {i for (b2, p2), st in seen.items() if b2 == bi and p2 != pj for i in st}
                    else:
                        others = {i for (b2, p2), lst in got.items() if b2 == bi and p2 != pj for i, _ in lst}
                    if must:
                        q = True
                    elif item in others:
                        q = False
                    else:
                        q = True if policy == "skipco_greedy" else u01(salt, bi, pj, c) < min(1.0, lab_rem / it_rem)
                else:
                    raise ValueError(policy)
                seen[f].add(item)
                if q:
                    got[f].append((item, float(p["stream_y"][c])))
            snaps[(k, f)] = len(got[f])
    return got, snaps


def score(ep, got, snaps, M):
    out = []
    for bi, bd in enumerate(ep):
        P = bd["pairs"]; C = bd["C"]
        br = np.zeros((len(P), len(BUDGETS)))
        for j, p in enumerate(P):
            br[j, 0] = np.mean((0.5 - p["ev_y"]) ** 2)
        for kb, k in enumerate(BUDGETS[1:], start=1):
            si, ii, y = [], [], []
            for j in range(len(P)):
                lst = got[(bi, j)][:snaps[(k, (bi, j))]]
                si += [j] * len(lst); ii += [i for i, _ in lst]; y += [v for _, v in lst]
            sx, ix = {}, {}
            for s_, i_ in zip(si, ii):
                sx.setdefault(s_, len(sx)); ix.setdefault(i_, len(ix))
            items = list(ix)
            fit = M.fit_core(np.array([sx[s_] for s_ in si], int), np.array([ix[i_] for i_ in ii], int), np.array(y, float),
                             len(sx), len(ix), np.zeros(len(sx)), C[items], [{} for _ in items])
            for j, p in enumerate(P):
                e = np.zeros(len(p["ev_i"])); lab = np.array([i_ in ix for i_ in p["ev_i"].tolist()])
                if lab.any():
                    e[lab] = fit["eps"][[ix[i_] for i_ in p["ev_i"][lab].tolist()]]
                if (~lab).any() and fit["prior"] is not None:
                    e[~lab] += fit["prior"].predict(C[p["ev_i"][~lab]])
                d = fit["delta"][sx[j]] if j in sx else 0.0
                pr = np.clip(1 / (1 + np.exp(-(fit["beta"] + d + e))), M.P_MIN, M.P_MAX)
                br[j, kb] = np.mean((pr - p["ev_y"]) ** 2)
        out += [(len(P) > 1, br[j]) for j in range(len(P))]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", default="practice"); ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--seeds", default="0,1,2"); ap.add_argument("--tau-true", type=float, default=1.5)
    ap.add_argument("--sd-within", type=float, default=0.4); ap.add_argument("--sub", default="submission_v6")
    ap.add_argument("--policies", default="random,skipco,skipco_greedy")
    a = ap.parse_args()
    M = hs.load_sub({"sub": a.sub}); pols = a.policies.split(",")
    res = {p: {"lone": [], "shared": []} for p in pols}; per_seed = {p: [] for p in pols}
    for seed in map(int, a.seeds.split(",")):
        acc = {p: [] for p in pols}
        for e in range(a.episodes):
            for sample in (("v3", "v4", "v5") if a.regime == "practice" else ("x",)):
                rng = np.random.default_rng([seed, 77, e, {"v3": 3, "v4": 4, "v5": 5, "x": 0}[sample], int(a.tau_true * 100)])
                if a.regime == "practice":
                    ep = episode([h for h in hs.H if h[0] == sample], rng, a.tau_true)
                elif a.regime == "boot":
                    ep = episode(boot_rows(rng, "boot"), rng, a.tau_true)
                else:
                    ep = episode_k(int(a.regime[1:]), rng, a.tau_true, a.sd_within)
                st = rng.bit_generator.state
                for pol in pols:
                    r2 = np.random.default_rng(); r2.bit_generator.state = st   # identical coordinator order for every policy
                    got, snaps = acquire(ep, pol, r2)
                    for sh, b in score(ep, got, snaps, M):
                        res[pol]["shared" if sh else "lone"].append(b); acc[pol].append(b @ W)
        for pol in pols:
            per_seed[pol].append(float(np.mean(acc[pol])))
    print(f"regime {a.regime}, tau_true {a.tau_true}, sd_within {a.sd_within}, episodes {a.episodes}, seeds {a.seeds}, sub {a.sub}")
    for pol in pols:
        allb = np.array(res[pol]["lone"] + res[pol]["shared"])
        line = f"  {pol:14s} ALC {float(np.mean(allb @ W)):.4f}  per seed " + " ".join(f"{v:.4f}" for v in per_seed[pol])
        for k in ("lone", "shared"):
            if res[pol][k]:
                B = np.array(res[pol][k]); line += f"  | {k} {len(B)} ALC {float(np.mean(B @ W)):.4f} B " + " ".join(f"{x:.4f}" for x in B.mean(0)[1:])
        print(line, flush=True)


if __name__ == "__main__":
    main()
