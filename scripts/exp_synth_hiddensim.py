"""exp_synth_hiddensim (tag 'synth'): hidden-calibrated synthetic evaluation of the model core.

Episodes replicate the three practice-board samples we were scored on (v3, v4, v5 feedback, see exp_synth_hidden_feedback.py): same
number of pairs, same benchmark sharing (2 systems on some benchmarks), same eval-pool sizes, and each pair's accuracy ybar = 0.5 +/- ECE@B0
with ONE random reflection sign per benchmark per episode (systems of a benchmark sit on the same side). Items: a benchmark pool with true
Rasch effects eps_i ~ N(0, tau_true^2) shared by its systems; pair level theta_s solved so the pair's mean P = ybar; binary labels; the
acquisition half is revealed in random order (first k at budget k). Item texts are random hashed-count vectors (no text signal), subjects
unknown (subject prior 0). Every candidate sees identical episodes (paired). The model is scripts/exp_pooled_core.py with a cfg dict (the
same math as submission_v6/model.py); baselines: Beta(a,a) posterior mean per pair, oracle constant ybar.
  .venv/bin/python scripts/exp_synth_hiddensim.py --episodes 60 --cfgs '[["v5", {"beta_prior": true}], ...]'
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "src"))
sys.dont_write_bytecode = True
import exp_pooled_core as core  # noqa: E402
from exp_synth_hidden_feedback import H, BENCH_ABS_LOGIT  # noqa: E402

BUDGETS = [0, 1, 3, 7, 15, 31]; W = {0: .1, 1: .2, 3: .2, 7: .2, 15: .2, 31: .1}


def solve_theta(eps, target):
    lo, hi = -15.0, 15.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if np.mean(1 / (1 + np.exp(-(mid + eps)))) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def make_episode(sample, rng, tau_true):
    rows = [h for h in H if h[0] == sample]
    benches = {}
    for h in rows:
        benches.setdefault(h[2], []).append(h)
    ep = []
    for b, hs in benches.items():
        sign = 1.0 if rng.random() < 0.5 else -1.0
        n_pool = max(2 * h[3] for h in hs) + int(rng.integers(0, 12))
        eps = rng.normal(0, tau_true, n_pool)
        C = (rng.random((n_pool, 4096)) < 0.012) * rng.integers(1, 4, (n_pool, 4096))  # random "texts"
        XA = core.tfidf_all(C.astype(np.float32))
        pairs = []
        for h in hs:
            ybar = float(np.clip(0.5 + sign * h[4], 0.004, 0.996))
            its = rng.choice(n_pool, size=min(n_pool, 2 * h[3]), replace=False)
            theta = solve_theta(eps[its], ybar)
            p = 1 / (1 + np.exp(-(theta + eps[its])))
            ylab = (rng.random(len(its)) < p).astype(float)
            n_acq = len(its) // 2
            order = rng.permutation(len(its))
            acq, ev = order[:n_acq], order[n_acq:]
            pairs.append(dict(acq_i=its[acq][:31], acq_y=ylab[acq][:31], ev_i=its[ev], ev_y=ylab[ev], ybar=ybar))
        ep.append(dict(nI=n_pool, XA=XA, C=C.astype(np.float32), pairs=pairs))
    return ep




def make_episode_k(n_sys, rng, tau_true, sd_within, n_items=120):
    """Hidden-structured regime with n_sys systems per benchmark: 7 benchmarks with the observed hidden mean |logit| levels (random sign),
    system marginal levels = benchmark level + N(0, sd_within^2), every system sees all n_items items (own 50/50 split)."""
    ep = []
    for mu_abs in BENCH_ABS_LOGIT:
        sign = 1.0 if rng.random() < 0.5 else -1.0
        eps = rng.normal(0, tau_true, n_items)
        C = (rng.random((n_items, 4096)) < 0.012) * rng.integers(1, 4, (n_items, 4096))
        XA = core.tfidf_all(C.astype(np.float32)); pairs = []
        for _ in range(n_sys):
            lvl = sign * mu_abs + rng.normal(0, sd_within)
            ybar = float(1 / (1 + np.exp(-lvl)))
            theta = solve_theta(eps, ybar)
            p = 1 / (1 + np.exp(-(theta + eps))); ylab = (rng.random(n_items) < p).astype(float)
            order = rng.permutation(n_items); acq, ev = order[:n_items // 2], order[n_items // 2:]
            pairs.append(dict(acq_i=acq[:31], acq_y=ylab[acq][:31], ev_i=ev, ev_y=ylab[ev], ybar=ybar))
        ep.append(dict(nI=n_items, XA=XA, C=C.astype(np.float32), pairs=pairs))
    return ep


def run_model(ep, cfg):
    """Returns list of per-pair ALC for one episode."""
    out = []
    for bd in ep:
        prior = core.ItemPrior(bd["XA"], None, alpha=cfg.get("alpha", 1.0))
        P = bd["pairs"]; nS = len(P)
        briers = np.zeros((nS, len(BUDGETS)))
        for j, p in enumerate(P):
            briers[j, 0] = np.mean((0.5 - p["ev_y"]) ** 2)
        for bi, k in enumerate(BUDGETS[1:], start=1):
            si = np.concatenate([[j] * min(k, len(p["acq_i"])) for j, p in enumerate(P)]).astype(int)
            ii = np.concatenate([p["acq_i"][:k] for p in P]).astype(int)
            y = np.concatenate([p["acq_y"][:k] for p in P]).astype(float)
            ts = np.concatenate([[j] * len(p["ev_i"]) for j, p in enumerate(P)]).astype(int)
            ti = np.concatenate([p["ev_i"] for p in P]).astype(int)
            fit = core.fit_snapshot(si, ii, y, nS, bd["nI"], np.zeros(nS), prior, cfg, None)
            pr = core.predict_targets(fit, ts, ti, cfg)
            off = 0
            for j, p in enumerate(P):
                m = len(p["ev_i"]); briers[j, bi] = np.mean((pr[off:off + m] - p["ev_y"]) ** 2); off += m
        out += [briers[j] for j in range(nS)]
    return out


def run_sub(ep, M):
    """Same episodes through the EXACT submission core (M.fit_core, batch form of M.predict_core); texts = the random count rows."""
    out = []
    for bd in ep:
        C = bd["C"]; P = bd["pairs"]; nS = len(P)
        briers = np.zeros((nS, len(BUDGETS)))
        for j, p in enumerate(P):
            briers[j, 0] = np.mean((0.5 - p["ev_y"]) ** 2)
        for bi, k in enumerate(BUDGETS[1:], start=1):
            si = np.concatenate([[j] * min(k, len(p["acq_i"])) for j, p in enumerate(P)]).astype(int)
            ii_g = np.concatenate([p["acq_i"][:k] for p in P]).astype(int)
            y = np.concatenate([p["acq_y"][:k] for p in P]).astype(float)
            sx, ix = {}, {}
            for s_, i_ in zip(si.tolist(), ii_g.tolist()):
                sx.setdefault(s_, len(sx)); ix.setdefault(i_, len(ix))
            items = list(ix)
            fit = M.fit_core(np.array([sx[s_] for s_ in si.tolist()]), np.array([ix[i_] for i_ in ii_g.tolist()]), y, len(sx), len(ix),
                             np.zeros(len(sx)), C[items], [{} for _ in items])
            for j, p in enumerate(P):
                e = np.zeros(len(p["ev_i"])); lab = np.array([i_ in ix for i_ in p["ev_i"].tolist()])
                if lab.any():
                    e[lab] = fit["eps"][[ix[i_] for i_ in p["ev_i"][lab].tolist()]]
                if (~lab).any() and fit["prior"] is not None:
                    e[~lab] += fit["prior"].predict(C[p["ev_i"][~lab]])
                d = fit["delta"][sx[j]] if j in sx else 0.0
                pr = np.clip(1 / (1 + np.exp(-(fit["beta"] + d + e))), M.P_MIN, M.P_MAX)
                briers[j, bi] = np.mean((pr - p["ev_y"]) ** 2)
        out += [briers[j] for j in range(nS)]
    return out


def run_beta(ep, a):
    out = []
    for bd in ep:
        for p in bd["pairs"]:
            b = []
            for k in BUDGETS:
                s = p["acq_y"][:k].sum(); n = min(k, len(p["acq_y"]))
                q = (a + s) / (2 * a + n)
                b.append(np.mean((q - p["ev_y"]) ** 2))
            out.append(np.array(b))
    return out


def run_oracle(ep):
    out = []
    for bd in ep:
        for p in bd["pairs"]:
            yb = p["ev_y"].mean()
            out.append(np.array([np.mean((0.5 - p["ev_y"]) ** 2)] + [np.mean((yb - p["ev_y"]) ** 2)] * 5))
    return out


_SUBS = {}
def load_sub(c):
    """c = {"sub": dir, "set": {CONST: value}} -> the submission module with overridden constants."""
    key = json.dumps(c, sort_keys=True)
    if key not in _SUBS:
        import importlib.util, os
        d = str((ROOT / c["sub"]).resolve()); sys.path.insert(0, d)
        try:
            spec = importlib.util.spec_from_file_location("_hsim_sub_%d" % len(_SUBS), os.path.join(d, "model.py"))
            m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
        finally:
            sys.path.remove(d)
        for k, v in c.get("set", {}).items():
            setattr(m, k, v)
        _SUBS[key] = m
    return _SUBS[key]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfgs", required=True); ap.add_argument("--episodes", type=int, default=40); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tau-true", type=float, default=1.5); ap.add_argument("--baselines", action="store_true")
    ap.add_argument("--regime", default="practice", help="practice (the three dev samples) | kN (hidden-structured, N systems per benchmark)")
    ap.add_argument("--sd-within", type=float, default=0.4)
    a = ap.parse_args()
    cfgs = json.loads(a.cfgs)
    eps_all = []
    if a.regime == "practice":
        for sample in ("v3", "v4", "v5"):
            for e in range(a.episodes):
                rng = np.random.default_rng([a.seed, {"v3": 3, "v4": 4, "v5": 5}[sample], e, int(a.tau_true * 100)])
                eps_all.append((sample, make_episode(sample, rng, a.tau_true)))
    else:
        n_sys = int(a.regime[1:])
        for e in range(a.episodes):
            rng = np.random.default_rng([a.seed, 1000 + n_sys, e, int(a.tau_true * 100), int(a.sd_within * 100)])
            eps_all.append((("v3", "v4", "v5")[e % 3], make_episode_k(n_sys, rng, a.tau_true, a.sd_within)))
    cands = [(n, ("cfg", c)) if not (isinstance(c, dict) and "sub" in c) else (n, ("sub", c)) for n, c in cfgs]
    if a.baselines:
        cands = [("oracle-constant", ("oracle", None)), ("Beta(1,1) Laplace", ("beta", 1.0)), ("Beta(0.7,0.7)", ("beta", 0.7))] + cands
    wv = np.array([W[k] for k in BUDGETS])
    for name, (kind, c) in cands:
        t0 = time.time(); per = {"v3": [], "v4": [], "v5": []}
        for sample, ep in eps_all:
            if kind == "sub":
                r = run_sub(ep, load_sub(c))
            else:
                r = run_model(ep, c) if kind == "cfg" else (run_beta(ep, c) if kind == "beta" else run_oracle(ep))
            per[sample] += r
        allb = np.array(per["v3"] + per["v4"] + per["v5"])
        alc_s = {s: float(np.mean(np.array(v) @ wv)) for s, v in per.items()}
        print(f"{name:30s} ALC {float(np.mean(allb @ wv)):.4f} (v3 {alc_s['v3']:.4f} v4 {alc_s['v4']:.4f} v5 {alc_s['v5']:.4f})  B: " +
              " ".join(f"{x:.4f}" for x in allb.mean(0)[1:]) + f"  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
