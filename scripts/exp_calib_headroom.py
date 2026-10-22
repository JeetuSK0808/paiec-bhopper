"""exp_calib: headroom analysis on a dump (single or pooled): model vs subject oracle vs subject-only Bayes posterior means with
(a) a cheating prior (KDE of the held-out benchmark's own pair accuracies) and (b) fair priors (fit on the other benchmarks' pairs)."""
import sys, numpy as np
sys.path.insert(0, "scripts")
from exp_calib_lib import load, fmt, sigmoid, logit, eb_posterior, TH, BUDGETS, WEIGHTS

D = load(sys.argv[1])
print(f"rows {len(D['y'])}, pairs {len(D['pair_names'])}")
p_model = sigmoid(D["z"]); fmt_ = lambda p, l: print(fmt(D, p, l)[1])
fmt_(p_model, "model (sigmoid z)")
# subject oracle: the pair's true evaluation-pool accuracy at every budget
npair = len(D["pair_names"]); acc = np.zeros(npair)
m0 = D["k"] == 0
acc = np.bincount(D["pair"][m0], D["y"][m0], npair) / np.maximum(np.bincount(D["pair"][m0], None, npair), 1)
fmt_(acc[D["pair"]], "subject oracle (eval acc, all budgets)")
po = np.where(D["k"] == 0, 0.5, acc[D["pair"]]); fmt_(po, "subject oracle, B0 = 0.5")
# per-pair (k_own, sum_own) at each budget
pb = D["pair"]; kk = D["k_own"]; ss = D["sum_own"]; bench_of_pair = np.zeros(npair, int); bench_of_pair[D["pair"]] = D["bench"]
def bayes_pred(prior_for_bench, b0_half=True):
    out = np.zeros(len(D["y"]))
    for k in BUDGETS:
        mk = D["k"] == k
        prs = np.unique(pb[mk])
        # one representative row per pair
        first = {}
        idx = np.nonzero(mk)[0]
        _, fi = np.unique(pb[idx], return_index=True); rep = idx[fi]
        lp = np.stack([prior_for_bench(bench_of_pair[p]) for p in pb[rep]])
        w = eb_posterior(kk[rep], ss[rep], lp)
        pm = (w * sigmoid(TH)[None, :]).sum(1)
        val = dict(zip(pb[rep], pm))
        out[mk] = np.array([val[p] for p in pb[mk]])
        if k == 0 and b0_half:
            out[mk] = 0.5
    return out
names = D["bench_names"]
def kde_logpdf(vals, bw):
    d = np.exp(-0.5 * ((TH[:, None] - vals[None, :]) / bw) ** 2).sum(1); d /= d.sum(); return np.log(d + 1e-300)
cheat = {bi: kde_logpdf(logit(acc[bench_of_pair == bi], 0.01), 0.3) for bi in np.unique(D["bench"])}
fmt_(bayes_pred(lambda b: cheat[b], b0_half=True), "Bayes, CHEATING prior (own bench KDE), B0=.5")
fmt_(bayes_pred(lambda b: cheat[b], b0_half=False), "Bayes, CHEATING prior, B0=prior mean")
for s in (0.8, 1.0, 1.2, 1.5, 2.0):
    lp = -0.5 * (TH / s) ** 2
    fmt_(bayes_pred(lambda b: lp), f"Bayes, N(0,{s}^2) prior")
# fair: logit-normal fit to other benchmarks' pairs, benchmarks weighted equally
for b0h in (True,):
    def fair(b):
        mus, vs = [], []
        for bb in np.unique(D["bench"]):
            if bb == b: continue
            v = logit(acc[bench_of_pair == bb], 0.01); mus.append(v.mean()); vs.append(v.var() if len(v) > 1 else 0.0)
        mu = np.mean(mus); var = np.mean(vs) + np.var(mus)
        return -0.5 * (TH - mu) ** 2 / var
    fmt_(bayes_pred(fair, b0_half=b0h), f"Bayes, fair logit-normal (bench-equal), B0=.5")
    def fair0(b):
        mus, vs = [], []
        for bb in np.unique(D["bench"]):
            if bb == b: continue
            v = logit(acc[bench_of_pair == bb], 0.01); mus.append(v.mean()); vs.append(v.var() if len(v) > 1 else 0.0)
        var = np.mean(vs) + np.mean(np.square(mus))
        return -0.5 * TH ** 2 / var
    fmt_(bayes_pred(fair0, b0_half=b0h), f"Bayes, fair N(0, s_fit^2) (bench-equal), B0=.5")
# per-budget view
for lab, p in (("model", p_model), ("oracle-prior Bayes", bayes_pred(lambda b: cheat[b])), ("subject oracle", po)):
    m, s, alcs, pbb = fmt(D, p, lab)
    print(f"  {lab:20s} " + " ".join(f"B{k}={np.mean([pbb[bi][k] for bi in pbb]):.4f}" for k in BUDGETS))
