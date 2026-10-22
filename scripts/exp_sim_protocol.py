#!/usr/bin/env python
"""Faithful local simulator of the PAIEC streaming evaluation ("streaming_alc_v1") for ANY submission directory.

Drives <sub>/model.py predict(input, labeled) -> float (and optional <sub>/labeling.py acquisition_function) through the
official interface, reconstructed from upstream/paiec_baseline/tools/streaming_ingestion.py, check_submission_zip.py and the
competition rules:
  * input = [subject, item]. subject: exactly the 8 public keys (normalized_name, provider, release_date, access_date, harness,
    harness_version, reasoning_effort, subject_features_extra), strings, "" when missing. item: item_content, item_features,
    interactors ("" -- None in the public data) and an anonymous stable benchmark_id ("benchmark_NNNNNN").
  * pairs = (subject, benchmark) with >= 80 distinct items; per pair a random 50/50 item split into acquisition / evaluation pools.
  * acquisition: a fixed random stream order per pair. With a labeling.py hook the call shape is upstream's AcquisitionHook
    (predictions for the hook come from ONE coordinator-side model instance per episode); without one, the evaluator's own rule
    sha256(json([context, input])) -> uniform < labels_remaining / items_remaining (selection sampling up to max_labels = 31).
    context = {subject_id, benchmark_id, labels_remaining, labels_acquired, max_labels, items_remaining}.
  * checkpoints at 0, 1, 3, 7, 15, 31 labels per pair; `labeled` = [[subject, item], 0/1] for ALL pairs of the episode, in
    acquisition order. "Evaluation workers are recreated with frozen evidence at each checkpoint": the model module is re-imported
    (fresh module state) per checkpoint and per worker (--workers, round-robin), identical inputs are predicted once (group_inputs),
    every prediction is validated (finite, in [0, 1]). --persist keeps one module per episode instead (to measure state effects).
  * score: per-pair Brier over its evaluation targets at each checkpoint, ALC = .1 B0 + .2 (B1+B3+B7+B15) + .1 B31, pairs equal.
Leave-one-benchmark-out: only pairs of the held-out benchmark are scored. Regimes:
  practice : ~practice board -- the held-out benchmark contributes 1 system (p = 5/7) or 2 (p = 2/7; 9 pairs over 7 benchmarks),
             88-152 items per pair (44-76 evaluation items), plus one unscored "filler" pair from every other public benchmark so that
             `labeled` and the worker's module state look like a multi-benchmark submission.
  kN       : N systems of the held-out benchmark (all reveal labels into the shared `labeled`).
Labels: every response is binarised at 0.5 (as src/paie/data.py). --unit trial (default): one target per (subject, item) whose label
is ONE random recorded response; majority: round(mean of responses); response: every response is its own target / stream candidate.
Priors (--prior): shipped (submission as is; its subject_prior.json was fit on all 6 public benchmarks -> leaks in LOBO), fold (copy of
the submission with subject_prior.json rebuilt WITHOUT the held-out benchmark, same builder as scripts/item_prior_cv.py), none.

  nice -n 10 env VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python scripts/exp_sim_protocol.py \
      --sub submission_v5 --regimes practice,k1,k2,k4,k8 --episodes 20 --prior fold --tag v5_fold
  ... --leak-report          (builder check against the shipped prior + per-fold leak diagnostics, no simulation)
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True  # never write .pyc files into submission folders
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "hf"
EXP = ROOT / "src" / "paie_exp" / "sim"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "upstream" / "paiec_baseline" / "tools"))
from streaming_ingestion import AcquisitionHook  # noqa: E402  (exact call-shape logic of the official evaluator)

BUDGETS = (0, 1, 3, 7, 15, 31)
WEIGHTS = {0: 0.1, 1: 0.2, 3: 0.2, 7: 0.2, 15: 0.2, 31: 0.1}
MAX_LABELS, MIN_ITEMS = 31, 80
PRACTICE_ITEMS = (88, 152)
SUBJ_KEYS = ("normalized_name", "provider", "release_date", "access_date", "harness", "harness_version", "reasoning_effort",
             "subject_features_extra")
PRIOR_KEYS = ("normalized_name", "provider", "release_date", "reasoning_effort", "subject_features_extra")


def _s(v) -> str:
    if v is None:
        return ""
    try:
        if pd.isna(v):
            return ""
    except (TypeError, ValueError):
        pass
    return str(v)


# ----------------------------------------------------------------------------------------------------------------- data
class Bench:
    def __init__(self, idx, name, anon, subj, items, ikey, resp):
        self.idx, self.name, self.anon, self.subj, self.items, self.ikey, self.resp = idx, name, anon, subj, items, ikey, resp
        self.elig = sorted(s for s, d in resp.items() if sum(1 for i in d if i in items) >= MIN_ITEMS)


def load_public(salt: str = "paiec-sim"):
    benches, subj_first = [], {}
    dirs = sorted(p for p in DATA.iterdir() if p.is_dir() and not p.name.startswith("."))
    for d in dirs:
        rfs = sorted(d.glob("response*.parquet"))
        if not rfs:
            continue
        r = pd.read_parquet(rfs[0], columns=["subject_id", "item_id", "response"]).dropna(subset=["response"])
        r = r.sort_values(["subject_id", "item_id"], kind="stable")
        sids, iids = r["subject_id"].to_numpy(), r["item_id"].to_numpy()
        ys = (r["response"].to_numpy(dtype=float) >= 0.5).astype(np.int8)
        chg = np.r_[True, (sids[1:] != sids[:-1]) | (iids[1:] != iids[:-1])]
        st = np.flatnonzero(chg); en = np.r_[st[1:], len(ys)]
        resp: dict = {}
        for a, b in zip(st.tolist(), en.tolist()):
            resp.setdefault(sids[a], {})[iids[a]] = ys[a:b]
        s = pd.read_parquet(d / "subjects.parquet").drop_duplicates("subject_id")
        subj = {}
        for rec in s.to_dict("records"):
            subj[rec["subject_id"]] = {k: _s(rec.get(k)) for k in SUBJ_KEYS}
            subj_first.setdefault(rec["subject_id"], rec)
        anon = "benchmark_%06d" % (int(hashlib.sha256(f"{salt}:{d.name}".encode()).hexdigest(), 16) % 10 ** 6)
        it = pd.read_parquet(d / "items.parquet", columns=["item_id", "content", "item_features"]).drop_duplicates("item_id")
        items, ikey, canon = {}, {}, {}
        for rec in it.to_dict("records"):
            dct = {"item_content": _s(rec.get("content")), "item_features": _s(rec.get("item_features")), "interactors": "",
                   "benchmark_id": anon}
            items[rec["item_id"]] = dct
            h = hashlib.sha1(json.dumps(dct, sort_keys=True).encode()).hexdigest()
            ikey[rec["item_id"]] = canon.setdefault(h, len(canon))  # identical visible items share one canonical input
        benches.append(Bench(len(benches), d.name, anon, subj, items, ikey, resp))
    return benches, subj_first


# ------------------------------------------------------------------------------------------------------- prior builder
def build_prior_json(benches, subj_first, exclude=(), kw=0.6, mw=0.5) -> dict:
    """Same construction as scripts/item_prior_cv.py --sp: per (benchmark, subject) mean of per-item mean binarised response over ALL
    pairs (no 80-item filter), subject metadata from the first subjects table (sorted benchmark order) containing the id."""
    from paie.subjprior import SubjectPrior
    rows = []
    for bd in benches:
        if bd.name in exclude:
            continue
        for sid in sorted(bd.resp):
            d = bd.resp[sid]
            ys = [float(v.mean()) for v in d.values()]
            rec = subj_first.get(sid, {})
            rows.append(({k: _s(rec.get(k)) for k in PRIOR_KEYS}, bd.name, float(np.mean(ys)), len(ys)))
    sp = SubjectPrior(known_weight=kw, meta_weight=mw).fit(rows)
    return {"table": {k: [float(v[0]), int(v[1])] for k, v in sp.table.items()}, "mu": sp.mu.tolist(), "sd": sp.sd.tolist(),
            "w": sp.w.tolist(), "kw": sp.kw, "mw": sp.mw}


def neutral_prior_json(n_feat: int, kw=0.6, mw=0.5) -> dict:
    return {"table": {}, "mu": [0.0] * n_feat, "sd": [1.0] * n_feat, "w": [0.0] * n_feat, "kw": kw, "mw": mw}


def prepare_submission(sub_dir: Path, prior: str, held: str, benches, subj_first) -> Path:
    """shipped -> the directory itself; fold/none -> a copy under src/paie_exp/sim/folds/ with a replaced subject_prior.json."""
    if prior == "shipped":
        return sub_dir
    sp_path = sub_dir / "subject_prior.json"
    if not sp_path.exists():
        return sub_dir  # nothing to rebuild: the submission carries no subject prior
    shipped = json.loads(sp_path.read_text())
    dst = EXP / "folds" / f"{sub_dir.name}__{prior}__{held}"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(sub_dir, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    if prior == "fold":
        j = build_prior_json(benches, subj_first, exclude={held}, kw=shipped.get("kw", 0.6), mw=shipped.get("mw", 0.5))
    elif prior == "none":
        j = neutral_prior_json(len(shipped["w"]), shipped.get("kw", 0.6), shipped.get("mw", 0.5))
    else:
        raise ValueError(prior)
    (dst / "subject_prior.json").write_text(json.dumps(j))
    return dst


# --------------------------------------------------------------------------------------------------- module isolation
_PURGE: set = set()
_COUNTER = [0]


def load_module(sub_dir: Path, fname: str = "model.py"):
    """Fresh import of <sub_dir>/<fname> (a new worker process): sibling modules imported by earlier loads are purged first, the
    submission dir is on sys.path only while the entry point executes (as in check_submission_zip._load_module)."""
    for name in list(_PURGE):
        sys.modules.pop(name, None)
    _PURGE.clear()
    d = str(Path(sub_dir).resolve())
    before = set(sys.modules)
    _COUNTER[0] += 1
    sys.path.insert(0, d)
    try:
        spec = importlib.util.spec_from_file_location(f"_paiec_sim_{Path(fname).stem}_{_COUNTER[0]}", os.path.join(d, fname))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        sys.path.remove(d)
    for name in set(sys.modules) - before:
        f = getattr(sys.modules[name], "__file__", None) or ""
        if os.path.realpath(f).startswith(d + os.sep):
            _PURGE.add(name)
    return mod


def validate_score(v) -> float:
    if isinstance(v, bool):
        raise ValueError(f"predict returned a bool: {v!r}")
    x = float(v)
    if not math.isfinite(x) or x < 0.0 or x > 1.0:
        raise ValueError(f"predict returned an invalid probability: {v!r}")
    return x


_HC_MEMO: dict = {}


def accelerate(mod):
    """--fast: memoise two PURE helpers of the v3/v5 model family (outputs are bit-identical; only runtime changes):
    _key(dict) -> json string (memo by object identity, the object is kept alive) and hash_counts(text) (memo by content)."""
    if callable(getattr(mod, "_key", None)):
        orig, memo = mod._key, {}

        def _key(d, _orig=orig, _memo=memo):
            hit = _memo.get(id(d))
            if hit is not None and hit[0] is d:
                return hit[1]
            v = _orig(d)
            _memo[id(d)] = (d, v)
            return v
        mod._key = _key
    if callable(getattr(mod, "hash_counts", None)):
        orig_hc = mod.hash_counts

        def hash_counts(*args, _orig=orig_hc, **kw):
            k = (args, tuple(sorted(kw.items())))
            if k not in _HC_MEMO:
                _HC_MEMO[k] = _orig(*args, **kw)
            return _HC_MEMO[k]
        mod.hash_counts = hash_counts
    return mod


def random_query(context: dict, current_input: list) -> bool:
    """The evaluator's default policy (streaming_ingestion.run_streaming, hook.mode == 'random')."""
    key = json.dumps([context, current_input], sort_keys=True, separators=(",", ":"))
    uniform = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big") / 2 ** 64
    return uniform < min(1, context["labels_remaining"] / context["items_remaining"])


# ------------------------------------------------------------------------------------------------------------ episodes
def _label_rows(bd: Bench, sid, item_ids, unit, rng):
    out = []
    for iid in item_ids:
        r = bd.resp[sid][iid]
        if unit == "trial":
            out.append((iid, int(r[int(rng.integers(len(r)))])))
        elif unit == "majority":
            out.append((iid, int(r.mean() >= 0.5)))
        elif unit == "response":
            out.extend((iid, int(y)) for y in r)
        else:
            raise ValueError(unit)
    return out


def _choose_items(bd: Bench, sids, items_mode, rng):
    sets = [sorted(i for i in bd.resp[s] if i in bd.items) for s in sids]
    if items_mode == "full":
        return sets
    n = int(rng.integers(PRACTICE_ITEMS[0], PRACTICE_ITEMS[1] + 1)) if items_mode == "practice" else int(items_mode)
    inter = sorted(set.intersection(*map(set, sets)))
    if len(inter) >= n:  # one common item set for all systems of the benchmark (a small hidden benchmark)
        sub = rng.choice(np.array(inter, dtype=object), size=n, replace=False).tolist()
        return [sorted(sub) for _ in sids]
    return [sorted(rng.choice(np.array(s, dtype=object), size=min(n, len(s)), replace=False).tolist()) for s in sets]


def build_episode(benches, held: Bench, regime: str, items_mode: str, unit: str, rng, fillers: bool):
    if regime == "practice":
        m = 1 if len(held.elig) == 1 else (2 if rng.random() < 2.0 / 7.0 else 1)
    else:
        m = min(int(regime[1:]), len(held.elig))
    groups = [(held, rng.choice(np.array(held.elig, dtype=object), size=m, replace=False).tolist(), True)]
    if regime == "practice" and fillers:
        for bd in benches:
            if bd is not held and bd.elig:
                groups.append((bd, rng.choice(np.array(bd.elig, dtype=object), size=1).tolist(), False))
    pairs = []
    for bd, sids, scored in groups:
        for sid, its in zip(sids, _choose_items(bd, sids, items_mode, rng)):
            perm = rng.permutation(len(its)); n_acq = len(its) // 2
            acq = [its[j] for j in perm[:n_acq]]; ev = [its[j] for j in perm[n_acq:]]
            stream = _label_rows(bd, sid, acq, unit, rng)
            stream = [stream[j] for j in rng.permutation(len(stream))]  # fixed random stream order
            pairs.append({"b": bd, "sid": sid, "scored": scored, "subject": bd.subj.get(sid, {k: "" for k in SUBJ_KEYS}),
                          "stream": stream, "evals": _label_rows(bd, sid, ev, unit, rng), "n_items": len(its)})
    order = rng.permutation(len(pairs))
    pairs = [pairs[j] for j in order]
    for j, p in enumerate(pairs):
        p["anon_sid"] = "subject_%03d" % (j + 1)
    return pairs


def run_episode(sub_dir: Path, pairs, rng, workers: int = 1, persist: bool = False, timing: dict | None = None, fast: bool = False):
    lm = (lambda d: accelerate(load_module(d))) if fast else load_module
    lab_path = sub_dir / "labeling.py"
    hook = AcquisitionHook(load_module(sub_dir, "labeling.py").acquisition_function if lab_path.exists() else None)
    if hook.mode == "legacy_rank":
        raise NotImplementedError("one-argument (legacy rank) acquisition_function is not simulated")
    coord = lm(sub_dir) if (hook.mode == "stream" and hook.needs_prediction) else None
    persistent = lm(sub_dir) if persist else None
    # unique evaluation inputs (group_inputs) and each pair's (input index, label) targets
    canon, inputs, targets = {}, [], []
    for p in pairs:
        sk = json.dumps(p["subject"], sort_keys=True)
        rows = []
        for iid, y in p["evals"]:
            key = (sk, p["b"].idx, p["b"].ikey[iid])
            if key not in canon:
                canon[key] = len(inputs); inputs.append([p["subject"], p["b"].items[iid]])
            rows.append((canon[key], y))
        targets.append(rows)
    order = rng.permutation(len(inputs)).tolist()  # coordinator's target order (fixed per episode)
    labeled, acquired, cursor = [], [0] * len(pairs), [0] * len(pairs)
    B = [dict() for _ in pairs]
    for budget in BUDGETS:
        t0 = time.time()
        for pi, p in enumerate(pairs):
            while acquired[pi] < budget and cursor[pi] < len(p["stream"]):
                iid, y = p["stream"][cursor[pi]]
                ctx = {"subject_id": p["anon_sid"], "benchmark_id": p["b"].anon, "labels_remaining": MAX_LABELS - acquired[pi],
                       "labels_acquired": acquired[pi], "max_labels": MAX_LABELS, "items_remaining": len(p["stream"]) - cursor[pi]}
                cursor[pi] += 1
                inp = [p["subject"], p["b"].items[iid]]
                if hook.mode == "random":
                    q = random_query(ctx, inp)
                else:
                    prob = validate_score(coord.predict(copy.deepcopy(inp), copy.deepcopy(labeled))) if hook.needs_prediction else None
                    q = hook(inp, prob, labeled, ctx)
                if q:
                    labeled.append([[copy.deepcopy(p["subject"]), copy.deepcopy(p["b"].items[iid])], int(y)])
                    acquired[pi] += 1
        preds = [None] * len(inputs)
        for w in range(workers):
            mod = persistent if persist else lm(sub_dir)
            frozen = copy.deepcopy(labeled)
            for j in order[w::workers]:
                preds[j] = validate_score(mod.predict(copy.deepcopy(inputs[j]), frozen))
        for pi, rows in enumerate(targets):
            B[pi][budget] = float(np.mean([(preds[j] - y) ** 2 for j, y in rows]))
        if timing is not None:
            timing[budget] = timing.get(budget, 0.0) + time.time() - t0
    out = []
    for pi, (p, b) in enumerate(zip(pairs, B)):
        if p["scored"]:
            out.append({"bench": p["b"].name, "sid": p["sid"], "n_items": p["n_items"], "n_eval": len(p["evals"]),
                        "labels": acquired[pi], "B": b, "alc": sum(WEIGHTS[k] * b[k] for k in BUDGETS)})
    return out


# ------------------------------------------------------------------------------------------------------------- reports
def leak_report(benches, subj_first, sub_dir: Path):
    from paie.subjprior import SubjectPrior
    shipped = json.loads((sub_dir / "subject_prior.json").read_text())
    rebuilt = build_prior_json(benches, subj_first, kw=shipped["kw"], mw=shipped["mw"])
    diff = {"table_keys_equal": set(shipped["table"]) == set(rebuilt["table"]),
            "max_abs_table": max(abs(shipped["table"][k][0] - rebuilt["table"][k][0]) for k in shipped["table"] if k in rebuilt["table"]),
            "max_abs_w": float(np.max(np.abs(np.array(shipped["w"]) - np.array(rebuilt["w"])))),
            "max_abs_mu": float(np.max(np.abs(np.array(shipped["mu"]) - np.array(rebuilt["mu"]))))}
    print("builder check (all 6 benchmarks) vs shipped subject_prior.json:", diff)

    def mk(j):
        sp = SubjectPrior(); sp.table = {k: tuple(v) for k, v in j["table"].items()}; sp.mu = np.array(j["mu"]); sp.sd = np.array(j["sd"])
        sp.w = np.array(j["w"]); sp.kw = j["kw"]; sp.mw = j["mw"]; return sp

    sp_ship = mk(shipped); rep = {"builder_check": diff, "folds": {}}
    lg = lambda a: np.log(np.clip(a, .02, .98) / (1 - np.clip(a, .02, .98)))  # noqa: E731
    print(f"{'held-out':18s} {'n':>3s} {'r(shipped)':>10s} {'r(fold)':>8s} {'rmse ship':>9s} {'rmse fold':>9s} {'names only in held-out':>22s}")
    for bd in benches:
        sids_all = sorted(bd.resp)
        acc = np.array([np.mean([float(v.mean()) for v in bd.resp[s].values()]) for s in sids_all]); rel = lg(acc) - lg(acc).mean()
        relmap = dict(zip(sids_all, rel))
        sp_fold = mk(build_prior_json(benches, subj_first, exclude={bd.name}, kw=shipped["kw"], mw=shipped["mw"]))
        el = bd.elig
        t = np.array([relmap[s] for s in el])
        o_s = np.array([sp_ship.offset(bd.subj.get(s, {})) for s in el]); o_f = np.array([sp_fold.offset(bd.subj.get(s, {})) for s in el])
        from paie.subjprior import norm_name
        other_names = {norm_name(bd2.subj.get(s, {}).get("normalized_name", "")) for bd2 in benches if bd2 is not bd for s in bd2.resp}
        only = sum(1 for s in el if norm_name(bd.subj.get(s, {}).get("normalized_name", "")) not in other_names)
        r_s = float(np.corrcoef(t, o_s)[0, 1]) if len(el) > 2 and o_s.std() > 0 else float("nan")
        r_f = float(np.corrcoef(t, o_f)[0, 1]) if len(el) > 2 and o_f.std() > 0 else float("nan")
        rm_s = float(np.sqrt(np.mean((0.5 * o_s - t) ** 2))); rm_f = float(np.sqrt(np.mean((0.5 * o_f - t) ** 2)))
        rep["folds"][bd.name] = {"n": len(el), "r_shipped": r_s, "r_fold": r_f, "rmse_shipped_x0.5": rm_s, "rmse_fold_x0.5": rm_f,
                                 "names_only_in_heldout": only}
        print(f"{bd.name:18s} {len(el):3d} {r_s:10.3f} {r_f:8.3f} {rm_s:9.3f} {rm_f:9.3f} {only:22d}")
    return rep


def summarize(results, label):
    by = {}
    for r in results:
        by.setdefault(r["bench"], []).append(r)
    rows = {}
    for b in sorted(by):
        rs = by[b]; alcs = np.array([r["alc"] for r in rs])
        rows[b] = {"n_pairs": len(rs), "alc": float(alcs.mean()), "se": float(alcs.std(ddof=1) / np.sqrt(len(rs))) if len(rs) > 1 else 0.0,
                   "B": {k: float(np.mean([r["B"][k] for r in rs])) for k in BUDGETS}}
    macro = float(np.mean([v["alc"] for v in rows.values()])) if rows else float("nan")
    micro = float(np.mean([r["alc"] for r in results])) if results else float("nan")
    print(f"--- {label}: macro ALC {macro:.4f} (mean over {len(rows)} held-out benchmarks), micro {micro:.4f} ({len(results)} pairs)")
    for b, v in rows.items():
        print(f"    {b:18s} pairs {v['n_pairs']:4d}  ALC {v['alc']:.4f} +- {v['se']:.4f}   " + " ".join(f"B{k}={v['B'][k]:.3f}" for k in BUDGETS))
    return {"macro": macro, "micro": micro, "per_bench": rows}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sub", default="submission_v5")
    ap.add_argument("--regimes", default="practice,k1,k2,k4,k8")
    ap.add_argument("--episodes", type=int, default=20, help="episodes per held-out benchmark and regime")
    ap.add_argument("--items", default="practice", help="practice (88-152 per pair) | full | <int>")
    ap.add_argument("--unit", default="trial", choices=["trial", "majority", "response"])
    ap.add_argument("--prior", default="fold", choices=["shipped", "fold", "none"])
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--persist", action="store_true", help="one model module per episode (no worker recreation per checkpoint)")
    ap.add_argument("--no-fillers", action="store_true")
    ap.add_argument("--only", default="", help="comma-separated held-out benchmarks")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--leak-report", action="store_true")
    ap.add_argument("--fast", action="store_true", help="memoise the pure _key/hash_counts helpers of v3/v5 (identical outputs)")
    a = ap.parse_args()
    sub_dir = (ROOT / a.sub).resolve() if not Path(a.sub).is_absolute() else Path(a.sub)
    t0 = time.time()
    benches, subj_first = load_public()
    print(f"loaded {len(benches)} benchmarks in {time.time() - t0:.1f}s: " + ", ".join(f"{b.name}({len(b.elig)} eligible)" for b in benches), flush=True)
    (EXP / "results").mkdir(parents=True, exist_ok=True)
    tag = a.tag or f"{sub_dir.name}_{a.prior}_{a.unit}_{a.items}"
    if a.leak_report:
        rep = leak_report(benches, subj_first, sub_dir)
        (EXP / "results" / f"leak_{tag}.json").write_text(json.dumps(rep, indent=1))
        return
    only = set(filter(None, a.only.split(",")))
    out = {"config": vars(a), "regimes": {}}
    for regime in a.regimes.split(","):
        code = 0 if regime == "practice" else int(regime[1:])
        results, timing = [], {}
        for held in benches:
            if (only and held.name not in only) or not held.elig:
                continue
            sd = prepare_submission(sub_dir, a.prior, held.name, benches, subj_first)
            tb = time.time(); res_b = []
            for ep in range(a.episodes):
                rng = np.random.default_rng([a.seed, code, held.idx, ep])
                pairs = build_episode(benches, held, regime, a.items, a.unit, rng, fillers=not a.no_fillers)
                res_b += run_episode(sd, pairs, rng, workers=a.workers, persist=a.persist, timing=timing, fast=a.fast)
            results += res_b
            print(f"[{regime}] {held.name:18s} pairs {len(res_b):4d}  ALC {np.mean([r['alc'] for r in res_b]):.4f}  ({time.time() - tb:.0f}s)", flush=True)
        out["regimes"][regime] = {"summary": summarize(results, f"{tag} [{regime}]"), "pairs": results,
                                  "seconds_per_checkpoint": timing}
        (EXP / "results" / f"sim_{tag}.json").write_text(json.dumps(out, indent=1, default=str))
    print(f"done in {time.time() - t0:.0f}s -> {EXP / 'results' / f'sim_{tag}.json'}")


if __name__ == "__main__":
    main()
