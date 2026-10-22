"""exp_verify_contract (tag 'verify'): adversarial contract checks of a submission directory (default: the extracted
submissions/paie_v6.zip). Runs model.predict(input, labeled) through the public interface only.

  .venv/bin/python -B scripts/exp_verify_contract.py <submission_dir>            # all checks, prints a summary
  .venv/bin/python -B scripts/exp_verify_contract.py <submission_dir> --scenario  # (internal) deterministic scenario dump

Checks: return type/range/finiteness on edge cases (None/empty labeled, unseen or missing benchmark_id, non-dict and None fields,
bool/int/float/numpy/fractional/NaN/inf/None/string labels, malformed entries, tuples, extra keys, huge inputs), no network use
(socket patched to raise), no file writes during predict, determinism across fresh processes with different PYTHONHASHSEED and
across prediction orders, runtime with a cache-defeating deep copy of `labeled` per call.
"""
from __future__ import annotations
import builtins, copy, importlib.util, json, math, os, socket, subprocess, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load(sub: str):
    d = str(Path(sub).resolve()); sys.path.insert(0, d)
    try:
        spec = importlib.util.spec_from_file_location("_verify_model", os.path.join(d, "model.py"))
        m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    finally:
        sys.path.remove(d)
    return m


def subj(i, **kw):
    s = {"normalized_name": f"model-{i}", "provider": ["openai", "anthropic", "google"][i % 3], "release_date": f"2025-0{1 + i % 9}-15",
         "access_date": "2026-01-01", "harness": "h", "harness_version": "1", "reasoning_effort": ["low", "high", ""][i % 3],
         "subject_features_extra": ""}
    s.update(kw); return s


def item(j, bench="benchmark_000001", **kw):
    words = ["alpha", "beta", "gamma", "delta", "eps", "zeta", "eta", "theta", "iota", "kappa"]
    it = {"item_content": " ".join(words[(j * k) % 10] for k in range(1, 30)) + f" q{j}", "item_features": f"cat=c{j % 4};idx={j}",
          "interactors": None, "benchmark_id": bench}
    it.update(kw); return it


def synth_labeled(rng, n_bench=3, n_sys=4, n_items=60, k=15, soft=False):
    lab, truth = [], {}
    for b in range(n_bench):
        bench = f"benchmark_{b:06d}"; lvl = rng.normal(0, 1.5); eps = rng.normal(0, 1.2, n_items)
        for s in range(n_sys):
            th = lvl + rng.normal(0, 0.5)
            for j in rng.permutation(n_items)[:k]:
                p = 1 / (1 + np.exp(-(th + eps[j])))
                y = float(p) if soft else int(rng.random() < p)
                lab.append([[subj(s + 10 * b), item(int(j), bench)], y])
    return lab


class Guard:
    """Patch socket creation to raise and record any file opened for writing while active."""
    def __enter__(self):
        self.writes = []; self._sock = socket.socket; self._open = builtins.open; outer = self

        class NoSock(socket.socket):
            def __init__(self, *a, **k):
                raise RuntimeError("network use attempted")
        socket.socket = NoSock

        def op(file, mode="r", *a, **k):
            if any(c in str(mode) for c in "wax+"):
                outer.writes.append(str(file))
            return outer._open(file, mode, *a, **k)
        builtins.open = op
        return self

    def __exit__(self, *e):
        socket.socket = self._sock; builtins.open = self._open


def check_value(v):
    return (type(v) is float) and math.isfinite(v) and 0.0 <= v <= 1.0


def edge_cases(M):
    rng = np.random.default_rng(0)
    base = synth_labeled(rng)
    S, I = subj(0), item(3)
    cases = {
        "labeled None": ([S, I], None),
        "labeled []": ([S, I], []),
        "checker smoke input": ([{"normalized_name": "sample-model", "provider": "sample-org", "release_date": "2026-01-01",
                                  "access_date": "2026-02-01", "harness": "sample-harness", "reasoning_effort": "medium",
                                  "harness_version": "1.0", "subject_features_extra": ""},
                                 {"item_content": "A sample yes/no evaluation item.", "item_features": "tier=near-term", "interactors": ""}], []),
        "normal": ([S, I], base),
        "unseen subject": ([subj(99), item(3)], base),
        "unseen item": ([S, item(999)], base),
        "unseen benchmark": ([S, item(3, "benchmark_999999")], base),
        "target without benchmark_id": ([S, {k: v for k, v in I.items() if k != "benchmark_id"}], base),
        "benchmark_id None everywhere": ([S, item(3, None)], [[[s_, dict(i_, benchmark_id=None)], y] for (s_, i_), y in base]),
        "benchmark_id int": ([S, item(3, 7)], [[[s_, dict(i_, benchmark_id=7)], y] for (s_, i_), y in base]),
        "subject None": ([None, I], base),
        "subject str": (["gpt", I], base),
        "item None": ([S, None], base),
        "item str": ([S, "what is 2+2"], base),
        "fields None": ([{k: None for k in S}, {"item_content": None, "item_features": None, "interactors": None, "benchmark_id": "benchmark_000000"}], base),
        "features dict": ([S, item(3, item_features={"cat": "c3", "idx": 3})], base),
        "features json": ([S, item(3, item_features='{"cat": "c3", "idx": 3}')], base),
        "features garbage": ([S, item(3, item_features="%%%;;==;x")], base),
        "features nan/inf": ([S, item(3, item_features="idx=nan;cat=inf")], base),
        "labels bool": ([S, I], [[e[0], bool(e[1])] for e in base]),
        "labels float": ([S, I], [[e[0], float(e[1])] for e in base]),
        "labels numpy": ([S, I], [[e[0], np.int64(e[1]) if i % 3 == 0 else (np.float64(e[1]) if i % 3 == 1 else np.bool_(e[1]))] for i, e in enumerate(base)]),
        "labels fractional": ([S, I], synth_labeled(np.random.default_rng(1), soft=True)),
        "labels NaN/inf/None/str": ([S, I], [[e[0], [float("nan"), float("inf"), None, "abc", "1", -3, 7][i % 7]] for i, e in enumerate(base)]),
        "entries malformed": ([S, I], [None, [], [1], [[S], 1], [S, I, 1], {"a": 1}, [[S, I]], "x"] + base[:20]),
        "entries tuples": ((S, I), tuple(((e[0][0], e[0][1]), e[1]) for e in base)),
        "input wrong length": ([S], base),
        "input None": (None, base),
        "extra keys": ([dict(S, subject_id="s1"), dict(I, item_id="i1", extra=[1, 2])], base),
        "all labels 1": ([S, I], [[e[0], 1] for e in base]),
        "all labels 0": ([S, I], [[e[0], 0] for e in base]),
        "one label": ([S, I], base[:1]),
        "huge item_content 5MB": ([S, item(3, item_content="x y z " * 900_000)], base),
        "huge features": ([S, item(3, item_features=";".join(f"k{i}=v{i}" for i in range(20000)))], base),
    }
    out = {}
    for name, (inp, lab) in cases.items():
        try:
            with Guard() as g:
                t0 = time.time(); v = M.predict(copy.deepcopy(inp), lab); dt = time.time() - t0
            ok = check_value(v) and not g.writes
            out[name] = dict(ok=ok, value=(v if isinstance(v, float) else repr(v)), sec=round(dt, 3), writes=g.writes)
        except Exception as exc:  # predict raised
            out[name] = dict(ok=False, value=f"RAISED {type(exc).__name__}: {exc}")
    return out


def scenario(M):
    """Deterministic scenario: 3 benchmarks x 4 systems, 15 labels each; predict every (system, item) pair in a fixed order and in reverse."""
    lab = synth_labeled(np.random.default_rng(42))
    targets = [[subj(s + 10 * b), item(j, f"benchmark_{b:06d}")] for b in range(3) for s in range(5) for j in range(0, 60, 3)]
    fwd = [M.predict(copy.deepcopy(t), lab) for t in targets]
    rev = [M.predict(copy.deepcopy(t), lab) for t in reversed(targets)][::-1]
    return fwd, rev


def timing(M, n_bench=7, n_sys=10, k=31, n_items=120, n_calls=1500, deep=True):
    rng = np.random.default_rng(7)
    lab = synth_labeled(rng, n_bench=n_bench, n_sys=n_sys, n_items=n_items, k=k)
    for e in lab:  # make items long (like researchcodebench ~100 KB) for a subset
        if rng.random() < 0.1:
            e[0][1]["item_content"] = e[0][1]["item_content"] + (" token" * 15000)
    targets = [[subj(int(rng.integers(n_sys)) + 10 * b), item(int(rng.integers(n_items)), f"benchmark_{b:06d}")] for b in [int(rng.integers(n_bench)) for _ in range(n_calls)]]
    t0 = time.time()
    for t in targets:
        M.predict(copy.deepcopy(t), copy.deepcopy(lab) if deep else lab)
    return len(lab), (time.time() - t0) / n_calls


def main():
    sub = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "submission_v6")
    if "--scenario" in sys.argv:
        M = load(sub); fwd, rev = scenario(M); print(json.dumps({"fwd": fwd, "rev": rev})); return
    M = load(sub)
    res = edge_cases(M)
    bad = [k for k, v in res.items() if not v["ok"]]
    for k, v in res.items():
        print(f"{'OK ' if v['ok'] else 'BAD'} {k:32s} {v}")
    # determinism across fresh processes / hash seeds and prediction order
    outs = []
    for seed in ("0", "12345", "random"):
        env = dict(os.environ, PYTHONHASHSEED=seed)
        r = subprocess.run([sys.executable, "-B", __file__, sub, "--scenario"], capture_output=True, text=True, env=env)
        outs.append(json.loads(r.stdout))
    fwd0 = np.array(outs[0]["fwd"])
    dmax = max(float(np.max(np.abs(np.array(o[k]) - fwd0))) for o in outs for k in ("fwd", "rev"))
    print(f"determinism: max |diff| across 3 fresh processes (PYTHONHASHSEED 0/12345/random) x 2 orders over {len(fwd0)} preds = {dmax:.3g}; "
          f"pred range [{fwd0.min():.3f}, {fwd0.max():.3f}]")
    # nan guard probe: a non-finite linear predictor must not become a confident prediction
    try:
        p = M.predict_core(dict(beta=float("nan"), delta=np.zeros(1), eps=np.zeros(1), prior=None, grp=None, g=None), 0, 0.0, 0, None, {})
        print(f"predict_core with beta=NaN returns {p!r} (a NaN z should not map to ~1.0)")
    except Exception as exc:
        print(f"predict_core with beta=NaN raises {type(exc).__name__} (predict() then falls back)")
    n_lab, per = timing(M, deep=True)
    print(f"timing: {n_lab} labels over 7 benchmarks, deep-copied labeled per call: {per * 1000:.2f} ms/call")
    n_lab, per = timing(M, deep=False)
    print(f"timing: {n_lab} labels, same labeled object per call: {per * 1000:.3f} ms/call")
    print(f"SUMMARY: {len(res) - len(bad)}/{len(res)} edge cases OK; bad: {bad}")


if __name__ == "__main__":
    main()
