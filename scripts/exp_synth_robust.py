"""Robustness, determinism and runtime checks for a submission's predict(input, labeled) (tag 'synth').
  .venv/bin/python scripts/exp_synth_robust.py submission_v6"""
import copy, importlib.util, json, math, os, random, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
_N = [0]


def fresh(sub):
    d = str((ROOT / sub).resolve())
    for name in ("textprior", "subjprior"):
        sys.modules.pop(name, None)
    sys.path.insert(0, d)
    try:
        _N[0] += 1
        spec = importlib.util.spec_from_file_location(f"_rob_{_N[0]}", os.path.join(d, "model.py"))
        m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    finally:
        sys.path.remove(d)
    return m


def ok(p):
    return isinstance(p, float) and math.isfinite(p) and 0.0 <= p <= 1.0


def main():
    sub = sys.argv[1] if len(sys.argv) > 1 else "submission_v6"
    M = fresh(sub)
    S = {"normalized_name": "gpt-5", "provider": "openai", "release_date": "2025-08-07", "access_date": "2026-01-01", "harness": "h",
         "harness_version": "1", "reasoning_effort": "high", "subject_features_extra": ""}
    def item(i, b="benchmark_000001", feat="tier=medium;points=1"):
        return {"item_content": f"question {i} about topic {i % 7} " * 5, "item_features": feat, "interactors": "", "benchmark_id": b}
    lab = [[[S, item(i)], i % 2] for i in range(12)]
    cases = {
        "no labels": ([S, item(99)], []),
        "labels None": ([S, item(99)], None),
        "normal": ([S, item(99)], lab),
        "labelled item": ([S, item(3)], lab),
        "unseen benchmark": ([S, item(1, "benchmark_999999")], lab),
        "other subject": ([dict(S, normalized_name="claude-x"), item(99)], lab),
        "subject None fields": ([{k: None for k in S}, item(99)], lab),
        "subject not dict": (["gpt-5", item(99)], lab),
        "item missing benchmark_id": ([S, {"item_content": "x"}], lab),
        "item not dict": ([S, "raw item text"], lab),
        "item_content None": ([S, dict(item(99), item_content=None)], lab),
        "item_features dict": ([S, dict(item(99), item_features={"tier": "hard"})], lab),
        "item_features JSON": ([S, dict(item(99), item_features='{"tier": "hard", "points": 3}')], lab),
        "item_features garbage": ([S, dict(item(99), item_features=[1, 2, {"a": None}])], lab),
        "interactors list": ([S, dict(item(99), interactors=[{"role": "user"}])], lab),
        "fractional labels": ([S, item(99)], [[[S, item(i)], 0.25 * (i % 5)] for i in range(12)]),
        "bool labels": ([S, item(99)], [[[S, item(i)], bool(i % 2)] for i in range(12)]),
        "garbage labeled entries": ([S, item(99)], lab + [None, [1, 2, 3], [[S], 1], [[S, item(50)], "x"], [[S, item(51)], float("nan")]]),
        "labels only other benchmark": ([S, item(99, "benchmark_000002")], lab),
        "input too short": ([S], lab),
        "all-one labels": ([S, item(99)], [[[S, item(i)], 1] for i in range(31)]),
        "all-zero labels": ([S, item(99)], [[[S, item(i)], 0] for i in range(31)]),
    }
    bad = 0
    for name, (inp, labeled) in cases.items():
        try:
            p = M.predict(copy.deepcopy(inp), copy.deepcopy(labeled))
            flag = "OK " if ok(p) else "BAD"
        except Exception as e:  # noqa: BLE001
            p, flag = repr(e), "EXC"
        bad += flag != "OK "
        print(f"  {flag} {name:30s} -> {p}")
    # determinism across fresh imports and input order of predictions
    rng = random.Random(0)
    lab2 = [[[dict(S, normalized_name=f"m{s}"), item(i, f"benchmark_{s % 3}", f"tier={'abc'[i % 3]};points={i % 4}")], rng.random() < 0.4]
            for s in range(6) for i in range(31)]
    lab2 = [[x, int(y)] for x, y in lab2]
    targets = [[dict(S, normalized_name=f"m{s}"), item(100 + i, f"benchmark_{s % 3}", f"tier={'abc'[i % 3]};points={i % 4}")] for s in range(6) for i in range(20)]
    runs = []
    for order in (0, 1):
        M2 = fresh(sub)
        ts = list(range(len(targets)))
        if order:
            ts = ts[::-1]
        out = {}
        for t in ts:
            out[t] = M2.predict(copy.deepcopy(targets[t]), lab2)
        runs.append([out[t] for t in range(len(targets))])
    same = max(abs(a - b) for a, b in zip(*runs))
    print(f"  determinism: max |diff| across fresh imports + reversed prediction order = {same:.3g}")
    bad += same > 1e-12
    # runtime: 60 pairs x 31 labels on 4 benchmarks with ~100 KB items, 50 targets per pair (fresh module, one labeled object)
    M3 = fresh(sub)
    big = lambda i, b: {"item_content": ("tok%d " % (i % 97)) * 20000 + f" id {i}", "item_features": "", "interactors": "", "benchmark_id": b}
    labs = [[[dict(S, normalized_name=f"s{s}"), big(i + 1000 * (s % 4), f"benchmark_{s % 4}")], int(rng.random() < 0.5)] for s in range(60) for i in range(31)]
    tg = [[dict(S, normalized_name=f"s{s}"), big(500 + i + 1000 * (s % 4), f"benchmark_{s % 4}")] for s in range(60) for i in range(50)]
    t0 = time.time()
    for t in tg:
        M3.predict(t, labs)
    dt = time.time() - t0
    print(f"  runtime: {len(tg)} predictions with {len(labs)} labels (~100 KB items): {dt:.1f}s ({1000 * dt / len(tg):.2f} ms/call, incl. first-time hashing)")
    print("ALL OK" if bad == 0 else f"{bad} PROBLEMS")


if __name__ == "__main__":
    main()
