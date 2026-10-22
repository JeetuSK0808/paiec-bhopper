"""exp_v7_robust_labeling (tag 'v7'): edge cases for submission_v7/labeling.py through the organisers' own AcquisitionHook.

Checks: call shape (mode 'stream', no prediction requested), the rule itself (skip co-acquired items, same decision as the evaluator's
random rule otherwise, forced take, exhausted budget), malformed inputs never raise and always return a bool, decisions are identical to
the evaluator's default policy whenever nothing is skipped, and the hook leaves its arguments unchanged. Timing on a large `labeled`.
  .venv/bin/python -B scripts/exp_v7_robust_labeling.py [submission_v7]
"""
import copy, hashlib, importlib.util, json, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "upstream" / "paiec_baseline" / "tools"))
from streaming_ingestion import AcquisitionHook  # noqa: E402

sub = ROOT / (sys.argv[1] if len(sys.argv) > 1 else "submission_v7")
spec = importlib.util.spec_from_file_location("v7_labeling", sub / "labeling.py")
lab = importlib.util.module_from_spec(spec); spec.loader.exec_module(lab)
hook = AcquisitionHook(lab.acquisition_function)
ok = []


def check(name, cond):
    ok.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name)


def default_rule(context, inp):
    key = json.dumps([context, inp], sort_keys=True, separators=(",", ":"))
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big") / 2 ** 64 < min(1, context["labels_remaining"] / context["items_remaining"])


check("mode is stream", hook.mode == "stream")
check("hook does not request a prediction", not hook.needs_prediction)
A = {"normalized_name": "a", "provider": "p", "release_date": "", "access_date": "", "harness": "", "harness_version": "",
     "reasoning_effort": "", "subject_features_extra": ""}
B = dict(A, normalized_name="b")
X = {"item_content": "x", "item_features": "", "interactors": "", "benchmark_id": "bm1"}
Y = dict(X, item_content="y")
X_other_bench = dict(X, benchmark_id="bm2")
ctx = {"subject_id": "s1", "benchmark_id": "bm1", "labels_acquired": 3, "labels_remaining": 28, "max_labels": 31, "items_remaining": 50}
lab_B_X = [[[B, X], 1]]
check("skips an item another system of the benchmark revealed", hook([A, X], None, lab_B_X, ctx) is False)
check("own earlier label of the same item does not trigger a skip", hook([A, X], None, [[[A, X], 0]], ctx) == default_rule(ctx, [A, X]))
check("same item text on another benchmark does not trigger a skip",
      hook([A, X], None, [[[B, X_other_bench], 1]], ctx) == default_rule(ctx, [A, X]))
check("forced take when items_remaining <= labels_remaining", hook([A, X], None, lab_B_X, dict(ctx, items_remaining=28)) is True)
check("never acquires with an exhausted allowance", hook([A, Y], None, [], dict(ctx, labels_remaining=0)) is False)
agree = all(hook([A, dict(Y, item_content=f"y{j}")], None, lab_B_X, dict(ctx, items_remaining=n)) ==
            default_rule(dict(ctx, items_remaining=n), [A, dict(Y, item_content=f"y{j}")]) for j in range(200) for n in (29, 40, 80))
check("matches the evaluator's default random rule whenever nothing is skipped (600 cases)", agree)
bad = [(None, None, None), ([A, X], "garbage", ctx), ([A, X], [["bad"]], ctx), ([A, X], [[[B, None], 1]], ctx),
       ([None, X], lab_B_X, ctx), ([A, "notadict"], lab_B_X, ctx), ([A, X], lab_B_X, {}), ([A, X], lab_B_X, None),
       ([A, X], lab_B_X, dict(ctx, labels_remaining="x")), ([A, X], [[[B, X], float("nan")]], ctx), ("short", [], ctx)]
res = []
for inp, l_, c_ in bad:
    try:
        r = lab.acquisition_function(inp, labeled=l_, context=c_); res.append(type(r) is bool)
    except Exception as exc:  # noqa: BLE001
        print("   raised:", type(exc).__name__, exc); res.append(False)
check(f"malformed inputs never raise and return a bool ({len(bad)} cases)", all(res))
before = copy.deepcopy([[A, X], lab_B_X, ctx]); lab.acquisition_function([A, X], labeled=lab_B_X, context=ctx)
check("arguments are left unchanged", before == [[A, X], lab_B_X, ctx])
big_item = dict(X, item_content="z" * 100_000)
big = [[[dict(B, normalized_name=f"s{j % 9}"), dict(big_item, item_content=big_item["item_content"] + str(j))], j % 2] for j in range(279)]
t0 = time.time()
for j in range(500):
    hook([A, dict(big_item, item_content=big_item["item_content"] + str(j))], None, big, ctx)
dt = (time.time() - t0) / 500
check(f"fast with 279 labels of 100 KB items (incl. the evaluator's deepcopy): {dt * 1000:.2f} ms per call", dt < 0.5)
print(f"{sum(ok)}/{len(ok)} checks passed")
sys.exit(0 if all(ok) else 1)
