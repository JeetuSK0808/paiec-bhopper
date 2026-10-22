"""exp_calib: sequential in-process grid over shrinkage settings of scripts/exp_calib_cv.py; stores per-benchmark per-budget Brier
in one json so budget-dependent settings can be chosen by nested leave-one-benchmark-out (scripts/exp_calib_select.py).
  .venv/bin/python scripts/exp_calib_grid.py OUT.json [--single] --grid "tau=0.5,0.8;tau-i=1.0,1.5;text-scale=1,2" [--extra "--beta-prior"]"""
import sys, json, itertools, io, contextlib, os, argparse
sys.path.insert(0, "scripts")
import exp_calib_cv
ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--single", action="store_true"); ap.add_argument("--grid", required=True); ap.add_argument("--extra", default="")
a = ap.parse_args()
axes = [(kv.split("=")[0], kv.split("=")[1].split(",")) for kv in a.grid.split(";")]
res = json.load(open(a.out)) if os.path.exists(a.out) else {}
tmp = a.out + ".tmp.json"
for combo in itertools.product(*[v for _, v in axes]):
    key = " ".join(f"--{n} {v}" for (n, _), v in zip(axes, combo)) + (" --single" if a.single else "") + (" " + a.extra if a.extra else "")
    if key in res:
        continue
    sys.argv = ["exp_calib_cv.py"] + key.split() + ["--json", tmp]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exp_calib_cv.main()
    res[key] = json.load(open(tmp)); last = [l for l in buf.getvalue().splitlines() if l.startswith("mean ALC")][0]
    print(f"{key:60s} {last[:40]}", flush=True)
    json.dump(res, open(a.out, "w"), indent=0)
os.remove(tmp) if os.path.exists(tmp) else None
