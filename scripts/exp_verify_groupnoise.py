"""exp_verify_groupnoise (tag 'verify'): hidden-benchmark risk of v6's item_features group effects. Runs the exact submission code through
scripts/exp_synth_v6cv.py, but appends PURE-NOISE item_features to every public item (deterministic per benchmark):
  uid : 'uid=<random integer>'  (unique numeric id -> becomes a z-scored numeric slope in v6's Groups)
  cat : 'grp=g<0..19>'          (random 20-level category -> one-hot group effects)
  both: both of the above
  heavy: 'n=<int(exp(N(5, 2)))>' (heavy-tailed numeric noise, like a token or file count)
and compares with the unmodified features (and optionally TAU_G=0). Reports mean ALC per regime and per benchmark.
  .venv/bin/python -B scripts/exp_verify_groupnoise.py --sub submission_v6 --modes none,uid,cat,both --regimes single,pooled --seeds 0
"""
import argparse, sys, zlib
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "src"))
sys.dont_write_bytecode = True
import exp_synth_v6cv as V  # noqa: E402
import exp_pooled_cv as H  # noqa: E402

_ORIG = V.raw_items
MODE = {"m": "none"}


def noisy_raw_items(held):
    texts, feats = _ORIG(held)
    if MODE["m"] == "none":
        return texts, feats
    rng = np.random.default_rng(zlib.crc32(held.encode()))
    uid = rng.integers(0, 1_000_000, len(feats)); cat = rng.integers(0, 20, len(feats))
    heavy = np.exp(rng.normal(5, 2, len(feats))).astype(int)
    out = []
    for f, u, c, hv in zip(feats, uid, cat, heavy):
        parts = [f] if isinstance(f, str) and f else []
        if MODE["m"] == "heavy":
            parts.append(f"n={int(hv)}")
        if MODE["m"] in ("uid", "both"):
            parts.append(f"uid={int(u)}")
        if MODE["m"] in ("cat", "both"):
            parts.append(f"grp=g{int(c)}")
        out.append(";".join(parts))
    return texts, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", default="submission_v6"); ap.add_argument("--modes", default="none,uid,cat,both")
    ap.add_argument("--regimes", default="single,pooled"); ap.add_argument("--seeds", default="0"); ap.add_argument("--set", default="")
    a = ap.parse_args()
    M = V.load_model(a.sub)
    for kv in filter(None, a.set.split(",")):
        k, v = kv.split("="); setattr(M, k, type(getattr(M, k))(v))
    V.raw_items = noisy_raw_items
    for mode in a.modes.split(","):
        MODE["m"] = mode
        for seed in [int(s) for s in a.seeds.split(",")]:
            benches = H.prepare(seed, 0.5)
            for reg in a.regimes.split(","):
                mean, res = V.run(M, benches, reg, seed)
                print(f"{a.sub} {a.set or ''} noise={mode:5s} seed={seed} {reg:7s} ALC {mean:.4f}  " +
                      " ".join(f"{h[:6]}:{v:.4f}" for h, v in res.items()), flush=True)


if __name__ == "__main__":
    main()
