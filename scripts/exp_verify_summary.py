"""exp_verify_summary (tag 'verify'): per-benchmark 3-seed means from the verification logs in src/paie_exp/verify/
(lines of scripts/exp_synth_v6cv.py and scripts/exp_synth_cv.py --perbench output).
  .venv/bin/python -B scripts/exp_verify_summary.py log_v5shipped_s012.txt log_v6cv_s012.txt log_v6final_s012.txt"""
import re, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
D = Path(__file__).resolve().parents[1] / "src" / "paie_exp" / "verify"
PAT = re.compile(r"seed=(\d+)\s+(\w+)\s+ALC ([0-9.]+)\s+(.*)")
for name in sys.argv[1:]:
    tab = defaultdict(lambda: defaultdict(list))
    for line in (D / name).read_text().splitlines():
        m = PAT.search(line)
        if not m:
            continue
        seed, reg, alc, rest = m.groups()
        tab[reg]["MEAN"].append(float(alc))
        for h, v in re.findall(r"(\w+):([0-9.]+)", rest):
            tab[reg][h].append(float(v))
    for reg, cols in tab.items():
        print(f"{name:26s} {reg:7s} " + "  ".join(f"{h}:{np.mean(v):.4f}(n={len(v)})" for h, v in cols.items()))
