"""Rebuild <dir>/subject_prior.json from ALL 6 public benchmarks with the same builder as scripts/item_prior_cv.py --sp (via
scripts/exp_sim_protocol.build_prior_json, verified to reproduce submission_v5/subject_prior.json to 1e-16).
  .venv/bin/python scripts/exp_synth_build_prior.py submission_v6 [--exclude bench]"""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import exp_sim_protocol as S  # noqa: E402

out = ROOT / sys.argv[1] / "subject_prior.json"
excl = set(sys.argv[sys.argv.index("--exclude") + 1].split(",")) if "--exclude" in sys.argv else set()
benches, subj_first = S.load_public()
j = S.build_prior_json(benches, subj_first, exclude=excl)
out.write_text(json.dumps(j))
ref = json.loads((ROOT / "submission_v5" / "subject_prior.json").read_text())
if not excl:
    d = max(abs(ref["table"][k][0] - j["table"][k][0]) for k in ref["table"])
    print(f"wrote {out}; keys equal to v5: {set(ref['table']) == set(j['table'])}; max |table diff| {d:.2e}; max |w diff| "
          f"{max(abs(a - b) for a, b in zip(ref['w'], j['w'])):.2e}")
else:
    print(f"wrote {out} excluding {excl}")
