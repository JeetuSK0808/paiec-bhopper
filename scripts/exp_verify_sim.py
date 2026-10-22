"""exp_verify_sim (tag 'verify'): run the faithful streaming simulator scripts/exp_sim_protocol.py unchanged, with its output folder
(per-fold subject_prior.json copies, results JSON) redirected to src/paie_exp/verify/sim/. Same CLI as exp_sim_protocol.py."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.dont_write_bytecode = True
import exp_sim_protocol as S  # noqa: E402

S.EXP = ROOT / "src" / "paie_exp" / "verify" / "sim"
S.EXP.mkdir(parents=True, exist_ok=True)
if __name__ == "__main__":
    S.main()
