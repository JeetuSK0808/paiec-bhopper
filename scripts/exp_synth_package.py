"""Package a submission directory as a Codabench ZIP with the files at the archive root (same layout as submissions/paie_v5.zip),
then run the organisers' validator upstream/paiec_baseline/check_submission_zip.py on it.
  .venv/bin/python scripts/exp_synth_package.py submission_v6 submissions/paie_v6.zip"""
import subprocess, sys, zipfile
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
FILES = ["model.py", "labeling.py", "textprior.py", "subjprior.py", "subject_prior.json", "requirements.txt", "README.md"]  # labeling.py only if present (v7+)
src, dst = ROOT / sys.argv[1], ROOT / sys.argv[2]
if dst.exists():
    raise SystemExit(f"refusing to overwrite {dst}")
with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as zf:
    for f in FILES:
        p = src / f
        if p.exists():
            zf.write(p, arcname=f)
print(f"wrote {dst}:", [i.filename for i in zipfile.ZipFile(dst).infolist()])
r = subprocess.run([sys.executable, "-B", str(ROOT / "upstream/paiec_baseline/check_submission_zip.py"), str(dst)], capture_output=True, text=True)
print(r.stdout.strip(), r.stderr.strip())
sys.exit(r.returncode)
