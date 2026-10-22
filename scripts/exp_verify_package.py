"""exp_verify_package (tag 'verify'): package a submission directory as a Codabench ZIP (files at the archive root, same file list and
layout as scripts/exp_synth_package.py), validate it with upstream/paiec_baseline/check_submission_zip.py, and only then move it to the
destination (the previous archive, if any, is kept as a backup under src/paie_exp/verify/).
  .venv/bin/python -B scripts/exp_verify_package.py submission_v6 submissions/paie_v6.zip
"""
import hashlib, shutil, subprocess, sys, zipfile
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
FILES = ["model.py", "textprior.py", "subjprior.py", "subject_prior.json", "requirements.txt", "README.md"]
src, dst = ROOT / sys.argv[1], ROOT / sys.argv[2]
tmp = ROOT / "src" / "paie_exp" / "verify" / ("new_" + dst.name)
if tmp.exists():
    tmp.unlink()
with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zf:
    for f in FILES:
        p = src / f
        if p.exists():
            zf.write(p, arcname=f)
names = [i.filename for i in zipfile.ZipFile(tmp).infolist()]
r = subprocess.run([sys.executable, "-B", str(ROOT / "upstream/paiec_baseline/check_submission_zip.py"), str(tmp)], capture_output=True, text=True)
print(r.stdout.strip(), r.stderr.strip())
if r.returncode != 0:
    raise SystemExit(f"checker failed; {dst} left unchanged")
if dst.exists():
    bak = ROOT / "src" / "paie_exp" / "verify" / (dst.stem + "_before_verify" + dst.suffix)
    if not bak.exists():
        shutil.copy2(dst, bak); print(f"backup of previous archive: {bak}")
shutil.move(str(tmp), str(dst))
print(f"wrote {dst}: {names}; sha256 {hashlib.sha256(dst.read_bytes()).hexdigest()}")
