"""Embed item content with plain transformers (no sentence-transformers): CLS pooling for bge, mean pooling otherwise.
Run with a venv that has torch+transformers:  ~/kvpress-attempt/.venv/bin/python scripts/embed_items_hf.py BAAI/bge-small-en-v1.5"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd, glob
ROOT = Path(__file__).resolve().parents[1]
name = sys.argv[1] if len(sys.argv) > 1 else "BAAI/bge-small-en-v1.5"
frames = []
for b in sorted(glob.glob(str(ROOT / "data/hf") + "/*/")):
    f = glob.glob(f"{b}items.parquet")
    if f: frames.append(pd.read_parquet(f[0], columns=["item_id", "content"]))
items = pd.concat(frames).drop_duplicates("item_id")
txt = items["content"].fillna("").astype(str).str.slice(0, 6000).tolist()
import torch
from transformers import AutoTokenizer, AutoModel
dev = "mps" if torch.backends.mps.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(name); m = AutoModel.from_pretrained(name).to(dev).eval()
t0 = time.time(); out = []
with torch.no_grad():
    for i in range(0, len(txt), 32):
        enc = tok(txt[i:i+32], padding=True, truncation=True, max_length=512, return_tensors="pt").to(dev)
        h = m(**enc).last_hidden_state
        if "bge" in name.lower():
            e = h[:, 0]
        else:
            mask = enc["attention_mask"].unsqueeze(-1).float(); e = (h * mask).sum(1) / mask.sum(1)
        e = torch.nn.functional.normalize(e, dim=-1); out.append(e.float().cpu().numpy())
        if i % 3200 == 0: print(f"  {i}/{len(txt)} {time.time()-t0:.0f}s", flush=True)
E = np.concatenate(out)
tag = name.split("/")[-1]; outp = ROOT / "results" / (f"item_emb.npz" if tag == "bge-small-en-v1.5" else f"item_emb_{tag}.npz")
np.savez(outp, item_id=items["item_id"].values, emb=E.astype(np.float32))
print(f"embedded {len(txt)} items with {name} on {dev} in {time.time()-t0:.0f}s -> {outp}"); print("END")
