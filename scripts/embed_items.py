"""Embed every item's content with a small open encoder (bge-small-en-v1.5, 33M params, Apache-2.0) -> results/item_emb.npz."""
import sys, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from paie.data import load_all
model_name = sys.argv[1] if len(sys.argv) > 1 else "BAAI/bge-small-en-v1.5"
_, _, items = load_all(); items = items.drop_duplicates("item_id")
txt = items["content"].fillna("").astype(str).str.slice(0, 4000).tolist()
from sentence_transformers import SentenceTransformer
import torch
dev = "mps" if torch.backends.mps.is_available() else "cpu"
m = SentenceTransformer(model_name, device=dev); m.max_seq_length = 512
t0 = time.time()
E = m.encode(txt, batch_size=64, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True)
out = ROOT / "results" / ("item_emb.npz" if "bge-small" in model_name else f"item_emb_{model_name.split('/')[-1]}.npz")
np.savez(out, item_id=items["item_id"].values, emb=E.astype(np.float32))
print(f"embedded {len(txt)} items with {model_name} on {dev} in {time.time()-t0:.0f}s -> {out}"); print("END")
