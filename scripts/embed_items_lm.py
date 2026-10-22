"""Hidden-state features of a small causal LM for the public items (feasibility probe for a cross-benchmark difficulty prior).

For each item of the five public benchmarks with >= 3 systems, run the LM on the item text (first 512 + last 128 tokens) and keep the
mean-pooled hidden state of a middle and the last layer, the last-token state of the last layer, and the mean token NLL.
Memory-safe (7 Oct, after an out-of-memory reboot): batches are capped at TOKENS_PER_BATCH tokens, the base model is run without the
LM head, the NLL is computed chunk-wise (never a full [batch x seq x vocab] logits tensor), the MPS cache is emptied after every batch,
and the run must use PYTORCH_MPS_HIGH_WATERMARK_RATIO (hard cap on Metal allocations). Peak RSS is printed; Metal memory is not in RSS,
so watch `footprint -p PID` (IOAccelerator) from outside.
Local files only (no download). Run with a venv that has torch + transformers, e.g.
  nice -n 10 env OMP_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4 HF_HUB_OFFLINE=1 PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.13 \
      PYTORCH_MPS_LOW_WATERMARK_RATIO=0.1 ~/kvpress-attempt/.venv/bin/python scripts/embed_items_lm.py Qwen/Qwen2.5-Coder-1.5B
"""
import glob, resource, sys, time
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
name = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-Coder-1.5B"
HEAD, TAIL, TOKENS_PER_BATCH, CHUNK = 512, 128, 1024, 128
frames = []
for b in sorted(glob.glob(str(ROOT / "data/hf") + "/*/")):
    if "swe_rebench" in b:
        continue
    f = glob.glob(f"{b}items.parquet")
    if f:
        frames.append(pd.read_parquet(f[0], columns=["item_id", "content"]))
items = pd.concat(frames).drop_duplicates("item_id").reset_index(drop=True)
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
dev = "mps" if torch.backends.mps.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(name, local_files_only=True)
m = AutoModelForCausalLM.from_pretrained(name, local_files_only=True, dtype=torch.float16, device_map=dev,
                                         low_cpu_mem_usage=True).eval()   # straight to the device: no CPU copy of the weights
base, head = m.model, m.lm_head
L = m.config.num_hidden_layers; mid = L // 2
ids = []
for t in items["content"].fillna("").astype(str).tolist():
    x = tok(t, add_special_tokens=False)["input_ids"] or [tok.eos_token_id]
    ids.append(x if len(x) <= HEAD + TAIL else x[:HEAD] + x[-TAIL:])
order = np.argsort([len(x) for x in ids])
H = m.config.hidden_size
F_mid = np.zeros((len(ids), H), np.float32); F_last = np.zeros_like(F_mid); F_tok = np.zeros_like(F_mid)
nll = np.zeros(len(ids), np.float32)
t0 = time.time(); s = 0; nb = 0
with torch.no_grad():
    while s < len(order):
        mx = len(ids[order[s]]); e = s
        while e < len(order) and (e - s + 1) * max(mx, len(ids[order[e]])) <= TOKENS_PER_BATCH:
            mx = max(mx, len(ids[order[e]])); e += 1
        e = max(e, s + 1); idx = order[s:e]; mx = max(len(ids[j]) for j in idx)
        inp = torch.full((len(idx), mx), tok.pad_token_id or 0, dtype=torch.long); att = torch.zeros((len(idx), mx), dtype=torch.long)
        for r, j in enumerate(idx):
            inp[r, :len(ids[j])] = torch.tensor(ids[j]); att[r, :len(ids[j])] = 1
        inp, att = inp.to(dev), att.to(dev)
        out = base(input_ids=inp, attention_mask=att, output_hidden_states=True)
        hs = out.hidden_states; a = att.unsqueeze(-1).float()
        hm = (hs[mid].float() * a).sum(1) / a.sum(1); hl = (hs[L].float() * a).sum(1) / a.sum(1)
        last = att.sum(1) - 1
        ht = hs[L].float()[torch.arange(len(idx), device=dev), last]
        tot = torch.zeros(len(idx), device=dev); cnt = torch.zeros(len(idx), device=dev)
        hid = out.last_hidden_state[:, :-1].reshape(-1, H); tgt = inp[:, 1:].reshape(-1); msk = att[:, 1:].reshape(-1).float()
        row_of = torch.arange(len(idx), device=dev).repeat_interleave(mx - 1)
        for c in range(0, hid.shape[0], CHUNK):   # at most CHUNK rows of vocabulary logits at a time
            lg = head(hid[c:c + CHUNK]).float()
            lp = lg.gather(-1, tgt[c:c + CHUNK, None])[:, 0] - torch.logsumexp(lg, -1)
            tot.index_add_(0, row_of[c:c + CHUNK], -lp * msk[c:c + CHUNK]); cnt.index_add_(0, row_of[c:c + CHUNK], msk[c:c + CHUNK])
            del lg, lp
        F_mid[idx] = hm.cpu().numpy(); F_last[idx] = hl.cpu().numpy(); F_tok[idx] = ht.cpu().numpy()
        nll[idx] = (tot / cnt.clamp(min=1)).cpu().numpy()
        del out, hs
        if dev == "mps":
            torch.mps.empty_cache()   # the MPS caching allocator otherwise keeps one buffer set per sequence shape (23 GB on 7 Oct)
        s = e; nb += 1
        if nb % 100 == 0:
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 30
            print(f"  {s}/{len(ids)} {time.time() - t0:.0f}s peak RSS {rss:.2f} GB", flush=True)
tag = name.split("/")[-1]
np.savez(ROOT / "results" / f"item_lm_{tag}.npz", item_id=items["item_id"].values, mid=F_mid, last=F_last, tok=F_tok, nll=nll,
         ntok=np.array([len(x) for x in ids]))
print(f"done {len(ids)} items with {name} on {dev} in {time.time() - t0:.0f}s; peak RSS "
      f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 30:.2f} GB"); print("END")
