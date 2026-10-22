"""Zero-label item difficulty from a local open-weight LLM (MLX, Qwen3-VL-8B-Instruct-4bit used as a text model).

For a sample of items per public benchmark, ask for the probability (0-100) that a typical current AI system solves the item, then
compare with the item's empirical solve rate across the benchmark's systems (Spearman, within benchmark). If the correlation holds
up across benchmarks, the rating becomes an item-prior feature for the budget-0/1/3 predictions (where our label-only model has
nothing but the benchmark mean). Run with the ~/quantiphy venv (has mlx_vlm):
  ~/quantiphy/.venv/bin/python scripts/llm_difficulty.py 300
"""
import sys, os, re, time, glob, json, numpy as np, pandas as pd
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
N = int(sys.argv[1]) if len(sys.argv) > 1 else 300
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); OUT = os.path.join(ROOT, "results", "llm_difficulty.csv")
MODEL = "mlx-community/Qwen3-VL-8B-Instruct-4bit"
rows = []
for bdir in sorted(glob.glob(os.path.join(ROOT, "data/hf/*"))):
    b = os.path.basename(bdir); it = os.path.join(bdir, "items.parquet")
    rp = [p for p in (os.path.join(bdir, "responses.parquet"), os.path.join(bdir, "response.parquet")) if os.path.exists(p)]
    if not os.path.exists(it) or not rp:
        continue
    items = pd.read_parquet(it, columns=["item_id", "content"]); resp = pd.read_parquet(rp[0])
    ycol = next(c for c in ("score", "correct", "label", "is_correct", "value", "response") if c in resp.columns)
    acc = resp.groupby("item_id")[ycol].mean().rename("acc")
    items = items.merge(acc, on="item_id"); items = items.sample(min(N, len(items)), random_state=0)
    items["benchmark"] = b; rows.append(items)
df = pd.concat(rows, ignore_index=True)
done = pd.read_csv(OUT) if os.path.exists(OUT) else pd.DataFrame(columns=["item_id", "llm"])
todo = df[~df.item_id.isin(done.item_id)]
model, processor = load(MODEL); t0 = time.time(); out = done.to_dict("records")
for j, r in enumerate(todo.itertuples()):
    text = str(r.content)[:2500]
    q = ("Below is one task from an AI evaluation benchmark. Estimate the probability, from 0 to 100, that a typical current AI "
         "system (a mix of frontier and open models, with standard tools) solves it correctly. Reply with the number only.\n\n"
         "TASK:\n" + text)
    prompt = apply_chat_template(processor, model.config, q, num_images=0)
    try:
        o = generate(model, processor, prompt, [], max_tokens=6, verbose=False); txt = o.text if hasattr(o, "text") else str(o)
        m = re.findall(r"\d+(?:\.\d+)?", txt); v = float(m[0]) if m else np.nan
    except Exception as e:
        v = np.nan
    out.append({"item_id": r.item_id, "llm": v})
    if j % 25 == 0:
        pd.DataFrame(out).to_csv(OUT, index=False); print(f"{j}/{len(todo)} {time.time()-t0:.0f}s", flush=True)
pd.DataFrame(out).to_csv(OUT, index=False)
res = df.merge(pd.DataFrame(out), on="item_id")
from scipy.stats import spearmanr
for b, g in res.groupby("benchmark"):
    g = g.dropna(subset=["llm"]); rho = spearmanr(g.llm, g.acc).correlation if len(g) > 10 else np.nan
    print(f"{b:20s} n={len(g):4d} spearman(llm, solve rate) = {rho:+.3f}  mean acc {g.acc.mean():.2f}  mean llm {g.llm.mean():.1f}")
print("END")
