import sys, time, numpy as np, pandas as pd
sys.path.insert(0, "src"); sys.path.insert(0, sys.argv[2] if len(sys.argv) > 2 else "submission_v2")
from paie.data import load_all
import model as M
held = sys.argv[1] if len(sys.argv) > 1 else "matharena"
pairs, subs, items = load_all(); items = items.drop_duplicates("item_id"); subs = subs.drop_duplicates("subject_id")
test = pairs[pairs.benchmark == held]
itxt = dict(zip(items.item_id, items.content.fillna("").astype(str)))
FIELDS = ["normalized_name","provider","release_date","access_date","harness","harness_version","reasoning_effort","subject_features_extra"]
sdict = {r["subject_id"]: {f: str(r.get(f, "") if pd.notna(r.get(f, "")) else "") for f in FIELDS} | {"_sid": r["subject_id"]} for _, r in subs.iterrows()}
ifeat = dict(zip(items.item_id, items.item_features.fillna("").astype(str)))
def item_dict(i): return {"item_content": itxt[i], "benchmark_id": held, "item_features": ifeat[i], "interactors": ""}
rng = np.random.default_rng(0); pools = []
for sid, g in test.groupby("subject_id"):
    if len(g) < 80: continue
    g = g.sample(frac=1.0, random_state=int(rng.integers(1e9))); h = len(g)//2; pools.append((sid, g.iloc[:h], g.iloc[h:]))
B = {}; t0 = time.time()
for k in (0, 1, 3, 7, 15, 31):
    labeled = [[[sdict[sid], item_dict(i)], int(y)] for sid, acq, _ in pools for i, y in zip(acq.item_id.values[:k], acq.y.values[:k])]
    bs = []
    for sid, acq, ev in pools:
        p = np.array([M.predict([sdict[sid], item_dict(i)], labeled) for i in ev.item_id.values])
        bs.append(np.mean((p - ev.y.values) ** 2))
    B[k] = float(np.mean(bs)); print(f"{held} k={k}: Brier {B[k]:.4f}  ({time.time()-t0:.0f}s)", flush=True)
W = {0: .1, 1: .2, 3: .2, 7: .2, 15: .2, 31: .1}; print(held, "ALC", round(sum(W[k]*B[k] for k in B), 4))
