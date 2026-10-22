# Team BHopper — Predictive AI Evaluation Competition (NeurIPS 2026)

Code and technical report for team **BHopper** (Satyajeeth Suresh Kannan) in the
[Predictive AI Evaluation Competition](https://aims.stanford.edu/competition) run by Stanford's AIMS lab.

The task: predict the probability that an AI system answers an item of a never-seen benchmark correctly, with 0, 1, 3, 7, 15
or 31 of its outcomes on that benchmark revealed. Scored by Brier ALC (lower is better).

**Selected submission: v6**, Codabench submission #949120 (`submission_v6/`, packaged as `submissions/paie_v6.zip`, SHA-256
`45c6596a2631b4ed714916f0da35a2461b8e53c910867f816ee6b55bda7aea08`). The report is `report/paiec_bhopper_report.pdf`.

## What's here

| Path | What it is |
|---|---|
| `submission_v6/` | The selected submission: `model.py` (hierarchical Bayesian logistic model, numpy only), `textprior.py`, `subjprior.py`, `subject_prior.json` (per-model relative strengths computed from the public training pool) |
| `submission_v7/` | v6 plus a stateless coordinated-acquisition `labeling.py` (tested, not selected; see Section 6 of the report) |
| `submissions/` | The two zips exactly as built for Codabench |
| `src/paie/` | Shared data loading and prior-building code |
| `scripts/` | Every experiment: the leave-one-benchmark-out harness, the faithful streaming simulator, the hidden-calibrated simulator, transfer probes, robustness checks, packaging |
| `results_logs/` | The logs behind the numbers in the report (aggregates over simulated or held-out data) |
| `report/` | LaTeX source (NeurIPS 2026 template), the figure script and the PDF |

## Setup

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

The submission itself needs only `numpy`. The experiments also need `pandas` and `pyarrow`; the language-model probes need
`torch` and `transformers`; the figure needs `matplotlib`.

**Data.** The public training pool is the gated Hugging Face dataset
[`aims-foundations/measurement-db`](https://huggingface.co/datasets/aims-foundations/measurement-db). Accept its terms, then:

```python
from huggingface_hub import snapshot_download
snapshot_download("aims-foundations/measurement-db", repo_type="dataset", local_dir="data/hf")
```

No data is redistributed here.

**Organisers' kit.** `check_submission_zip.py` comes from the competition's starting kit (Codabench competition 17828). Put it in
`upstream/paiec_baseline/` to run the packaging check.

## Reproduce

All commands run from the repository root with
`nice -n 10 env VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -B` in front.

```bash
# oracle Brier scores on the public data (report Table 1)
scripts/oracle_table.py
# v6 in the three model-selection tools (report Table 4)
scripts/exp_synth_v6cv.py --sub submission_v6 --regimes single,mix2,mix8,pooled --seeds 0,1,2
scripts/exp_synth_sim.py --sub submission_v6 --regimes practice --episodes 20 --prior fold --fast
scripts/exp_synth_sim.py --sub submission_v6 --regimes k1,k2,k4,k8 --episodes 12 --prior fold --fast
scripts/exp_synth_hiddensim.py --episodes 60 --baselines --cfgs '[["v6", {"sub": "submission_v6"}]]'
# lone vs shared pairs, the Bayes limit and best-of-N (Tables 5-6, Figure 1)
scripts/exp_v7_hsplit.py --episodes 60 --seeds 0,1,2
scripts/exp_v7_hsplit.py --boot --episodes 200 --seeds 0,1,2
python report/make_figures.py report/fig_curves.pdf
# coordinated acquisition (v7)
scripts/exp_v7_acq.py --regime practice --episodes 60 --seeds 0,1,2 --policies random,skipco,skipseen
scripts/exp_v7_sim.py --sub submission_v7 --regimes practice --episodes 20 --prior fold --fast --seed 0
scripts/exp_v7_robust_labeling.py submission_v7
# transfer probes (Table 7)
scripts/exp_v7_xfer_emb.py
scripts/embed_items_lm.py Qwen/Qwen2.5-Coder-1.5B
# packaging check
python upstream/paiec_baseline/check_submission_zip.py submissions/paie_v6.zip
```

**One file is not included.** The hidden-calibrated simulator (`scripts/exp_synth_hidden_feedback.py`,
`exp_synth_hiddensim.py`, `exp_v7_hsplit.py`, `exp_v7_acq.py`) reads `private/hidden_feedback.json`. That table is derived from our
own Codabench scoring logs of the hidden test (per-pair accuracies of the 26 hidden pairs we were scored on). The competition rules
forbid sharing private test data, so it is not published; the organisers can have it on request. Without it those scripts stop with
a clear message. Everything else runs on the public data alone.

## Disclosure

- Training data: `measurement-db` only (the public training pool).
- Pretrained models: none in the submission. BAAI/bge-small-en-v1.5, BAAI/bge-base-en-v1.5, Qwen/Qwen2.5-Coder-1.5B and
  Qwen3-VL-8B-Instruct (4-bit) were used only in negative-result experiments.
- Formative feedback: only our own three scoring logs, used only to set one global prior width and build the synthetic validation set.
- AI assistance: an LLM-based coding assistant (Claude Code) was used for code, experiments and drafting. All numbers come from the
  scripts here.

## Licence

MIT (see `LICENSE`). The NeurIPS style file in `report/` belongs to the NeurIPS Foundation and is included only to build the PDF.
