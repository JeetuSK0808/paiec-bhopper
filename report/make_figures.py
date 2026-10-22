# figure for the technical report: (a) Brier by label budget, (b) expected best-of-N practice-board score
# every number is parsed from the hidden-calibrated simulator log, nothing typed in by hand except the board observations
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
LOG = next(p for p in (ROOT / "results_logs/v7/log_hsplit_boot.txt", ROOT / "src/paie_exp/v7/log_hsplit_boot.txt")
           if p.exists())
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "report/fig_curves.pdf"

txt = LOG.read_text()
curves, best = {}, {}
for line in txt.splitlines():
    m = re.match(r"\s+(\S+)\s+(lone|shared)\s+pairs\s+\d+\s+ALC\s+([\d.]+)\s+B0\.\.B31\s+(.*)$", line)
    if m:
        curves[(m.group(1), m.group(2))] = [float(v) for v in m.group(4).split()]
    m = re.match(r"\s+(\S+)\s+sample ALC mean.*expected best-of-N: (.*)$", line)
    if m:
        best[m.group(1)] = {int(k): float(v) for k, v in re.findall(r"N=(\d+):([\d.]+)", m.group(2))}

# practice board, read on 7 Oct 2026 (Codabench 17828)
BOARD_1, BOARD_24, OURS, UPLOADS_PER_TEAM = 0.117, (0.148, 0.152), 0.1845, 37

# categorical slots in fixed order (validated: adjacent CVD dE >= 9.1, normal-vision >= 19.6 on a light surface)
C = {"v6": "#2a78d6", "emp": "#eb6834", "oracle": "#1baf7a", "beta": "#eda100", "v6s": "#e87ba4"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#dcdbd7"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"], "font.size": 8,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.6, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelcolor": INK, "ytick.labelcolor": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
    "legend.frameon": False, "pdf.fonttype": 42,
})
fig, (a, b) = plt.subplots(1, 2, figsize=(5.5, 2.35), constrained_layout=True)

budgets = [0, 1, 3, 7, 15, 31]
x = list(range(len(budgets)))
series_a = [
    ("emp", ("emp_mean", "lone"), "organisers' empirical mean", "o"),
    ("beta", ("beta1", "lone"), "Beta(1,1) posterior mean", "D"),
    ("v6", ("submission_v6", "lone"), "v6, system alone on its benchmark", "o"),
    ("v6s", ("submission_v6", "shared"), "v6, benchmark shared by 2 systems", "s"),
    ("oracle", ("oracle", "lone"), "oracle constant (true accuracy)", "^"),
]
for key, ck, label, mk in series_a:
    y = curves[ck]
    a.plot(x, y, color=C[key], lw=1.4, marker=mk, ms=4.2, mec="white", mew=0.7, label=label,
           zorder=3 if key != "beta" else 2)
a.set_xticks(x, [str(k) for k in budgets])
a.set_xlabel("revealed labels per pair (budget)")
a.set_ylabel("Brier score (lower is better)")
a.set_title("(a) Brier score by label budget", fontsize=8.5, color=INK, loc="left")
a.legend(fontsize=6.3, loc="upper right", handlelength=1.8, borderaxespad=0.2)
a.set_ylim(0.15, 0.335)

Ns = [1, 6, 20, 40, 100]
for key, mk, name, label in [("emp", "o", "emp_mean", "organisers' mean"),
                             ("v6", "o", "submission_v6", "v6"),
                             ("oracle", "^", "oracle", "oracle constant")]:
    b.plot(Ns, [best[name][n] for n in Ns], color=C[key], lw=1.4, marker=mk, ms=4.2, mec="white", mew=0.7,
           label=label, zorder=3)
b.axhspan(*BOARD_24, color=MUTED, alpha=0.13, lw=0, zorder=1)
b.text(11, sum(BOARD_24) / 2, "board #2–#4", fontsize=6.3, color=MUTED, va="center")
b.axhline(BOARD_1, color=MUTED, lw=0.8, zorder=1)
b.text(1.05, BOARD_1 + 0.001, "board #1 (0.117)", fontsize=6.3, color=MUTED, va="bottom")
b.axvline(UPLOADS_PER_TEAM, color=MUTED, lw=0.8, zorder=1)
b.text(UPLOADS_PER_TEAM * 1.07, BOARD_1 + 0.001, "~37 uploads/team", fontsize=6.3, color=MUTED, va="bottom")
b.plot([6], [OURS], marker="*", ms=7, color=INK, mec="white", mew=0.6, ls="none", zorder=4, label="ours (best of 6 uploads)")
b.set_xscale("log")
b.set_xticks(Ns, [str(n) for n in Ns])
b.minorticks_off()
b.set_xlabel("uploads by a team (board keeps the best)")
b.set_ylabel("expected best ALC")
b.set_title("(b) What the practice board rewards", fontsize=8.5, color=INK, loc="left")
leg = b.legend(fontsize=6.3, loc="upper right", handlelength=1.8, borderaxespad=0.3, frameon=True, facecolor="white", edgecolor="none", framealpha=1.0)
leg.set_zorder(5)
b.set_ylim(0.11, 0.226)

fig.savefig(OUT)
fig.savefig(OUT.with_suffix(".png"), dpi=220)
print("wrote", OUT)
