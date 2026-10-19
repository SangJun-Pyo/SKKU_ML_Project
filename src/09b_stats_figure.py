"""Render the statsmodels OLS summary (points ~ log_price) as a slide image.

Reads outputs/models/stats_ols_price_summary.txt (written by 03c_stats_ols.py), drops the
Date/Time lines, highlights the rows the slide talks about, and saves
outputs/figures_ko/fig12_ols_summary.png (2240 x 1400 px, shown at 1120 x 700 on the slide).
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import MODEL_DIR, ROOT, ensure_dirs  # noqa: E402

ensure_dirs()
OUT = ROOT / "outputs" / "figures_ko"
OUT.mkdir(parents=True, exist_ok=True)

lines = (MODEL_DIR / "stats_ols_price_summary.txt").read_text(encoding="utf-8").splitlines()
keep = [ln for ln in lines if not ln.startswith(("Date:", "Time:"))]
keep = [ln for ln in keep if ln.strip()]                     # drop blank lines
keep = [ln for ln in keep if not ln.startswith(("Notes:", "[1]"))]
HIGHLIGHT = ("R-squared", "log_price", "Prob (F-statistic)", "Durbin-Watson", "Skew:", "Kurtosis:")

W, H, DPI = 2240, 1400, 200
fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI)
fig.patch.set_facecolor("#fdfbf8")
ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
n = len(keep)
top, bottom = 0.965, 0.035
step = (top - bottom) / (n - 1)
for i, ln in enumerate(keep):
    y = top - i * step
    hit = any(h in ln for h in HIGHLIGHT)
    if hit:
        ax.add_patch(plt.Rectangle((0.015, y - step * 0.5), 0.97, step * 0.95,
                                   transform=ax.transAxes, color="#f1e3e6", lw=0))
    ax.text(0.03, y, ln, transform=ax.transAxes, family="DejaVu Sans Mono", fontsize=15.2,
            va="center", ha="left", color="#7b2335" if hit else "#231a1d",
            fontweight="bold" if hit else "normal")
fig.savefig(OUT / "fig12_ols_summary.png", dpi=DPI, facecolor=fig.get_facecolor())
print("saved outputs/figures_ko/fig12_ols_summary.png", n, "lines")
