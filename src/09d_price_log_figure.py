"""Why log price: price distribution before vs after the log transform (cleaning slide).

Saves outputs/figures_ko/fig15_price_log.png (1632 x 776 px, shown at 816 x 388 on the slide).
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CLEAN_CSV, ROOT, ensure_dirs, setup_fonts  # noqa: E402

ensure_dirs()
setup_fonts()
OUT = ROOT / "outputs" / "figures_ko"
d = pd.read_csv(CLEAN_CSV, usecols=["price", "log_price"])
skew_raw, skew_log = d["price"].skew(), d["log_price"].skew()
cut = 300
share_over = 100 * (d["price"] > cut).mean()

fig, axes = plt.subplots(1, 2, figsize=(8.16, 3.88), dpi=200)
fig.patch.set_facecolor("#fdfbf8")
for ax in axes:
    ax.set_facecolor("#fdfbf8")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#d9d0c4")
    ax.set_yticks([])
    ax.tick_params(axis="x", labelsize=15, colors="#5b5257")

axes[0].hist(d["price"].clip(upper=cut), bins=40, color="#c9a3aa")
axes[0].set_title(f"원래 가격 (왜도 {skew_raw:.1f})", fontsize=18, color="#231a1d", loc="left", pad=8)
axes[0].set_xticks([0, 100, 200, 300]); axes[0].set_xticklabels(["$0", "$100", "$200", f"${cut}+"])
axes[0].text(0.97, 0.9, f"${cut} 초과 {share_over:.1f}%가\n$3,300까지 긴 꼬리", transform=axes[0].transAxes,
             ha="right", va="top", fontsize=14, color="#5b5257")

axes[1].hist(d["log_price"], bins=28, color="#7b2335")
axes[1].set_title(f"로그 가격 (왜도 {skew_log:.1f})", fontsize=18, color="#7b2335", loc="left", pad=8,
                  fontweight="bold")
ticks = [5, 20, 100, 500, 3000]
axes[1].set_xticks(np.log(ticks)); axes[1].set_xticklabels([f"${t:,}" for t in ticks])
fig.subplots_adjust(left=0.03, right=0.98, top=0.84, bottom=0.14, wspace=0.12)
fig.savefig(OUT / "fig15_price_log.png", dpi=200, facecolor=fig.get_facecolor())
print("saved outputs/figures_ko/fig15_price_log.png", round(skew_raw, 2), round(skew_log, 2), round(share_over, 2))
