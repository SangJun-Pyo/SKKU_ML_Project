"""Over-fitting chart: train / valid / test R² per regression model (from 03d_overfit_check.py).

Saves outputs/figures_ko/fig14_overfit.png (2368 x 1332 px, shown at 1184 x 666 on the slide).
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import MODEL_DIR, ROOT, ensure_dirs, setup_fonts  # noqa: E402

ensure_dirs()
setup_fonts()
OUT = ROOT / "outputs" / "figures_ko"
d = pd.read_csv(MODEL_DIR / "overfit_check.csv")
d = d[d["task"] == "regression"].reset_index(drop=True)

C = {"train": "#d8b9be", "valid": "#b8893a", "test": "#7b2335"}
NAME = {"train": "학습", "valid": "검증", "test": "시험"}
fig, ax = plt.subplots(figsize=(11.84, 6.66), dpi=200)
fig.patch.set_facecolor("#fdfbf8"); ax.set_facecolor("#fdfbf8")
H = 0.26
n = len(d)
for i, (_, r) in enumerate(d.iterrows()):
    y0 = n - 1 - i
    for k, key in enumerate(("train", "valid", "test")):
        v = r[f"{key}_R2"]
        y = y0 + (1 - k) * H
        ax.barh(y, v, height=H * 0.9, color=C[key], label=NAME[key] if i == 0 else None)
        ax.text(v + 0.006, y, f"{v:.2f}", va="center", fontsize=14, color="#231a1d",
                fontweight="bold" if key == "test" else "normal")
    gap = r.train_R2 - r.valid_R2
    ax.text(0.665, y0, f"학습−검증\n{gap:+.2f}", va="center", ha="center", fontsize=15,
            color="#7b2335" if gap > 0.05 else "#5b5257", fontweight="bold" if gap > 0.05 else "normal")
ax.set_yticks(range(n)); ax.set_yticklabels(list(d["label"])[::-1], fontsize=17)
ax.set_xlim(0, 0.70); ax.set_ylim(-0.6, n - 0.4)
ax.set_xlabel("설명력 R² (높을수록 좋음)", fontsize=15, color="#5b5257")
ax.tick_params(axis="x", labelsize=14, colors="#5b5257"); ax.tick_params(axis="y", length=0)
for s_ in ("top", "right", "left"):
    ax.spines[s_].set_visible(False)
ax.spines["bottom"].set_color("#d9d0c4")
ax.legend(loc="upper center", bbox_to_anchor=(0.45, 1.09), fontsize=15, frameon=False, ncol=3)
fig.subplots_adjust(left=0.2, right=0.98, top=0.9, bottom=0.12)
fig.savefig(OUT / "fig14_overfit.png", dpi=200, facecolor=fig.get_facecolor())
print("saved outputs/figures_ko/fig14_overfit.png")
