"""P1b: descriptive EDA on the full cleaned data (all splits)."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CLEAN_CSV, EDA_DIR, ensure_dirs, setup_fonts  # noqa: E402

ensure_dirs()
setup_fonts()
df = pd.read_csv(CLEAN_CSV)
review_words = df["description"].str.split().str.len()   # whitespace-separated tokens

# ---- summary
summary = {
    "pearson_review_words_points": float(stats.pearsonr(review_words, df["points"])[0]),
    "review_words_mean": float(review_words.mean()), "review_words_median": float(review_words.median()),
    "n_rows": int(len(df)),
    "pearson_price_points": float(stats.pearsonr(df["price"], df["points"])[0]),
    "pearson_logprice_points": float(stats.pearsonr(df["log_price"], df["points"])[0]),
    "spearman_price_points": float(stats.spearmanr(df["price"], df["points"])[0]),
    "price_min": float(df["price"].min()), "price_median": float(df["price"].median()),
    "price_max": float(df["price"].max()),
    "points_min": int(df["points"].min()), "points_median": float(df["points"].median()),
    "points_max": int(df["points"].max()),
    "share_points_ge_90": float((df["points"] >= 90).mean()),
    "vintage_nonmissing_share": float(df["vintage"].notna().mean()),
    "n_countries": int(df["country"].nunique()), "n_varieties": int(df["variety"].nunique()),
    "n_provinces": int(df["province"].nunique()),
}
with open(EDA_DIR / "eda_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

# ---- price buckets
edges = [0, 10, 20, 30, 50, 100, np.inf]
labels = ["<=$10", "($10,$20]", "($20,$30]", "($30,$50]", "($50,$100]", ">$100"]
df["price_bucket"] = pd.cut(df["price"], bins=edges, labels=labels, right=True)
bk = (df.groupby("price_bucket", observed=False)["points"]
      .agg(n="size", median_points="median", mean_points="mean",
           share_ge_90=lambda s: (s >= 90).mean()).reset_index())
bk.to_csv(EDA_DIR / "price_buckets.csv", index=False, encoding="utf-8")

# ---- country / variety
for col, fn in (("country", "country_summary.csv"), ("variety", "variety_summary.csv")):
    t = (df.groupby(col).agg(n=("points", "size"), median_price=("price", "median"),
                             mean_points=("points", "mean")).reset_index()
         .sort_values(["n", col], ascending=[False, True]))
    t.to_csv(EDA_DIR / fn, index=False, encoding="utf-8")

# ---- quiz pair (opening quiz: Opus One 2006 vs Montes Alpha Cabernet Sauvignon 2006)
QUIZ_IDS = (28150, 26694)
quiz = {}
fields = ["title", "country", "province", "variety", "price", "points", "taster_name", "description"]
for rid in QUIZ_IDS:
    r = df[df["source_row_id"] == rid]
    if r.empty:
        quiz[str(rid)] = None
        continue
    row = r.iloc[0]
    quiz[str(rid)] = {k: (row[k].item() if hasattr(row[k], "item") else (None if pd.isna(row[k]) else row[k])) for k in fields}
    quiz[str(rid)]["split"] = row["split"]
if all(quiz[str(k)] for k in QUIZ_IDS):
    a, b = quiz[str(QUIZ_IDS[0])], quiz[str(QUIZ_IDS[1])]
    quiz["_check"] = {"prices": [a["price"], b["price"]], "points": [a["points"], b["points"]],
                      "price_ratio": round(a["price"] / b["price"], 2),
                      "cheaper_scores_higher": b["price"] < a["price"] and b["points"] > a["points"]}
else:
    quiz["_check"] = {"missing_after_cleaning": [str(k) for k in QUIZ_IDS if not quiz[str(k)]]}
with open(EDA_DIR / "quiz_pair.json", "w", encoding="utf-8") as f:
    json.dump(quiz, f, indent=2, ensure_ascii=False)

# ---- draft figures
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(df["points"], bins=np.arange(79.5, 100.6, 1), color="#4C72B0", edgecolor="white")
ax.set_xlabel("points"); ax.set_ylabel("wines"); ax.set_title("Points distribution")
fig.tight_layout(); fig.savefig(EDA_DIR / "fig_points_hist.png", dpi=150); plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 5))
hb = ax.hexbin(df["price"], df["points"], xscale="log", gridsize=45, mincnt=1, bins="log", cmap="viridis")
fig.colorbar(hb, ax=ax, label="wines (log count)")
ax.set_xlabel("price (USD, log scale)"); ax.set_ylabel("points")
ax.set_title(f"Price vs points (Pearson r on log price = {summary['pearson_logprice_points']:.2f})")
fig.tight_layout(); fig.savefig(EDA_DIR / "fig_logprice_vs_points_hexbin.png", dpi=150); plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(bk["price_bucket"].astype(str), bk["median_points"], color="#4C72B0")
for i, (m, n) in enumerate(zip(bk["median_points"], bk["n"])):
    ax.text(i, m + 0.1, f"{m:.0f}\n(n={n:,})", ha="center", va="bottom", fontsize=8)
ax.set_ylim(80, max(bk["median_points"]) + 2)
ax.set_xlabel("price bucket"); ax.set_ylabel("median points"); ax.set_title("Median points by price bucket")
fig.tight_layout(); fig.savefig(EDA_DIR / "fig_price_bucket_points.png", dpi=150); plt.close(fig)

print(json.dumps(summary, indent=2)); print(bk.to_string(index=False))
print(json.dumps(quiz, indent=2, ensure_ascii=False)[:2500])
