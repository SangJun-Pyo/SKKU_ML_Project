"""P3b: value map ("가성비 지도") + under-$20 picks, on the test set.

Value = actual points - points predicted from price alone (Ridge on log_price, fit on train+valid).
Positive mean residual for a country/variety = it scores higher than its price would suggest.
Under-$20 picks are ranked by the MAIN model's predicted points; actual points are shown only after ranking.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CLEAN_CSV, MODEL_DIR, VALUE_DIR, ensure_dirs, setup_fonts  # noqa: E402

ensure_dirs()
setup_fonts()
MIN_N = 100
CHEAP_CAP = 20
TOP_K = 10

cfg = json.loads((MODEL_DIR / "model_config.json").read_text(encoding="utf-8"))
df = pd.read_csv(CLEAN_CSV)
trval = df[df["split"].isin(["train", "valid"])]
test = df[df["split"] == "test"].reset_index(drop=True)

# price-only model (same spec as the ladder's ridge_price), refit on train+valid
alpha = cfg["ridge_price_alpha"]
m = make_pipeline(StandardScaler(), Ridge(alpha=alpha)).fit(trval[["log_price"]], trval["points"])
test["pred_price_only"] = m.predict(test[["log_price"]])
test["residual"] = test["points"] - test["pred_price_only"]
preds = pd.read_csv(MODEL_DIR / "test_predictions.csv")
chk = test[["source_row_id", "pred_price_only"]].merge(preds[["source_row_id", "pred_ridge_price"]], on="source_row_id")
assert np.allclose(chk["pred_price_only"], chk["pred_ridge_price"], atol=1e-6), "price-only model mismatch vs P2"
slope_per_doubling = float(m[-1].coef_[0] / m[0].scale_[0] * np.log(2))


def value_table(col):
    rows = []
    for level, g in test.groupby(col):
        n = len(g)
        if n < MIN_N:
            continue
        r = g["residual"].to_numpy()
        mean, se = r.mean(), r.std(ddof=1) / np.sqrt(n)
        h = stats.t.ppf(0.975, n - 1) * se
        rows.append({col: level, "n": n, "mean_residual": mean, "ci95_low": mean - h, "ci95_high": mean + h,
                     "ci_excludes_zero": bool((mean - h) > 0 or (mean + h) < 0),
                     "median_price_usd": float(g["price"].median()), "mean_points": float(g["points"].mean()),
                     "mean_pred_price_only": float(g["pred_price_only"].mean())})
    t = pd.DataFrame(rows).sort_values(["mean_residual", col], ascending=[False, True]).reset_index(drop=True)
    t["rank_desc"] = np.arange(1, len(t) + 1)
    t["min_n"] = MIN_N
    return t


vc = value_table("country")
vv = value_table("variety")
vc.to_csv(VALUE_DIR / "value_by_country.csv", index=False, encoding="utf-8")
vv.to_csv(VALUE_DIR / "value_by_variety.csv", index=False, encoding="utf-8")

# under-$20 picks by main model
MAIN = cfg["main_model"]   # same key as 04_sim.py (single source: model_config.json)
main_col = f"pred_{MAIN}"
cheap = test[test["price"] <= CHEAP_CAP].merge(preds[["source_row_id", main_col]], on="source_row_id")
cheap = cheap.sort_values([main_col, "price", "title"], ascending=[False, True, True])
cheap = cheap.drop_duplicates("title_norm", keep="first").head(TOP_K).reset_index(drop=True)
top = pd.DataFrame({"rank": np.arange(1, len(cheap) + 1), "source_row_id": cheap["source_row_id"],
                    "title": cheap["title"], "country": cheap["country"], "province": cheap["province"],
                    "variety": cheap["variety"], "price_usd": cheap["price"],
                    "pred_points_main": cheap[main_col], "actual_points": cheap["points"],
                    "main_model": MAIN})
top.to_csv(VALUE_DIR / "under20_top10.csv", index=False, encoding="utf-8")

pool = test[test["price"] <= CHEAP_CAP]
with open(VALUE_DIR / "valuemap_config.json", "w", encoding="utf-8") as f:
    json.dump({"price_model": "StandardScaler + Ridge on log_price, fit on train+valid", "alpha": alpha,
               "points_per_price_doubling": slope_per_doubling, "min_n_test_rows": MIN_N,
               "ci": "t-interval on mean residual", "n_test_rows": int(len(test)),
               "under20_pool_rows": int(len(pool)), "under20_pool_share_ge90_pct": float(100 * (pool["points"] >= 90).mean()),
               "under20_pool_mean_points": float(pool["points"].mean()),
               "under20_top10_mean_actual_points": float(top["actual_points"].mean()),
               "under20_top10_n_ge90": int((top["actual_points"] >= 90).sum()),
               "under20_rank_rule": "main-model predicted points desc, then price asc, then title asc; "
                                    "one row per title_norm; actual points not used for ranking"},
              f, indent=2, ensure_ascii=False)

print(vc.head(8).to_string(index=False)); print(vc.tail(5).to_string(index=False))
print(vv.head(5).to_string(index=False)); print(vv.tail(5).to_string(index=False))
print(top.to_string(index=False))

# draft figure
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
for ax, t, col in ((axes[0], vc, "country"), (axes[1], vv, "variety")):
    show = pd.concat([t.head(8), t.tail(8)]).drop_duplicates(col) if len(t) > 16 else t
    show = show.iloc[::-1]
    y = np.arange(len(show))
    colors = ["#4C72B0" if v > 0 else "#DD8452" for v in show["mean_residual"]]
    ax.barh(y, show["mean_residual"], color=colors,
            xerr=[show["mean_residual"] - show["ci95_low"], show["ci95_high"] - show["mean_residual"]],
            error_kw=dict(lw=0.8, capsize=2))
    ax.set_yticks(y, [f"{a} (n={n})" for a, n in zip(show[col], show["n"])], fontsize=8)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("mean (actual - price-predicted) points")
    ax.set_title(f"Value by {col} (test, n >= {MIN_N})")
fig.tight_layout(); fig.savefig(VALUE_DIR / "fig_valuemap_draft.png", dpi=150); plt.close(fig)
