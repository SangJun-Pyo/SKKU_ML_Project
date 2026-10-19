"""P3a: budget selection simulation on the test set, using saved test predictions only.

For each budget cap B: eligible = test rows with price <= B, deduped to one row per title_norm (seeded).
1,000 markets of 20 distinct titles each (seeded per budget). Each strategy picks one wine per market.
Paired bootstrap over markets (10,000 resamples) gives 95% CIs for strategy differences.

Note: markets are re-sampled from ONE fixed test pool, so the CIs describe market-sampling variability
for this pool, not a guarantee about the wine population.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import MODEL_DIR, SEED, SIM_DIR, ensure_dirs, setup_fonts  # noqa: E402

ensure_dirs()
setup_fonts()

BUDGETS = [20, 30, 50]
N_MARKETS = 1000
MARKET_SIZE = 20
N_BOOT = 10000
BOOT_CHUNK = 1000
CI_NOTE = ("Markets are re-sampled from one fixed test pool; the CI reflects market-sampling variability "
           "only, not uncertainty about the wider wine population or a different test sample.")

cfg = json.loads((MODEL_DIR / "model_config.json").read_text(encoding="utf-8"))
pred = pd.read_csv(MODEL_DIR / "test_predictions.csv")
clf = pd.read_csv(MODEL_DIR / "classify_test_predictions.csv")[["source_row_id", "prob_logit_all", "prob_xgb_clf"]]
n0 = len(pred)
pred = pred.merge(clf, on="source_row_id", how="left", validate="one_to_one")   # left merge keeps row order
assert len(pred) == n0 and pred[["prob_logit_all", "prob_xgb_clf"]].notna().all().all()
MAIN = cfg["main_model"]   # same key as 05_valuemap.py (single source: model_config.json)
STRATS = {"MaxPrice": None, "Ridge": "pred_ridge_meta_price", "Main": f"pred_{MAIN}",
          "Logistic": "prob_logit_all", "XGBClf": "prob_xgb_clf"}
STRAT_ORDER = ["Random", "MaxPrice", "Ridge", "Main", "Logistic", "XGBClf"]
STRAT_LABEL = {"Random": "Random pick (expected value over the 20)",
               "MaxPrice": "Most expensive in the market",
               "Ridge": "Highest predicted points: Ridge meta+price",
               "Main": f"Highest predicted points: {MAIN}",
               "Logistic": "Highest P(points >= 90): logistic regression, all main features (03b)",
               "XGBClf": "Highest P(points >= 90): XGBoost classifier (03b)"}
COMPARISONS = [("Main", "MaxPrice"), ("Main", "Ridge"), ("Main", "Random"), ("Ridge", "MaxPrice"),
               ("MaxPrice", "Random"),
               # classification strategies (added after the original five; markets/seeds/tie draws unchanged)
               ("Main", "Logistic"), ("Logistic", "MaxPrice"), ("Logistic", "Ridge"), ("XGBClf", "Logistic"),
               ("Main", "XGBClf"), ("XGBClf", "MaxPrice")]

summary_rows, ci_rows, choice_rows, market_rows = [], [], [], []
for B in BUDGETS:
    elig = pred[pred["price"] <= B]
    pool = (elig.sample(frac=1.0, random_state=SEED + B).drop_duplicates("title_norm", keep="first")
            .sort_values("source_row_id").reset_index(drop=True))
    rng = np.random.default_rng([SEED, B])
    mk = np.stack([rng.choice(len(pool), size=MARKET_SIZE, replace=False) for _ in range(N_MARKETS)])
    tb_max = rng.random(mk.shape)
    tb_model = rng.random(mk.shape)
    price = pool["price"].to_numpy()[mk]
    points = pool["points"].to_numpy()[mk]
    rid = pool["source_row_id"].to_numpy()[mk]

    per_market = {}  # strategy -> dict(points, ge90, price) arrays over markets
    per_market["Random"] = {"points": points.mean(axis=1), "ge90": (points >= 90).mean(axis=1),
                            "price": price.mean(axis=1)}
    for s, col in STRATS.items():
        pick = np.empty(N_MARKETS, dtype=int)
        for m in range(N_MARKETS):
            if col is None:   # max price, ties -> random
                order = np.lexsort((tb_max[m], -price[m]))
            else:             # max predicted points / probability, ties -> lower price -> random
                p = pool[col].to_numpy()[mk[m]]
                order = np.lexsort((tb_model[m], price[m], -p))
            pick[m] = order[0]
        ar = np.arange(N_MARKETS)
        per_market[s] = {"points": points[ar, pick].astype(float), "ge90": (points[ar, pick] >= 90).astype(float),
                         "price": price[ar, pick], "rid": rid[ar, pick]}

    for m in range(N_MARKETS):
        for slot in range(MARKET_SIZE):
            market_rows.append((B, m, slot, int(rid[m, slot])))
    for s in STRAT_ORDER:
        d = per_market[s]
        for m in range(N_MARKETS):
            choice_rows.append({"budget": B, "market_id": m, "strategy": s,
                                "source_row_id": int(d["rid"][m]) if "rid" in d else pd.NA,
                                "price": float(d["price"][m]), "points": float(d["points"][m]),
                                "ge90": float(d["ge90"][m])})
        summary_rows.append({"budget": B, "strategy": s, "strategy_label": STRAT_LABEL[s],
                             "n_markets": N_MARKETS, "market_size": MARKET_SIZE,
                             "pool_rows_le_budget": int(len(elig)), "pool_unique_titles": int(len(pool)),
                             "mean_points": float(d["points"].mean()),
                             "share_ge90_pct": float(100 * d["ge90"].mean()),
                             "mean_price_usd": float(d["price"].mean())})

    brng = np.random.default_rng([SEED, B, 1])
    boot_idx = [brng.integers(0, N_MARKETS, size=(BOOT_CHUNK, N_MARKETS)) for _ in range(N_BOOT // BOOT_CHUNK)]
    for a, b in COMPARISONS:
        for metric, scale, unit in (("ge90", 100.0, "pct_points"), ("points", 1.0, "points"),
                                    ("price", 1.0, "usd")):
            diff = scale * (per_market[a][metric] - per_market[b][metric])
            bs = np.concatenate([diff[ix].mean(axis=1) for ix in boot_idx])
            ci_rows.append({"budget": B, "comparison": f"{a} - {b}", "metric": metric, "unit": unit,
                            "diff": float(diff.mean()), "ci95_low": float(np.percentile(bs, 2.5)),
                            "ci95_high": float(np.percentile(bs, 97.5)),
                            "share_markets_a_better_pct": float(100 * (diff > 0).mean()),
                            "share_markets_tie_pct": float(100 * (diff == 0).mean()),
                            "n_boot": N_BOOT, "note": CI_NOTE})

summ = pd.DataFrame(summary_rows)
summ.to_csv(SIM_DIR / "sim_summary.csv", index=False, encoding="utf-8")
cis = pd.DataFrame(ci_rows)
cis.to_csv(SIM_DIR / "sim_bootstrap_ci.csv", index=False, encoding="utf-8")
pd.DataFrame(choice_rows).to_csv(SIM_DIR / "market_choices.csv", index=False, encoding="utf-8")
pd.DataFrame(market_rows, columns=["budget", "market_id", "slot", "source_row_id"]).to_csv(
    SIM_DIR / "markets.csv", index=False, encoding="utf-8")
with open(SIM_DIR / "sim_config.json", "w", encoding="utf-8") as f:
    json.dump({"budgets_usd": BUDGETS, "n_markets": N_MARKETS, "market_size": MARKET_SIZE, "n_boot": N_BOOT,
               "seed": SEED, "main_model": MAIN, "ridge_model": "ridge_meta_price",
               "logistic_model": "logit_all (03b_classify.py)", "xgb_classifier_model": "xgb_clf (03b_classify.py)",
               "dedupe": "one row per title_norm per budget pool, chosen by seeded shuffle (random_state=SEED+B)",
               "market_rng": "numpy default_rng([SEED, B])", "bootstrap_rng": "numpy default_rng([SEED, B, 1])",
               "tie_rules": {"MaxPrice": "random (seeded)", "model": "lower price, then random (seeded)"},
               "random_strategy": "expected value over the 20 wines in the market (source_row_id empty in "
                                  "market_choices.csv; market composition in markets.csv)",
               "ci_note": CI_NOTE}, f, indent=2, ensure_ascii=False)

print(summ[["budget", "strategy", "pool_unique_titles", "mean_points", "share_ge90_pct", "mean_price_usd"]]
      .to_string(index=False))
print(cis[cis["metric"] == "ge90"][["budget", "comparison", "diff", "ci95_low", "ci95_high"]].to_string(index=False))

# draft figure: share >= 90 by budget and strategy
fig, ax = plt.subplots(figsize=(8, 4.5))
order = ["Random", "MaxPrice", "Ridge", "Main"]
colors = ["#BBBBBB", "#DD8452", "#9DB4D8", "#4C72B0"]
w = 0.2
for i, (s, c) in enumerate(zip(order, colors)):
    v = summ[summ["strategy"] == s]
    xs = np.arange(len(BUDGETS)) + (i - 1.5) * w
    ax.bar(xs, v["share_ge90_pct"], width=w, color=c, label=s)
    for x, y in zip(xs, v["share_ge90_pct"]):
        ax.text(x, y + 0.8, f"{y:.0f}", ha="center", fontsize=7)
ax.set_xticks(np.arange(len(BUDGETS)), [f"<= ${b}" for b in BUDGETS])
ax.set_ylabel("share of picks >= 90 points (%)")
ax.set_title(f"Budget simulation: {N_MARKETS} markets x {MARKET_SIZE} wines (test set)")
ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(SIM_DIR / "fig_sim_share_ge90_draft.png", dpi=150); plt.close(fig)
