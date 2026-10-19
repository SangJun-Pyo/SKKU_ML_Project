"""P3c: critic (taster) confounding check for the value map. Descriptive only.

- critic_summary.csv / country_critic_share.csv: all cleaned rows.
- within_critic_check.csv, within_critic_cells.csv: TEST rows with the same price-only residual as the
  value map (residual = points - pred_ridge_price from P2, identical to 05_valuemap's price-only model).
Within-critic premium for a country = n-weighted average over critics of
  [critic's mean residual in that country - the same critic's mean residual on their OTHER qualifying
   countries (leave-one-out: the target country is excluded from the critic's baseline)],
using only critics with >= CELL_MIN_N test rows in 2+ value-map countries. Missing taster -> "Unknown",
excluded from the within-critic analysis (it is a mix of critics).
CI: bootstrap over test rows only. With 1-2 critics per country it ignores critic-level uncertainty and
critic x country interaction, so it is narrower than any "controlled for critic" reading would need.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CLEAN_CSV, CRITIC_DIR, MODEL_DIR, SEED, VALUE_DIR, ensure_dirs  # noqa: E402

ensure_dirs()
CELL_MIN_N = 30          # min test rows for a critic x country cell to count as "reviews that country"
N_BOOT = 1000
CI_NOTE = ("row-bootstrap CI over test rows; ignores critic-level uncertainty (only 1-2 critics per country) "
           "and critic x country interaction, so it is too narrow for a 'controlled for critic' claim")

df = pd.read_csv(CLEAN_CSV)
df["taster"] = df["taster_name"].fillna("Unknown")

# ---------------------------------------------------------------- 1. critic summary (all rows)
rows = []
for t, g in df.groupby("taster"):
    vc = g["country"].value_counts()
    rows.append({"taster_name": t, "n": len(g), "mean_points": g["points"].mean(),
                 "median_price_usd": g["price"].median(), "n_countries": int(vc.size),
                 "top_country": vc.index[0], "top_country_share_pct": 100 * vc.iloc[0] / len(g)})
cs = pd.DataFrame(rows).sort_values(["n", "taster_name"], ascending=[False, True])
cs.to_csv(CRITIC_DIR / "critic_summary.csv", index=False, encoding="utf-8")
critic_mean = cs.set_index("taster_name")["mean_points"]

# ---------------------------------------------------------------- 2. top critic share per value-map country (all rows)
vmap = pd.read_csv(VALUE_DIR / "value_by_country.csv")
countries = list(vmap["country"])
rows = []
for c in countries:
    g = df[df["country"] == c]
    vc = g["taster"].value_counts()
    named = vc.drop("Unknown", errors="ignore")
    rows.append({"country": c, "n_all_rows": len(g), "top_critic": vc.index[0],
                 "top_critic_share_pct": 100 * vc.iloc[0] / len(g),
                 "top_critic_overall_mean_points": critic_mean[vc.index[0]],
                 "top_named_critic": named.index[0], "top_named_critic_share_pct": 100 * named.iloc[0] / len(g),
                 "top_named_critic_overall_mean_points": critic_mean[named.index[0]],
                 "unknown_share_pct": 100 * vc.get("Unknown", 0) / len(g),
                 "value_map_mean_residual": float(vmap.loc[vmap["country"] == c, "mean_residual"].iloc[0])})
pd.DataFrame(rows).to_csv(CRITIC_DIR / "country_critic_share.csv", index=False, encoding="utf-8")

# ---------------------------------------------------------------- 3. within-critic check (test rows)
pred = pd.read_csv(MODEL_DIR / "test_predictions.csv")[["source_row_id", "pred_ridge_price"]]
test = df[df["split"] == "test"].merge(pred, on="source_row_id")
test["residual"] = test["points"] - test["pred_ridge_price"]
named = test[(test["taster"] != "Unknown") & test["country"].isin(countries)]
cell_n = named.groupby(["taster", "country"]).size()
qual_cells = cell_n[cell_n >= CELL_MIN_N]
multi = qual_cells.groupby(level=0).size()
multi_critics = sorted(multi[multi >= 2].index)
qual = set(qual_cells.index)
r = named[[(t, c) in qual for t, c in zip(named["taster"], named["country"])]]
r = r[r["taster"].isin(multi_critics)].reset_index(drop=True)

cell_codes, cell_keys = pd.factorize(r["taster"] + "||" + r["country"], sort=True)
cells = [tuple(k.split("||")) for k in cell_keys]
crit_codes, crits = pd.factorize(r["taster"], sort=True)
cell_crit = np.array([list(crits).index(t) for t, _ in cells])
cell_ctry = np.array([c for _, c in cells])
resid = r["residual"].to_numpy()


def cell_deviation(idx):
    """Per cell: mean residual minus the same critic's mean residual on their OTHER qualifying countries."""
    n_cell = np.bincount(cell_codes[idx], minlength=len(cells))
    s_cell = np.bincount(cell_codes[idx], weights=resid[idx], minlength=len(cells))
    n_crit = np.bincount(crit_codes[idx], minlength=len(crits))[cell_crit]
    s_crit = np.bincount(crit_codes[idx], weights=resid[idx], minlength=len(crits))[cell_crit]
    n_oth = n_crit - n_cell
    with np.errstate(invalid="ignore", divide="ignore"):
        dev = s_cell / n_cell - (s_crit - s_cell) / n_oth
    return dev, n_cell


def premiums(idx):
    """country -> n-weighted mean of leave-one-out deviations for a row index sample."""
    dev, n_cell = cell_deviation(idx)
    out = {}
    for c in np.unique(cell_ctry):
        m = cell_ctry == c
        ok = m & np.isfinite(dev)
        out[c] = float(np.sum(dev[ok] * n_cell[ok]) / np.sum(n_cell[ok])) if ok.any() else np.nan
    return out


point = premiums(np.arange(len(r)))
rng = np.random.default_rng(SEED)
boot = {c: [] for c in point}
for _ in range(N_BOOT):
    b = premiums(rng.integers(0, len(r), len(r)))
    for c in point:
        boot[c].append(b.get(c, np.nan))

rows = []
for c in countries:
    base = {"country": c, "n_test": int(vmap.loc[vmap["country"] == c, "n"].iloc[0]),
            "raw_mean_residual": float(vmap.loc[vmap["country"] == c, "mean_residual"].iloc[0])}
    if c in point:
        sub = r[r["country"] == c]
        cr = sorted(sub["taster"].unique())
        rows.append({**base, "estimable": True, "n_rows_within": len(sub), "n_critics": len(cr),
                     "critics": "; ".join(cr), "mean_residual_overlap_critics": sub["residual"].mean(),
                     "within_critic_premium": point[c],
                     "ci95_low": float(np.nanpercentile(boot[c], 2.5)),
                     "ci95_high": float(np.nanpercentile(boot[c], 97.5)),
                     "note": "premium vs the same critics' own average on their other qualifying countries "
                             "(leave-one-out)", "ci_note": CI_NOTE})
    else:
        rows.append({**base, "estimable": False, "n_rows_within": 0, "n_critics": 0, "critics": "",
                     "mean_residual_overlap_critics": np.nan, "within_critic_premium": np.nan,
                     "ci95_low": np.nan, "ci95_high": np.nan,
                     "note": f"not estimable: no named critic has >= {CELL_MIN_N} test rows in this country "
                             f"AND in another value-map country", "ci_note": ""})
wc = pd.DataFrame(rows)
wc["cell_min_n"] = CELL_MIN_N
wc["n_boot"] = N_BOOT
wc["multi_country_critics"] = "; ".join(multi_critics)
wc.to_csv(CRITIC_DIR / "within_critic_check.csv", index=False, encoding="utf-8")
all_idx = np.arange(len(r))
dev_all, n_all = cell_deviation(all_idx)
s_all = np.bincount(cell_codes, weights=resid, minlength=len(cells))
n_crit_all = np.bincount(crit_codes, minlength=len(crits))[cell_crit]
s_crit_all = np.bincount(crit_codes, weights=resid, minlength=len(crits))[cell_crit]
pd.DataFrame({"taster": [t for t, _ in cells], "country": cell_ctry, "n": n_all,
              "mean_residual": s_all / n_all, "n_other_countries_rows": n_crit_all - n_all,
              "critic_mean_residual_other_countries": (s_crit_all - s_all) / (n_crit_all - n_all),
              "within_critic_deviation": dev_all}).to_csv(
    CRITIC_DIR / "within_critic_cells.csv", index=False, encoding="utf-8")

# A taster fixed-effect regression (residual ~ country + taster dummies) was tried and dropped: country and
# critic are nearly collinear, and e.g. Schachner's countries (Argentina/Chile/Spain) connect to the rest only
# through a handful of rows (5 in France, 15 in Italy), so the country coefficients were unstable artefacts.
# The within-critic comparison above (cells with >= CELL_MIN_N rows only) is the reported check.

print(wc[["country", "raw_mean_residual", "estimable", "critics", "within_critic_premium", "ci95_low",
          "ci95_high"]].to_string(index=False))
