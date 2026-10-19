"""Audit 4: value map from my own price-only fit; within-critic numbers (builder definition + 'other countries only');
under-$20 ranking rule."""
import json
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "audit" / "results"
df = pd.read_csv(ROOT / "data" / "processed" / "wine_clean.csv")
tv = df[df.split.isin(["train", "valid"])]
te = df[df.split == "test"].copy()
# OLS on log price (Ridge alpha=0.01 on 89k rows is numerically OLS)
b1, b0 = np.polyfit(tv.log_price, tv.points, 1)
te["resid"] = te.points - (b0 + b1 * te.log_price)
res = {"slope_per_doubling": float(b1 * np.log(2)), "intercept": float(b0)}
P = pd.read_csv(ROOT / "outputs" / "models" / "test_predictions.csv")
mm = te.merge(P[["source_row_id", "pred_ridge_price", "pred_xgb_meta_price"]], on="source_row_id")
res["max_abs_diff_vs_saved_price_only_pred"] = float(np.abs((b0 + b1 * mm.log_price) - mm.pred_ridge_price).max())

def tab(col):
    g = te.groupby(col)["resid"].agg(["size", "mean"]).rename(columns={"size": "n"})
    return g[g.n >= 100].sort_values("mean", ascending=False)
for col in ("country", "variety"):
    mine = tab(col)
    theirs = pd.read_csv(ROOT / "outputs" / "valuemap" / f"value_by_{col}.csv").set_index(col)
    j = mine.join(theirs[["n", "mean_residual"]], rsuffix="_b", how="outer")
    res[f"{col}_levels_mine"] = int(len(mine)); res[f"{col}_levels_builder"] = int(len(theirs))
    res[f"{col}_max_abs_mean_diff"] = float((j["mean"] - j["mean_residual"]).abs().max())
    res[f"{col}_n_mismatch"] = int((j["n"] != j["n_b"]).sum())
    res[f"{col}_top5"] = {k: round(v, 4) for k, v in mine["mean"].head(5).items()}
    res[f"{col}_bottom5"] = {k: round(v, 4) for k, v in mine["mean"].tail(5)[::-1].items()}

# ---- within-critic
te["taster"] = te.taster_name.fillna("Unknown")
countries = list(tab("country").index)
named = te[(te.taster != "Unknown") & te.country.isin(countries)]
cell = named.groupby(["taster", "country"])["resid"].agg(["size", "mean"])
q = cell[cell["size"] >= 30]
nq = q.groupby(level=0).size()
q = q[q.index.get_level_values(0).isin(nq[nq >= 2].index)]
res["multi_country_critics"] = sorted(set(q.index.get_level_values(0)))
crit_mean_all = (q["size"] * q["mean"]).groupby(level=0).sum() / q["size"].groupby(level=0).sum()
out = {}
for c in ["Austria", "Germany", "Portugal", "Argentina", "Chile", "Spain", "France", "US"]:
    rows = q[q.index.get_level_values(1) == c]
    if rows.empty:
        out[c] = None; continue
    dev_incl, dev_excl, w, per = [], [], [], {}
    for (t, _), r in rows.iterrows():
        others = q.loc[t].drop(c)
        excl = float((others["size"] * others["mean"]).sum() / others["size"].sum())
        dev_incl.append(r["mean"] - crit_mean_all[t]); dev_excl.append(r["mean"] - excl); w.append(r["size"])
        per[t] = {"n": int(r["size"]), "cell_mean_resid": round(float(r["mean"]), 4),
                  "critic_other_countries": {k: int(v) for k, v in others["size"].items()},
                  "premium_vs_others_only": round(float(r["mean"] - excl), 4)}
    w = np.array(w)
    out[c] = {"raw_resid_all_test_rows": round(float(te.loc[te.country == c, "resid"].mean()), 4),
              "builder_def_premium(incl_self)": round(float(np.dot(dev_incl, w) / w.sum()), 4),
              "premium_vs_other_countries_only": round(float(np.dot(dev_excl, w) / w.sum()), 4),
              "per_critic": per}
res["within_critic"] = out
# zero-sum property of the builder definition: n-weighted devs sum to 0 within each critic
zs = {}
for t in crit_mean_all.index:
    s = q.loc[t]
    zs[t] = round(float((s["size"] * (s["mean"] - crit_mean_all[t])).sum()), 10)
res["builder_def_zero_sum_within_critic"] = zs

# ---- under-$20 list
u = pd.read_csv(ROOT / "outputs" / "valuemap" / "under20_top10.csv")
ch = mm[mm.price <= 20].sort_values(["pred_xgb_meta_price", "price", "title"], ascending=[False, True, True]).drop_duplicates("title_norm").head(10)
res["under20_reproduced_by_pred"] = list(ch.source_row_id) == list(u.source_row_id)
res["under20_pred_monotone_desc"] = bool((np.diff(u.pred_points_main) <= 0).all())
by_actual = mm[mm.price <= 20].sort_values(["points", "pred_xgb_meta_price"], ascending=False).head(10)
res["under20_overlap_with_actual_top10"] = int(len(set(by_actual.source_row_id) & set(u.source_row_id)))
res["under20_mean_actual"] = float(u.actual_points.mean()); res["under20_n_ge90"] = int((u.actual_points >= 90).sum())
res["under20_countries"] = u.country.value_counts().to_dict()
pool = te[te.price <= 20]
res["under20_pool_rows"] = int(len(pool)); res["under20_pool_share_ge90_pct"] = float(100 * (pool.points >= 90).mean())
res["under20_pool_mean_points"] = float(pool.points.mean())
# how lucky is 10/10? share >=90 among top-50 by prediction
top50 = mm[mm.price <= 20].sort_values("pred_xgb_meta_price", ascending=False).drop_duplicates("title_norm").head(50)
res["under20_top50_share_ge90_pct"] = float(100 * (top50.points >= 90).mean())

json.dump(res, open(OUT / "a4_value_critic.json", "w"), indent=2, ensure_ascii=False)
print(json.dumps(res, indent=1, ensure_ascii=False))
