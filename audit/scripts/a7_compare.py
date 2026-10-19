"""Audit 7: compare key_numbers.json against the auditor's recomputations -> audit/results/comparison.csv + markdown."""
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "audit" / "results"
K = json.loads((ROOT / "outputs" / "key_numbers.json").read_text(encoding="utf-8"))
a1, a2, a3, a4, a5, a6 = (json.load(open(R / f)) for f in
    ["a1_clean_split.json", "a2_models.json", "a3_sim.json", "a4_value_critic.json", "a5_eda_quiz_text.json", "a6_shap.json"])
rows = []
def add(check, exp, got, tol, warn_tol=None, note=""):
    if isinstance(exp, (int, float)) and isinstance(got, (int, float)) and not isinstance(exp, bool):
        d = abs(exp - got)
        st = "PASS" if d <= tol else ("WARN" if warn_tol is not None and d <= warn_tol else "FAIL")
    else:
        st = "PASS" if exp == got else "FAIL"
    rows.append({"check": check, "expected": exp, "recomputed": round(got, 4) if isinstance(got, float) else got,
                 "tolerance": tol, "status": st, "note": note})

D = K["data"]; cs = {s["step"]: s["rows_remaining"] for s in D["cleaning_steps"]}
add("raw rows", D["raw_rows"], a1["raw_rows"], 0)
add("after exact-dup (excl. Unnamed: 0)", cs["b_drop_exact_duplicates"], a1["after_exact_dup"], 0)
add("after required fields", cs["c_require_valid_fields"], a1["after_required"], 0)
add("after dup normalized descriptions", cs["d_drop_duplicate_descriptions"], a1["after_desc_dup"], 0)
add("clean row-id set identical to own cleaning", True, a1["row_id_set_identical"], 0)
for s in ("train", "valid", "test"):
    add(f"split rows {s}", D["split_rows"][s], a1["split_rows"][s], 0)
    add(f"unique titles {s}", D["split_unique_titles"][s], a1["unique_titles"][s], 0)
for k, v in a1["title_overlap"].items():
    add(f"title_norm overlap {k}", 0, v, 0)
add("split reproduced from own cleaning (row mismatches)", 0, a1["split_reproduced_mismatch_rows"], 0)

E = K["eda"]
add("Pearson r (raw price)", E["pearson_r_price_points"], a5["pearson_raw"], 1e-4)
add("Pearson r (log price)", E["pearson_r_logprice_points"], a5["pearson_log"], 1e-4)
add("r^2 log price in-sample", E["r2_logprice_points_in_sample"], a5["r2_log_insample"], 1e-4)
add("Spearman rho", E["spearman_rho_price_points"], a5["spearman"], 1e-4)
add("share >= 90 (%)", E["share_ge90_pct"], a5["share_ge90_pct"], 1e-3)
for b in E["price_buckets"]:
    mb = a5["price_buckets"][b["price_bucket"]]
    add(f"bucket {b['price_bucket']} n", b["n"], mb["n"], 0)
    add(f"bucket {b['price_bucket']} median / share>=90", (b["median_points"], b["share_ge90_pct"]), (mb["median"], mb["share_ge90_pct"]), 0)
for rid in [r for r in ("15428", "37412") if r in K["quiz_pair"]]:
    q, m = K["quiz_pair"][rid], a5["quiz"][rid]
    add(f"quiz row {rid} (title, price, points)", (q["title"], q["price"], q["points"]), (m["title"], m["price"], m["points"]), 0)

fp = a2["from_predictions"]
for r in K["models"]["ladder"]:
    for sp_ in ("valid", "test"):
        for met, key in (("MAE", "MAE_points"), ("RMSE", "RMSE_points"), ("R2", "R2")):
            add(f"{r['model']} {sp_} {met} (from saved preds)", r[f"{sp_}_{key}"], fp[r["model"]][sp_][met], 1e-4)
for k in ("ridge_price", "ridge_meta", "ridge_meta_price"):
    t = next(r for r in K["models"]["ladder"] if r["model"] == k)
    add(f"{k} test MAE (own refit)", t["test_MAE_points"], a2["ridge_refit"][k]["test"]["MAE"], 0.005, 0.02)
    add(f"{k} test R2 (own refit)", t["test_R2"], a2["ridge_refit"][k]["test"]["R2"], 0.005, 0.02)
add("xgb_meta_price test MAE (own refit, seed 7)", K["models"]["test_MAE_main_points"], a2["xgb_refit"]["test"]["MAE"], 0.01, 0.03)
add("xgb_meta_price test R2 (own refit, seed 7)", K["models"]["test_R2_main"], a2["xgb_refit"]["test"]["R2"], 0.01, 0.03)
g = K["models"]["gap_main_vs_ridge_meta_price"]
add("test MAE gap ridge - main", g["test_MAE_ridge_minus_main_points"], a2["mae_gap_ridge_minus_main"]["diff"], 1e-4)
add("test MAE gap CI low (own seed)", g["test_MAE_ridge_minus_main_ci95"][0], a2["mae_gap_ridge_minus_main"]["ci95"][0], 0.005)
add("test MAE gap CI high (own seed)", g["test_MAE_ridge_minus_main_ci95"][1], a2["mae_gap_ridge_minus_main"]["ci95"][1], 0.005)

for B in ("20", "30", "50"):
    S = K["simulation"]["by_budget"][f"usd_{B}"]; m = a3["by_budget"][B]
    for s in ("Random", "MaxPrice", "Ridge", "Main"):
        ms = m["strategies"][s]
        suf = "_EV" if s == "Random" else "_saved_picks"
        add(f"sim ${B} {s} mean points", S["strategies"][s]["mean_points"], ms["mean_points" + suf], 1e-3)
        add(f"sim ${B} {s} share>=90 %", S["strategies"][s]["share_ge90_pct"], ms["share_ge90_pct" + suf], 1e-3)
        add(f"sim ${B} {s} mean price", S["strategies"][s]["mean_price_usd"], ms["mean_price" + suf], 1e-3)
        if s != "Random":
            add(f"sim ${B} {s} share>=90 % (tie-EV re-pick)", S["strategies"][s]["share_ge90_pct"], ms["share_ge90_pct_EV"], 1.0,
                note="expected value over tied candidates; differs only by tie-break luck")
    c = m["checks"]
    add(f"sim ${B} all candidates within budget", True, c["all_within_budget"], 0)
    add(f"sim ${B} every market 20 distinct titles", True, c["all_20_distinct_titles"] and c["all_size_20"], 0)
    for s in ("MaxPrice", "Ridge", "Main"):
        add(f"sim ${B} {s} saved pick is a valid argmax (of 1000)", 1000, m["saved_pick_in_my_candidate_set"][s], 0)
S30 = K["simulation"]["by_budget"]["usd_30"]["diffs"]; b30 = a3["by_budget"]["30"]["bootstrap_own_seed"]
for kk, mine in (("Main_minus_MaxPrice__ge90_pct_points", "Main-MaxPrice__ge90_pctp"), ("Ridge_minus_MaxPrice__ge90_pct_points", "Ridge-MaxPrice__ge90_pctp"),
                 ("Main_minus_MaxPrice__points_points", "Main-MaxPrice__points"), ("Ridge_minus_MaxPrice__points_points", "Ridge-MaxPrice__points"),
                 ("Main_minus_MaxPrice__price_usd", "Main-MaxPrice__price_usd")):
    tol = 0.5 if "ge90" in kk else 0.03
    add(f"$30 {kk} diff", S30[kk]["diff"], b30[mine]["diff"], 1e-3)
    add(f"$30 {kk} CI low (own seed)", S30[kk]["ci95_low"], b30[mine]["ci95"][0], tol)
    add(f"$30 {kk} CI high (own seed)", S30[kk]["ci95_high"], b30[mine]["ci95"][1], tol)

V = K["value_map"]
add("points per price doubling", V["points_per_price_doubling"], a4["slope_per_doubling"], 1e-3)
add("value map: n countries shown (n>=100)", V["n_countries_shown"], a4["country_levels_mine"], 0)
add("value map: n varieties shown (n>=100)", V["n_varieties_shown"], a4["variety_levels_mine"], 0)
for grp, col in (("country", "country"), ("variety", "variety")):
    for r in V[f"{grp}_top5"]:
        add(f"value {grp} {r[col]} mean residual", r["mean_residual_points"], a4[f"{grp}_top5"][r[col]], 1e-3)
    for r in V[f"{grp}_bottom5"]:
        add(f"value {grp} {r[col]} mean residual", r["mean_residual_points"], a4[f"{grp}_bottom5"][r[col]], 1e-3)
CC = K["critic_check"]
for c in ("Austria", "Germany", "Portugal", "Argentina", "Chile"):
    w = a4["within_critic"][c]
    add(f"critic {c} within-critic premium (builder def.)", CC[f"critic_{c.lower()}_within_critic_premium_points"], w["builder_def_premium(incl_self)"], 1e-3)
    add(f"critic {c} premium, 'vs OTHER countries' def. (as labelled)", CC[f"critic_{c.lower()}_within_critic_premium_points"],
        w["premium_vs_other_countries_only"], 0.05, 10, note="label says 'other countries'; code includes the country itself")
add("under-$20 list reproduced by pred-only ranking", True, a4["under20_reproduced_by_pred"], 0)
add("under-$20 top10 mean actual", V["under20_top10_mean_actual_points"], a4["under20_mean_actual"], 1e-3)
add("under-$20 top10 n>=90", V["under20_top10_n_ge90"], a4["under20_n_ge90"], 0)
add("under-$20 pool share>=90 %", V["under20_pool_share_ge90_pct"], a4["under20_pool_share_ge90_pct"], 1e-3)
add("under-$20 Austrian picks (HARNESS_PLAN v1.1 says 7/10)", 7, a4["under20_countries"].get("Austria"), 0, 10, note="plan text, not key_numbers")
SH = K["shap"]["feature_share_pct"]
for f in SH:
    add(f"SHAP share {f} (saved sample)", SH[f], a6["saved_sample_share_pct"][f], 1e-3)
    add(f"SHAP share {f} (own refit, own sample)", SH[f], a6["own_refit_share_pct"][f], 1.5)
T = K["text_aux_model"]
add("text model test R2 (meta+price+TF-IDF)", T["test_R2"], fp["text_ridge"]["test"]["R2"], 1e-4)

df = pd.DataFrame(rows)
df.to_csv(R / "comparison.csv", index=False, encoding="utf-8")
print(df.status.value_counts().to_string())
print(df[df.status != "PASS"].to_string(index=False))
with open(R / "comparison_table.md", "w", encoding="utf-8") as f:
    f.write("| check | expected (key_numbers) | recomputed | tolerance | status |\n|---|---|---|---|---|\n")
    for r in rows:
        e = r["expected"]; g_ = r["recomputed"]
        fmt = lambda x: (f"{x:.4f}" if isinstance(x, float) else str(x)).replace("|", "/")
        f.write(f"| {r['check']} | {fmt(e)} | {fmt(g_)} | {r['tolerance']} | {r['status']} |\n")
