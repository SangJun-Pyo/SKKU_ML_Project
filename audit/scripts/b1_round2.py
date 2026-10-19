"""Audit round 2: classification AUCs + own L2-logistic refit, Logistic/XGBClf simulation re-picks, Lasso/ENet
metrics + zero counts, raw profile, new quiz pair. Independent code; reads only data/ and outputs/."""
import json, time
from pathlib import Path
import numpy as np, pandas as pd, scipy.sparse as sp
from sklearn.linear_model import LogisticRegression, Lasso, ElasticNet, Ridge
from sklearn.metrics import roc_auc_score, log_loss

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "audit" / "results"
K = json.loads((ROOT / "outputs" / "key_numbers.json").read_text(encoding="utf-8"))
res = {}
def auc(y, s):   # own Mann-Whitney AUC (ties = 0.5), cross-checked against sklearn
    r = pd.Series(s).rank().to_numpy(); n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))

# ---------------- 1. classification AUCs from saved predictions
CP = pd.read_csv(ROOT / "outputs" / "models" / "classify_test_predictions.csv")
y = CP.y_ge90.to_numpy()
res["y_matches_points"] = bool((y == (CP.points >= 90)).all())
res["test_base_rate_pct"] = float(100 * y.mean())
res["auc_saved"] = {c: auc(y, CP[c].to_numpy()) for c in CP.columns if c.startswith(("prob_", "score_"))}
res["auc_saved_sklearn_check"] = {c: float(roc_auc_score(y, CP[c])) for c in res["auc_saved"]}
TP = pd.read_csv(ROOT / "outputs" / "models" / "test_predictions.csv")
assert (TP.source_row_id.to_numpy() == CP.source_row_id.to_numpy()).all()
res["auc_ridge_meta_price_score"] = auc(y, TP.pred_ridge_meta_price.to_numpy())
res["auc_ridge_price_score"] = auc(y, TP.pred_ridge_price.to_numpy())
res["logloss_saved"] = {c: float(log_loss(y, np.clip(CP[c], 1e-15, 1 - 1e-15))) for c in CP.columns if c.startswith("prob_")}

# paired bootstrap AUC differences, own seed
rng = np.random.default_rng(777)
pairs = {"xgb_clf-logit_all": ("prob_xgb_clf", "prob_logit_all"), "xgb_clf-xgb_reg": ("prob_xgb_clf", "score_xgb_reg"),
         "logit_all-ridge_meta_price": ("prob_logit_all", None)}
cols = {k: CP[k].to_numpy() for k in ("prob_xgb_clf", "prob_logit_all", "score_xgb_reg")}
cols[None] = TP.pred_ridge_meta_price.to_numpy()
bs = {k: [] for k in pairs}
for _ in range(2000):
    ix = rng.integers(0, len(y), len(y))
    for k, (a, b) in pairs.items():
        bs[k].append(roc_auc_score(y[ix], cols[a][ix]) - roc_auc_score(y[ix], cols[b][ix]))
res["auc_diff_own_boot"] = {k: {"diff": float(roc_auc_score(y, cols[a]) - roc_auc_score(y, cols[b])),
                                "ci95": [float(np.percentile(bs[k], 2.5)), float(np.percentile(bs[k], 97.5))]}
                            for k, (a, b) in pairs.items()}

# own L2 logistic refit (own encoder; C chosen on valid log loss; refit train+valid; test once)
df = pd.read_csv(ROOT / "data" / "processed" / "wine_clean.csv")
tr, va, te = (df[df.split == s].reset_index(drop=True) for s in ("train", "valid", "test"))
tv = pd.concat([tr, va], ignore_index=True)
def build(fit, app):
    F, A = [], []
    for c in ["country", "province", "variety"]:
        vc = fit[c].value_counts(); keep = sorted(vc[vc >= 20].index); idx = {l: i for i, l in enumerate(keep)}
        for d, L in ((fit, F), (app, A)):
            j = d[c].map(idx).fillna(len(keep)).astype(int).to_numpy()
            L.append(sp.csr_matrix((np.ones(len(d)), (np.arange(len(d)), j)), shape=(len(d), len(keep) + 1)))
    med = fit.vintage.median()
    N = [np.column_stack([d.log_price, d.vintage.fillna(med), d.vintage.isna()]).astype(float) for d in (fit, app)]
    mu, sd = N[0].mean(0), N[0].std(0)
    F.append(sp.csr_matrix((N[0] - mu) / sd)); A.append(sp.csr_matrix((N[1] - mu) / sd))
    return sp.hstack(F).tocsr(), sp.hstack(A).tocsr()
ytr, yva, yte, ytv = ((d.points >= 90).astype(int).to_numpy() for d in (tr, va, te, tv))
Xf, Xa = build(tr, va)
grid = {}
for C in [0.01, 0.1, 1.0, 10.0]:
    p = LogisticRegression(C=C, max_iter=5000).fit(Xf, ytr).predict_proba(Xa)[:, 1]
    grid[C] = float(log_loss(yva, p))
Cb = min(grid, key=grid.get)
Xf, Xa = build(tv, te)
p = LogisticRegression(C=Cb, max_iter=5000).fit(Xf, ytv).predict_proba(Xa)[:, 1]
res["logit_l2_refit"] = {"valid_logloss_grid": grid, "C_chosen": Cb, "test_auc": auc(yte, p),
                         "test_logloss": float(log_loss(yte, p)), "corr_with_saved": float(np.corrcoef(p, CP.prob_logit_all)[0, 1])}

# ---------------- 2. simulation re-picks for Logistic and XGBClf
M = pd.read_csv(ROOT / "outputs" / "sim" / "markets.csv")
CH = pd.read_csv(ROOT / "outputs" / "sim" / "market_choices.csv")
J0 = M.merge(CP[["source_row_id", "title_norm", "price", "points", "prob_logit_all", "prob_xgb_clf"]], on="source_row_id", how="left")
J0 = J0.merge(TP[["source_row_id", "pred_xgb_meta_price"]], on="source_row_id", how="left")
res["sim"] = {}
for B, J in J0.groupby("budget"):
    out = {"within_budget": bool((J.price <= B).all()), "distinct20": bool((J.groupby("market_id").title_norm.nunique() == 20).all())}
    picks = {}
    for s, col in (("Logistic", "prob_logit_all"), ("XGBClf", "prob_xgb_clf"), ("Main", "pred_xgb_meta_price")):
        ev, valid, real, ties = [], 0, [], 0
        sv = CH[(CH.budget == B) & (CH.strategy == s)].set_index("market_id")
        for mid, mm in J.groupby("market_id"):
            pr, pt, sc = mm.price.to_numpy(), mm.points.to_numpy(), mm[col].to_numpy()
            msk = sc == sc.max(); msk &= pr == pr[msk].min(); ties += int(msk.sum() > 1)
            ev.append((pt[msk] >= 90).mean())
            valid += int(int(sv.loc[mid, "source_row_id"]) in set(mm.source_row_id[msk]))
            real.append(float(sv.loc[mid, "ge90"]))
        picks[s] = np.array(real)
        out[s] = {"share_ge90_pct_EV": 100 * float(np.mean(ev)), "share_ge90_pct_saved": 100 * float(np.mean(real)),
                  "key_numbers": K["simulation"]["by_budget"][f"usd_{B}"]["strategies"][s]["share_ge90_pct"],
                  "saved_pick_valid_argmax": valid, "markets_with_ties": ties}
    if B == 30:
        d = 100 * (picks["Main"] - picks["Logistic"])
        ix = np.random.default_rng(31337).integers(0, len(d), (10000, len(d)))
        out["Main_minus_Logistic_ge90"] = {"diff": float(d.mean()), "ci95": [float(np.percentile(d[ix].mean(1), 2.5)), float(np.percentile(d[ix].mean(1), 97.5))],
                                           "key_numbers": K["simulation"]["by_budget"]["usd_30"]["diffs"]["Main_minus_Logistic__ge90_pct_points"]}
    res["sim"][int(B)] = out

# ---------------- 3. Lasso / ElasticNet
VP = pd.read_csv(ROOT / "outputs" / "models" / "valid_predictions.csv")
def mets(yv, p):
    e = yv - p; return {"MAE": float(np.abs(e).mean()), "R2": float(1 - (e ** 2).sum() / ((yv - yv.mean()) ** 2).sum())}
res["sparse_from_preds"] = {k: {"test": mets(TP.points.to_numpy(float), TP[f"pred_{k}"].to_numpy()),
                                "valid": mets(VP.points.to_numpy(float), VP[f"pred_{k}"].to_numpy())}
                            for k in ("lasso_meta_price", "enet_meta_price", "ridge_meta_price")}
t = time.time()
Xf, Xa = build(tv, te)   # same design: one-hot (unscaled) + standardized numerics
nonehot = Xf.shape[1] - 3
for k, mdl in (("lasso", Lasso(alpha=1e-4, max_iter=50000)), ("enet", ElasticNet(alpha=3e-4, l1_ratio=0.2, max_iter=50000))):
    mdl.fit(Xf, tv.points.to_numpy(float)); c = mdl.coef_
    res[f"{k}_refit"] = {"n_coef": int(len(c)), "n_onehot": int(nonehot), "n_zero": int((c == 0).sum()),
                         "n_abs_lt_1e-3": int((np.abs(c) < 1e-3).sum()), "test": mets(te.points.to_numpy(float), mdl.predict(Xa))}
# sensitivity: zero count at even smaller alpha (grid floor dependence)
for a in (3e-5, 1e-5):
    c = Lasso(alpha=a, max_iter=50000).fit(Xf, tv.points.to_numpy(float)).coef_
    res[f"lasso_alpha_{a}_n_zero"] = int((c == 0).sum())
res["lasso_seconds"] = round(time.time() - t, 1)
# train vs test gap for Ridge meta+price (overfitting check for claim c)
r = Ridge(alpha=10).fit(Xf, tv.points); res["ridge_trainvalid_vs_test"] = {"fit_rows": mets(tv.points.to_numpy(float), r.predict(Xf)),
                                                                            "test": mets(te.points.to_numpy(float), r.predict(Xa)),
                                                                            "n_over_p": float(Xf.shape[0] / Xf.shape[1])}

# ---------------- 4. raw profile + quiz
raw = pd.read_csv(ROOT / "data" / "raw" / "winemag-data-130k-v2.csv")
res["raw_missing_pct"] = {c: round(float(100 * raw[c].isna().mean()), 2) for c in
                          ["price", "designation", "region_1", "region_2", "taster_name", "taster_twitter_handle", "country", "province", "variety"]}
prof = pd.read_csv(ROOT / "outputs" / "eda" / "raw_profile.csv").set_index("column")["missing_pct"].to_dict()
res["raw_profile_csv"] = {c: prof[c] for c in res["raw_missing_pct"]}
d_ = raw[~raw.drop(columns=["Unnamed: 0"]).duplicated()]
res["missing_pct_after_dedup"] = {c: round(float(100 * d_[c].isna().mean()), 2) for c in ["price", "taster_name"]}
q = {}
for rid in (28150, 26694):
    rr = raw[raw["Unnamed: 0"] == rid].iloc[0]
    q[rid] = {"title": rr.title, "price": float(rr.price), "points": int(rr.points),
              "taster": None if pd.isna(rr.taster_name) else rr.taster_name, "in_clean": bool(rid in set(df.source_row_id)),
              "raw_rows_same_title": int((raw.title == rr.title).sum()),
              "split": df.loc[df.source_row_id == rid, "split"].iloc[0] if rid in set(df.source_row_id) else None}
res["quiz"] = q
json.dump(res, open(OUT / "b1_round2.json", "w"), indent=1, ensure_ascii=False, default=str)
print(json.dumps(res, indent=1, ensure_ascii=False, default=str))
