"""Audit 2: recompute metrics from saved predictions; independently refit Ridge ladder and main XGBoost."""
import json, time
from pathlib import Path
import numpy as np, pandas as pd, scipy.sparse as sp
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "audit" / "results"
K = json.loads((ROOT / "outputs" / "key_numbers.json").read_text(encoding="utf-8"))
cfg = json.loads((ROOT / "outputs" / "models" / "model_config.json").read_text(encoding="utf-8"))
res = {"from_predictions": {}}

def mets(y, p):
    e = y - p
    return {"MAE": float(np.mean(np.abs(e))), "RMSE": float(np.sqrt(np.mean(e ** 2))),
            "R2": float(1 - np.sum(e ** 2) / np.sum((y - y.mean()) ** 2))}

for split in ("valid", "test"):
    P = pd.read_csv(ROOT / "outputs" / "models" / f"{split}_predictions.csv")
    y = P["points"].to_numpy(float)
    for c in [c for c in P.columns if c.startswith("pred_")]:
        res["from_predictions"].setdefault(c[5:], {})[split] = mets(y, P[c].to_numpy(float))
    res[f"{split}_n"] = len(P)

# ---- independent feature builder
df = pd.read_csv(ROOT / "data" / "processed" / "wine_clean.csv")
tr, va, te = (df[df.split == s].reset_index(drop=True) for s in ("train", "valid", "test"))
tv = pd.concat([tr, va], ignore_index=True)
CATS = ["country", "province", "variety"]

def build(fit, app, use_price, use_meta, linear=True):
    mats_f, mats_a, names = [], [], []
    if use_meta:
        for c in CATS:
            vc = fit[c].value_counts()
            keep = sorted(vc[vc >= 20].index)
            levels = keep + ["__rare__"]
            idx = {l: i for i, l in enumerate(levels)}
            def enc(s):
                j = s.map(idx).fillna(len(keep)).astype(int).to_numpy()
                return sp.csr_matrix((np.ones(len(s), np.float32), (np.arange(len(s)), j)), shape=(len(s), len(levels)))
            ef, ea = enc(fit[c]), enc(app[c])
            # drop the rare column if nothing in fit is rare (mirrors 'infrequent only if exists')
            if (fit[c].map(idx).isna().sum()) == 0:
                ef, ea = ef[:, :-1], ea[:, :-1]; levels = levels[:-1]
            mats_f.append(ef); mats_a.append(ea); names += [f"{c}={l}" for l in levels]
        med = fit["vintage"].median()
        vf, va_ = fit["vintage"].fillna(med).to_numpy(), app["vintage"].fillna(med).to_numpy()
        mf, ma = fit["vintage"].isna().to_numpy(float), app["vintage"].isna().to_numpy(float)
        num_f, num_a = [vf, mf], [va_, ma]; names += ["vintage", "vintage_missing"]
    else:
        num_f, num_a = [], []
    if use_price:
        num_f.append(fit["log_price"].to_numpy()); num_a.append(app["log_price"].to_numpy()); names.append("log_price")
    if num_f:
        Nf, Na = np.column_stack(num_f), np.column_stack(num_a)
        if linear:
            mu, sd = Nf.mean(0), Nf.std(0); sd[sd == 0] = 1
            Nf, Na = (Nf - mu) / sd, (Na - mu) / sd
        mats_f.append(sp.csr_matrix(Nf)); mats_a.append(sp.csr_matrix(Na))
    return sp.hstack(mats_f).tocsr(), sp.hstack(mats_a).tocsr(), names

res["ridge_refit"] = {}
for key, (up, um) in {"ridge_price": (True, False), "ridge_meta": (False, True), "ridge_meta_price": (True, True)}.items():
    Xf, Xa, _ = build(tr, va, up, um)
    grid = {}
    for a in [0.01, 0.1, 1.0, 10.0, 100.0]:
        p = Ridge(alpha=a).fit(Xf, tr.points).predict(Xa)
        grid[a] = float(np.mean(np.abs(va.points - p)))
    a_best = min(grid, key=grid.get)
    Xf, Xa, _ = build(tv, te, up, um)
    p = Ridge(alpha=a_best).fit(Xf, tv.points).predict(Xa)
    res["ridge_refit"][key] = {"alpha_chosen_on_valid": a_best, "builder_alpha": cfg[f"{key}_alpha"],
                               "valid_MAE_at_alpha": grid[a_best], "test": mets(te.points.to_numpy(float), p)}

# ---- main XGBoost refit (one fit, params from model_config.json)
import xgboost as xgb
t = time.time()
Xf, Xa, names = build(tv, te, True, True, linear=False)
xc = cfg["xgb_chosen"]
m = xgb.XGBRegressor(learning_rate=0.1, subsample=0.8, tree_method="hist", max_depth=xc["max_depth"],
                     min_child_weight=xc["min_child_weight"], colsample_bytree=xc["colsample_bytree"],
                     n_estimators=xc["n_estimators"], n_jobs=2, random_state=7)
m.fit(Xf.toarray().astype(np.float32), tv.points)
p = m.predict(Xa.toarray().astype(np.float32))
res["xgb_refit"] = {"n_features": Xf.shape[1], "builder_n_features": cfg["n_tree_features_trainvalid_fit"],
                    "seed_used": 7, "test": mets(te.points.to_numpy(float), p), "seconds": round(time.time() - t, 1)}
P = pd.read_csv(ROOT / "outputs" / "models" / "test_predictions.csv")
res["xgb_refit"]["corr_with_saved_preds"] = float(np.corrcoef(P["pred_xgb_meta_price"], p)[0, 1])
res["xgb_refit"]["row_order_same"] = bool((P["source_row_id"].to_numpy() == te["source_row_id"].to_numpy()).all())

# ---- paired bootstrap Main vs Ridge test MAE (own seed)
y = P.points.to_numpy(float)
d = np.abs(y - P.pred_ridge_meta_price) - np.abs(y - P.pred_xgb_meta_price)
rng = np.random.default_rng(2026)
bs = np.array([d.to_numpy()[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)])
res["mae_gap_ridge_minus_main"] = {"diff": float(d.mean()), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}

# ---- share-of-gain decomposition, for the "most of the improvement is linear" claim
fp = res["from_predictions"]
g = lambda k, m_: fp[k]["test"][m_]
res["gain_share"] = {
    "R2_ridge_share_of_price_to_xgb_gain": (g("ridge_meta_price", "R2") - g("ridge_price", "R2")) / (g("xgb_meta_price", "R2") - g("ridge_price", "R2")),
    "MAE_ridge_share_of_price_to_xgb_gain": (g("ridge_price", "MAE") - g("ridge_meta_price", "MAE")) / (g("ridge_price", "MAE") - g("xgb_meta_price", "MAE")),
}
json.dump(res, open(OUT / "a2_models.json", "w"), indent=2)
print(json.dumps(res, indent=2))
