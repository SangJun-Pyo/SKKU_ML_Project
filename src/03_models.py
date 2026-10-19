"""P2: model ladder (mean -> Ridge -> RF -> XGBoost), winery TE experiment, SHAP, text aux model.

Protocol
- Every choice (Ridge alpha, XGB config, main model, winery keep/drop, text alpha) uses `valid` only.
- Chosen configs are then refit on train+valid and scored on `test` exactly once.
- Main model features are pre-review product info only: country, province, variety, log_price, vintage
  (+ vintage-missing flag). Winery appears only in the experiment, via out-of-fold target encoding.
- The text model uses the review description (post-review info) and is an auxiliary analysis only.
"""
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy.sparse as sp  # noqa: E402
import shap  # noqa: E402
import xgboost as xgb  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.ensemble import RandomForestRegressor  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import ElasticNet, Lasso, Ridge  # noqa: E402
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CLEAN_CSV, MODEL_DIR, SEED, ensure_dirs, setup_fonts  # noqa: E402

T0 = time.time()
ensure_dirs()
setup_fonts()

# ---------------------------------------------------------------- settings
RARE_MIN = 20                 # category levels with < 20 rows in the fitting data -> one "infrequent" level
CAT = ["country", "province", "variety"]
ORIG_FEATURES = ["country", "province", "variety", "log_price", "vintage"]
FEATS = {
    "price": ["log_price"],
    "meta": ["country", "province", "variety", "vintage"],
    "meta_price": ["country", "province", "variety", "log_price", "vintage"],
}
RIDGE_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0]
LASSO_ALPHAS = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]          # "why Ridge and not Lasso?" check (meta+price)
ENET_ALPHAS = [3e-4, 1e-3, 3e-3, 1e-2]
ENET_L1_RATIOS = [0.2, 0.5, 0.8]
SPARSE_CLEAR_GAIN = 0.02      # Lasso/ElasticNet "clearly better" than Ridge = valid MAE lower by >= 0.02
RF_PARAMS = dict(n_estimators=200, min_samples_leaf=5, max_features=0.33)
XGB_FIXED = dict(learning_rate=0.1, subsample=0.8, n_estimators=2000, tree_method="hist")
XGB_GRID = [dict(max_depth=d, min_child_weight=m, colsample_bytree=c)
            for d in (4, 6, 8) for m in (1, 10) for c in (0.5, 0.8)]          # 12 configs
XGB_EARLY_STOP = 50
TE_SMOOTH = 20                # winery target encoding: shrink toward global mean with weight 20 rows
TE_FOLDS = 5
WINERY_KEEP_MIN_GAIN = 0.02   # "clearly helps" = valid MAE improves by >= 0.02 points
TFIDF = dict(ngram_range=(1, 2), min_df=5, stop_words="english", sublinear_tf=True,
             max_features=20000, dtype=np.float32)
TEXT_ALPHAS = [1.0, 3.0, 10.0, 30.0]
SHAP_N = 5000
SHAP_MIN_N = 30               # min sampled rows for per-country / per-variety SHAP means
BOOT_N = 2000                 # paired bootstrap of test MAE differences (rows)

# ---------------------------------------------------------------- data
df = pd.read_csv(CLEAN_CSV)
train = df[df["split"] == "train"].reset_index(drop=True)
valid = df[df["split"] == "valid"].reset_index(drop=True)
test = df[df["split"] == "test"].reset_index(drop=True)
trval = pd.concat([train, valid], ignore_index=True)
print(f"train={len(train)} valid={len(valid)} test={len(test)}", flush=True)


def make_ct(feats, kind):
    """kind='linear' -> sparse one-hot + standardized numerics; kind='tree' -> dense, raw numerics."""
    lin = kind == "linear"
    parts = []
    cats = [c for c in CAT if c in feats]
    if cats:
        parts.append(("cat", OneHotEncoder(min_frequency=RARE_MIN, handle_unknown="infrequent_if_exist",
                                           sparse_output=lin, dtype=np.float32), cats))
    if "log_price" in feats:
        parts.append(("num", StandardScaler() if lin else "passthrough", ["log_price"]))
    if "vintage" in feats:
        steps = [("imp", SimpleImputer(strategy="median", add_indicator=True))]
        if lin:
            steps.append(("sc", StandardScaler()))
        parts.append(("vint", Pipeline(steps), ["vintage"]))
    return ColumnTransformer(parts, sparse_threshold=1.0 if lin else 0.0)


def to_dense32(X):
    return np.asarray(X.toarray() if sp.issparse(X) else X, dtype=np.float32)


def metrics(y, p):
    return dict(MAE=float(mean_absolute_error(y, p)), RMSE=float(np.sqrt(mean_squared_error(y, p))),
                R2=float(r2_score(y, p)))


def winery_te(fit_df, apply_df):
    """Out-of-fold (grouped by title_norm) smoothed mean points per winery for fit rows;
    full-fit-data encoding for apply rows. Unseen wineries -> global mean."""
    def enc(src, tgt):
        gm = src["points"].mean()
        g = src.groupby("winery")["points"].agg(["sum", "count"])
        s = tgt["winery"].map(g["sum"]).fillna(0.0).to_numpy()
        c = tgt["winery"].map(g["count"]).fillna(0.0).to_numpy()
        return (s + TE_SMOOTH * gm) / (c + TE_SMOOTH)
    fit_df = fit_df[["winery", "points", "title_norm"]].reset_index(drop=True)
    oof = np.zeros(len(fit_df))
    gkf = GroupKFold(n_splits=TE_FOLDS, shuffle=True, random_state=SEED)
    # integer group codes (sorted = same order as np.unique on the strings); string groups are very slow here
    codes = pd.factorize(fit_df["title_norm"], sort=True)[0]
    for tr_i, va_i in gkf.split(np.zeros(len(fit_df)), groups=codes):
        oof[va_i] = enc(fit_df.iloc[tr_i], fit_df.iloc[va_i])
    return oof.reshape(-1, 1).astype(np.float32), enc(fit_df, apply_df).reshape(-1, 1).astype(np.float32)


def fit_ridge(feats, fit_df, apply_df, alpha):
    ct = make_ct(feats, "linear")
    Xf = ct.fit_transform(fit_df)
    m = Ridge(alpha=alpha).fit(Xf, fit_df["points"].to_numpy())
    return m.predict(ct.transform(apply_df))


def fit_rf(Xf, yf, Xa):
    m = RandomForestRegressor(**RF_PARAMS, n_jobs=2, random_state=SEED).fit(Xf, yf)
    m.set_params(n_jobs=1)    # multi-threaded predict sums trees in completion order -> last-bit float noise
    return m, m.predict(Xa)


def fit_xgb(Xf, yf, Xa, cfg, n_estimators=None, eval_set=None):
    params = {**XGB_FIXED, **cfg, "n_jobs": 2, "random_state": SEED, "eval_metric": "mae"}
    if n_estimators is not None:
        params["n_estimators"] = n_estimators
    if eval_set is not None:
        params["early_stopping_rounds"] = XGB_EARLY_STOP
    m = xgb.XGBRegressor(**params)
    m.fit(Xf, yf, eval_set=eval_set, verbose=False) if eval_set is not None else m.fit(Xf, yf)
    return m, m.predict(Xa)


y_tr, y_va, y_te, y_tv = (d["points"].to_numpy() for d in (train, valid, test, trval))
pred_va, pred_te = {}, {}
config = {"seed": SEED, "rare_level_min_count": RARE_MIN,
          "rare_level_rule": "levels with < rare_level_min_count rows in the fitting data are merged into one "
                             "'infrequent' level (OneHotEncoder min_frequency); unseen levels map there too",
          "vintage_handling": "median imputation (fitting data) + missing indicator",
          "selection_split": "valid", "final_fit": "train+valid", "test_used": "once, after all choices"}
labels = {
    "mean": "Mean baseline",
    "ridge_price": "Ridge: log_price only",
    "ridge_meta": "Ridge: meta only (country, province, variety, vintage)",
    "ridge_meta_price": "Ridge: meta + log_price",
    "rf_meta_price": "RandomForest: meta + log_price",
    "xgb_meta_price": "XGBoost: meta + log_price (tuned on valid)",
}

# ---------------------------------------------------------------- 1. mean baseline
pred_va["mean"] = np.full(len(valid), y_tr.mean())
pred_te["mean"] = np.full(len(test), y_tv.mean())

# ---------------------------------------------------------------- 2-4. Ridge ladder (alpha on valid)
ridge_grid = []
for key, fs in (("ridge_price", "price"), ("ridge_meta", "meta"), ("ridge_meta_price", "meta_price")):
    best = None
    for a in RIDGE_ALPHAS:
        p = fit_ridge(FEATS[fs], train, valid, a)
        mae = mean_absolute_error(y_va, p)
        ridge_grid.append({"model": key, "alpha": a, "valid_MAE": mae})
        if best is None or mae < best[1] - 1e-12:
            best = (a, mae, p)
    pred_va[key] = best[2]
    pred_te[key] = fit_ridge(FEATS[fs], trval, test, best[0])
    config[f"{key}_alpha"] = best[0]
    print(f"{key}: alpha={best[0]} valid MAE={best[1]:.4f}", flush=True)
pd.DataFrame(ridge_grid).to_csv(MODEL_DIR / "ridge_alpha_grid_valid.csv", index=False, encoding="utf-8")

# ---------------------------------------------------------------- 4b. Lasso / ElasticNet on meta+price (comparison)
# Same design matrix as Ridge(meta+price). Ridge stays the linear model of record (simulation) unless one of these
# is clearly better on valid; that decision is only flagged here, never switched automatically.
t = time.time()


def fit_linear(model, fit_df, apply_df):
    ct = make_ct(FEATS["meta_price"], "linear")
    m = model.fit(ct.fit_transform(fit_df), fit_df["points"].to_numpy())
    return m, ct, m.predict(ct.transform(apply_df))


def zero_report(m, ct):
    names = list(ct.get_feature_names_out())
    coef = np.asarray(m.coef_).ravel()
    rep = {"n_coef_total": int(len(coef)), "n_coef_zero": int((coef == 0).sum())}
    onehot = [i for i, n in enumerate(names) if n.startswith("cat__")]
    rep["n_onehot_total"] = len(onehot)
    rep["n_onehot_zero"] = int((coef[onehot] == 0).sum())
    rep["onehot_zero_pct"] = 100 * rep["n_onehot_zero"] / max(len(onehot), 1)
    for c in CAT:
        idx = [i for i, n in enumerate(names) if n.startswith(f"cat__{c}_")]
        rep[f"{c}_levels_total"] = len(idx)
        rep[f"{c}_levels_zero"] = int((coef[idx] == 0).sum())
    num = [i for i, n in enumerate(names) if not n.startswith("cat__")]
    rep["numeric_zero"] = [names[i] for i in num if coef[i] == 0]
    return rep


sparse_grid = []
for key, grid in (("lasso_meta_price", [{"alpha": a} for a in LASSO_ALPHAS]),
                  ("enet_meta_price", [{"alpha": a, "l1_ratio": r} for r in ENET_L1_RATIOS for a in ENET_ALPHAS])):
    best = None
    for g in grid:
        mdl = Lasso(**g, max_iter=50000) if key.startswith("lasso") else ElasticNet(**g, max_iter=50000)
        _, _, p = fit_linear(mdl, train, valid)
        mae = mean_absolute_error(y_va, p)
        sparse_grid.append({"model": key, **g, "valid_MAE": mae})
        if best is None or mae < best[1] - 1e-12:
            best = (g, mae, p)
    pred_va[key] = best[2]
    mdl = Lasso(**best[0], max_iter=50000) if key.startswith("lasso") else ElasticNet(**best[0], max_iter=50000)
    m_fin, ct_fin, pred_te[key] = fit_linear(mdl, trval, test)
    config[f"{key}_params"] = best[0]
    config[f"{key}_zero_coefficients_final_fit"] = zero_report(m_fin, ct_fin)
    print(f"{key}: {best[0]} valid MAE={best[1]:.4f}", flush=True)
pd.DataFrame(sparse_grid).to_csv(MODEL_DIR / "lasso_enet_grid_valid.csv", index=False, encoding="utf-8")
m_r, ct_r, _ = fit_linear(Ridge(alpha=config["ridge_meta_price_alpha"]), trval, test)
config["ridge_meta_price_zero_coefficients_final_fit"] = zero_report(m_r, ct_r)
labels_sparse = {"lasso_meta_price": "Lasso: meta + log_price (alpha on valid)",
                 "enet_meta_price": "ElasticNet: meta + log_price (alpha x l1_ratio on valid)"}
_rv = mean_absolute_error(y_va, pred_va["ridge_meta_price"])
config["sparse_vs_ridge"] = {
    k: {"valid_MAE_gain_vs_ridge": float(_rv - mean_absolute_error(y_va, pred_va[k])),
        "clearly_better_on_valid": bool(_rv - mean_absolute_error(y_va, pred_va[k]) >= SPARSE_CLEAR_GAIN)}
    for k in labels_sparse}
config["sparse_vs_ridge"]["threshold_valid_MAE_gain"] = SPARSE_CLEAR_GAIN
config["sparse_vs_ridge"]["note"] = ("Ridge(meta+price) remains the linear model used in the simulation; a 'clearly "
                                     "better' flag is reported for a human decision, never applied automatically")
print(f"lasso/enet done ({time.time() - t:.0f}s)", flush=True)

# ---------------------------------------------------------------- tree design matrices
ct_tr = make_ct(FEATS["meta_price"], "tree")
X_tr = to_dense32(ct_tr.fit_transform(train))
X_va = to_dense32(ct_tr.transform(valid))
ct_tv = make_ct(FEATS["meta_price"], "tree")
X_tv = to_dense32(ct_tv.fit_transform(trval))
X_te = to_dense32(ct_tv.transform(test))
tree_feature_names = list(ct_tv.get_feature_names_out())
config["n_tree_features_train_fit"] = int(X_tr.shape[1])
config["n_tree_features_trainvalid_fit"] = int(X_tv.shape[1])

# ---------------------------------------------------------------- 5. RandomForest (fixed moderate config)
t = time.time()
_, pred_va["rf_meta_price"] = fit_rf(X_tr, y_tr, X_va)
rf_final, pred_te["rf_meta_price"] = fit_rf(X_tv, y_tv, X_te)
config["rf_params"] = RF_PARAMS
print(f"rf: valid MAE={mean_absolute_error(y_va, pred_va['rf_meta_price']):.4f} ({time.time() - t:.0f}s)", flush=True)

# ---------------------------------------------------------------- 6. XGBoost grid (early stopping on valid)
t = time.time()
xgb_rows = []
best = None
for i, cfg in enumerate(XGB_GRID):
    m, p = fit_xgb(X_tr, y_tr, X_va, cfg, eval_set=[(X_va, y_va)])
    mae = mean_absolute_error(y_va, p)
    xgb_rows.append({"config_id": i, **cfg, "best_n_estimators": int(m.best_iteration) + 1, "valid_MAE": mae})
    if best is None or mae < best["valid_MAE"] - 1e-12:
        best = {**xgb_rows[-1], "pred": p}
pd.DataFrame(xgb_rows).to_csv(MODEL_DIR / "xgb_grid_valid.csv", index=False, encoding="utf-8")
xgb_cfg = {k: best[k] for k in ("max_depth", "min_child_weight", "colsample_bytree")}
xgb_n = best["best_n_estimators"]
pred_va["xgb_meta_price"] = best["pred"]
xgb_final, pred_te["xgb_meta_price"] = fit_xgb(X_tv, y_tv, X_te, xgb_cfg, n_estimators=xgb_n)
config["xgb_fixed_params"] = {k: v for k, v in XGB_FIXED.items() if k != "n_estimators"}
config["xgb_grid_size"] = len(XGB_GRID)
config["xgb_chosen"] = {**xgb_cfg, "n_estimators": xgb_n}
config["xgb_note"] = ("n_estimators chosen by early stopping on valid MAE, so the XGB valid score is slightly "
                      "optimistic; the test score is clean. Final refit on train+valid uses the same n_estimators.")
print(f"xgb: best {config['xgb_chosen']} valid MAE={best['valid_MAE']:.4f} ({time.time() - t:.0f}s)", flush=True)

# ---------------------------------------------------------------- main model choice (valid MAE)
ladder = ["ridge_price", "ridge_meta", "ridge_meta_price", "rf_meta_price", "xgb_meta_price"]
valid_mae = {k: float(mean_absolute_error(y_va, pred_va[k])) for k in ladder}
main_key = min(ladder, key=lambda k: valid_mae[k])
tree_key = min(["rf_meta_price", "xgb_meta_price"], key=lambda k: valid_mae[k])
config["main_model"] = main_key
config["main_tree_model"] = tree_key
config["main_model_rule"] = "lowest valid MAE among ladder models 2-6 (pre-review features only)"
config["valid_MAE_by_model"] = valid_mae

# ---------------------------------------------------------------- 7. winery OOF TE experiment on best tree
t = time.time()
te_tr, te_va = winery_te(train, valid)
te_tv, te_te = winery_te(trval, test)
Xw_tr, Xw_va = np.hstack([X_tr, te_tr]), np.hstack([X_va, te_va])
Xw_tv, Xw_te = np.hstack([X_tv, te_tv]), np.hstack([X_te, te_te])
wkey = tree_key.replace("_meta_price", "_meta_price_winery")
if tree_key == "xgb_meta_price":
    mw, pred_va[wkey] = fit_xgb(Xw_tr, y_tr, Xw_va, xgb_cfg, eval_set=[(Xw_va, y_va)])
    wn = int(mw.best_iteration) + 1
    _, pred_te[wkey] = fit_xgb(Xw_tv, y_tv, Xw_te, xgb_cfg, n_estimators=wn)
    config["winery_experiment_n_estimators"] = wn
else:
    _, pred_va[wkey] = fit_rf(Xw_tr, y_tr, Xw_va)
    _, pred_te[wkey] = fit_rf(Xw_tv, y_tv, Xw_te)
w_gain = valid_mae[tree_key] - mean_absolute_error(y_va, pred_va[wkey])
labels[wkey] = f"(Experiment) {labels[tree_key].split(':')[0]}: meta + log_price + winery OOF target encoding"
config["winery_experiment"] = {
    "model": wkey, "base_model": tree_key, "encoding": f"smoothed mean points per winery (m={TE_SMOOTH}), "
    f"{TE_FOLDS}-fold out-of-fold inside the fitting data, folds grouped by title_norm; unseen -> global mean",
    "valid_MAE_gain": float(w_gain), "keep_threshold": WINERY_KEEP_MIN_GAIN,
    "clearly_helps_on_valid": bool(w_gain >= WINERY_KEEP_MIN_GAIN),
    "used_in_main_model": False,
    "note": "Main model stays pre-review product info only (country/province/variety/price/vintage) by design; "
            "the winery model is reported as an experiment regardless of the result."}
print(f"winery experiment: valid MAE gain={w_gain:.4f} ({time.time() - t:.0f}s)", flush=True)

# ---------------------------------------------------------------- text aux model (post-review)
t = time.time()


def text_design(fit_df, apply_df):
    """Returns (vectorizer, TF-IDF fit, TF-IDF apply, TF-IDF+meta+price fit, TF-IDF+meta+price apply)."""
    vec = TfidfVectorizer(**TFIDF)
    Tf = vec.fit_transform(fit_df["description"])
    Ta = vec.transform(apply_df["description"])
    ct = make_ct(FEATS["meta_price"], "linear")
    Mf = ct.fit_transform(fit_df)
    Ma = ct.transform(apply_df)
    return vec, Tf.tocsr(), Ta.tocsr(), sp.hstack([Tf, Mf]).tocsr(), sp.hstack([Ta, Ma]).tocsr()


def pick_alpha(Xf, Xa, grid_csv):
    grid, best = [], None
    for a in TEXT_ALPHAS:
        p = Ridge(alpha=a).fit(Xf, y_tr).predict(Xa)
        mae = mean_absolute_error(y_va, p)
        grid.append({"alpha": a, "valid_MAE": mae})
        if best is None or mae < best[1] - 1e-12:
            best = (a, mae, p)
    pd.DataFrame(grid).to_csv(MODEL_DIR / grid_csv, index=False, encoding="utf-8")
    return best


_, W_tr, W_va, T_tr, T_va = text_design(train, valid)
best_wo = pick_alpha(W_tr, W_va, "text_only_alpha_grid_valid.csv")      # TF-IDF words only
best = pick_alpha(T_tr, T_va, "text_alpha_grid_valid.csv")               # TF-IDF + meta + price
pred_va["text_ridge"] = best[2]
vec_tv, W_tv, W_te, T_tv, T_te = text_design(trval, test)
text_model = Ridge(alpha=best[0]).fit(T_tv, y_tv)
pred_te["text_ridge"] = text_model.predict(T_te)
labels["text_ridge"] = "(Aux, post-review) Ridge: TF-IDF(description) + meta + log_price"
pred_va["text_only_ridge"] = best_wo[2]
pred_te["text_only_ridge"] = Ridge(alpha=best_wo[0]).fit(W_tv, y_tv).predict(W_te)
labels["text_only_ridge"] = "(Aux, post-review) Ridge: TF-IDF(description) only, no meta/price"
config["text_only_model"] = {"alpha": best_wo[0], "features": "TF-IDF of description only (same vectorizer)"}
config["text_model"] = {"tfidf": {k: (str(v) if k == "dtype" else v) for k, v in TFIDF.items()},
                        "alpha": best[0], "n_terms": int(len(vec_tv.vocabulary_)),
                        "note": "description is written by the critic who gave the score -> post-review, "
                                "auxiliary analysis only; never used in the simulation"}
terms = vec_tv.get_feature_names_out()
coefs = text_model.coef_[:len(terms)]
doc_freq = np.asarray((T_tv[:, :len(terms)] > 0).sum(axis=0)).ravel()
rows = []
for direction, order in (("positive", np.argsort(-coefs)[:25]), ("negative", np.argsort(coefs)[:25])):
    for r, j in enumerate(order, 1):
        rows.append({"direction": direction, "rank": r, "term": terms[j], "coef": float(coefs[j]),
                     "doc_freq_trainvalid": int(doc_freq[j])})
pd.DataFrame(rows).to_csv(MODEL_DIR / "text_top_terms.csv", index=False, encoding="utf-8")
print(f"text: alpha={best[0]} valid MAE={best[1]:.4f}; text-only alpha={best_wo[0]} valid MAE={best_wo[1]:.4f} "
      f"({time.time() - t:.0f}s)", flush=True)

# ---------------------------------------------------------------- metrics table
family = {k: "ladder" for k in ["mean"] + ladder}
family[wkey] = "experiment"
family["lasso_meta_price"] = "linear_alt"
family["enet_meta_price"] = "linear_alt"
labels.update(labels_sparse)
family["text_ridge"] = "aux_post_review"
family["text_only_ridge"] = "aux_post_review"
order = ["mean"] + ladder + ["lasso_meta_price", "enet_meta_price", wkey, "text_only_ridge", "text_ridge"]
mrows = []
for k in order:
    for split_name, y, p in (("valid", y_va, pred_va[k]), ("test", y_te, pred_te[k])):
        mrows.append({"model": k, "label": labels[k], "family": family[k], "split": split_name,
                      "n": int(len(y)), **metrics(y, p), "is_main": k == main_key, "is_main_tree": k == tree_key})
met = pd.DataFrame(mrows)
met.to_csv(MODEL_DIR / "metrics.csv", index=False, encoding="utf-8")
print(met[["model", "split", "MAE", "RMSE", "R2"]].to_string(index=False), flush=True)

# paired bootstrap on test rows: MAE(other) - MAE(main) (positive = main is better)
rng = np.random.default_rng(SEED)
boot_idx_chunks = [rng.integers(0, len(test), size=(200, len(test))) for _ in range(BOOT_N // 200)]
crow = []
ae_main = np.abs(y_te - pred_te[main_key])
for k in [c for c in order if c != main_key]:
    d = np.abs(y_te - pred_te[k]) - ae_main
    bs = np.concatenate([d[ix].mean(axis=1) for ix in boot_idx_chunks])
    crow.append({"comparison": f"MAE({k}) - MAE({main_key})", "other_model": k, "main_model": main_key,
                 "split": "test", "diff_MAE": float(d.mean()), "ci95_low": float(np.percentile(bs, 2.5)),
                 "ci95_high": float(np.percentile(bs, 97.5)), "n_boot": BOOT_N,
                 "note": "paired bootstrap over test rows; positive = main model has lower error"})
pd.DataFrame(crow).to_csv(MODEL_DIR / "test_mae_diff_ci.csv", index=False, encoding="utf-8")

# ---------------------------------------------------------------- predictions
base_cols = ["source_row_id", "title_norm", "price", "points"]
for name, d, preds in (("valid", valid, pred_va), ("test", test, pred_te)):
    out = d[base_cols].copy()
    for k in order:
        out[f"pred_{k}"] = preds[k]
    out.to_csv(MODEL_DIR / f"{name}_predictions.csv", index=False, encoding="utf-8")

# ---------------------------------------------------------------- SHAP on main tree model (final refit)
t = time.time()
tree_final = xgb_final if tree_key == "xgb_meta_price" else rf_final
rng = np.random.default_rng(SEED)
s_idx = np.sort(rng.choice(len(test), size=min(SHAP_N, len(test)), replace=False))
Xs = X_te[s_idx]
explainer = shap.TreeExplainer(tree_final)
sv = np.asarray(explainer.shap_values(Xs))
base_value = float(np.ravel(explainer.expected_value)[0])


def orig_feature(name):
    if name.startswith("cat__"):
        rest = name[len("cat__"):]
        for c in CAT:
            if rest.startswith(c + "_"):
                return c
    if name.startswith("num__"):
        return "log_price"
    if name.startswith("vint__"):
        return "vintage"
    raise ValueError(name)


groups = [orig_feature(n) for n in tree_feature_names]
G = np.zeros((len(groups), len(ORIG_FEATURES)))
for j, g in enumerate(groups):
    G[j, ORIG_FEATURES.index(g)] = 1.0
sv_g = sv @ G                                           # per-row SHAP summed over each feature's columns
mean_abs = np.abs(sv_g).mean(axis=0)
imp = pd.DataFrame({"feature": ORIG_FEATURES, "n_columns": G.sum(axis=0).astype(int),
                    "mean_abs_shap": mean_abs, "share_pct": 100 * mean_abs / mean_abs.sum()})
imp = imp.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
imp["rank"] = np.arange(1, len(imp) + 1)
imp["model"] = tree_key
imp["n_rows_sampled"] = len(s_idx)
imp.to_csv(MODEL_DIR / "shap_feature_importance.csv", index=False, encoding="utf-8")

sample = test.iloc[s_idx].reset_index(drop=True)
gs = pd.DataFrame(sv_g, columns=[f"shap_{c}" for c in ORIG_FEATURES])
gs.insert(0, "source_row_id", sample["source_row_id"].to_numpy())
gs["base_value"] = base_value
gs["pred"] = tree_final.predict(Xs)
gs.to_csv(MODEL_DIR / "shap_grouped_sample.csv", index=False, encoding="utf-8")
assert np.allclose(gs[[f"shap_{c}" for c in ORIG_FEATURES]].sum(axis=1) + base_value, gs["pred"], atol=1e-2)

for col, fn in (("country", "shap_by_country.csv"), ("variety", "shap_by_variety.csv")):
    tmp = pd.DataFrame({col: sample[col], "shap": sv_g[:, ORIG_FEATURES.index(col)],
                        "points": sample["points"], "price": sample["price"]})
    agg = (tmp.groupby(col).agg(n=("shap", "size"), mean_shap=("shap", "mean"),
                                mean_points=("points", "mean"), median_price=("price", "median"))
           .reset_index())
    agg = agg[agg["n"] >= SHAP_MIN_N].sort_values(["mean_shap", col], ascending=[False, True])
    agg["rank_desc"] = np.arange(1, len(agg) + 1)
    agg["min_n"] = SHAP_MIN_N
    agg.to_csv(MODEL_DIR / fn, index=False, encoding="utf-8")

config["shap"] = {"model": tree_key, "n_rows_sampled": int(len(s_idx)), "sample_seed": SEED,
                  "base_value": base_value, "min_n_group": SHAP_MIN_N,
                  "aggregation": "per row, sum SHAP over all one-hot columns of a feature (vintage = value + "
                                 "missing flag), then mean of absolute values"}

fig, ax = plt.subplots(figsize=(7, 4))
ax.barh(imp["feature"][::-1], imp["share_pct"][::-1], color="#4C72B0")
for i, v in enumerate(imp["share_pct"][::-1]):
    ax.text(v + 0.5, i, f"{v:.1f}%", va="center", fontsize=9)
ax.set_xlabel("share of mean |SHAP| (%)")
ax.set_title(f"Feature importance ({tree_key}, n={len(s_idx):,} test rows)")
fig.tight_layout(); fig.savefig(MODEL_DIR / "fig_shap_bar_draft.png", dpi=150); plt.close(fig)

plt.figure()
shap.summary_plot(sv, Xs, feature_names=tree_feature_names, max_display=15, show=False)
plt.title("SHAP beeswarm (top 15 columns)")
plt.tight_layout(); plt.savefig(MODEL_DIR / "fig_shap_beeswarm_draft.png", dpi=150); plt.close("all")
print(imp.to_string(index=False), f"({time.time() - t:.0f}s)", flush=True)

# ---------------------------------------------------------------- ladder draft figure
fig, ax = plt.subplots(figsize=(8, 4))
mt = met[met["family"] == "ladder"]
xs = np.arange(len(mt["model"].unique()))
for off, s, col in ((-0.2, "valid", "#9DB4D8"), (0.2, "test", "#4C72B0")):
    v = mt[mt["split"] == s]
    ax.bar(xs + off, v["MAE"], width=0.4, label=s, color=col)
ax.set_xticks(xs, mt["model"].unique(), rotation=20, fontsize=8)
ax.set_ylim(1.5, mt["MAE"].max() + 0.1); ax.set_ylabel("MAE (points)"); ax.legend()
ax.set_title("Model ladder (lower is better)")
fig.tight_layout(); fig.savefig(MODEL_DIR / "fig_ladder_draft.png", dpi=150); plt.close(fig)

with open(MODEL_DIR / "model_config.json", "w", encoding="utf-8") as f:
    json.dump(config, f, indent=2, ensure_ascii=False)
print(f"main={main_key} main_tree={tree_key} runtime={time.time() - T0:.1f}s", flush=True)
