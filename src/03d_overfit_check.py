"""P2d: over-fitting check — train vs valid vs test performance for the chosen configs.

Protocol (no new choices are made here; every hyperparameter is read from 03_models / 03b_classify output)
- "train" / "valid": the chosen config fit on TRAIN only, scored on the rows it learned from (train) and on
  valid. This valid score must equal the one 03_models / 03b_classify reported (asserted below).
- "trval_fit" / "test": the final model refit on TRAIN+VALID, scored on its own fitting rows and on test.
  The test score is the one already reported; it is re-computed here only as a check.
- Gap = train - valid (how much the model memorises), valid vs test (whether choosing on valid over-fitted valid).

Outputs: outputs/models/overfit_check.csv, overfit_check.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CLEAN_CSV, MODEL_DIR, SEED, ensure_dirs  # noqa: E402

ensure_dirs()
RARE_MIN = 20
CAT = ["country", "province", "variety"]
FEATS = {"price": ["log_price"],
         "meta_price": ["country", "province", "variety", "log_price", "vintage"]}
mcfg = json.loads((MODEL_DIR / "model_config.json").read_text(encoding="utf-8"))
ccfg = json.loads((MODEL_DIR / "classify_config.json").read_text(encoding="utf-8"))
reported = pd.read_csv(MODEL_DIR / "metrics.csv")
reported_c = pd.read_csv(MODEL_DIR / "classify_metrics.csv")

df = pd.read_csv(CLEAN_CSV)
train = df[df["split"] == "train"].reset_index(drop=True)
valid = df[df["split"] == "valid"].reset_index(drop=True)
test = df[df["split"] == "test"].reset_index(drop=True)
trval = pd.concat([train, valid], ignore_index=True)


def make_ct(feats, kind):
    """Identical to 03_models.make_ct."""
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


def dense(X):
    return np.asarray(X.toarray() if sp.issparse(X) else X, dtype=np.float32)


def run(fit_df, eval_df, builder, kind, feats):
    ct = make_ct(feats, kind)
    Xf = ct.fit_transform(fit_df)
    Xe = ct.transform(eval_df)
    if kind == "tree":
        Xf, Xe = dense(Xf), dense(Xe)
    m = builder()
    return m, Xf, Xe


REG = {
    "ridge_price": ("Ridge (가격만)", "linear", "price",
                    lambda: Ridge(alpha=mcfg["ridge_price_alpha"])),
    "ridge_meta_price": ("Ridge (전체 변수)", "linear", "meta_price",
                         lambda: Ridge(alpha=mcfg["ridge_meta_price_alpha"])),
    "rf_meta_price": ("랜덤포레스트", "tree", "meta_price",
                      lambda: RandomForestRegressor(**mcfg["rf_params"], n_jobs=2, random_state=SEED)),
    "xgb_meta_price": ("XGBoost", "tree", "meta_price",
                       lambda: xgb.XGBRegressor(**mcfg["xgb_fixed_params"], **mcfg["xgb_chosen"], n_jobs=2,
                                                random_state=SEED, eval_metric="mae")),
}
rows = []
for key, (label, kind, fs, builder) in REG.items():
    out = {"task": "regression", "model": key, "label": label, "metric": "R2"}
    for stage, fit_df, eval_df, names in (("first", train, valid, ("train", "valid")),
                                          ("final", trval, test, ("trval_fit", "test"))):
        m, Xf, Xe = run(fit_df, eval_df, builder, kind, FEATS[fs])
        yf, ye = fit_df["points"].to_numpy(), eval_df["points"].to_numpy()
        m.fit(Xf, yf)
        if hasattr(m, "set_params") and isinstance(m, RandomForestRegressor):
            m.set_params(n_jobs=1)
        pf, pe = m.predict(Xf), m.predict(Xe)
        out[f"{names[0]}_R2"], out[f"{names[1]}_R2"] = r2_score(yf, pf), r2_score(ye, pe)
        out[f"{names[0]}_MAE"], out[f"{names[1]}_MAE"] = mean_absolute_error(yf, pf), mean_absolute_error(ye, pe)
    for s in ("valid", "test"):
        rep = reported[(reported["model"] == key) & (reported["split"] == s)]["R2"].iloc[0]
        assert abs(rep - out[f"{s}_R2"]) < 2e-3, (key, s, rep, out[f"{s}_R2"])
    rows.append(out)
    print(f"{label}: train {out['train_R2']:.3f} valid {out['valid_R2']:.3f} test {out['test_R2']:.3f}", flush=True)

# ---------------------------------------------------------------- classification (points >= 90), ROC-AUC
lc, xc = ccfg["logit_all"], ccfg["xgb_clf"]
CLF = {
    "logit_all": ("로지스틱 (전체 변수)", "linear",
                  lambda: LogisticRegression(C=lc["C"], l1_ratio=0, solver="lbfgs", max_iter=5000)),
    "xgb_clf": ("XGBoost 분류", "tree",
                lambda: xgb.XGBClassifier(**{k: xc[k] for k in ("learning_rate", "subsample", "colsample_bytree",
                                                                  "tree_method", "max_depth", "min_child_weight",
                                                                  "n_estimators")},
                                          n_jobs=2, random_state=SEED, eval_metric="logloss")),
}
for key, (label, kind, builder) in CLF.items():
    out = {"task": "classification", "model": key, "label": label, "metric": "ROC_AUC"}
    for fit_df, eval_df, names in ((train, valid, ("train", "valid")), (trval, test, ("trval_fit", "test"))):
        m, Xf, Xe = run(fit_df, eval_df, builder, kind, FEATS["meta_price"])
        yf, ye = (fit_df["points"] >= 90).astype(int).to_numpy(), (eval_df["points"] >= 90).astype(int).to_numpy()
        m.fit(Xf, yf)
        out[f"{names[0]}_AUC"] = roc_auc_score(yf, m.predict_proba(Xf)[:, 1])
        out[f"{names[1]}_AUC"] = roc_auc_score(ye, m.predict_proba(Xe)[:, 1])
    for s in ("valid", "test"):
        rep = reported_c[(reported_c["model"] == key) & (reported_c["split"] == s)]["roc_auc"].iloc[0]
        assert abs(rep - out[f"{s}_AUC"]) < 2e-3, (key, s, rep, out[f"{s}_AUC"])
    rows.append(out)
    print(f"{label}: train AUC {out['train_AUC']:.3f} valid {out['valid_AUC']:.3f} test {out['test_AUC']:.3f}",
          flush=True)

res = pd.DataFrame(rows)
res.to_csv(MODEL_DIR / "overfit_check.csv", index=False, encoding="utf-8")
(MODEL_DIR / "overfit_check.json").write_text(json.dumps({
    "_note": ("train/valid = chosen config fit on train only; trval_fit/test = final model refit on train+valid. "
              "XGB n_estimators were picked by early stopping on valid, so its valid score is slightly optimistic."),
    "rows": json.loads(res.to_json(orient="records"))}, ensure_ascii=False, indent=2), encoding="utf-8")
print(res.round(3).to_string())
