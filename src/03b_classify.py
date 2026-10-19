"""P2b: classification framing - target y = (points >= 90), the outcome the budget simulation scores.

Benchmark question: does a tree model beat plain logistic regression (the credit-scoring standard)?
Same splits and the same pre-review features as the main model (country, province, variety with the same
rare-level grouping, log_price, vintage + missing flag).
  (a) logit_price : logistic regression, standardized log_price only (C fixed at 1.0; one feature, nothing to tune)
  (b) logit_all   : logistic regression, L2, one-hot categoricals + standardized numerics, C tuned on valid
  (b2) logit_l1  : logistic regression, L1 (liblinear; note liblinear also penalizes the intercept), C on valid
  (c) xgb_clf     : XGBoost classifier, small grid tuned on valid; n_estimators by early stopping on valid
                    log loss (so its valid scores are slightly optimistic; test is clean)
  (d) xgb_reg_score: reference only, no refit - the existing XGBoost regressor's predicted points used as a
                    ranking score (AUC only; it is not a probability)
Selection on valid (log loss, a proper scoring rule) -> refit on train+valid -> test scored once.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss, log_loss,
                             roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CLEAN_CSV, MODEL_DIR, SEED, ensure_dirs  # noqa: E402

T0 = time.time()
ensure_dirs()

THRESH = 90
CAT = ["country", "province", "variety"]
FEATS_ALL = ["country", "province", "variety", "log_price", "vintage"]
mcfg = json.loads((MODEL_DIR / "model_config.json").read_text(encoding="utf-8"))
RARE_MIN = mcfg["rare_level_min_count"]          # same rare-level grouping as the main model
C_GRID = [0.01, 0.1, 1.0, 10.0]
XGB_FIXED = dict(learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, n_estimators=2000, tree_method="hist")
XGB_GRID = [dict(max_depth=d, min_child_weight=m) for d in (6, 8) for m in (1, 10)]     # 4 configs
XGB_EARLY_STOP = 50
BOOT_N = 2000


def make_ct(feats, kind):
    """Identical to 03_models.make_ct: 'linear' = sparse one-hot + standardized numerics; 'tree' = dense raw."""
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


def dense32(X):
    return np.asarray(X.toarray() if sp.issparse(X) else X, dtype=np.float32)


def cls_metrics(y, p, prob=True):
    out = {"roc_auc": float(roc_auc_score(y, p)), "base_rate": float(y.mean())}
    if prob:
        pc = np.clip(p, 1e-15, 1 - 1e-15)
        out.update({"pr_auc": float(average_precision_score(y, p)), "log_loss": float(log_loss(y, pc)),
                    "brier": float(brier_score_loss(y, p)), "accuracy_at_0_5": float(accuracy_score(y, p >= 0.5))})
    else:
        out.update({"pr_auc": np.nan, "log_loss": np.nan, "brier": np.nan, "accuracy_at_0_5": np.nan})
    return out


df = pd.read_csv(CLEAN_CSV)
train = df[df["split"] == "train"].reset_index(drop=True)
valid = df[df["split"] == "valid"].reset_index(drop=True)
test = df[df["split"] == "test"].reset_index(drop=True)
trval = pd.concat([train, valid], ignore_index=True)
y_tr, y_va, y_te, y_tv = ((d["points"] >= THRESH).astype(int).to_numpy() for d in (train, valid, test, trval))
pva, pte, cfg = {}, {}, {"target": f"points >= {THRESH}", "seed": SEED, "rare_level_min_count": RARE_MIN,
                         "selection_metric": "valid log loss", "final_fit": "train+valid", "test_used": "once"}


def fit_logit(feats, fit_df, y_fit, apply_dfs, C, l1=False, return_model=False):
    ct = make_ct(feats, "linear")
    X = ct.fit_transform(fit_df)
    if l1:
        m = LogisticRegression(C=C, l1_ratio=1, class_weight=None, solver="liblinear", max_iter=5000)
    else:
        m = LogisticRegression(C=C, l1_ratio=0, class_weight=None, solver="lbfgs", max_iter=5000)
    m.fit(X, y_fit)
    preds = [m.predict_proba(ct.transform(a))[:, 1] for a in apply_dfs]
    return (preds, m, ct) if return_model else preds


def zero_report(m, ct):
    names = list(ct.get_feature_names_out())
    coef = np.asarray(m.coef_).ravel()
    onehot = [i for i, n in enumerate(names) if n.startswith("cat__")]
    rep = {"n_coef_total": int(len(coef)), "n_coef_zero": int((coef == 0).sum()), "n_onehot_total": len(onehot),
           "n_onehot_zero": int((coef[onehot] == 0).sum())}
    for c in CAT:
        idx = [i for i, n in enumerate(names) if n.startswith(f"cat__{c}_")]
        rep[f"{c}_levels_total"], rep[f"{c}_levels_zero"] = len(idx), int((coef[idx] == 0).sum())
    return rep


# (a) logistic, price only
pva["logit_price"], = fit_logit(["log_price"], train, y_tr, [valid], 1.0)
pte["logit_price"], = fit_logit(["log_price"], trval, y_tv, [test], 1.0)
cfg["logit_price"] = {"C": 1.0, "features": ["log_price"], "note": "single standardized feature; C not tuned"}

# (b) logistic, all main features, C on valid
grid = []
for C in C_GRID:
    p, = fit_logit(FEATS_ALL, train, y_tr, [valid], C)
    grid.append({"C": C, "valid_log_loss": float(log_loss(y_va, p)), "valid_roc_auc": float(roc_auc_score(y_va, p)),
                 "pred": p})
best = min(grid, key=lambda g: (g["valid_log_loss"], g["C"]))
pva["logit_all"] = best["pred"]
pte["logit_all"], = fit_logit(FEATS_ALL, trval, y_tv, [test], best["C"])
pd.DataFrame([{k: v for k, v in g.items() if k != "pred"} for g in grid]).to_csv(
    MODEL_DIR / "classify_logit_C_grid_valid.csv", index=False, encoding="utf-8")
cfg["logit_all"] = {"C": best["C"], "C_grid": C_GRID, "penalty": "L2", "class_weight": None, "solver": "lbfgs",
                    "features": FEATS_ALL + ["vintage missing flag"]}
print(f"logit_all: C={best['C']} valid logloss={best['valid_log_loss']:.4f} AUC={best['valid_roc_auc']:.4f}", flush=True)

# (b2) L1 logistic, all main features, C on valid
grid1 = []
for C in C_GRID:
    p, = fit_logit(FEATS_ALL, train, y_tr, [valid], C, l1=True)
    grid1.append({"C": C, "valid_log_loss": float(log_loss(y_va, p)), "valid_roc_auc": float(roc_auc_score(y_va, p)),
                  "pred": p})
best1 = min(grid1, key=lambda g: (g["valid_log_loss"], g["C"]))
pva["logit_l1"] = best1["pred"]
(pl1,), m_l1, ct_l1 = fit_logit(FEATS_ALL, trval, y_tv, [test], best1["C"], l1=True, return_model=True)
pte["logit_l1"] = pl1
pd.DataFrame([{k: v for k, v in g.items() if k != "pred"} for g in grid1]).to_csv(
    MODEL_DIR / "classify_logit_l1_C_grid_valid.csv", index=False, encoding="utf-8")
_, m_l2, ct_l2 = fit_logit(FEATS_ALL, trval, y_tv, [test], best["C"], return_model=True)
cfg["logit_l1"] = {"C": best1["C"], "C_grid": C_GRID, "penalty": "L1", "solver": "liblinear",
                   "note": "liblinear also penalizes the intercept", "zero_coefficients_final_fit": zero_report(m_l1, ct_l1)}
cfg["logit_all"]["zero_coefficients_final_fit"] = zero_report(m_l2, ct_l2)
print(f"logit_l1: C={best1['C']} valid logloss={best1['valid_log_loss']:.4f} AUC={best1['valid_roc_auc']:.4f}", flush=True)

# (c) XGBoost classifier
t = time.time()
ct_tr = make_ct(FEATS_ALL, "tree")
X_tr, X_va = dense32(ct_tr.fit_transform(train)), dense32(ct_tr.transform(valid))
ct_tv = make_ct(FEATS_ALL, "tree")
X_tv, X_te = dense32(ct_tv.fit_transform(trval)), dense32(ct_tv.transform(test))
rows, bestx = [], None
for i, g in enumerate(XGB_GRID):
    m = xgb.XGBClassifier(**XGB_FIXED, **g, n_jobs=2, random_state=SEED, eval_metric="logloss",
                          early_stopping_rounds=XGB_EARLY_STOP)
    m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    p = m.predict_proba(X_va)[:, 1]
    r = {"config_id": i, **g, "best_n_estimators": int(m.best_iteration) + 1,
         "valid_log_loss": float(log_loss(y_va, p)), "valid_roc_auc": float(roc_auc_score(y_va, p))}
    rows.append(r)
    if bestx is None or r["valid_log_loss"] < bestx[0]["valid_log_loss"] - 1e-12:
        bestx = (r, p)
pd.DataFrame(rows).to_csv(MODEL_DIR / "classify_xgb_grid_valid.csv", index=False, encoding="utf-8")
xc = {k: bestx[0][k] for k in ("max_depth", "min_child_weight")}
xn = bestx[0]["best_n_estimators"]
pva["xgb_clf"] = bestx[1]
mf = xgb.XGBClassifier(**{**XGB_FIXED, "n_estimators": xn}, **xc, n_jobs=2, random_state=SEED, eval_metric="logloss")
pte["xgb_clf"] = mf.fit(X_tv, y_tv).predict_proba(X_te)[:, 1]
cfg["xgb_clf"] = {**{k: v for k, v in XGB_FIXED.items() if k != "n_estimators"}, **xc, "n_estimators": xn,
                  "grid_size": len(XGB_GRID),
                  "note": "n_estimators by early stopping on valid log loss -> valid scores slightly optimistic"}
print(f"xgb_clf: {xc} n={xn} valid logloss={bestx[0]['valid_log_loss']:.4f} ({time.time() - t:.0f}s)", flush=True)

# (d) reference: regressor's predicted points as a ranking score
vp = pd.read_csv(MODEL_DIR / "valid_predictions.csv")
tp = pd.read_csv(MODEL_DIR / "test_predictions.csv")
assert (vp["source_row_id"].to_numpy() == valid["source_row_id"].to_numpy()).all()
assert (tp["source_row_id"].to_numpy() == test["source_row_id"].to_numpy()).all()
REG = mcfg["main_model"]
pva["xgb_reg_score"] = vp[f"pred_{REG}"].to_numpy()
pte["xgb_reg_score"] = tp[f"pred_{REG}"].to_numpy()
cfg["xgb_reg_score"] = {"source": f"pred_{REG} from P2 (regressor of points, not refit)", "metrics": "ROC-AUC only"}

LABELS = {"logit_price": "Logistic: log_price only", "logit_all": "Logistic (L2): all main features",
          "logit_l1": "Logistic (L1): all main features",
          "xgb_clf": "XGBoost classifier: all main features",
          "xgb_reg_score": f"Reference: {REG} regressor predicted points as score (no refit)"}
ORDER = ["logit_price", "logit_all", "logit_l1", "xgb_clf", "xgb_reg_score"]
mrows = []
for k in ORDER:
    for s, y, p in (("valid", y_va, pva[k]), ("test", y_te, pte[k])):
        mrows.append({"model": k, "label": LABELS[k], "split": s, "n": int(len(y)),
                      **cls_metrics(y, p, prob=k != "xgb_reg_score")})
met = pd.DataFrame(mrows)
met.to_csv(MODEL_DIR / "classify_metrics.csv", index=False, encoding="utf-8")

# paired bootstrap over test rows for AUC differences vs logistic(all)
rng = np.random.default_rng(SEED)
diffs = {k: [] for k in ("xgb_clf", "xgb_reg_score", "logit_price", "logit_l1")}
for _ in range(BOOT_N):
    ix = rng.integers(0, len(y_te), len(y_te))
    a = roc_auc_score(y_te[ix], pte["logit_all"][ix])
    for k in diffs:
        diffs[k].append(roc_auc_score(y_te[ix], pte[k][ix]) - a)
auc_te = {k: roc_auc_score(y_te, pte[k]) for k in ORDER}
crow = [{"comparison": f"AUC({k}) - AUC(logit_all)", "model": k, "reference": "logit_all", "split": "test",
         "diff_auc": float(auc_te[k] - auc_te["logit_all"]), "ci95_low": float(np.percentile(diffs[k], 2.5)),
         "ci95_high": float(np.percentile(diffs[k], 97.5)), "n_boot": BOOT_N,
         "note": "paired bootstrap over test rows (same resample for both models)"} for k in diffs]
pd.DataFrame(crow).to_csv(MODEL_DIR / "classify_auc_diff_ci.csv", index=False, encoding="utf-8")

out = test[["source_row_id", "title_norm", "price", "points"]].copy()
out["y_ge90"] = y_te
out["prob_logit_price"] = pte["logit_price"]
out["prob_logit_all"] = pte["logit_all"]
out["prob_logit_l1"] = pte["logit_l1"]
out["prob_xgb_clf"] = pte["xgb_clf"]
out["score_xgb_reg"] = pte["xgb_reg_score"]
out.to_csv(MODEL_DIR / "classify_test_predictions.csv", index=False, encoding="utf-8")
with open(MODEL_DIR / "classify_config.json", "w", encoding="utf-8") as f:
    json.dump(cfg, f, indent=2, ensure_ascii=False)

print(met[["model", "split", "roc_auc", "pr_auc", "log_loss", "brier", "accuracy_at_0_5"]].to_string(index=False))
print(pd.DataFrame(crow)[["comparison", "diff_auc", "ci95_low", "ci95_high"]].to_string(index=False))
print(f"runtime {time.time() - T0:.0f}s", flush=True)
