"""P2e: does the country-province overlap (multicollinearity) change the results?

Country and province are nearly the same information (Cramer's V 0.82, GVIF^(1/2df) 26.9 for country, see
02b_associations.py). Multicollinearity makes INDIVIDUAL coefficients / attributions unstable, but it does not
by itself hurt prediction. This script checks both sides:

1) Prediction: Ridge(meta+price) and XGBoost refit WITHOUT country, same hyperparameters as the main run
   (no new tuning). valid = fit on train; test = fit on train+valid. Compared with the main models.
2) Attribution: SHAP shares of the main XGBoost with country and province read TOGETHER as one "origin" group
   (per-row SHAP of the two groups summed before taking |.|), and SHAP shares of the no-country XGBoost.

Outputs: outputs/models/collinearity_check.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import shap
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CLEAN_CSV, MODEL_DIR, SEED, ensure_dirs  # noqa: E402

ensure_dirs()
RARE_MIN, SHAP_N = 20, 5000
mcfg = json.loads((MODEL_DIR / "model_config.json").read_text(encoding="utf-8"))
reported = pd.read_csv(MODEL_DIR / "metrics.csv")
df = pd.read_csv(CLEAN_CSV)
train = df[df["split"] == "train"].reset_index(drop=True)
valid = df[df["split"] == "valid"].reset_index(drop=True)
test = df[df["split"] == "test"].reset_index(drop=True)
trval = pd.concat([train, valid], ignore_index=True)
FULL = ["country", "province", "variety", "log_price", "vintage"]
NOC = ["province", "variety", "log_price", "vintage"]


def make_ct(feats, kind):
    """Identical to 03_models.make_ct."""
    lin = kind == "linear"
    parts = []
    cats = [c for c in ["country", "province", "variety"] if c in feats]
    if cats:
        parts.append(("cat", OneHotEncoder(min_frequency=RARE_MIN, handle_unknown="infrequent_if_exist",
                                           sparse_output=lin, dtype=np.float32), cats))
    parts.append(("num", StandardScaler() if lin else "passthrough", ["log_price"]))
    steps = [("imp", SimpleImputer(strategy="median", add_indicator=True))] + ([("sc", StandardScaler())] if lin else [])
    parts.append(("vint", Pipeline(steps), ["vintage"]))
    return ColumnTransformer(parts, sparse_threshold=1.0 if lin else 0.0)


def dense(X):
    return np.asarray(X.toarray() if sp.issparse(X) else X, dtype=np.float32)


def run(feats, kind, builder, fit_df, eval_df):
    ct = make_ct(feats, kind)
    Xf, Xe = ct.fit_transform(fit_df), ct.transform(eval_df)
    if kind == "tree":
        Xf, Xe = dense(Xf), dense(Xe)
    m = builder().fit(Xf, fit_df["points"].to_numpy())
    p = m.predict(Xe)
    return m, ct, Xe, {"R2": float(r2_score(eval_df["points"], p)), "MAE": float(mean_absolute_error(eval_df["points"], p))}


MODELS = {
    "ridge_meta_price": ("linear", lambda: Ridge(alpha=mcfg["ridge_meta_price_alpha"])),
    "xgb_meta_price": ("tree", lambda: xgb.XGBRegressor(**mcfg["xgb_fixed_params"], **mcfg["xgb_chosen"], n_jobs=2,
                                                         random_state=SEED, eval_metric="mae")),
}
out = {"prediction": {}}
final_noc = None
for key, (kind, builder) in MODELS.items():
    rec = {}
    for tag, feats in (("with_country", FULL), ("without_country", NOC)):
        _, _, _, v = run(feats, kind, builder, train, valid)
        m, ct, Xe, t = run(feats, kind, builder, trval, test)
        rec[tag] = {"valid": v, "test": t}
        if key == "xgb_meta_price" and tag == "without_country":
            final_noc = (m, ct, Xe)
    for s in ("valid", "test"):        # sanity: the with-country run reproduces the reported scores
        r = reported[(reported["model"] == key) & (reported["split"] == s)]["R2"].iloc[0]
        assert abs(r - rec["with_country"][s]["R2"]) < 2e-3, (key, s, r, rec["with_country"][s]["R2"])
    rec["test_R2_change"] = rec["without_country"]["test"]["R2"] - rec["with_country"]["test"]["R2"]
    out["prediction"][key] = rec
    print(key, {k: round(v["test"]["R2"], 4) for k, v in rec.items() if isinstance(v, dict)}, flush=True)

# ---------------------------------------------------------------- attribution
gs = pd.read_csv(MODEL_DIR / "shap_grouped_sample.csv")          # main XGB, per-row SHAP per original feature
grp = {"가격": gs["shap_log_price"], "빈티지": gs["shap_vintage"], "품종": gs["shap_variety"],
       "산지(국가+세부 산지)": gs["shap_country"] + gs["shap_province"]}
ma = {k: float(np.abs(v).mean()) for k, v in grp.items()}
out["shap_main_origin_combined_pct"] = {k: round(100 * v / sum(ma.values()), 2) for k, v in ma.items()}

m, ct, Xe = final_noc
rng = np.random.default_rng(SEED)
s_idx = np.sort(rng.choice(len(test), size=min(SHAP_N, len(test)), replace=False))   # same rows as 03_models
sv = np.asarray(shap.TreeExplainer(m).shap_values(Xe[s_idx]))
names = list(ct.get_feature_names_out())


def orig(n):
    if n.startswith("cat__"):
        return "세부 산지" if n[5:].startswith("province_") else "품종"
    return "가격" if n.startswith("num__") else "빈티지"


g = pd.DataFrame(sv, columns=names).T.groupby([orig(n) for n in names]).sum().T
ma2 = g.abs().mean()
out["shap_without_country_pct"] = {k: round(100 * float(v) / float(ma2.sum()), 2) for k, v in ma2.items()}
out["_note"] = ("Same hyperparameters as the main run, no re-tuning. Multicollinearity affects how credit is split "
                "between country and province, not how well the model predicts.")
(MODEL_DIR / "collinearity_check.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(out, ensure_ascii=False, indent=1))
