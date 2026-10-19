"""Audit 6: SHAP share — (a) recompute from the saved grouped sample, (b) recompute on my own XGB refit."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "audit" / "results"
res = {}
g = pd.read_csv(ROOT / "outputs" / "models" / "shap_grouped_sample.csv")
cols = [c for c in g.columns if c.startswith("shap_")]
ma = g[cols].abs().mean()
res["saved_sample_share_pct"] = {c[5:]: round(float(100 * v / ma.sum()), 4) for c, v in ma.items()}
res["saved_sample_additivity_maxerr"] = float((g[cols].sum(axis=1) + g.base_value - g.pred).abs().max())

# (b) own refit + TreeExplainer on 5,000 random test rows (own seed)
df = pd.read_csv(ROOT / "data" / "processed" / "wine_clean.csv")
tv, te = df[df.split.isin(["train", "valid"])].reset_index(drop=True), df[df.split == "test"].reset_index(drop=True)
import scipy.sparse as sp, xgboost as xgb, shap
def build(fit, app):
    F, A, grp = [], [], []
    for c in ["country", "province", "variety"]:
        vc = fit[c].value_counts(); keep = sorted(vc[vc >= 20].index); idx = {l: i for i, l in enumerate(keep)}
        n = len(keep) + 1
        for d, L in ((fit, F), (app, A)):
            j = d[c].map(idx).fillna(len(keep)).astype(int).to_numpy()
            L.append(sp.csr_matrix((np.ones(len(d)), (np.arange(len(d)), j)), shape=(len(d), n)))
        grp += [c] * n
    med = fit.vintage.median()
    for d, L in ((fit, F), (app, A)):
        L.append(sp.csr_matrix(np.column_stack([d.log_price, d.vintage.fillna(med), d.vintage.isna()]).astype(float)))
    grp += ["log_price", "vintage", "vintage"]
    return sp.hstack(F).toarray().astype(np.float32), sp.hstack(A).toarray().astype(np.float32), np.array(grp)
Xf, Xa, grp = build(tv, te)
m = xgb.XGBRegressor(learning_rate=0.1, subsample=0.8, tree_method="hist", max_depth=8, min_child_weight=1,
                     colsample_bytree=0.8, n_estimators=420, n_jobs=2, random_state=7).fit(Xf, tv.points)
rng = np.random.default_rng(99)
s = rng.choice(len(te), 5000, replace=False)
sv = shap.TreeExplainer(m).shap_values(Xa[s])
G = pd.DataFrame(sv).T.groupby(grp).sum().T
ma2 = G.abs().mean()
res["own_refit_share_pct"] = {k: round(float(100 * v / ma2.sum()), 4) for k, v in ma2.items()}
# mean-|SHAP| share is NOT variance explained; report how a 'drop-price' comparison looks for context
json.dump(res, open(OUT / "a6_shap.json", "w"), indent=2)
print(json.dumps(res, indent=1))
