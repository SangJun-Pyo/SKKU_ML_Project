"""P2c: classical statistical inference (statsmodels) alongside the ML models.

Why this script exists
- ML metrics (test R², MAE, AUC) say how well we PREDICT.
- Classical OLS / Logit output (coefficients, standard errors, p-values, F-test, adj. R²,
  residual diagnostics) says whether the price effect is statistically distinguishable from 0
  and how large it is. Both views are reported side by side.

Protocol
- Fit on train+valid (same rows the final ML models are refit on); test R² reported once.
- Standard errors are heteroskedasticity-robust (HC3) because Breusch-Pagan rejects constant variance.
- With ~89k rows almost any effect is "significant" (p < 0.001): read effect sizes and R², not p alone.

Outputs (outputs/models/)
- stats_ols.json                       : numbers used by 08_key_numbers.py and the slides
- stats_ols_price_summary.txt          : full statsmodels summary, points ~ log_price
- stats_ols_full_summary_head.txt      : header + numeric-coefficient part of the full model summary
- stats_logit_price_summary.txt        : full statsmodels summary, (points >= 90) ~ log_price
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson, jarque_bera

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CLEAN_CSV, MODEL_DIR, ensure_dirs  # noqa: E402

ensure_dirs()
RARE_MIN = 20  # same rule as 03_models.py: levels with < 20 fitting rows -> "Other"

df = pd.read_csv(CLEAN_CSV)
fit = df[df["split"].isin(["train", "valid"])].reset_index(drop=True)
test = df[df["split"] == "test"].reset_index(drop=True)


def prep(frame, ref):
    out = frame.copy()
    med = ref["vintage"].median()
    out["vintage_missing"] = out["vintage"].isna().astype(int)
    out["vintage_c"] = out["vintage"].fillna(med) - med          # centered year
    for c in ["country", "province", "variety"]:
        keep = ref[c].value_counts()
        keep = set(keep[keep >= RARE_MIN].index)
        out[c] = np.where(out[c].isin(keep), out[c], "Other")
    return out


fit = prep(fit, fit)
test = prep(test, df[df["split"].isin(["train", "valid"])])
LN2 = np.log(2)


def r2(y, p):
    return 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)


def diag(res, X):
    jb, jb_p, skew, kurt = jarque_bera(res.resid)
    bp = het_breuschpagan(res.resid, X)
    return {"durbin_watson": round(float(durbin_watson(res.resid)), 3),
            "jarque_bera": round(float(jb), 1), "jarque_bera_p": float(jb_p),
            "resid_skew": round(float(skew), 3), "resid_kurtosis": round(float(kurt), 3),
            "breusch_pagan_lm": round(float(bp[0]), 1), "breusch_pagan_p": float(bp[1]),
            "condition_number": round(float(np.linalg.cond(res.model.exog)), 1)}


# ---------------------------------------------------------------- 1) correlation tests
pr = stats.pearsonr(fit["log_price"], fit["points"])
sr = stats.spearmanr(fit["price"], fit["points"])
corr = {"n": int(len(fit)), "pearson_r_logprice": round(float(pr.statistic), 4), "pearson_p": float(pr.pvalue),
        "pearson_ci95": [round(float(v), 4) for v in pr.confidence_interval(0.95)],
        "spearman_rho": round(float(sr.statistic), 4), "spearman_p": float(sr.pvalue)}

# ---------------------------------------------------------------- 2) OLS: points ~ log_price
m1_ols = smf.ols("points ~ log_price", data=fit).fit()
m1 = m1_ols.get_robustcov_results(cov_type="HC3")
names1 = m1_ols.model.exog_names
b = dict(zip(names1, m1.params)); se = dict(zip(names1, m1.bse))
tv = dict(zip(names1, m1.tvalues)); pv = dict(zip(names1, m1.pvalues))
ci = dict(zip(names1, m1.conf_int()))
ols_price = {
    "formula": "points ~ log_price", "n": int(m1.nobs), "se_type": "HC3",
    "r2": round(float(m1.rsquared), 4), "adj_r2": round(float(m1.rsquared_adj), 4),
    "f_stat": round(float(m1.fvalue), 1), "f_p": float(m1.f_pvalue),
    "aic": round(float(m1_ols.aic), 1), "bic": round(float(m1_ols.bic), 1),
    "coef": {k: {"b": round(float(b[k]), 4), "se": round(float(se[k]), 4), "t": round(float(tv[k]), 1),
                 "p": float(pv[k]), "ci95": [round(float(ci[k][0]), 4), round(float(ci[k][1]), 4)]}
             for k in names1},
    "points_per_price_doubling": round(float(b["log_price"] * LN2), 3),
    "points_per_price_doubling_ci95": [round(float(ci["log_price"][0] * LN2), 3),
                                       round(float(ci["log_price"][1] * LN2), 3)],
    "test_r2": round(float(r2(test["points"], m1_ols.predict(test))), 4),
    "diagnostics": diag(m1_ols, m1_ols.model.exog),
}
(MODEL_DIR / "stats_ols_price_summary.txt").write_text(str(m1.summary()), encoding="utf-8")

# ---------------------------------------------------------------- 3) OLS: full pre-review model
F_FULL = ("points ~ log_price + vintage_c + vintage_missing"
          " + C(country) + C(province) + C(variety)")
m2_ols = smf.ols(F_FULL, data=fit).fit()
m2 = m2_ols.get_robustcov_results(cov_type="HC3")
names2 = m2_ols.model.exog_names
b2 = dict(zip(names2, m2.params)); se2 = dict(zip(names2, m2.bse)); p2 = dict(zip(names2, m2.pvalues))
ci2 = dict(zip(names2, m2.conf_int()))
cat_terms = [n for n in names2 if n.startswith("C(")]
pt = test.copy()
for c in ["country", "province", "variety"]:          # unseen levels -> "Other"
    pt[c] = np.where(pt[c].isin(set(fit[c])), pt[c], "Other")
anova = anova_lm(m1_ols, m2_ols)                      # nested F-test: does meta add to price?
ols_full = {
    "formula": F_FULL, "n": int(m2.nobs), "se_type": "HC3",
    "n_params": int(len(names2)), "df_model": int(m2_ols.df_model),
    "r2": round(float(m2.rsquared), 4), "adj_r2": round(float(m2.rsquared_adj), 4),
    "f_stat": round(float(m2.fvalue), 1), "f_p": float(m2.f_pvalue),
    "aic": round(float(m2_ols.aic), 1), "bic": round(float(m2_ols.bic), 1),
    "coef_numeric": {k: {"b": round(float(b2[k]), 4), "se": round(float(se2[k]), 4),
                         "p": float(p2[k]), "ci95": [round(float(ci2[k][0]), 4), round(float(ci2[k][1]), 4)]}
                     for k in ["log_price", "vintage_c", "vintage_missing"]},
    "points_per_price_doubling": round(float(b2["log_price"] * LN2), 3),
    "n_category_dummies": len(cat_terms),
    "collinearity_note": ("province is nested in country, so category dummies are not separately identified "
                          "(huge condition number, rank-deficient design). R², F and the nested F-test are valid; "
                          "individual category coefficients / p-values are not interpreted."),
    "nested_f_vs_price_only": {"f": round(float(anova["F"].iloc[1]), 1), "p": float(anova["Pr(>F)"].iloc[1]),
                               "df_added": int(anova["df_diff"].iloc[1])},
    "test_r2": round(float(r2(pt["points"], m2_ols.predict(pt))), 4),
    "diagnostics": diag(m2_ols, m2_ols.model.exog),
}
full_txt = str(m2.summary())
head = full_txt.split("C(country)")[0]
num_rows = [ln for ln in full_txt.splitlines() if ln.strip().startswith(("log_price", "vintage_c", "vintage_missing"))]
tail = full_txt.split("==============================================================================")[-2:]
(MODEL_DIR / "stats_ols_full_summary_head.txt").write_text(
    head + "\n".join(num_rows) + "\n... (category dummies omitted) ...\n" + "=" * 78 + "\n" + tail[-2],
    encoding="utf-8")

# ---------------------------------------------------------------- 3b) robustness: cluster-robust SE by winery
# Reviews of the same winery are not independent (house style, same critic). Clustering by winery
# widens the SE of the price effect; if it stays far from 0, the conclusion does not hinge on independence.
g = pd.factorize(fit["winery"])[0]
m1_cl = smf.ols("points ~ log_price", data=fit).fit(cov_type="cluster", cov_kwds={"groups": g})
m2_cl = smf.ols(F_FULL, data=fit).fit(cov_type="cluster", cov_kwds={"groups": g})
cluster = {"groups": "winery", "n_groups": int(len(np.unique(g))),
           "price_only": {"se": round(float(m1_cl.bse["log_price"]), 4), "t": round(float(m1_cl.tvalues["log_price"]), 1),
                          "p": float(m1_cl.pvalues["log_price"]),
                          "ci95": [round(float(v), 4) for v in m1_cl.conf_int().loc["log_price"]]},
           "full": {"se": round(float(m2_cl.bse["log_price"]), 4), "t": round(float(m2_cl.tvalues["log_price"]), 1),
                    "p": float(m2_cl.pvalues["log_price"]),
                    "ci95": [round(float(v), 4) for v in m2_cl.conf_int().loc["log_price"]]}}
ols_price["cluster_robust"] = cluster["price_only"]
ols_full["cluster_robust"] = cluster["full"]

# ---------------------------------------------------------------- 4) Logit: (points >= 90) ~ log_price
fit["ge90"] = (fit["points"] >= 90).astype(int)
lg = smf.logit("ge90 ~ log_price", data=fit).fit(disp=0, cov_type="HC3")
lb = lg.params["log_price"]; lci = lg.conf_int().loc["log_price"]
logit_price = {
    "formula": "(points >= 90) ~ log_price", "n": int(lg.nobs),
    "coef_log_price": round(float(lb), 4), "p": float(lg.pvalues["log_price"]),
    "odds_ratio_per_doubling": round(float(np.exp(lb * LN2)), 3),
    "odds_ratio_per_doubling_ci95": [round(float(np.exp(lci.iloc[0] * LN2)), 3),
                                     round(float(np.exp(lci.iloc[1] * LN2)), 3)],
    "mcfadden_pseudo_r2": round(float(lg.prsquared), 4), "llr_p": float(lg.llr_pvalue),
}
(MODEL_DIR / "stats_logit_price_summary.txt").write_text(str(lg.summary()), encoding="utf-8")

out = {"_note": ("Classical inference on train+valid (n rows above). HC3 robust SEs. p-values printed as 0.0 "
                 "are below float precision (< 1e-300). With n ~ 89k, significance is expected; effect "
                 "size and R² carry the meaning."),
       "correlation": corr, "cluster_info": {k: cluster[k] for k in ("groups", "n_groups")}, "ols_price": ols_price, "ols_full": ols_full, "logit_price": logit_price}
(MODEL_DIR / "stats_ols.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({k: out[k] for k in ("correlation", "cluster_info")}, ensure_ascii=False))
print("price_only cluster:", ols_price["cluster_robust"]); print("full cluster:", ols_full["cluster_robust"])
print("full r2/adj/test:", ols_full["r2"], ols_full["adj_r2"], ols_full["test_r2"], "nestedF:", ols_full["nested_f_vs_price_only"])
