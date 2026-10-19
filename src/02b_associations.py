"""P1b: basic statistics per split + association / covariance / multicollinearity check.

1) Basic statistics (describe) for the whole cleaned data and for each split (train/valid/test):
   the three splits should look alike, otherwise the test score would not represent the data.
2) Association matrix across ALL model variables (target + 5 inputs). Most inputs are categorical, so one
   coefficient type cannot cover every pair:
     numeric-numeric   : Pearson r (signed)
     categorical-numeric: correlation ratio eta (0..1; = |r| when the categorical has 2 levels)
     categorical-categorical: bias-corrected Cramer's V (0..1)
   All three are on a 0..1 strength scale (|r| for the numeric pair), so they can share one heatmap.
3) Covariance matrix of the numeric variables (points, log_price, vintage). Covariance depends on units
   (points vs log-dollars vs years) and is not comparable across pairs; correlation = standardized covariance.
4) Multicollinearity: generalized VIF (Fox & Monette 1992) per input variable on the one-hot design,
   GVIF^(1/(2*df)) is comparable to sqrt(VIF) (rule of thumb: > sqrt(10) ~ 3.2 is a problem).

Outputs (outputs/eda/): split_describe.csv, describe_numeric.csv, assoc_matrix.csv, assoc_long.csv,
cov_numeric.csv, gvif.csv, assoc_summary.json; outputs/figures_ko/fig13_assoc.png
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CLEAN_CSV, EDA_DIR, ROOT, ensure_dirs, setup_fonts  # noqa: E402

ensure_dirs()
setup_fonts()
FIG_DIR = ROOT / "outputs" / "figures_ko"
FIG_DIR.mkdir(parents=True, exist_ok=True)
RARE_MIN = 20

df = pd.read_csv(CLEAN_CSV)
fitmask = df["split"].isin(["train", "valid"])
for c in ["country", "province", "variety"]:                       # same rare-level rule as the models
    vc = df.loc[fitmask, c].value_counts()
    keep = set(vc[vc >= RARE_MIN].index)
    df[c + "_g"] = np.where(df[c].isin(keep), df[c], "기타")

# ---------------------------------------------------------------- 1) basic statistics
def block(d, name):
    return {"split": name, "n": len(d),
            "points_mean": d["points"].mean(), "points_std": d["points"].std(), "points_min": d["points"].min(),
            "points_median": d["points"].median(), "points_max": d["points"].max(),
            "ge90_pct": 100 * (d["points"] >= 90).mean(),
            "price_mean": d["price"].mean(), "price_median": d["price"].median(), "price_std": d["price"].std(),
            "price_min": d["price"].min(), "price_max": d["price"].max(),
            "log_price_mean": d["log_price"].mean(), "log_price_std": d["log_price"].std(),
            "vintage_median": d["vintage"].median(), "vintage_missing_pct": 100 * d["vintage"].isna().mean(),
            "n_countries": d["country"].nunique(), "n_varieties": d["variety"].nunique(),
            "us_share_pct": 100 * (d["country"] == "US").mean()}

split_desc = pd.DataFrame([block(df, "all")] + [block(df[df["split"] == s], s) for s in ("train", "valid", "test")])
split_desc.to_csv(EDA_DIR / "split_describe.csv", index=False, encoding="utf-8")
df[["points", "price", "log_price", "vintage"]].describe().T.to_csv(EDA_DIR / "describe_numeric.csv", encoding="utf-8")
# are the splits alike? KS test of points distribution train vs test (descriptive only)
ks = stats.ks_2samp(df.loc[df["split"] == "train", "points"], df.loc[df["split"] == "test", "points"])

# ---------------------------------------------------------------- 2) association matrix
VARS = [("points", "평점", "num"), ("log_price", "로그 가격", "num"), ("vintage", "빈티지", "num"),
        ("country_g", "국가", "cat"), ("province_g", "세부 산지", "cat"), ("variety_g", "품종", "cat")]


def eta(cat, num):
    m = num.notna()
    cat, num = cat[m], num[m]
    g = num.groupby(cat)
    ss_between = (g.count() * (g.mean() - num.mean()) ** 2).sum()
    ss_total = ((num - num.mean()) ** 2).sum()
    return float(np.sqrt(ss_between / ss_total))


def cramers_v(a, b):
    ct = pd.crosstab(a, b).to_numpy()
    chi2 = stats.chi2_contingency(ct, correction=False)[0]
    n = ct.sum()
    r, k = ct.shape
    phi2 = chi2 / n
    phi2c = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))       # Bergsma (2013) bias correction
    rc, kc = r - (r - 1) ** 2 / (n - 1), k - (k - 1) ** 2 / (n - 1)
    return float(np.sqrt(phi2c / min(kc - 1, rc - 1)))


names = [v[1] for v in VARS]
M = pd.DataFrame(np.eye(len(VARS)), index=names, columns=names)
long = []
for i, (ci, ni, ti) in enumerate(VARS):
    for j, (cj, nj, tj) in enumerate(VARS):
        if j <= i:
            continue
        if ti == "num" and tj == "num":
            d = df[[ci, cj]].dropna()
            val, kind = float(stats.pearsonr(d[ci], d[cj]).statistic), "Pearson r"
        elif ti == "cat" and tj == "cat":
            val, kind = cramers_v(df[ci], df[cj]), "Cramer's V"
        else:
            cat, num = (df[ci], df[cj]) if ti == "cat" else (df[cj], df[ci])
            val, kind = eta(cat, num), "correlation ratio (eta)"
        M.iloc[i, j] = M.iloc[j, i] = val
        long.append({"var1": ni, "var2": nj, "measure": kind, "value": val})
M.to_csv(EDA_DIR / "assoc_matrix.csv", encoding="utf-8")
pd.DataFrame(long).to_csv(EDA_DIR / "assoc_long.csv", index=False, encoding="utf-8")

# ---------------------------------------------------------------- 3) covariance (numeric)
cov = df[["points", "log_price", "vintage"]].cov()
cov.to_csv(EDA_DIR / "cov_numeric.csv", encoding="utf-8")
corr_num = df[["points", "log_price", "vintage"]].corr()

# ---------------------------------------------------------------- 4) generalized VIF on the fitting rows
fit = df[fitmask].copy()
fit["vintage_missing"] = fit["vintage"].isna().astype(float)
fit["vintage_f"] = fit["vintage"].fillna(fit["vintage"].median())
groups = {"국가": pd.get_dummies(fit["country_g"], prefix="c", drop_first=True, dtype=float),
          "세부 산지": pd.get_dummies(fit["province_g"], prefix="p", drop_first=True, dtype=float),
          "품종": pd.get_dummies(fit["variety_g"], prefix="v", drop_first=True, dtype=float),
          "로그 가격": fit[["log_price"]].astype(float),
          "빈티지": fit[["vintage_f", "vintage_missing"]].astype(float)}


def gvif_table(gs):
    X = pd.concat(gs.values(), axis=1)
    X = X.loc[:, X.std() > 0]
    R = np.corrcoef(X.to_numpy(), rowvar=False)
    cols = list(X.columns)
    _, logdet_all = np.linalg.slogdet(R)
    out = []
    for name, block_ in gs.items():
        idx = [cols.index(c) for c in block_.columns if c in cols]
        rest = [k for k in range(len(cols)) if k not in idx]
        _, ld1 = np.linalg.slogdet(R[np.ix_(idx, idx)])
        _, ld2 = np.linalg.slogdet(R[np.ix_(rest, rest)])
        log_gvif = ld1 + ld2 - logdet_all
        dfj = len(idx)
        out.append({"variable": name, "df": dfj, "GVIF": float(np.exp(min(log_gvif, 700))),
                    "GVIF_1_2df": float(np.exp(min(log_gvif / (2 * dfj), 700)))})
    return out


g_all = gvif_table(groups)
g_noc = gvif_table({k: v for k, v in groups.items() if k != "국가"})
gv = pd.DataFrame([{**r, "design": "all 5 inputs"} for r in g_all] +
                  [{**r, "design": "without country (province already implies it)"} for r in g_noc])
gv.to_csv(EDA_DIR / "gvif.csv", index=False, encoding="utf-8")

# ---------------------------------------------------------------- figure: heatmap
fig, ax = plt.subplots(figsize=(11.84, 6.66), dpi=200)
fig.patch.set_facecolor("#fdfbf8")
A = M.abs().to_numpy()
n = len(names)
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
cmap = LinearSegmentedColormap.from_list("wine", ["#fbf6ef", "#e6c3c9", "#b5606f", "#7b2335"])
A_show = A.copy()
np.fill_diagonal(A_show, np.nan)
cmap.set_bad("#ece5da")
ax.imshow(A_show, cmap=cmap, vmin=0, vmax=1, aspect="auto")
for i in range(n):
    for j in range(n):
        if i == j:
            txt, col = "—", "#7a6f74"
        else:
            v = M.iloc[i, j]
            txt = f"{v:+.2f}" if (VARS[i][2] == "num" and VARS[j][2] == "num") else f"{v:.2f}"
            col = "#fdfbf8" if A[i, j] > 0.55 else "#231a1d"
        ax.text(j, i, txt, ha="center", va="center", fontsize=17, color=col,
                fontweight="bold" if (i != j and A[i, j] >= 0.5) else "normal")
ax.set_xticks(range(n)); ax.set_yticks(range(n))
ax.set_xticklabels(names, fontsize=16); ax.set_yticklabels(names, fontsize=16)
ax.tick_params(length=0)
ax.xaxis.tick_top()
for s in ax.spines.values():
    s.set_visible(False)
ax.set_xticks(np.arange(-.5, n, 1), minor=True); ax.set_yticks(np.arange(-.5, n, 1), minor=True)
ax.grid(which="minor", color="#fdfbf8", linewidth=3); ax.tick_params(which="minor", length=0)
fig.text(0.5, 0.025, "숫자-숫자: 피어슨 r · 범주-숫자: 상관비 η · 범주-범주: 크래머 V  (0 = 무관, 1 = 완전히 겹침)",
         ha="center", fontsize=13, color="#5b5257")
fig.subplots_adjust(left=0.13, right=0.98, top=0.9, bottom=0.08)
fig.savefig(FIG_DIR / "fig13_assoc.png", dpi=200, facecolor=fig.get_facecolor())

summary = {
    "split_describe": json.loads(split_desc.round(4).to_json(orient="records")),
    "ks_points_train_vs_test": {"stat": float(ks.statistic), "p": float(ks.pvalue)},
    "assoc": {f"{r['var1']}~{r['var2']}": {"measure": r["measure"], "value": round(r["value"], 4)} for r in long},
    "cov_numeric": json.loads(cov.round(4).to_json()),
    "corr_numeric": json.loads(corr_num.round(4).to_json()),
    "gvif": json.loads(gv.round(4).to_json(orient="records")),
}
(EDA_DIR / "assoc_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(split_desc.round(2).T.to_string())
print(M.round(3).to_string())
print(cov.round(3).to_string())
print(gv.round(2).to_string())
print("KS train vs test points:", ks)
