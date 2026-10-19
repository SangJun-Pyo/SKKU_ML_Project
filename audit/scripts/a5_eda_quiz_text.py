"""Audit 5: EDA correlations, price buckets, share>=90, quiz rows (from RAW csv); text-only sensitivity model."""
import json
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "audit" / "results"
df = pd.read_csv(ROOT / "data" / "processed" / "wine_clean.csv")
res = {}
lp = np.log(df.price)
res["max_abs_logprice_col_diff"] = float(np.abs(lp - df.log_price).max())
res["pearson_raw"] = float(np.corrcoef(df.price, df.points)[0, 1])
res["pearson_log"] = float(np.corrcoef(lp, df.points)[0, 1])
res["r2_log_insample"] = res["pearson_log"] ** 2
res["spearman"] = float(stats.spearmanr(df.price, df.points).statistic)
res["share_ge90_pct"] = float(100 * (df.points >= 90).mean())
res["price_min_med_max"] = [float(df.price.min()), float(df.price.median()), float(df.price.max())]
res["points_median"] = float(df.points.median())
res["n_countries"], res["n_provinces"], res["n_varieties"] = int(df.country.nunique()), int(df.province.nunique()), int(df.variety.nunique())
res["vintage_present_pct"] = float(100 * df.vintage.notna().mean())
edges = [0, 10, 20, 30, 50, 100, np.inf]
b = np.digitize(df.price, edges[1:-1], right=True)   # right=True -> (lo, hi]
names = ["<=$10", "($10,$20]", "($20,$30]", "($30,$50]", "($50,$100]", ">$100"]
res["price_buckets"] = {names[i]: {"n": int((b == i).sum()), "median": float(df.points[b == i].median()),
                                    "mean": round(float(df.points[b == i].mean()), 4),
                                    "share_ge90_pct": round(float(100 * (df.points[b == i] >= 90).mean()), 4)} for i in range(6)}
# quiz rows straight from the RAW file
raw = pd.read_csv(ROOT / "data" / "raw" / "winemag-data-130k-v2.csv")
q = {}
for rid in (15428, 37412):
    r = raw[raw["Unnamed: 0"] == rid].iloc[0]
    q[rid] = {"title": r.title, "price": float(r.price), "points": int(r.points), "variety": r.variety,
              "taster": r.taster_name, "in_clean": bool(rid in set(df.source_row_id)),
              "split": df.loc[df.source_row_id == rid, "split"].iloc[0]}
res["quiz"] = q
# other rows with identical title as the quiz wines (could a different vintage/score confuse the quiz?)
for rid in (15428, 37412):
    t = raw.loc[raw["Unnamed: 0"] == rid, "title"].iloc[0]
    res[f"quiz_{rid}_raw_rows_same_title"] = int((raw.title == t).sum())

# ---- text-only sensitivity: how much of the 77% is words alone? (fit train+valid, eval test; alpha fixed=1 like builder)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
tv, te = df[df.split.isin(["train", "valid"])], df[df.split == "test"]
vec = TfidfVectorizer(ngram_range=(1, 2), min_df=5, stop_words="english", sublinear_tf=True, max_features=20000, dtype=np.float32)
Xf, Xa = vec.fit_transform(tv.description), vec.transform(te.description)
p = Ridge(alpha=1.0).fit(Xf, tv.points).predict(Xa)
e = te.points - p
res["text_only_test_R2"] = float(1 - (e ** 2).sum() / ((te.points - te.points.mean()) ** 2).sum())
res["text_only_test_MAE"] = float(e.abs().mean())
# does description length alone carry signal? (known WE artefact: longer notes for higher scores)
L = df.description.str.len()
res["corr_desc_length_points"] = float(np.corrcoef(L, df.points)[0, 1])
json.dump(res, open(OUT / "a5_eda_quiz_text.json", "w"), indent=2, ensure_ascii=False, default=str)
print(json.dumps(res, indent=1, ensure_ascii=False, default=str))
