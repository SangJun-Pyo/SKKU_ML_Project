"""Builds notebooks/wine_stats_colab.ipynb (run: python notebooks/build_notebook.py)."""
from pathlib import Path

import nbformat as nbf

nb = nbf.v4.new_notebook()
C = []


def md(s):
    C.append(nbf.v4.new_markdown_cell(s.strip()))


def code(s):
    C.append(nbf.v4.new_code_cell(s.strip()))


md("""
# 비싼 와인은 진짜 맛있을까? — 통계 지표 한눈에 보기

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SangJun-Pyo/SKKU_ML_Project/blob/main/notebooks/wine_stats_colab.ipynb)

발표에 쓴 통계 지표를 `print`로 모두 확인하는 노트북입니다. 세 부분으로 되어 있습니다.

| 부분 | 내용 | 데이터 필요 | 시간 |
|---|---|---|---|
| **1부** | 저장된 결과(`outputs/key_numbers.json`)를 주제별로 출력 | 없음 | 즉시 |
| **2부** | 원본 데이터로 핵심 통계를 직접 다시 계산 (OLS 요약표, 상관, 모델 성능 등) | Kaggle CSV | 약 2분 (트리 모델 포함 시 10~15분) |
| **3부** | 전체 분석 파이프라인 재실행 (시뮬레이션·SHAP 포함) | Kaggle CSV | 약 10~15분 |

Colab에서는 위에서부터 차례로 실행(`Shift + Enter`)하면 됩니다.
""")

md("## 0. 준비: 저장소 내려받기")
code(r'''
import os, sys, subprocess
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules
REPO_URL = "https://github.com/SangJun-Pyo/SKKU_ML_Project.git"

if IN_COLAB:
    if not Path("SKKU_ML_Project").exists():
        subprocess.run(["git", "clone", "--depth", "1", REPO_URL], check=True)
    os.chdir("SKKU_ML_Project")
elif not Path("src").exists() and Path("../src").exists():   # 로컬에서 notebooks/ 안에서 연 경우
    os.chdir("..")

ROOT = Path.cwd()
print("작업 폴더:", ROOT.name)
print("statsmodels, xgboost 설치 확인:")
for pkg in ["pandas", "numpy", "scipy", "sklearn", "statsmodels", "xgboost"]:
    try:
        m = __import__(pkg)
        print(f"  {pkg:12s} {m.__version__}")
    except ImportError:
        print(f"  {pkg:12s} 없음 → 설치합니다")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg if pkg != "sklearn" else "scikit-learn"], check=True)
''')

md("""
---
# 1부. 저장된 결과 출력 (데이터 없이 바로 실행)

`outputs/key_numbers.json`은 `src/08_key_numbers.py`가 모든 분석 결과를 모아 둔 파일로, 슬라이드의 숫자는 전부 여기서 나왔습니다.
""")
code(r'''
import json
import numpy as np
import pandas as pd

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 30)
pd.set_option("display.float_format", lambda v: f"{v:,.4f}")

K = json.loads((ROOT / "outputs" / "key_numbers.json").read_text(encoding="utf-8"))


def title(t):
    print("\n" + "=" * 78 + f"\n {t}\n" + "=" * 78)


def kv(d, keys=None, indent=2):
    for k in (keys or d):
        v = d[k]
        if isinstance(v, float):
            v = f"{v:,.4f}"
        print(" " * indent + f"{k:<38s} {v}")

print("불러온 항목:", ", ".join(k for k in K if not k.startswith("_")))
''')

md("### 1-1. 데이터: 원본, 정제, 분할")
code(r'''
title("원본 데이터")
kv(K["raw"], ["raw_rows", "raw_cols", "price_missing", "exact_duplicates_excl_id"])
print("\n  열별 결측률(%)")
print(pd.Series(K["raw"]["missing_pct_by_column"]).drop("Unnamed: 0").sort_values(ascending=False).to_string())

title("정제 단계")
print(pd.DataFrame(K["data"]["cleaning_steps"]).to_string(index=False))

title("학습 / 검증 / 시험 분할 (와인 이름 기준 그룹 분할)")
sr = pd.Series(K["data"]["split_rows"])
print(pd.DataFrame({"건수": sr, "비율(%)": (sr / sr.sum() * 100).round(1)}).to_string())
print("\n  분할 사이 겹치는 와인 이름 수:", K["data"]["title_overlap_between_splits"])
''')

md("### 1-2. 기초 통계량과 분포 비교 (pandas describe, KS 검정)")
code(r'''
title("분할별 기초 통계량")
d = pd.DataFrame(K["associations"]["split_describe"]).set_index("split")
cols = ["n", "points_mean", "points_std", "points_min", "points_median", "points_max", "ge90_pct",
        "price_median", "price_mean", "log_price_mean", "log_price_std", "vintage_median", "vintage_missing_pct", "us_share_pct"]
print(d[cols].T.to_string())

ks = K["associations"]["ks_points_train_vs_test"]
title("KS 검정: 학습 vs 시험 평점 분포")
print(f"  KS 통계량 = {ks['stat']:.4f},  p-value = {ks['p']:.4f}  → p가 크므로 두 분포가 다르다고 볼 근거 없음")
''')

md("### 1-3. 가격과 평점")
code(r'''
title("가격과 평점의 상관")
kv(K["eda"], ["pearson_r_price_points", "pearson_r_logprice_points", "spearman_rho_price_points",
              "r2_logprice_points_in_sample", "share_ge90_pct", "price_usd_min", "price_usd_median", "price_usd_max"])

title("가격대별 평점과 90점 이상 비율")
print(pd.DataFrame(K["eda"]["price_buckets"]).to_string(index=False))
''')

md("### 1-4. 변수 간 상관, 공분산, 다중공선성(GVIF)")
code(r'''
title("변수 쌍별 연관도 (숫자-숫자: 피어슨 r, 숫자-범주: 상관비 eta, 범주-범주: 크래머 V)")
a = pd.DataFrame([{"쌍": k, "척도": v["measure"], "값": v["value"]} for k, v in K["associations"]["assoc"].items()])
print(a.sort_values("값", ascending=False).to_string(index=False))

title("공분산 행렬 (단위가 달라 크기 비교 불가)")
print(pd.DataFrame(K["associations"]["cov_numeric"]).to_string())
title("상관 행렬 (공분산을 표준화한 것)")
print(pd.DataFrame(K["associations"]["corr_numeric"]).to_string())

title("GVIF^(1/(2·df)) — √10 ≈ 3.2를 넘으면 다중공선성 의심")
g = pd.DataFrame(K["associations"]["gvif"])[["design", "variable", "df", "GVIF_1_2df"]]
print(g.to_string(index=False))
''')

md("### 1-5. 통계 검정 (statsmodels OLS · 로지스틱)")
code(r'''
S = K["stats"]
title("상관 검정 (학습+검증)")
kv(S["correlation"])

for key, name in [("ols_price", "OLS: 평점 ~ 로그 가격"), ("ols_full", "OLS: 평점 ~ 로그 가격 + 빈티지 + 국가 + 세부 산지 + 품종")]:
    o = S[key]
    title(name + f"   (n = {o['n']:,}, 표준오차 {o['se_type']})")
    print(f"  R² = {o['r2']:.4f}   수정 R² = {o['adj_r2']:.4f}   F = {o['f_stat']:,}  (p = {o['f_p']:.3g})   시험 데이터 R² = {o['test_r2']:.4f}")
    coefs = o.get("coef") or o.get("coef_numeric")
    rows = [{"변수": k, "계수": v["b"], "표준오차": v["se"], "t": v.get("t", v["b"] / v["se"]), "p-value": v["p"],
             "95% CI 하한": v["ci95"][0], "95% CI 상한": v["ci95"][1]} for k, v in coefs.items()]
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\n  → 가격이 2배가 되면 평점 +{o['points_per_price_doubling']:.2f}점")
    print("  잔차 진단:"); kv(o["diagnostics"], indent=4)
    print("  와이너리 군집 표준오차:"); kv(o["cluster_robust"], indent=4)
    if key == "ols_full":
        print("  가격만 모델 대비 F-검정(산지·품종·빈티지 추가 효과):", o["nested_f_vs_price_only"])

title("로지스틱 회귀: (평점 ≥ 90) ~ 로그 가격")
kv(S["logit_price"])
''')

md("### 1-6. 회귀 모델 성능 (평점 숫자 맞히기)")
code(r'''
title("모델별 검증·시험 성능 (MAE = 평균 오차 점수, R² = 설명력)")
lad = pd.DataFrame(K["models"]["ladder"])[["model", "label", "valid_MAE_points", "test_MAE_points", "valid_R2", "test_R2"]]
print(lad.to_string(index=False))
print("\n  메인 모델:", K["models"]["main_model"], "/ 선택 기준:", K["models"]["main_model_rule"])
print("  XGBoost 최종 설정:", K["models"]["xgb_chosen_params"])

title("규제 방식 비교 (Ridge vs Lasso vs ElasticNet, 같은 변수)")
rg = K["regularization"]
rows = []
for m in ["ridge_meta_price", "lasso_meta_price", "enet_meta_price"]:
    r = rg[m]
    rows.append({"모델": m, "설정": r["params"], "시험 MAE": r["test_MAE_points"], "시험 R²": r["test_R2"],
                 "0이 된 계수": f"{r['zero_coefficients']['n_coef_zero']} / {r['zero_coefficients']['n_coef_total']}"})
print(pd.DataFrame(rows).to_string(index=False))
''')

md("### 1-7. 분류 모델 성능 (90점 이상 가려내기)")
code(r'''
cl = K["classification"]
title(f"90점 이상 분류 — 시험 데이터 90점 이상 비율 {cl['test_base_rate_ge90_pct']:.1f}%")
rows = []
for m, v in cl["models"].items():
    rows.append({"모델": m, "검증 AUC": v.get("valid_roc_auc"), "시험 AUC": v.get("test_roc_auc"),
                 "시험 PR-AUC": v.get("test_pr_auc"), "시험 log loss": v.get("test_log_loss"),
                 "시험 정확도(0.5)": v.get("test_accuracy_at_0_5")})
print(pd.DataFrame(rows).to_string(index=False))
print("\n  ※ xgb_reg_score = 회귀 XGBoost의 예측 평점을 순위로만 쓴 참고값 (분류 모델 아님)")

title("로지스틱(전체 변수) 대비 시험 AUC 차이와 95% 신뢰구간")
print(pd.DataFrame(cl["test_auc_diff_vs_logit_all"]).T.to_string())
''')

md("### 1-8. 과적합 점검 (학습 / 검증 / 시험)")
code(r'''
title("학습·검증·시험 성능 비교")
ov = pd.DataFrame(K["overfit"]["rows"])
reg = ov[ov.task == "regression"][["label", "train_R2", "valid_R2", "test_R2", "train_MAE", "valid_MAE", "test_MAE"]]
clf = ov[ov.task == "classification"][["label", "train_AUC", "valid_AUC", "test_AUC"]]
print(reg.to_string(index=False)); print(); print(clf.to_string(index=False))
print("\n ", K["overfit"]["_note"])
''')

md("### 1-9. 다중공선성 재학습과 SHAP (모델이 무엇을 보고 예측했나)")
code(r'''
co = K["collinearity"]
title("국가를 빼고 다시 학습했을 때 시험 R² 변화")
for m in ["ridge_meta_price", "xgb_meta_price"]:
    p = co["prediction"][m]
    print(f"  {m:18s} 국가 포함 {p['with_country']['test']['R2']:.4f} → 국가 제외 {p['without_country']['test']['R2']:.4f}  (변화 {p['test_R2_change']:+.4f})")

title("SHAP 변수별 몫(%) — 국가+세부 산지 합산 / 국가 제외 재학습")
print(pd.DataFrame({"합산": pd.Series(co["shap_main_origin_combined_pct"]),
                    "국가 제외": pd.Series(co["shap_without_country_pct"])}).to_string())
''')

md("### 1-10. 예산 안에서 고르기 시뮬레이션")
code(r'''
sim = K["simulation"]
print(f"시험 데이터에서 예산 이하 와인 {sim['market_size']}병을 뽑아 1병 고르기 × {sim['n_markets']:,}번, 부트스트랩 {sim['n_boot']:,}회")
for b, v in sim["by_budget"].items():
    title(f"예산 {b.replace('usd_', '$')}")
    st = pd.DataFrame(v["strategies"]).T[["share_ge90_pct", "mean_points", "mean_price_usd"]]
    st.columns = ["90점 이상(%)", "평균 평점", "평균 가격($)"]
    print(st.to_string())
    dd = {k: v2 for k, v2 in v["diffs"].items() if "ge90" in k and "Random" not in k}
    print("\n  90점 이상 비율 차이(%p)와 95% CI")
    print(pd.DataFrame(dd).T.to_string())
''')

md("### 1-11. 가성비 지도와 평론가 효과")
code(r'''
vm = K["value_map"]
title("가격만으로 예상한 평점 대비 실제 평점 차이 — 국가")
print(pd.DataFrame(vm["country_top5"] + vm["country_bottom5"]).to_string(index=False))
title("같은 기준 — 품종")
print(pd.DataFrame(vm["variety_top5"] + vm["variety_bottom5"]).to_string(index=False))

cc = K["critic_check"]
title("평론가 효과: 나라별 주 평론가와 같은 평론가 안에서 비교한 우위")
rows = []
for c in ["austria", "germany", "portugal", "argentina", "chile"]:
    rows.append({"국가": c, "주 평론가": cc[f"critic_{c}_top_critic"], "점유율(%)": cc[f"critic_{c}_top_critic_share_pct"],
                 "평론가 평균 평점": cc[f"critic_{c}_top_critic_overall_mean_points"],
                 "단순 비교": cc[f"critic_{c}_raw_residual_points"],
                 "같은 평론가 안": cc[f"critic_{c}_within_critic_premium_points"],
                 "95% CI": cc[f"critic_{c}_within_critic_ci95"]})
print(pd.DataFrame(rows).to_string(index=False))
print("\n ", cc["critic_ci_note"])
''')

md("### 1-12. 리뷰 본문 보조 분석과 퀴즈 와인")
code(r'''
t = K["text_aux_model"]
title("리뷰 본문(TF-IDF) 보조 모델 — 사후 정보라 예측 성능으로 보지 않음")
print(f"  리뷰 단어만: 시험 R² {t['text_only_test_R2']:.4f}, MAE {t['text_only_test_MAE_points']:.3f}")
print(f"  단어 + 가격·산지 등: 시험 R² {t['test_R2']:.4f}, MAE {t['test_MAE_points']:.3f}")
print("  평점을 올리는 단어:", ", ".join(x["term"] for x in t["top_positive_terms"]))
print("  평점을 깎는 단어:  ", ", ".join(x["term"] for x in t["top_negative_terms"]))

title("퀴즈 와인")
for k, v in K["quiz_pair"].items():
    if not k.startswith("_"):
        print(f"  {v['title']:<58s} ${v['price']:>6.0f}  {v['points']}점  평론가: {v['taster_name']}")
''')

md("""
---
# 2부. 원본 데이터로 직접 다시 계산하기

1부의 숫자를 저장 파일에서 읽는 대신 **직접 계산**해 봅니다. 먼저 Kaggle 원본 CSV(`winemag-data-130k-v2.csv`)가 필요합니다.

- **방법 A (자동)**: 아래 셀이 `kagglehub`로 내려받기를 시도합니다.
- **방법 B (업로드)**: 자동이 실패하면 Colab 파일 업로드 창이 뜹니다. [Kaggle 페이지](https://www.kaggle.com/datasets/zynicide/wine-reviews)에서 받은 CSV를 고르세요.
- 로컬에서는 `data/raw/`에 파일을 넣어 두면 됩니다.
""")
code(r'''
import shutil

RAW = ROOT / "data" / "raw" / "winemag-data-130k-v2.csv"
RAW.parent.mkdir(parents=True, exist_ok=True)

if not RAW.exists():
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "kagglehub"], check=True)
        import kagglehub
        p = Path(kagglehub.dataset_download("zynicide/wine-reviews"))
        shutil.copy(p / "winemag-data-130k-v2.csv", RAW)
        print("kagglehub로 내려받음")
    except Exception as e:
        print("자동 다운로드 실패:", type(e).__name__, e)
        if IN_COLAB:
            from google.colab import files
            print("winemag-data-130k-v2.csv 파일을 골라 주세요")
            up = files.upload()
            name = next(iter(up))
            shutil.move(name, RAW)

print("원본 파일:", RAW, "존재" if RAW.exists() else "없음 — 위 안내대로 파일을 넣어 주세요")
''')

md("### 2-1. 정제와 분할 (`src/01_clean.py`를 그대로 실행)")
code(r'''
r = subprocess.run([sys.executable, "src/01_clean.py"], capture_output=True, text=True)
print(r.stdout[-2000:]); print(r.stderr[-2000:])

df = pd.read_csv(ROOT / "data" / "processed" / "wine_clean.csv")
train, valid, test = (df[df.split == s].reset_index(drop=True) for s in ["train", "valid", "test"])
trval = pd.concat([train, valid], ignore_index=True)
print(f"정제 후 {len(df):,}건  →  학습 {len(train):,} / 검증 {len(valid):,} / 시험 {len(test):,}")
''')

md("### 2-2. 기초 통계량 (describe)과 KS 검정")
code(r'''
from scipy import stats

title("분할별 describe — 평점")
print(df.groupby("split")["points"].describe().loc[["train", "valid", "test"]].to_string())
title("분할별 describe — 가격(USD)")
print(df.groupby("split")["price"].describe().loc[["train", "valid", "test"]].to_string())
title("분할별 90점 이상 비율(%)")
print((df.assign(ge90=df.points >= 90).groupby("split")["ge90"].mean() * 100).round(2).to_string())

ks = stats.ks_2samp(train.points, test.points)
print(f"\nKS 검정(학습 vs 시험 평점): 통계량 {ks.statistic:.4f}, p-value {ks.pvalue:.4f}")
''')

md("### 2-3. 상관계수와 가격대별 90점 이상 비율")
code(r'''
title("상관계수 (전체 데이터)")
print(f"  피어슨 r (가격, 평점)      = {stats.pearsonr(df.price, df.points).statistic:.4f}")
print(f"  피어슨 r (로그 가격, 평점) = {stats.pearsonr(df.log_price, df.points).statistic:.4f}")
print(f"  스피어만 ρ (가격, 평점)    = {stats.spearmanr(df.price, df.points).statistic:.4f}")

title("숫자 변수 공분산 / 상관 행렬")
num = df[["points", "log_price", "vintage"]]
print(num.cov().to_string()); print(); print(num.corr().to_string())

bins = [0, 10, 20, 30, 50, 100, np.inf]
labels = ["$10 이하", "$10~20", "$20~30", "$30~50", "$50~100", "$100 초과"]
bk = df.assign(가격대=pd.cut(df.price, bins, labels=labels))
title("가격대별 평점")
print(bk.groupby("가격대", observed=True).agg(건수=("points", "size"), 평균평점=("points", "mean"),
                                             비율_90점이상=("points", lambda s: (s >= 90).mean() * 100)).to_string())
''')

md("""
### 2-4. statsmodels 회귀분석 요약표

통계 수업에서 보는 형태의 출력입니다. 표준오차는 이분산에 강한 **HC3**를 썼습니다.
읽는 법: `coef` = 기울기, `P>|z|` = p-value, `[0.025 0.975]` = 95% 신뢰구간, `R-squared` = 설명력, `Durbin-Watson` = 잔차의 독립성, `Cond. No.` = 다중공선성 지표.
""")
code(r'''
import statsmodels.formula.api as smf
from statsmodels.stats.diagnostic import het_breuschpagan

fit = df[df.split.isin(["train", "valid"])].reset_index(drop=True)   # 원래 파일 순서 유지 (Durbin-Watson은 행 순서에 따라 달라짐)
m1 = smf.ols("points ~ log_price", data=fit).fit(cov_type="HC3")
print(m1.summary())

bp = het_breuschpagan(m1.resid, m1.model.exog)
print("\n※ Durbin-Watson은 행 순서에 따라 값이 바뀝니다. 이 데이터는 시계열이 아니라서, 0.78은 '이웃 행(같은 와이너리 등)끼리 닮았다'는 신호로만 읽고 군집 표준오차로 재확인했습니다.")
print(f"Breusch-Pagan 검정 p = {bp[1]:.3g}  → 잔차 분산이 일정하지 않음 → HC3 보정 사용")
print(f"가격 2배일 때 평점 변화 = {m1.params['log_price'] * np.log(2):+.3f}점")
print(f"평점 1점을 올리는 가격 배수 = e^(1/{m1.params['log_price']:.2f}) = {np.exp(1 / m1.params['log_price']):.3f}배")
pred = m1.predict(test)
print(f"시험 데이터 R² = {1 - ((test.points - pred) ** 2).sum() / ((test.points - test.points.mean()) ** 2).sum():.4f}")
''')

md("### 2-5. 전체 변수 OLS, 와이너리 군집 표준오차, 로지스틱 오즈비")
code(r'''
RARE_MIN = 20


def prep(frame, ref):
    out = frame.copy()
    med = ref["vintage"].median()
    out["vintage_missing"] = out["vintage"].isna().astype(int)
    out["vintage_c"] = out["vintage"].fillna(med) - med
    for c in ["country", "province", "variety"]:
        vc = ref[c].value_counts()
        out[c] = np.where(out[c].isin(set(vc[vc >= RARE_MIN].index)), out[c], "Other")
    return out


fp = prep(fit, fit)
F_FULL = "points ~ log_price + vintage_c + vintage_missing + C(country) + C(province) + C(variety)"
m2 = smf.ols(F_FULL, data=fp).fit(cov_type="HC3")
title("전체 변수 OLS (범주 계수는 국가·세부 산지가 겹쳐 해석하지 않음)")
print(f"  R² = {m2.rsquared:.4f}, 수정 R² = {m2.rsquared_adj:.4f}, 파라미터 {len(m2.params)}개")
print(m2.summary2().tables[1].loc[["log_price", "vintage_c", "vintage_missing"]].to_string())

g = pd.factorize(fit["winery"])[0]
mc = smf.ols("points ~ log_price", data=fit).fit(cov_type="cluster", cov_kwds={"groups": g})
title("와이너리 군집 표준오차 (같은 와이너리 와인끼리 독립이 아닐 수 있음)")
print(f"  와이너리 {len(np.unique(g)):,}개 · 로그 가격 계수 {mc.params['log_price']:.4f}, 표준오차 {mc.bse['log_price']:.4f}, t = {mc.tvalues['log_price']:.1f}")

fit["ge90"] = (fit.points >= 90).astype(int)
lg = smf.logit("ge90 ~ log_price", data=fit).fit(disp=0, cov_type="HC3")
print(); print(lg.summary())
lo, hi = np.exp(lg.conf_int().loc["log_price"] * np.log(2))
print(f"\n가격 2배 → 90점 이상일 오즈 {np.exp(lg.params['log_price'] * np.log(2)):.2f}배 (95% CI {lo:.2f}~{hi:.2f})")
''')

md("""
### 2-6. 머신러닝 모델 다시 학습하기

발표에서 고른 설정(하이퍼파라미터)을 그대로 써서 다시 학습합니다. 튜닝 과정은 생략하고 결과만 재현합니다.
- 학습·검증 점수: **학습 데이터로만** 학습한 모델
- 시험 점수: **학습+검증**으로 다시 학습한 최종 모델 (발표와 같은 방식)

`RUN_TREES = False`로 바꾸면 오래 걸리는 랜덤포레스트·XGBoost(합쳐서 약 10분)를 건너뜁니다. 선형 모델만은 2분 안쪽입니다.
""")
code(r'''
import time
import scipy.sparse as sp
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, roc_auc_score

RUN_TREES = True
SEED = 42
CAT = ["country", "province", "variety"]
FEATS = ["country", "province", "variety", "log_price", "vintage"]


def make_ct(feats, kind):
    """src/03_models.py와 같은 전처리: 선형 = 원핫 + 표준화, 트리 = 원핫 + 원래 값"""
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


def run(name, model, feats, kind, task="reg"):
    t0 = time.time()
    res = {"모델": name}
    for stage, fit_df, evals in [("학습만", train, {"학습": train, "검증": valid}), ("최종", trval, {"시험": test})]:
        ct = make_ct(feats, kind)
        X = ct.fit_transform(fit_df)
        y = fit_df.points.to_numpy() if task == "reg" else (fit_df.points >= 90).astype(int).to_numpy()
        m = model().fit(X, y)
        for nm, d in evals.items():
            Xe = ct.transform(d)
            if task == "reg":
                p = m.predict(Xe)
                res[f"{nm} R²"] = r2_score(d.points, p)
                res[f"{nm} MAE"] = mean_absolute_error(d.points, p)
            else:
                res[f"{nm} AUC"] = roc_auc_score(d.points >= 90, m.predict_proba(Xe)[:, 1])
    res["초"] = round(time.time() - t0, 1)
    return res


reg_rows = [
    run("Ridge (가격만)", lambda: Ridge(alpha=10.0), ["log_price"], "linear"),
    run("Ridge (전체 변수)", lambda: Ridge(alpha=10.0), FEATS, "linear"),
    run("Lasso (전체 변수)", lambda: Lasso(alpha=1e-4, max_iter=20000), FEATS, "linear"),
    run("ElasticNet (전체 변수)", lambda: ElasticNet(alpha=3e-4, l1_ratio=0.2, max_iter=20000), FEATS, "linear"),
]
cls_rows = [
    run("로지스틱 (가격만)", lambda: LogisticRegression(C=1.0, max_iter=5000), ["log_price"], "linear", "cls"),
    run("로지스틱 (전체 변수)", lambda: LogisticRegression(C=1.0, max_iter=5000), FEATS, "linear", "cls"),
]
if RUN_TREES:
    import xgboost as xgb
    reg_rows.append(run("랜덤포레스트", lambda: RandomForestRegressor(n_estimators=200, min_samples_leaf=5, max_features=0.33,
                                                             n_jobs=-1, random_state=SEED), FEATS, "tree"))
    reg_rows.append(run("XGBoost", lambda: xgb.XGBRegressor(learning_rate=0.1, subsample=0.8, tree_method="hist", max_depth=8,
                                                           min_child_weight=1, colsample_bytree=0.8, n_estimators=420,
                                                           n_jobs=2, random_state=SEED), FEATS, "tree"))
    cls_rows.append(run("XGBoost 분류", lambda: xgb.XGBClassifier(learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
                                                                 tree_method="hist", max_depth=6, min_child_weight=1,
                                                                 n_estimators=571, n_jobs=2, random_state=SEED), FEATS, "tree", "cls"))

title("회귀: 평점 숫자 맞히기 (학습 → 검증 → 시험)")
print(pd.DataFrame(reg_rows).to_string(index=False))
title("분류: 90점 이상 가려내기 (AUC, 0.5 = 무작위)")
print(pd.DataFrame(cls_rows).to_string(index=False))
print("\n※ 트리 모델은 실행 환경(스레드 수, 라이브러리 버전)에 따라 소수점 셋째 자리가 조금 다를 수 있습니다.")
''')

md("""
---
# 3부. 전체 파이프라인 재실행 (선택, 약 10~15분)

XGBoost 튜닝, SHAP, 예산 시뮬레이션, 평론가 점검까지 모두 다시 계산하고 `outputs/key_numbers.json`을 새로 만듭니다.
실행이 끝난 뒤 **1부 셀을 다시 실행**하면 새로 계산한 숫자로 출력됩니다.

Colab에서 한글 그래프가 깨지면 이 셀 앞에 `!apt-get -qq install fonts-noto-cjk`를 실행하고 런타임을 다시 시작하세요.
""")
code(r'''
RUN_FULL_PIPELINE = False   # True로 바꾸고 실행

if RUN_FULL_PIPELINE:
    p = subprocess.Popen([sys.executable, "src/run_all.py"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in p.stdout:
        print(line, end="")
    p.wait()
else:
    print("건너뜀: RUN_FULL_PIPELINE = True로 바꾸면 전체 분석을 다시 실행합니다.")
''')

nb["cells"] = C
nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                  "language_info": {"name": "python"}, "colab": {"provenance": []}}
out = Path(__file__).with_name("wine_stats_colab.ipynb")
nbf.write(nb, out)
print("wrote", out)
