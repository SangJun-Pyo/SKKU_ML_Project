# Independent Audit — "비싼 와인은 진짜 맛있을까?"

Auditor: Independent Auditor (fresh context) · Date: 2026-10-19
Audited artifact: `outputs/key_numbers.json` (md5 `7d3b41d9fded860bb5a856ab5702b5a6`, written 15:48:11 KST), plus
`outputs/models/test_predictions.csv` (md5 `8c0dadc8…`) and `outputs/sim/markets.csv` (md5 `b85eb7a0…`).
Audit code: `audit/scripts/a1_clean_split.py` … `a7_compare.py`; raw results in `audit/results/*.json`,
full machine comparison in `audit/results/comparison.csv` (217 checks).
Nothing in `src/`, `data/` or `outputs/` was modified.

---

## 1. Summary verdict

**PASS with warnings. Gate G4 can be passed once the critic-check wording is fixed and the claims are reworded as below.**

- **Numbers: 212 of 217 checks PASS, 5 WARN, 0 FAIL.** Every number in `key_numbers.json` that I recomputed matches.
  - The cleaning counts, split, EDA, quiz rows, metrics, simulation, value map and SHAP shares all match exactly or within rounding.
  - My independent refits also land close: Ridge test MAE matches to 1e-4, and my XGBoost refit gives test MAE 1.7130 against 1.7118.
- **Leakage: none in the main pipeline.**
  - No `title_norm` or description overlaps across the splits.
  - The test set is never used for any choice.
  - The winery encoding is out-of-fold within the fitting data.
  - The main model uses only country, province, variety, log_price and vintage (+ a missing flag).
  - One looser risk, reported as context: 44% of test titles are a different vintage of a wine that is also in train/valid.
- **Warnings.**
  - The within-critic premium is *labelled* "vs the same critics' **other** countries", but the code subtracts the critic's average over **all** their countries, *including* the target country. That mislabel accounts for 4 WARNs. The numbers match the code. They do not match the label.
  - The plan text says "7/10 Austrian"; the actual count is 8/10.
- **Claims.** Two headline claims are supported as worded, and six need rewording. None is unsupported. The two that most need fixing:
  - "리뷰 단어로 77%": that model also contains price and region. Words alone give R² 0.70.
  - "평론가를 통제해도": this rests on one or two critics per country, and Germany's sign flips between its two critics.

---

## 2. Check table (key items; full 217-row table in Appendix A)

Tolerance: 0 means an exact match. "1e-4" means rounding only. For a refit, the tolerance is how much an independent implementation may differ.

| # | check | expected (key_numbers) | recomputed | tolerance | status |
|---|---|---|---|---|---|
| 1 | raw rows × cols | 129,971 | 129,971 × 14 | 0 | PASS |
| 2 | after exact duplicates (excl. `Unnamed: 0`) | 119,988 | 119,988 | 0 | PASS |
| 3 | after required fields (price>0, points 80–100, country/province/variety/title/description non-empty) | 111,537 | 111,537 | 0 | PASS |
| 4 | after duplicate normalized descriptions | 111,510 | 111,510 | 0 | PASS |
| 5 | clean `source_row_id` set = my own cleaning | identical | identical (sym. diff 0) | 0 | PASS |
| 6 | split rows train / valid / test | 71,381 / 17,843 / 22,286 | 71,381 / 17,843 / 22,286 | 0 | PASS |
| 7 | unique titles train / valid / test | 70,748 / 17,688 / 22,109 | same | 0 | PASS |
| 8 | `title_norm` overlap (tr-va, tr-te, va-te) | 0 / 0 / 0 | 0 / 0 / 0 | 0 | PASS |
| 9 | description overlap across splits | — | 0 / 0 / 0 | 0 | PASS |
| 10 | split re-run from my own cleaned frame | — | 0 rows differ | 0 | PASS |
| 11 | Pearson r (price) / (log price) | 0.4166 / 0.6178 | 0.4166 / 0.6178 | 1e-4 | PASS |
| 12 | Spearman ρ | 0.6126 | 0.6126 | 1e-4 | PASS |
| 13 | in-sample r² (log price) | 0.3816 | 0.3816 | 1e-4 | PASS |
| 14 | share ≥ 90 (all clean rows) | 37.75% | 37.75% | 1e-3 | PASS |
| 15 | price buckets: n, median, share ≥ 90 (6 buckets) | e.g. ($20,$30]: 24,798 / 88 / 34.02% | identical, all 6 | 0 | PASS |
| 16 | quiz row 15428 | Columbia Crest 2014 Grand Estates Cab, $12, 90 | same | 0 | PASS |
| 17 | quiz row 37412 | Laird 2013 Cab (Napa), $60, 90 | same | 0 | PASS |
| 18 | all 8 models × valid/test × MAE/RMSE/R² from saved predictions (48 numbers) | — | all equal | 1e-4 | PASS |
| 19 | Ridge price-only test MAE / R² (own refit, own encoder, α picked on valid) | 1.9331 / 0.3810 | 1.9331 / 0.3810 | 0.005 | PASS |
| 20 | Ridge meta-only test MAE / R² (own refit) | 2.2539 / 0.1804 | 2.2539 / 0.1804 | 0.005 | PASS |
| 21 | Ridge meta+price test MAE / R² (own refit; α = 10 chosen on valid, same as builder) | 1.8176 / 0.4477 | 1.8176 / 0.4477 | 0.005 | PASS |
| 22 | XGBoost main test MAE (one refit, config params, different seed 7) | 1.7118 | 1.7130 | 0.01 | PASS |
| 23 | XGBoost main test R² (same refit) | 0.5011 | 0.5008 | 0.01 | PASS |
| 24 | test MAE gap Ridge − Main, 95% CI (own seed) | 0.1058 [0.0976, 0.1143] | 0.1058 [0.0969, 0.1143] | 0.005 | PASS |
| 25 | sim: 4 strategies × 3 budgets × (points, share ≥ 90, price) | e.g. $30 Main 89.98 / 64.7% / $25.14 | identical (36 numbers) | 1e-3 | PASS |
| 26 | sim: saved pick is a valid argmax under the stated tie rules | 1000/1000 per strategy × budget | 1000/1000 everywhere | 0 | PASS |
| 27 | sim: MaxPrice $30 share ≥ 90 as tie-averaged expected value | 41.0% | 41.5% (457/1000 markets have ties at max price) | 1.0 | PASS |
| 28 | sim: all candidates ≤ budget; 20 rows and 20 distinct titles per market; all rows from test | true | true (3 budgets) | 0 | PASS |
| 29 | $30 Main − MaxPrice, share ≥ 90 (pp), CI (own seed) | 23.7 [20.0, 27.5] | 23.7 [20.1, 27.3] | 0.5 | PASS |
| 30 | $30 Ridge − MaxPrice, share ≥ 90 (pp), CI (own seed) | 19.5 [15.8, 23.2] | 19.5 [15.8, 23.2] | 0.5 | PASS |
| 31 | $30 Main − MaxPrice, points, CI | 1.298 [1.127, 1.476] | 1.298 [1.123, 1.477] | 0.03 | PASS |
| 32 | $30 Ridge − MaxPrice, points, CI | 1.090 [0.910, 1.275] | 1.090 [0.909, 1.274] | 0.03 | PASS |
| 33 | $30 Main − MaxPrice, price (USD), CI | −4.246 [−4.510, −3.985] | −4.246 [−4.511, −3.981] | 0.03 | PASS |
| 34 | points per price doubling (own OLS on train+valid) | 2.0117 | 2.0117 | 1e-3 | PASS |
| 35 | value map: levels with n ≥ 100 (countries / varieties) | 12 / 36 | 12 / 36 | 0 | PASS |
| 36 | value map: country mean residuals (all 12) | Austria +1.8162 … Argentina −0.7173 | max abs diff 1.5e-7 | 1e-3 | PASS |
| 37 | value map: variety mean residuals (all 36) | Grüner +2.0704 … Champagne Blend −0.8684 | max abs diff 2.5e-7 | 1e-3 | PASS |
| 38 | within-critic premium, builder's code definition: AT / DE / PT / AR / CL | 0.1375 / 0.7712 / 0.6652 / −0.2048 / −0.0568 | identical | 1e-3 | PASS |
| 39 | within-critic, Austria — definition **as labelled** ("vs other countries") | 0.1375 | 0.2245 | 0.05 | **WARN** |
| 40 | within-critic, Germany — as labelled | 0.7712 | 1.4287 | 0.05 | **WARN** |
| 41 | within-critic, Portugal — as labelled | 0.6652 | 0.8829 | 0.05 | **WARN** |
| 42 | within-critic, Argentina — as labelled | −0.2048 | −0.2816 | 0.05 | **WARN** |
| 43 | within-critic, Chile — as labelled | −0.0568 | −0.0815 | 0.05 | PASS |
| 44 | under-$20 list reproduced by ranking on predicted points only | 10 rows | identical order; prediction monotone ↓; 0/10 overlap with the actual-points top 10 | 0 | PASS |
| 45 | under-$20 top 10: mean actual / n ≥ 90 / pool share ≥ 90 | 91.2 / 10 / 11.02% | 91.2 / 10 / 11.02% | 1e-3 | PASS |
| 46 | under-$20 Austrian picks (HARNESS_PLAN §8 text, not key_numbers) | "7/10" | 8/10 (+ Israel 1, France 1) | 0 | **WARN** |
| 47 | SHAP share log_price (from saved grouped sample) | 57.06% | 57.06% | 1e-3 | PASS |
| 48 | SHAP share log_price / vintage / variety / province / country (own refit, own 5,000-row sample) | 57.06 / 14.66 / 10.95 / 9.76 / 7.58 | 56.83 / 14.44 / 10.78 / 9.62 / 8.32 | 1.5 pp | PASS |
| 49 | text aux model test R² (TF-IDF + meta + price) | 0.7713 | 0.7713 | 1e-4 | PASS |

---

## 3. Leakage findings

| Item | Finding | Verdict |
|---|---|---|
| Same `title_norm` in two splits | 0 / 0 / 0. The GroupShuffleSplit reproduces row for row from my own cleaning. | PASS |
| Same normalized description in two splits | 0 / 0 / 0 | PASS |
| Test used for selection | Code read (`03_models.py`): Ridge α (l.166–177), the XGB grid and early stopping (l.202–212, eval_set = valid), the main-model choice (l.221–223), the winery keep/drop decision (l.245) and the text α (l.273–278) all use **valid** only. Final models are refit on train+valid and predict test once (l.175, 194, 212, 240, 283). The simulation, value map and critic check read only the saved test predictions. | PASS |
| Winery target encoding | `winery_te` (l.101–117): 5-fold GroupKFold by `title_norm` *inside the fitting data* for the fitting rows. Valid/test rows are encoded from fitting-data statistics only. No valid or test targets are used. | PASS |
| Main-model features | `FEATS["meta_price"]` = country, province, variety, log_price, vintage (l.47). SHAP groups confirm exactly 5 features (348 columns). The winery and text models are kept out of the simulation (04_sim l.37 uses only `ridge_meta_price` and `xgb_meta_price`). | PASS |
| Preprocessing fitted on fitting data only | The OneHot rare-level merging, median imputation and scaler all sit inside ColumnTransformers fitted on train (or train+valid). | PASS |
| XGB valid score | n_estimators comes from early stopping on valid, so XGB's valid MAE is slightly optimistic when it is compared with RF for the main-model choice. This is documented. The test confirms the order: RF − XGB = 0.028 MAE, CI [0.022, 0.034]. | note |
| **Looser leakage: same wine, different vintage** (context, not a failure) | After stripping 4-digit years from `title_norm`, **9,727 of 22,109 test titles (44.0%; 9,771 rows, 43.8%)** match a train/valid title (valid vs train: 38.9%). For matched test rows, the mean absolute gap to that wine's train/valid mean score is 1.85 pts, so the other vintages are informative but far from identical. The **main model cannot exploit this**, because it has no winery or title feature; at most, specific province × variety × price × vintage cells repeat. The **winery experiment can**: 94.9% of test rows have a winery seen in train/valid. Its test MAE of 1.631 is therefore an "established wineries" number and would be optimistic for new wineries. | context |

---

## 4. Simulation: additional observations

- Ties: at $30, 457 of 1000 markets have two or more wines at the market's maximum price (typically $30). In practice, MaxPrice means "a random wine priced at the cap". This baseline is fair, but the slide should say so.
- The picks are diverse: the Main strategy picks 731 distinct wines over 1000 markets at $30, and the 10 most-picked wines make up 4.4% of picks. Re-use of wines across markets therefore does not shrink the market-bootstrap CI much. The CI is still conditional on this one test pool (the builder's `ci_note` is correct).
- Share of the ≥ 90 gain over MaxPrice that Ridge already captures: $20: 76% (15.2/19.9), $30: 82% (19.5/23.7), $50: 81% (13.9/17.1). Main − Ridge is still positive at every budget (CI excludes 0: $30 +4.2 pp [1.3, 7.1]).

## 5. Value map and critic check

- The country and variety residual tables reproduce to 1e-7 from my own OLS on log price (train+valid fit, test rows, n ≥ 100).
- **How the within-critic premium is defined.** The code is `07_critic_check.py` l.76–87. For each critic who has ≥ 30 test rows in 2+ value-map countries, it computes deviation = cell mean residual − *the critic's n-weighted mean over all their qualifying countries, including this one*. It then n-weight-averages those deviations across critics.
- **Is that sound?** It is a reasonable descriptive check, but it is weaker than the slide wording "평론가를 통제해도" implies:
  1. **Mislabelled.** The CSV note (l.110) and the `critic_note` in `08_key_numbers.py` (l.204–206) say "vs … **other** countries". The code includes the target country itself, which shrinks every premium toward 0 (Germany 0.77 against 1.43 under the labelled definition). Only the docstring (l.7) states the real definition. Fix one or the other; the simplest fix is to change the label to "vs the critic's average over all countries they review (including this one)".
  2. **Zero-sum within each critic.** I verified this: the n-weighted deviations sum to exactly 0 for every critic. A country's premium is therefore purely *relative to the other countries in the same critic's portfolio*. Schachner reviews only Argentina, Chile and Spain, which all have negative raw residuals. Within-critic, Spain turns positive (+0.17) and Argentina stays negative (−0.20). The check cannot tell whether Schachner's whole portfolio is low because of the critic or because of the countries.
  3. **The premium rests on one or two critics per country.**
     - Portugal is Roger Voss alone: Portugal +0.98 against his France +0.07 and Austria +0.56.
     - Germany is driven by Anna Lee C. Iijima (+1.78 against her US wines, 405 of 417 from New York). **Joe Czerwinski's German wines sit 0.48 below his own average** (n = 60). Germany's within-critic sign therefore depends on the critic.
     - Austria's +1.82 raw residual falls to +0.14 (or +0.22). Anne Krebiehl MW is about as generous to her French wines (+2.23; all 243 are from Alsace, mostly Riesling, Pinot Gris and Gewürztraminer) as to Austria (+2.44).
  4. **The CI is too narrow for a "controlled" claim.** The bootstrap (l.91–96) resamples rows, so with 1–2 critics the CI ignores critic-level uncertainty. Critic × country interactions (e.g. Voss liking Portugal) cannot be separated from a real Portugal effect. Variety mix (Riesling, Grüner, Portuguese Red) is also not controlled.
- **Under-$20 list.** The ranking is on predicted points only: the order reproduces from `pred_xgb_meta_price` desc → price asc → title asc, one row per title, and it shares 0 rows with the top 10 by actual points. The 10/10 ≥ 90 result is the extreme of the ranking. Over the top 50 by prediction, 74% score ≥ 90, against 11% in the pool. That is still strong, but 10/10 should not be shown as a typical hit rate.

## 6. EDA and quiz

- All correlations, the bucket table and share ≥ 90 reproduce exactly (Pearson computed with numpy, Spearman with scipy, buckets with my own `np.digitize`).
- Both quiz rows are in the raw data exactly as stated, and each is the only raw row with its title. Both rows are in the train split. Context for the reveal: the two wines were scored by **different critics** (Sean P. Sullivan and Virginie Boone).

## 7. Reproducibility and portability risks

- No absolute paths: all paths come from `common.py` via `Path(__file__)`. Good.
- Reads and writes pass `encoding="utf-8"` explicitly. Good. On Korean Windows, Excel opens UTF-8 CSVs without a BOM as mojibake. If teammates open the CSVs in Excel, write them with `utf-8-sig` (cosmetic).
- **Windows console encoding.** `05_valuemap.py` l.95–97 and other scripts print non-ASCII names (Grüner Veltliner, Gewürztraminer, Château Jolys, Rosé). On Windows, when output is redirected or piped (cp949), this can raise `UnicodeEncodeError`. Recommend `set PYTHONUTF8=1`, or `sys.stdout.reconfigure(encoding="utf-8")`.
- **Fonts.** `setup_fonts()` (`common.py` l.31–45) silently falls back to the matplotlib default when none of Noto Sans CJK KR, Malgun Gothic or AppleGothic is installed. Korean labels then render as boxes with no error. Linux teammates need `fonts-noto-cjk`.
- **Version pins.**
  - `GroupKFold(shuffle=True)` (`03_models.py` l.112) needs scikit-learn ≥ 1.6, and older versions raise an error.
  - `requirements.txt` pins very recent versions (pandas 3.0.5, numpy 2.5.3, Python 3.13 bytecode in `__pycache__`), which may not install on a teammate's older Python.
- **Threads.** XGBoost `hist` with `n_jobs=2` and RF with `n_jobs=2` are deterministic on a fixed machine, but last-digit float differences across CPUs, OS or thread counts are possible. My independent XGB refit (different seed) moved test MAE by only 0.0012, so headline numbers are robust to this.
- A new `src/09_figures_ko.py` (Viz Builder) appeared during the audit. It was not audited. `run_all.py` globs `0*_*.py`, so it now runs as part of the pipeline.

## 8. Bugs and inconsistencies (file:line)

| Severity | File:line | Issue |
|---|---|---|
| Medium (wording reaches slides) | `src/07_critic_check.py:110` and `src/08_key_numbers.py:204–206` | The label says the premium is "vs … other countries", but the computation at `07_critic_check.py:79–82` subtracts the critic's mean over **all** qualifying countries, including the target. Rename the label, or exclude the target country (that would change the values: AT 0.22, DE 1.43, PT 0.88, AR −0.28, CL −0.08). |
| Low | `HARNESS_PLAN.md:133` | "Under-$20 picks … (7/10 Austrian)" is actually **8/10** (Austria 8, Israel 1, France 1). |
| Low | `src/07_critic_check.py:91–96` | The row-level bootstrap ignores critic clustering, so the CIs are too narrow for any "controlled for critic" reading. Label them as row-sampling CIs. |
| Low (latent) | `src/04_sim.py:36` vs `src/05_valuemap.py:71` | The simulation's "Main" uses `cfg["main_tree_model"]`, while the value map uses `cfg["main_model"]`. They are the same today (`xgb_meta_price`), but would silently diverge if the main model ever became Ridge. |
| Cosmetic | `src/08_key_numbers.py:140` | SHAP `country_bottom5` includes Italy at **+0.03** (a positive value in a "bottom" list). Fine as a rank, but don't label it "negative". |

---

## 9. Claim review

| # | Planned claim | Verdict | Evidence | Safe wording (KO) |
|---|---|---|---|---|
| 1 | 가격만으로 점수 변동의 약 38%를 설명한다 | **Supported** (minor qualifier) | Test R² 0.381 (log price, Ridge); in-sample r² 0.382 | "(로그) 가격 하나로 Wine Enthusiast 점수 변동의 약 38%를 설명한다 (테스트셋 R² 0.38)." |
| 2 | 산지·품종을 더하면 45%, XGBoost로 50% | **Needs rewording** (vintage is also in the model) | Ridge meta+price R² 0.448 (country, province, variety **and vintage**); XGB 0.501 | "가격에 산지·품종·빈티지를 더하면 45%(선형), XGBoost로 50%까지 설명한다. 나머지 절반은 이 정보로 설명되지 않는다." |
| 3 | $30 예산에서 가장 비싼 와인을 고르면 90점 이상이 41%, 모델로 고르면 65%이고 평균 $4 더 싸다 | **Supported** (state the setup) | MaxPrice 41.0%, Main 64.7%; +23.7 pp [20.0, 27.5]; price $29.39 against $25.14 (−$4.25 [−4.51, −3.99]) | "$30 이하 와인 20병 중 1병을 고르는 실험(테스트셋 1,000회)에서, 가장 비싼 병은 90점 이상이 41%, 모델이 고른 병은 65%(+24%p, 95% CI 20~28%p)였고 평균 $4 더 쌌다." |
| 4 | 개선의 대부분은 선형모델(Ridge)로도 얻어진다 | **Needs rewording** (true for the simulation, borderline for prediction) | Simulation: Ridge captures 76–82% of the ≥ 90 gain. Prediction: Ridge captures only 56% of the R² gain from price-only to XGB (MAE 52%). XGB is still significantly better (+4.2 pp at $30, CI [1.3, 7.1]). | "예산 실험에서 개선의 약 80%는 단순 선형모델(Ridge)로도 얻어지고, XGBoost는 3~5%p를 더한다." |
| 5 | 오스트리아·독일·포르투갈은 가격 대비 점수가 높다 / 아르헨티나·칠레는 낮다 | **Needs rewording** (descriptive and critic-confounded) | Residuals +1.82 / +1.26 / +0.97 and −0.72 / −0.57, all CIs exclude 0. Austria falls to +0.14–0.22 within-critic, Chile to ≈ 0 (CI includes 0). | "Wine Enthusiast 평론가 점수 기준으로, 오스트리아·독일·포르투갈 와인은 같은 가격대 평균보다 1~2점 높게, 아르헨티나·칠레는 0.6~0.7점 낮게 평가됐다. 단, 나라마다 담당 평론가가 달라 이 차이의 상당 부분은 평론가 차이일 수 있다." |
| 6 | 평론가를 통제해도 포르투갈(일부 독일)의 프리미엄은 남는다 | **Needs rewording** ("통제" overclaims; Germany is critic-dependent) | Portugal: one critic (Voss), +0.67 (code) / +0.88 (vs his other countries). Germany: Iijima +1.78, Czerwinski −0.48. Row-bootstrap CIs; the measure is zero-sum within each critic. | "같은 평론가가 매긴 점수끼리 비교해도, Roger Voss는 포르투갈 와인을 자신이 평가한 프랑스·오스트리아 와인보다 가격 대비 약 0.7~0.9점 높게 줬다. 독일은 평론가에 따라 방향이 갈려 결론을 내리기 어렵다. 오스트리아의 큰 프리미엄은 평론가 차이로 대부분 설명된다." |
| 7 | 리뷰 단어로는 점수의 77%를 설명한다 | **Needs rewording** (the model is not words only, and it is post-review) | The 0.771 model = TF-IDF **+ price + region + vintage**. Words alone: R² **0.704** (my refit). Review length alone correlates r = 0.58 with points. The text is written by the critic who gave the score. | "점수를 매긴 평론가가 쓴 리뷰 문장만으로 점수 변동의 약 70%(가격·산지를 더하면 77%)가 재현된다. 리뷰는 점수와 같은 사람이 쓴 '사후 정보'라서 예측이 아니라 평론가가 무엇을 칭찬하는지 보여주는 분석이다." |
| 8 | 가격 영향(SHAP) 비중 57% | **Needs rewording** (it is a share of attribution, not of variance or "influence on the score") | Mean \|SHAP\| share 57.06% (reproduced; own refit 56.83%), XGB, 5,000 test rows. Vintage 14.7%, variety 10.9%, province 9.8%, country 7.6%. | "모델 예측에 대한 평균 기여도(|SHAP|)로 보면 가격이 57%, 빈티지·품종·산지·국가가 나머지 43%를 차지한다." |

### Other overclaim risks

- **"맛있다 / taste".** The target is one magazine's critic score: only wines scoring 80 or more were published, and each wine was reviewed by one critic. It is not consumer taste and not a blind-tasting panel. Answer the title question as "전문가 점수 기준으로는…" and keep "점수 ≠ 내 입맛" on the conclusion slide, not only in Limitations.
- **Causal language.** Avoid "가격이 점수를 올린다", "산지가 점수를 만든다" and "모델이 더 좋은 와인을 만든다/찾는다". Price and score are associated, and the causation may run the other way (good scores raise prices, reputation drives both). Use "함께 움직인다", "~와 관련된다" and "같은 가격대 평균보다".
- **Range restriction.** Only wines scoring 80–100 exist in the data, so the price–score correlation, and the "38%", describe published reviews only.
- **Time and market.** Prices are US USD prices at review time (data collected in 2017). They are not 2026 prices and not Korean prices: Korean shelf prices include import, liquor tax and distribution markups, and many of these wines are unavailable in Korea. Avoid "한국 마트에서 $30 이하라면…". Say "미국 리뷰 기준가 $30 이하".
- **Simulation generalization.** The CI covers market-sampling variability within one fixed test pool, so it is not a population guarantee. "모델로 고르면" means a model trained on this critic's scores, scored against the same critics. MaxPrice under a cap is effectively "a random wine at the cap price".
- **Under-$20 list.** 10/10 ≥ 90 is the very top of the ranking (top 50: 74%). It is an illustration, and 8 of the 10 are Austrian, which reflects the critic confound in §5.
- **Winery result** (if shown): 95% of test rows come from wineries seen in training, and 44% of test titles are another vintage of a training wine. It is valid for established wineries, not new ones.
- **Value map: country vs variety.** Austria's premium overlaps with Grüner Veltliner and Riesling (both top varieties), and Portugal's with Portuguese Red. A price-only residual does not separate country from variety, so do not present them as independent findings.
- **Quiz reveal.** The two 90-point wines were rated by different critics. "같은 90점" is the same number on a scale that may vary by critic. Fine for a hook; don't build an argument on it.

---

## Appendix A — full machine comparison (generated by `audit/scripts/a7_compare.py`)

| check | expected (key_numbers) | recomputed | tolerance | status |
|---|---|---|---|---|
| raw rows | 129971 | 129971 | 0 | PASS |
| after exact-dup (excl. Unnamed: 0) | 119988 | 119988 | 0 | PASS |
| after required fields | 111537 | 111537 | 0 | PASS |
| after dup normalized descriptions | 111510 | 111510 | 0 | PASS |
| clean row-id set identical to own cleaning | True | True | 0 | PASS |
| split rows train | 71381 | 71381 | 0 | PASS |
| unique titles train | 70748 | 70748 | 0 | PASS |
| split rows valid | 17843 | 17843 | 0 | PASS |
| unique titles valid | 17688 | 17688 | 0 | PASS |
| split rows test | 22286 | 22286 | 0 | PASS |
| unique titles test | 22109 | 22109 | 0 | PASS |
| title_norm overlap train-valid | 0 | 0 | 0 | PASS |
| title_norm overlap train-test | 0 | 0 | 0 | PASS |
| title_norm overlap valid-test | 0 | 0 | 0 | PASS |
| split reproduced from own cleaning (row mismatches) | 0 | 0 | 0 | PASS |
| Pearson r (raw price) | 0.4166 | 0.4166 | 0.0001 | PASS |
| Pearson r (log price) | 0.6178 | 0.6178 | 0.0001 | PASS |
| r^2 log price in-sample | 0.3816 | 0.3816 | 0.0001 | PASS |
| Spearman rho | 0.6126 | 0.6126 | 0.0001 | PASS |
| share >= 90 (%) | 37.7527 | 37.7527 | 0.001 | PASS |
| bucket <=$10 n | 5913 | 5913 | 0 | PASS |
| bucket <=$10 median / share>=90 | (85.0, 1.1838) | (85.0, 1.1838) | 0 | PASS |
| bucket ($10,$20] n | 36806 | 36806 | 0 | PASS |
| bucket ($10,$20] median / share>=90 | (87.0, 12.8756) | (87.0, 12.8756) | 0 | PASS |
| bucket ($20,$30] n | 24798 | 24798 | 0 | PASS |
| bucket ($20,$30] median / share>=90 | (88.0, 34.0229) | (88.0, 34.0229) | 0 | PASS |
| bucket ($30,$50] n | 25519 | 25519 | 0 | PASS |
| bucket ($30,$50] median / share>=90 | (90.0, 56.0563) | (90.0, 56.0563) | 0 | PASS |
| bucket ($50,$100] n | 15245 | 15245 | 0 | PASS |
| bucket ($50,$100] median / share>=90 | (91.0, 76.4185) | (91.0, 76.4185) | 0 | PASS |
| bucket >$100 n | 3229 | 3229 | 0 | PASS |
| bucket >$100 median / share>=90 | (93.0, 89.7182) | (93.0, 89.7182) | 0 | PASS |
| quiz row 15428 (title, price, points) | ('Columbia Crest 2014 Grand Estates Cabernet Sauvignon (Columbia Valley (WA))', 12.0, 90) | ('Columbia Crest 2014 Grand Estates Cabernet Sauvignon (Columbia Valley (WA))', 12.0, 90) | 0 | PASS |
| quiz row 37412 (title, price, points) | ('Laird 2013 Cabernet Sauvignon (Napa Valley)', 60.0, 90) | ('Laird 2013 Cabernet Sauvignon (Napa Valley)', 60.0, 90) | 0 | PASS |
| mean valid MAE (from saved preds) | 2.5142 | 2.5142 | 0.0001 | PASS |
| mean valid RMSE (from saved preds) | 3.0690 | 3.0690 | 0.0001 | PASS |
| mean valid R2 (from saved preds) | -0.0000 | -0.0000 | 0.0001 | PASS |
| mean test MAE (from saved preds) | 2.5336 | 2.5336 | 0.0001 | PASS |
| mean test RMSE (from saved preds) | 3.0943 | 3.0943 | 0.0001 | PASS |
| mean test R2 (from saved preds) | -0.0000 | -0.0000 | 0.0001 | PASS |
| ridge_price valid MAE (from saved preds) | 1.9288 | 1.9288 | 0.0001 | PASS |
| ridge_price valid RMSE (from saved preds) | 2.4179 | 2.4179 | 0.0001 | PASS |
| ridge_price valid R2 (from saved preds) | 0.3793 | 0.3793 | 0.0001 | PASS |
| ridge_price test MAE (from saved preds) | 1.9331 | 1.9331 | 0.0001 | PASS |
| ridge_price test RMSE (from saved preds) | 2.4345 | 2.4345 | 0.0001 | PASS |
| ridge_price test R2 (from saved preds) | 0.3810 | 0.3810 | 0.0001 | PASS |
| ridge_meta valid MAE (from saved preds) | 2.2408 | 2.2408 | 0.0001 | PASS |
| ridge_meta valid RMSE (from saved preds) | 2.7821 | 2.7821 | 0.0001 | PASS |
| ridge_meta valid R2 (from saved preds) | 0.1782 | 0.1782 | 0.0001 | PASS |
| ridge_meta test MAE (from saved preds) | 2.2539 | 2.2539 | 0.0001 | PASS |
| ridge_meta test RMSE (from saved preds) | 2.8013 | 2.8013 | 0.0001 | PASS |
| ridge_meta test R2 (from saved preds) | 0.1804 | 0.1804 | 0.0001 | PASS |
| ridge_meta_price valid MAE (from saved preds) | 1.8176 | 1.8176 | 0.0001 | PASS |
| ridge_meta_price valid RMSE (from saved preds) | 2.2840 | 2.2840 | 0.0001 | PASS |
| ridge_meta_price valid R2 (from saved preds) | 0.4462 | 0.4462 | 0.0001 | PASS |
| ridge_meta_price test MAE (from saved preds) | 1.8176 | 1.8176 | 0.0001 | PASS |
| ridge_meta_price test RMSE (from saved preds) | 2.2996 | 2.2996 | 0.0001 | PASS |
| ridge_meta_price test R2 (from saved preds) | 0.4477 | 0.4477 | 0.0001 | PASS |
| rf_meta_price valid MAE (from saved preds) | 1.7437 | 1.7437 | 0.0001 | PASS |
| rf_meta_price valid RMSE (from saved preds) | 2.2144 | 2.2144 | 0.0001 | PASS |
| rf_meta_price valid R2 (from saved preds) | 0.4794 | 0.4794 | 0.0001 | PASS |
| rf_meta_price test MAE (from saved preds) | 1.7398 | 1.7398 | 0.0001 | PASS |
| rf_meta_price test RMSE (from saved preds) | 2.2192 | 2.2192 | 0.0001 | PASS |
| rf_meta_price test R2 (from saved preds) | 0.4857 | 0.4857 | 0.0001 | PASS |
| xgb_meta_price valid MAE (from saved preds) | 1.7135 | 1.7135 | 0.0001 | PASS |
| xgb_meta_price valid RMSE (from saved preds) | 2.1781 | 2.1781 | 0.0001 | PASS |
| xgb_meta_price valid R2 (from saved preds) | 0.4963 | 0.4963 | 0.0001 | PASS |
| xgb_meta_price test MAE (from saved preds) | 1.7118 | 1.7118 | 0.0001 | PASS |
| xgb_meta_price test RMSE (from saved preds) | 2.1856 | 2.1856 | 0.0001 | PASS |
| xgb_meta_price test R2 (from saved preds) | 0.5011 | 0.5011 | 0.0001 | PASS |
| xgb_meta_price_winery valid MAE (from saved preds) | 1.6461 | 1.6461 | 0.0001 | PASS |
| xgb_meta_price_winery valid RMSE (from saved preds) | 2.0956 | 2.0956 | 0.0001 | PASS |
| xgb_meta_price_winery valid R2 (from saved preds) | 0.5337 | 0.5337 | 0.0001 | PASS |
| xgb_meta_price_winery test MAE (from saved preds) | 1.6307 | 1.6307 | 0.0001 | PASS |
| xgb_meta_price_winery test RMSE (from saved preds) | 2.0855 | 2.0855 | 0.0001 | PASS |
| xgb_meta_price_winery test R2 (from saved preds) | 0.5457 | 0.5457 | 0.0001 | PASS |
| text_ridge valid MAE (from saved preds) | 1.1699 | 1.1699 | 0.0001 | PASS |
| text_ridge valid RMSE (from saved preds) | 1.4913 | 1.4913 | 0.0001 | PASS |
| text_ridge valid R2 (from saved preds) | 0.7639 | 0.7639 | 0.0001 | PASS |
| text_ridge test MAE (from saved preds) | 1.1673 | 1.1673 | 0.0001 | PASS |
| text_ridge test RMSE (from saved preds) | 1.4796 | 1.4796 | 0.0001 | PASS |
| text_ridge test R2 (from saved preds) | 0.7713 | 0.7713 | 0.0001 | PASS |
| ridge_price test MAE (own refit) | 1.9331 | 1.9331 | 0.005 | PASS |
| ridge_price test R2 (own refit) | 0.3810 | 0.3810 | 0.005 | PASS |
| ridge_meta test MAE (own refit) | 2.2539 | 2.2539 | 0.005 | PASS |
| ridge_meta test R2 (own refit) | 0.1804 | 0.1804 | 0.005 | PASS |
| ridge_meta_price test MAE (own refit) | 1.8176 | 1.8176 | 0.005 | PASS |
| ridge_meta_price test R2 (own refit) | 0.4477 | 0.4477 | 0.005 | PASS |
| xgb_meta_price test MAE (own refit, seed 7) | 1.7118 | 1.7130 | 0.01 | PASS |
| xgb_meta_price test R2 (own refit, seed 7) | 0.5011 | 0.5008 | 0.01 | PASS |
| test MAE gap ridge - main | 0.1058 | 0.1058 | 0.0001 | PASS |
| test MAE gap CI low (own seed) | 0.0976 | 0.0969 | 0.005 | PASS |
| test MAE gap CI high (own seed) | 0.1143 | 0.1143 | 0.005 | PASS |
| sim $20 Random mean points | 86.5140 | 86.5140 | 0.001 | PASS |
| sim $20 Random share>=90 % | 11.0950 | 11.0950 | 0.001 | PASS |
| sim $20 Random mean price | 15.0726 | 15.0727 | 0.001 | PASS |
| sim $20 MaxPrice mean points | 87.5790 | 87.5790 | 0.001 | PASS |
| sim $20 MaxPrice share>=90 % | 22.0000 | 22.0000 | 0.001 | PASS |
| sim $20 MaxPrice mean price | 19.9490 | 19.9490 | 0.001 | PASS |
| sim $20 MaxPrice share>=90 % (tie-EV re-pick) | 22.0000 | 21.8816 | 1.0 | PASS |
| sim $20 Ridge mean points | 88.6480 | 88.6480 | 0.001 | PASS |
| sim $20 Ridge share>=90 % | 37.2000 | 37.2000 | 0.001 | PASS |
| sim $20 Ridge mean price | 18.2260 | 18.2260 | 0.001 | PASS |
| sim $20 Ridge share>=90 % (tie-EV re-pick) | 37.2000 | 37.2000 | 1.0 | PASS |
| sim $20 Main mean points | 88.8990 | 88.8990 | 0.001 | PASS |
| sim $20 Main share>=90 % | 41.9000 | 41.9000 | 0.001 | PASS |
| sim $20 Main mean price | 18.2360 | 18.2360 | 0.001 | PASS |
| sim $20 Main share>=90 % (tie-EV re-pick) | 41.9000 | 41.9000 | 1.0 | PASS |
| sim $20 all candidates within budget | True | True | 0 | PASS |
| sim $20 every market 20 distinct titles | True | True | 0 | PASS |
| sim $20 MaxPrice saved pick is a valid argmax (of 1000) | 1000 | 1000 | 0 | PASS |
| sim $20 Ridge saved pick is a valid argmax (of 1000) | 1000 | 1000 | 0 | PASS |
| sim $20 Main saved pick is a valid argmax (of 1000) | 1000 | 1000 | 0 | PASS |
| sim $30 Random mean points | 87.1553 | 87.1553 | 0.001 | PASS |
| sim $30 Random share>=90 % | 19.6450 | 19.6450 | 0.001 | PASS |
| sim $30 Random mean price | 18.9980 | 18.9980 | 0.001 | PASS |
| sim $30 MaxPrice mean points | 88.6820 | 88.6820 | 0.001 | PASS |
| sim $30 MaxPrice share>=90 % | 41.0000 | 41.0000 | 0.001 | PASS |
| sim $30 MaxPrice mean price | 29.3880 | 29.3880 | 0.001 | PASS |
| sim $30 MaxPrice share>=90 % (tie-EV re-pick) | 41.0000 | 41.4750 | 1.0 | PASS |
| sim $30 Ridge mean points | 89.7720 | 89.7720 | 0.001 | PASS |
| sim $30 Ridge share>=90 % | 60.5000 | 60.5000 | 0.001 | PASS |
| sim $30 Ridge mean price | 25.5050 | 25.5050 | 0.001 | PASS |
| sim $30 Ridge share>=90 % (tie-EV re-pick) | 60.5000 | 60.5000 | 1.0 | PASS |
| sim $30 Main mean points | 89.9800 | 89.9800 | 0.001 | PASS |
| sim $30 Main share>=90 % | 64.7000 | 64.7000 | 0.001 | PASS |
| sim $30 Main mean price | 25.1420 | 25.1420 | 0.001 | PASS |
| sim $30 Main share>=90 % (tie-EV re-pick) | 64.7000 | 64.7000 | 1.0 | PASS |
| sim $30 all candidates within budget | True | True | 0 | PASS |
| sim $30 every market 20 distinct titles | True | True | 0 | PASS |
| sim $30 MaxPrice saved pick is a valid argmax (of 1000) | 1000 | 1000 | 0 | PASS |
| sim $30 Ridge saved pick is a valid argmax (of 1000) | 1000 | 1000 | 0 | PASS |
| sim $30 Main saved pick is a valid argmax (of 1000) | 1000 | 1000 | 0 | PASS |
| sim $50 Random mean points | 87.8334 | 87.8334 | 0.001 | PASS |
| sim $50 Random share>=90 % | 29.8500 | 29.8500 | 0.001 | PASS |
| sim $50 Random mean price | 24.9464 | 24.9464 | 0.001 | PASS |
| sim $50 MaxPrice mean points | 90.1950 | 90.1950 | 0.001 | PASS |
| sim $50 MaxPrice share>=90 % | 63.6000 | 63.6000 | 0.001 | PASS |
| sim $50 MaxPrice mean price | 47.4060 | 47.4060 | 0.001 | PASS |
| sim $50 MaxPrice share>=90 % (tie-EV re-pick) | 63.6000 | 63.5350 | 1.0 | PASS |
| sim $50 Ridge mean points | 90.9670 | 90.9670 | 0.001 | PASS |
| sim $50 Ridge share>=90 % | 77.5000 | 77.5000 | 0.001 | PASS |
| sim $50 Ridge mean price | 41.7390 | 41.7390 | 0.001 | PASS |
| sim $50 Ridge share>=90 % (tie-EV re-pick) | 77.5000 | 77.5000 | 1.0 | PASS |
| sim $50 Main mean points | 91.1020 | 91.1020 | 0.001 | PASS |
| sim $50 Main share>=90 % | 80.7000 | 80.7000 | 0.001 | PASS |
| sim $50 Main mean price | 41.4610 | 41.4610 | 0.001 | PASS |
| sim $50 Main share>=90 % (tie-EV re-pick) | 80.7000 | 80.7500 | 1.0 | PASS |
| sim $50 all candidates within budget | True | True | 0 | PASS |
| sim $50 every market 20 distinct titles | True | True | 0 | PASS |
| sim $50 MaxPrice saved pick is a valid argmax (of 1000) | 1000 | 1000 | 0 | PASS |
| sim $50 Ridge saved pick is a valid argmax (of 1000) | 1000 | 1000 | 0 | PASS |
| sim $50 Main saved pick is a valid argmax (of 1000) | 1000 | 1000 | 0 | PASS |
| $30 Main_minus_MaxPrice__ge90_pct_points diff | 23.7000 | 23.7000 | 0.001 | PASS |
| $30 Main_minus_MaxPrice__ge90_pct_points CI low (own seed) | 20.0000 | 20.1000 | 0.5 | PASS |
| $30 Main_minus_MaxPrice__ge90_pct_points CI high (own seed) | 27.5000 | 27.3000 | 0.5 | PASS |
| $30 Ridge_minus_MaxPrice__ge90_pct_points diff | 19.5000 | 19.5000 | 0.001 | PASS |
| $30 Ridge_minus_MaxPrice__ge90_pct_points CI low (own seed) | 15.8000 | 15.8000 | 0.5 | PASS |
| $30 Ridge_minus_MaxPrice__ge90_pct_points CI high (own seed) | 23.2000 | 23.2000 | 0.5 | PASS |
| $30 Main_minus_MaxPrice__points_points diff | 1.2980 | 1.2980 | 0.001 | PASS |
| $30 Main_minus_MaxPrice__points_points CI low (own seed) | 1.1270 | 1.1230 | 0.03 | PASS |
| $30 Main_minus_MaxPrice__points_points CI high (own seed) | 1.4760 | 1.4770 | 0.03 | PASS |
| $30 Ridge_minus_MaxPrice__points_points diff | 1.0900 | 1.0900 | 0.001 | PASS |
| $30 Ridge_minus_MaxPrice__points_points CI low (own seed) | 0.9100 | 0.9090 | 0.03 | PASS |
| $30 Ridge_minus_MaxPrice__points_points CI high (own seed) | 1.2750 | 1.2740 | 0.03 | PASS |
| $30 Main_minus_MaxPrice__price_usd diff | -4.2460 | -4.2460 | 0.001 | PASS |
| $30 Main_minus_MaxPrice__price_usd CI low (own seed) | -4.5100 | -4.5110 | 0.03 | PASS |
| $30 Main_minus_MaxPrice__price_usd CI high (own seed) | -3.9850 | -3.9810 | 0.03 | PASS |
| points per price doubling | 2.0117 | 2.0117 | 0.001 | PASS |
| value map: n countries shown (n>=100) | 12 | 12 | 0 | PASS |
| value map: n varieties shown (n>=100) | 36 | 36 | 0 | PASS |
| value country Austria mean residual | 1.8162 | 1.8162 | 0.001 | PASS |
| value country Germany mean residual | 1.2634 | 1.2634 | 0.001 | PASS |
| value country Portugal mean residual | 0.9709 | 0.9709 | 0.001 | PASS |
| value country Australia mean residual | 0.3686 | 0.3686 | 0.001 | PASS |
| value country New Zealand mean residual | 0.2435 | 0.2435 | 0.001 | PASS |
| value country Argentina mean residual | -0.7173 | -0.7173 | 0.001 | PASS |
| value country Chile mean residual | -0.5676 | -0.5676 | 0.001 | PASS |
| value country Spain mean residual | -0.3579 | -0.3579 | 0.001 | PASS |
| value country US mean residual | -0.1639 | -0.1639 | 0.001 | PASS |
| value country Italy mean residual | -0.1075 | -0.1075 | 0.001 | PASS |
| value variety Grüner Veltliner mean residual | 2.0704 | 2.0704 | 0.001 | PASS |
| value variety Portuguese Red mean residual | 1.4459 | 1.4459 | 0.001 | PASS |
| value variety Riesling mean residual | 1.3478 | 1.3478 | 0.001 | PASS |
| value variety Pinot Gris mean residual | 0.8220 | 0.8220 | 0.001 | PASS |
| value variety Gewürztraminer mean residual | 0.6795 | 0.6795 | 0.001 | PASS |
| value variety Champagne Blend mean residual | -0.8684 | -0.8684 | 0.001 | PASS |
| value variety Merlot mean residual | -0.7609 | -0.7609 | 0.001 | PASS |
| value variety Cabernet Franc mean residual | -0.6873 | -0.6873 | 0.001 | PASS |
| value variety Sangiovese mean residual | -0.6472 | -0.6472 | 0.001 | PASS |
| value variety Pinot Grigio mean residual | -0.6224 | -0.6224 | 0.001 | PASS |
| critic Austria within-critic premium (builder def.) | 0.1375 | 0.1375 | 0.001 | PASS |
| critic Austria premium, 'vs OTHER countries' def. (as labelled) | 0.1375 | 0.2245 | 0.05 | WARN |
| critic Germany within-critic premium (builder def.) | 0.7712 | 0.7712 | 0.001 | PASS |
| critic Germany premium, 'vs OTHER countries' def. (as labelled) | 0.7712 | 1.4287 | 0.05 | WARN |
| critic Portugal within-critic premium (builder def.) | 0.6652 | 0.6652 | 0.001 | PASS |
| critic Portugal premium, 'vs OTHER countries' def. (as labelled) | 0.6652 | 0.8829 | 0.05 | WARN |
| critic Argentina within-critic premium (builder def.) | -0.2048 | -0.2048 | 0.001 | PASS |
| critic Argentina premium, 'vs OTHER countries' def. (as labelled) | -0.2048 | -0.2816 | 0.05 | WARN |
| critic Chile within-critic premium (builder def.) | -0.0568 | -0.0568 | 0.001 | PASS |
| critic Chile premium, 'vs OTHER countries' def. (as labelled) | -0.0568 | -0.0815 | 0.05 | PASS |
| under-$20 list reproduced by pred-only ranking | True | True | 0 | PASS |
| under-$20 top10 mean actual | 91.2000 | 91.2000 | 0.001 | PASS |
| under-$20 top10 n>=90 | 10 | 10 | 0 | PASS |
| under-$20 pool share>=90 % | 11.0184 | 11.0184 | 0.001 | PASS |
| under-$20 Austrian picks (HARNESS_PLAN v1.1 says 7/10) | 7 | 8 | 0 | WARN |
| SHAP share log_price (saved sample) | 57.0638 | 57.0638 | 0.001 | PASS |
| SHAP share log_price (own refit, own sample) | 57.0638 | 56.8310 | 1.5 | PASS |
| SHAP share vintage (saved sample) | 14.6557 | 14.6557 | 0.001 | PASS |
| SHAP share vintage (own refit, own sample) | 14.6557 | 14.4384 | 1.5 | PASS |
| SHAP share variety (saved sample) | 10.9465 | 10.9465 | 0.001 | PASS |
| SHAP share variety (own refit, own sample) | 10.9465 | 10.7841 | 1.5 | PASS |
| SHAP share province (saved sample) | 9.7552 | 9.7552 | 0.001 | PASS |
| SHAP share province (own refit, own sample) | 9.7552 | 9.6235 | 1.5 | PASS |
| SHAP share country (saved sample) | 7.5788 | 7.5788 | 0.001 | PASS |
| SHAP share country (own refit, own sample) | 7.5788 | 8.3229 | 1.5 | PASS |
| text model test R2 (meta+price+TF-IDF) | 0.7713 | 0.7713 | 0.0001 | PASS |

---

# Round 2 (2026-10-19, after the classification / regularization / quiz update)

Audited: `outputs/key_numbers.json` md5 `337ebd058a20371135056b7c5018ed12` (17:35 KST). New audit code: `audit/scripts/b1_round2.py` → `audit/results/b1_round2.json`. Round-1 results are archived in `audit/results/round1/`. All round-1 scripts were re-run against the new outputs (`results/comparison.csv`).

## R2.0 Summary

**PASS overall, with 0 FAIL. All new numbers reproduce.** Claims (b) and (c) need rewording, and two numbers are not yet in `key_numbers.json`.

| # | Item | Status |
|---|---|---|
| 1 | Classification AUCs, own L2 logistic refit, no test selection, XGBClf − Logistic CI | **PASS** |
| 2 | Simulation: Logistic / XGBClf re-picks, Main − Logistic CI at $30 | **PASS** |
| 3 | Lasso / ElasticNet metrics and zero counts | **PASS** (numbers); see claim (b) for wording |
| 4 | Raw profile missing rates; quiz rows 28150 / 26694 | **PASS** (numbers); **WARN**: neither the missing rates nor the quiz tasters are in `key_numbers.json` |
| — | Round-1 regression | 227 PASS. The round-1 WARNs are **resolved**: the critic premium is now leave-one-out and matches my "as labelled" numbers exactly (AT 0.2245, DE 1.4287, PT 0.8829, AR −0.2816, CL −0.0815), and the plan text now says 8/10. The 5 "FAIL" rows in `comparison.csv` are my stale check of the *old* definition, so they are expected. |

## R2.1 Classification (points ≥ 90)

| check | key_numbers | recomputed | tol. | status |
|---|---|---|---|---|
| test base rate | 37.79% | 37.79%; `y_ge90` equals `points ≥ 90` on all rows | 0 | PASS |
| test AUC logit_price / logit_all / logit_l1 | 0.8151 / 0.8412 / 0.8412 | 0.8151 / 0.8412 / 0.8412 (own rank-based AUC, equal to sklearn) | 1e-4 | PASS |
| test AUC xgb_clf / xgb_reg_score | 0.8571 / 0.8541 | 0.8571 / 0.8541 | 1e-4 | PASS |
| own L2 logistic refit (own encoder; C chosen on **valid** log loss; refit on train+valid) | C = 1.0, AUC 0.8412 | C = 1.0, AUC 0.8412, log loss 0.4742; corr with saved probabilities 1.0000 | 0.005 | PASS |
| AUC diff XGBClf − Logistic, paired bootstrap 95% CI (own seed) | 0.0159 [0.0139, 0.0179] | 0.0159 [0.0140, 0.0179] | 0.002 | PASS |
| extra: AUC XGBClf − XGB regressor score | — | +0.0029 [0.0015, 0.0046] | — | context |
| extra: AUC Logistic − Ridge(meta+price) as a score | — | +0.0027 [0.0017, 0.0037]; Ridge-as-score AUC 0.8385 | — | context |

**Test not used for selection (code read, `03b_classify.py`).**
- Logistic C is chosen by valid log loss (l.190–197 for L2, l.205–212 for L1).
- XGBClf's grid and early stopping use valid only (l.229–241).
- Final models are refit on train+valid and score test once (l.197, 212, 244).
- Test AUCs and CIs are only reported (l.274–286).
- The regressor score is read from the saved P2 predictions, with no refit.
- The features are the same 5 pre-review features.

Minor: the classifier framing and the two new simulation strategies were added *after* the first test-based results existed. Nothing was tuned on test, but the more strategies are shown, the more the Main headline should stay the one pre-planned comparison.

## R2.2 Simulation: new strategies

| check | key_numbers | recomputed (re-picked from `markets.csv` with argmax P, ties → lower price → random) | tol. | status |
|---|---|---|---|---|
| Logistic share ≥ 90 at $20 / $30 / $50 | 38.5 / 60.0 / 77.8 | 38.5 / 60.0 / 77.8 (tie-averaged: 38.55 / 60.0 / 77.8) | 1e-3 | PASS |
| XGBClf share ≥ 90 at $20 / $30 / $50 | 41.9 / 64.4 / 82.1 | 41.9 / 64.4 / 82.1 (tie-averaged: 41.9 / 64.4 / 82.15) | 1e-3 | PASS |
| saved pick is a valid argmax | — | 1000/1000 for both strategies at every budget; ≤ 3 tied markets per budget | 0 | PASS |
| `markets.csv` unchanged since round 1; within budget; 20 distinct titles | — | same md5 (`b85eb7a0…`); true / true | 0 | PASS |
| $30 Main − Logistic share ≥ 90, CI (own seed) | +4.7 [1.9, 7.5] | +4.7 [2.0, 7.5] | 0.5 | PASS |

## R2.3 Regularization

| check | key_numbers | recomputed | tol. | status |
|---|---|---|---|---|
| Lasso test MAE / R² (saved predictions) | 1.8178 / 0.4477 | 1.8178 / 0.4477 | 1e-4 | PASS |
| ElasticNet test MAE / R² (saved predictions) | 1.8184 / 0.4473 | 1.8184 / 0.4473 | 1e-4 | PASS |
| Lasso zero coefficients (own refit, α = 1e-4, same design) | 64 of 348 (all 64 are one-hot; 345 one-hot columns) | 64 (all one-hot) | 0 | PASS |
| ElasticNet zero coefficients (α = 3e-4, l1_ratio = 0.2) | 25 | 25 | 0 | PASS |
| L1 logistic zero coefficients | 56 | not refit (liblinear; not cheap to match exactly) | — | not checked |

**Context for claims (b) and (c):**
- **The Lasso and ElasticNet optima sit at the edge of the grid.** Lasso α = 1e-4 is the *smallest* α tried, and valid MAE falls monotonically as α falls. ElasticNet's α = 3e-4 and l1_ratio = 0.2 are both the smallest values in their grids. The true optimum is "this small or smaller".
- **The zero count depends on that grid floor.** My refits zero 26 coefficients at α = 3e-5 and 13 at α = 1e-5. The "64" is a property of the grid, not of the data.
- **A zero does not mean an irrelevant level.** All levels are one-hot encoded with an intercept, so a zero means "no different from the baseline the intercept absorbs".
- **Ridge is flat across α = 0.01–10.** Valid MAE varies by less than 0.0002 (best α = 10, an interior point) and gets worse at α = 100. Logistic C = 1 is interior; log loss for C = 0.1–10 stays within 0.0004.
- **There is no sign of overfitting in the linear models.** Ridge MAE is 1.818 on its fitting rows and 1.818 on test (R² 0.454 against 0.448), with n/p ≈ 256.

## R2.4 Raw profile and quiz

- **Raw missing rates (all 129,971 raw rows, before dedup).** I get price 6.92%, designation 28.83%, region_1 16.35%, region_2 61.14%, taster_name 20.19% and taster_twitter_handle 24.02%. `raw_profile.csv` shows 6.9 / 28.8 / 16.3 / 61.1 / 20.2 / 24.0. → **PASS** (region_1 is 21,247/129,971 = 16.348%, so 16.3 is the correct rounding).
  - Country and province are each missing in 63 rows (0.05%), which the CSV displays as "0.0".
  - After dedup, price is missing in 7.0% of rows and taster in 20.8%. If a slide puts these next to the cleaned n, say which base is used.
- **Quiz rows.**

  | row | wine | price / points | taster | status |
  |---|---|---|---|---|
  | 28150 | Opus One 2006 Red (Napa Valley) | $195 / 90 | **missing** | PASS |
  | 26694 | Montes 2006 Alpha Cabernet Sauvignon (Colchagua Valley) | $28 / 91 | **Michael Schachner** | PASS |

  Both rows are in the cleaned data (train split), and each is the only raw row with its title.
- **WARN (slide traceability rule).** The raw-profile missing rates and the quiz tasters are **not in `key_numbers.json`**. Its `quiz_pair` fields omit `taster_name`, and there is no raw-profile section. Either add them through `08_key_numbers.py` or keep them off the slides.
- **Context for the quiz reveal.**
  - The cheaper wine is the one with a named critic, Schachner. He is the critic whose countries have the *lowest* price-adjusted scores in the value map, so his 91 for a $28 wine is not a lenient score.
  - The Opus One score has no attributed critic, so the two scores may not come from the same scale.
  - Same vintage (2006) on both, which is good.

## R2.5 Claim verdicts

| claim | verdict | evidence | safe wording (KO) |
|---|---|---|---|
| (a) 로지스틱은 Ridge와 같은 수준, XGBoost가 AUC +0.016으로 소폭 우위 | **Supported** | Simulation Logistic − Ridge: −0.5 to +1.3 pp, all CIs include 0. AUC Logistic 0.841 against Ridge-as-score 0.838. XGBClf − Logistic AUC +0.016 [0.014, 0.018], and +3.4 to +4.4 pp in the simulation (CI excludes 0). | "90점 이상 판별에서 로지스틱(AUC 0.84)은 Ridge와 같은 수준이고, XGBoost(0.86)가 AUC +0.016으로 소폭 앞선다(예산 실험에서는 90점 이상 비율 +3~4%p)." |
| (b) Lasso·ElasticNet은 Ridge와 성능 차이 없음, Lasso는 345개 중 64개 계수를 0으로 | **Needs rewording** | First half supported: test MAE 1.8178 / 1.8184 against 1.8176, a difference of ≤ 0.001. The "64 of 345" is reproducible, but it holds at the *smallest α tried* and shrinks to 26 or 13 at smaller α, so it is a grid artefact. A zeroed one-hot level is not "irrelevant". | "Lasso·ElasticNet도 Ridge와 성능이 같다(MAE 차이 0.001 이내). 변수를 덜어내도 이득이 없어 Ridge를 유지했다." Drop the "64개", or write "가장 약한 규제에서도 일부 범주(64개)가 0이 됐지만 성능 차이는 없었다". |
| (c) 규제 강도는 거의 0이 최적 — 데이터가 충분해 과적합 위험이 작음 | **Needs rewording** | The direction is right: stronger regularization hurts, train and test fits are equal, and n/p ≈ 256. But "최적" is not shown. Ridge's best α = 10 is interior and the curve is flat over 0.01–10; Lasso and ElasticNet hit the grid floor. This applies to the linear models only, since XGBoost needs early stopping. | "선형모델은 규제 강도를 바꿔도 성능이 거의 변하지 않았고, 강하게 규제하면 오히려 나빠졌다. 학습·테스트 성능이 같아(관측치/변수 ≈ 256) 선형모델의 과적합 위험은 작다." |
| (d) 90점 이상 여부를 직접 맞히든 점수를 맞히든 결과는 비슷 | **Supported** (qualify: same model family) | Simulation Main − XGBClf: 0.0 / +0.3 / −1.4 pp (all CIs include 0). Logistic − Ridge: CIs include 0. AUC XGBClf − XGB regressor +0.003, statistically nonzero but tiny. | "같은 모델끼리 비교하면, '90점 이상인가'를 직접 맞히든 점수를 맞힌 뒤 고르든 결과는 거의 같았다(AUC 차이 0.003, 예산 실험 차이 유의하지 않음)." |

## R2.6 Notes for code and slides

- `03b_classify.py:164`: L1 logistic uses liblinear, which also penalizes the intercept. This is documented, and L1 and L2 results are identical, so it doesn't matter.
- If the deck shows the raw missing-rate table or the quiz tasters, add them to `key_numbers.json` first (rule in HARNESS_PLAN §3).
