"""P6: final Korean presentation charts -> outputs/figures_ko/fig01..fig11_*.png   (editorial "wine list" style)

Reads ONLY files in outputs/ (CSV + key_numbers.json) plus data/processed/wine_clean.csv (fig01 only).
No data value is hard-coded: the dictionaries below hold labels and colours, never numbers.
Canvas = the slide slot: 11.84 x 6.66 in at 200 dpi = 2368 x 1332 px (exactly 2x the 1184 x 666 px card), on the card
colour #fdfbf8. No in-image title and no source note (the slide headline and footer carry them).
Type sizes are set for the 2x canvas: 1 pt = 2.78 image px = 1.39 slide px.
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import font_manager as fm  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (CLEAN_CSV, CRITIC_DIR, EDA_DIR, KEY_NUMBERS_JSON, MODEL_DIR, OUT_DIR,  # noqa: E402
                    SIM_DIR, VALUE_DIR, SEED, setup_fonts)

FIG_DIR = OUT_DIR / "figures_ko"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------ fonts: serif for values/annotations, sans for axes/categories
SANS = setup_fonts()  # Noto Sans CJK KR -> Malgun Gothic -> AppleGothic (sets rcParams font.family)
_available = {f.name for f in fm.fontManager.ttflist}
SERIF = next((n for n in ("Noto Serif CJK KR", "Malgun Gothic", "AppleGothic") if n in _available), SANS)

# ------------------------------------------------------------------ labels (Korean) - one place
MODEL_KO = {
    "mean": "평균",
    "ridge_price": "가격만(Ridge)",
    "ridge_meta": "산지·품종·빈티지(Ridge)",
    "ridge_meta_price": "산지·품종·빈티지+가격(Ridge)",
    "lasso_meta_price": "Lasso (같은 변수)",
    "enet_meta_price": "ElasticNet (같은 변수)",
    "rf_meta_price": "랜덤포레스트",
    "xgb_meta_price": "XGBoost",
    "text_only_ridge": "리뷰 텍스트만(보조)",
    "text_ridge": "리뷰 텍스트 포함\n(보조·리뷰 후 정보)",
}
LADDER_ORDER = ["mean", "ridge_price", "ridge_meta", "ridge_meta_price", "lasso_meta_price", "enet_meta_price",
                "rf_meta_price", "xgb_meta_price"]
TEXT_MODELS = ["text_only_ridge", "text_ridge"]   # hatched, below the divider

STRATEGY_KO = {"Random": "무작위", "MaxPrice": "최고가 선택", "Ridge": "Ridge", "Logistic": "로지스틱", "Main": "XGBoost"}
# Logistic sits between Ridge and XGBoost so teal is never adjacent to the dark gray (validator: teal<->gray_dark
# normal-vision dE 13.3 < 15 when adjacent; teal<->ochre and teal<->wine pass).
STRATEGY_ORDER = ["Random", "MaxPrice", "Ridge", "Logistic", "Main"]

CLS_KO = {"logit_price": "로지스틱(가격만)", "logit_all": "로지스틱(전체 변수)", "xgb_clf": "XGBoost 분류",
          "xgb_reg_score": "XGBoost 회귀 예측(참고)"}
CLS_ORDER = ["logit_price", "logit_all", "xgb_clf", "xgb_reg_score"]

FEATURE_KO = {"log_price": "가격", "vintage": "빈티지", "variety": "품종", "province": "세부산지", "country": "국가"}

PRICE_BUCKET_KO = {"<=$10": "$10 이하", "($10,$20]": "$10~20", "($20,$30]": "$20~30",
                   "($30,$50]": "$30~50", "($50,$100]": "$50~100", ">$100": "$100 초과"}

COUNTRY_KO = {"Austria": "오스트리아", "Germany": "독일", "Portugal": "포르투갈", "Australia": "호주",
              "New Zealand": "뉴질랜드", "South Africa": "남아공", "France": "프랑스", "Italy": "이탈리아",
              "US": "미국", "Spain": "스페인", "Chile": "칠레", "Argentina": "아르헨티나"}

VARIETY_KO = {"Grüner Veltliner": "그뤼너 펠트리너", "Portuguese Red": "포르투갈 레드", "Riesling": "리슬링",
              "Pinot Gris": "피노 그리", "Gewürztraminer": "게뷔르츠트라미너", "Gamay": "가메",
              "Champagne Blend": "샴페인 블렌드", "Merlot": "메를로", "Cabernet Franc": "카베르네 프랑",
              "Sangiovese": "산지오베제", "Pinot Grigio": "피노 그리지오", "Cabernet Sauvignon": "카베르네 소비뇽"}

# ------------------------------------------------------------------ palette (single dictionary)
PALETTE = {
    "card": "#fdfbf8", "border": "#d9d0c4",
    "wine": "#7b2335",        # XGBoost / featured / positive
    "ochre": "#c08a2b",       # Ridge
    "ochre_light": "#d9b46a", # Lasso / ElasticNet (same linear family as Ridge)
    "teal": "#2f6f6a",        # Logistic regression
    "slate": "#3e6370",       # negative pole of the diverging charts
    "gray_light": "#b9b0a6",  # baselines: 무작위, 평균 / CI spans 0
    "gray_dark": "#8a8079",   # baselines: 최고가 선택, 랜덤포레스트
    "ink": "#231a1d", "muted": "#5b5257", "grid": "#e8e1d6",
    "hex_lo": "#f3ece2",      # hexbin ramp: cream -> wine
}
CARD, BORDER = PALETTE["card"], PALETTE["border"]
WINE, OCHRE, SLATE, TEAL = PALETTE["wine"], PALETTE["ochre"], PALETTE["slate"], PALETTE["teal"]
OCHRE_L = PALETTE["ochre_light"]
GRAY_L, GRAY_D = PALETTE["gray_light"], PALETTE["gray_dark"]
INK, MUTED, GRID = PALETTE["ink"], PALETTE["muted"], PALETTE["grid"]

STRATEGY_COLOR = {"Random": GRAY_L, "MaxPrice": GRAY_D, "Ridge": OCHRE, "Logistic": TEAL, "Main": WINE}
CLS_COLOR = {"logit_price": TEAL, "logit_all": TEAL, "xgb_clf": WINE, "xgb_reg_score": WINE}
MODEL_COLOR = {"mean": GRAY_L, "ridge_price": OCHRE, "ridge_meta": OCHRE, "ridge_meta_price": OCHRE,
               "lasso_meta_price": OCHRE_L, "enet_meta_price": OCHRE_L, "rf_meta_price": GRAY_D, "xgb_meta_price": WINE}

# ------------------------------------------------------------------ sizes: 200 dpi canvas, 1 pt = 2.78 image px
FIGSIZE, DPI = (11.84, 6.66), 200
FS_CAT = 19.5       # category labels            (54 px)
FS_TICK = 19        # numeric tick labels        (53 px)
FS_VAL = 22.5       # value labels, serif bold   (62 px)
FS_BIG = 29.5       # the one key annotation     (82 px)
FS_NOTE = 16.5      # axis titles, small notes   (46 px)
MINUS = "−"

plt.rcParams.update({
    "figure.dpi": DPI, "savefig.dpi": DPI, "figure.facecolor": CARD, "axes.facecolor": CARD, "savefig.facecolor": CARD,
    "text.color": INK, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelsize": FS_TICK, "ytick.labelsize": FS_TICK, "axes.labelsize": FS_NOTE,
    "axes.edgecolor": BORDER, "axes.linewidth": 1.0, "axes.unicode_minus": False,
})

KEY = json.loads(KEY_NUMBERS_JSON.read_text(encoding="utf-8"))
HALO = [pe.withStroke(linewidth=5, foreground=CARD)]


# ------------------------------------------------------------------ helpers
def sgn(v, d=2, plus=True):
    """Signed number with a real minus sign."""
    if round(v, d) == 0:
        v = 0.0
    s = f"{abs(v):.{d}f}"
    return (MINUS + s) if v < 0 else (("+" + s) if plus and v > 0 else s)


def new_fig(left, right, top, bottom):
    fig = plt.figure(figsize=FIGSIZE, facecolor=CARD)
    ax = fig.add_axes([left, bottom, right - left, top - bottom], facecolor=CARD)
    return fig, ax


def style_axes(ax, grid_axis=None, left_spine=False, xticks=False):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_visible(left_spine)
    ax.spines["bottom"].set_visible(True)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(BORDER)
        ax.spines[s].set_linewidth(1.0)
    ax.tick_params(axis="y", length=0, labelsize=FS_TICK, pad=6)
    ax.tick_params(axis="x", length=4 if xticks else 0, color=BORDER, width=1.0, pad=7, labelsize=FS_TICK)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, lw=0.9)
    ax.set_axisbelow(True)


def finish(fig, name):
    path = FIG_DIR / name
    fig.savefig(path, dpi=DPI, facecolor=CARD)
    plt.close(fig)
    print("saved", path.relative_to(OUT_DIR.parent))


def label(ax, x, y, s, dx=6, dy=0, ha="left", va="center", size=FS_VAL, weight="bold", color=INK, family=None, **kw):
    return ax.annotate(s, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va, fontsize=size,
                       fontweight=weight, color=color, fontfamily=family or SERIF, annotation_clip=False, **kw)


def pct(v, d=1):
    return f"{v:.{d}f}%"


def ylabel_top(ax, text, y=1.04):
    """Axis title laid flat above the y axis (editorial style) instead of rotated."""
    ax.text(0, y, text, transform=ax.transAxes, ha="left", va="bottom", fontsize=FS_NOTE, color=MUTED)


def int_ticks(lo, hi, max_n=5):
    ticks = list(range(int(np.ceil(lo)), int(np.floor(hi)) + 1))
    if len(ticks) > max_n:
        ticks = [t for t in ticks if t % 2 == 0]
    return ticks


def delta_bracket(ax, x_from, y_from, x_to, y_to, hw, gap):
    """Thin bracket right of bar `x_to`, reaching back (dashed) to the top of bar `x_from`. Returns bracket x."""
    xb = x_to + hw + gap
    ax.plot([x_from + hw, xb], [y_from, y_from], color=MUTED, lw=1.1, ls=(0, (4, 3)), zorder=6)
    ax.plot([xb, xb], [y_from, y_to], color=INK, lw=1.5, zorder=6, solid_capstyle="butt")
    for yy in (y_from, y_to):
        ax.plot([xb - gap * 0.7, xb], [yy, yy], color=INK, lw=1.5, zorder=6, solid_capstyle="butt")
    return xb


def key_block(ax, x, y, key, note=None, ha="right"):
    """Key annotation (serif bold, large) with an optional small CI note beneath it; bottom of block at (x, y)."""
    base = note if note else key
    if note:
        n = label(ax, x, y, note, dx=0, ha=ha, va="bottom", size=FS_NOTE, weight="normal", color=MUTED, family=SANS,
                  linespacing=1.25)
        k = ax.annotate(key, xy=(1.0 if ha == "right" else 0.5, 1.0), xycoords=n, xytext=(0, 4), textcoords="offset points",
                        ha=ha, va="bottom", fontsize=FS_BIG, fontweight="bold", color=INK, fontfamily=SERIF,
                        annotation_clip=False)
        return k
    return label(ax, x, y, key, dx=0, ha=ha, va="bottom", size=FS_BIG)


# ================================================================== fig01 price vs points (hexbin)
def fig01():
    df = pd.read_csv(CLEAN_CSV, usecols=["price", "points"])
    r = KEY["eda"]["pearson_r_logprice_points"]
    fig, ax = new_fig(left=0.115, right=0.955, top=0.96, bottom=0.215)
    cmap = LinearSegmentedColormap.from_list("cream_wine", [PALETTE["hex_lo"], WINE])
    rng = np.random.default_rng(SEED)   # +-0.5 uniform jitter on the integer score, for the hexbin only
    pts_j = df["points"].to_numpy() + rng.uniform(-0.5, 0.5, len(df))
    ax.hexbin(df["price"], pts_j, xscale="log", gridsize=(52, 20),
              extent=(np.log10(3.6), np.log10(3600), 79.5, 100.5), bins="log", mincnt=1, cmap=cmap,
              linewidths=0.3, edgecolors=CARD)
    # median points per log-price bin (computed here from the data, not typed in)
    edges = np.logspace(np.log10(4), np.log10(500), 17)
    idx = np.digitize(df["price"], edges)
    xs, ys = [], []
    for i in range(1, len(edges)):
        sel = df.loc[idx == i, "points"]
        if len(sel) >= 100:
            xs.append(np.sqrt(edges[i - 1] * edges[i]))
            ys.append(sel.median())
    ax.plot(xs, ys, color=INK, lw=2.6, solid_capstyle="round", zorder=5, path_effects=HALO)
    ax.set_xscale("log")
    ax.set_xlim(3.6, 3600)
    ax.set_ylim(79.5, 100.5)
    ax.xaxis.set_major_locator(FixedLocator([5, 20, 100, 500, 3000]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax.set_yticks([80, 90, 100])
    ax.set_xlabel("가격 (USD, 로그 눈금)", labelpad=8)
    ax.set_ylabel("평점 (점)", labelpad=8)
    style_axes(ax, grid_axis="y", xticks=True)
    ax.text(0.025, 0.975, f"r = {r:.2f}", transform=ax.transAxes, fontsize=FS_BIG, fontfamily=SERIF, fontweight="bold",
            va="top", ha="left", color=INK, path_effects=HALO)
    ax.text(0.025, 0.855, "로그 가격 기준 상관", transform=ax.transAxes, fontsize=FS_NOTE, va="top", ha="left",
            color=MUTED, path_effects=HALO)
    ax.annotate("중앙값", (xs[-1], ys[-1]), xytext=(8, 0), textcoords="offset points", ha="left", va="center",
                fontsize=FS_CAT, color=INK, fontfamily=SERIF, fontweight="bold", path_effects=HALO)
    finish(fig, "fig01_price_vs_points.png")


# ================================================================== fig02 share >= 90 by price bucket
def fig02():
    d = pd.read_csv(EDA_DIR / "price_buckets.csv")
    d["lab"] = d["price_bucket"].map(PRICE_BUCKET_KO).fillna(d["price_bucket"])
    overall = KEY["eda"]["share_ge90_pct"]
    fig, ax = new_fig(left=0.125, right=0.975, top=0.87, bottom=0.215)
    x = np.arange(len(d))
    vals = d["share_ge_90"] * 100
    ax.bar(x, vals, width=0.6, color=WINE, zorder=3)
    for xi, v in zip(x, vals):
        label(ax, xi, v, pct(v), dx=0, dy=5, ha="center", va="bottom", size=FS_VAL, path_effects=HALO, zorder=7)
    ax.axhline(overall, color=GRAY_D, lw=1.2, ls=(0, (5, 4)), zorder=2)
    ax.annotate(f"전체 평균 {overall:.1f}%", (-0.5, overall), xytext=(0, 5), textcoords="offset points", ha="left",
                va="bottom", fontsize=FS_NOTE + 1, color=MUTED, path_effects=HALO)
    ax.set_xticks(x)
    ax.set_xticklabels(list(d["lab"]), fontsize=FS_CAT, color=INK)
    ax.set_ylim(0, 100)
    ax.set_xlim(-0.55, len(d) - 0.45)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.set_xlabel("가격대 (USD)", labelpad=8)
    ylabel_top(ax, "90점 이상 와인 비율 (%)", y=1.05)
    style_axes(ax, grid_axis="y")
    finish(fig, "fig02_price_bucket_ge90.png")


# ================================================================== fig03 model ladder
def fig03():
    m = pd.read_csv(MODEL_DIR / "metrics.csv")
    m = m[m["split"] == "test"].set_index("model")
    rows = LADDER_ORDER + TEXT_MODELS
    fig, ax = new_fig(left=0.335, right=0.97, top=0.975, bottom=0.185)
    n0 = len(LADDER_ORDER)
    ypos = list(range(n0)) + [n0 + 0.9 + i for i in range(len(TEXT_MODELS))]
    h = 0.6
    for mdl, y in zip(rows, ypos):
        v = float(m.loc[mdl, "R2"])
        v = 0.0 if round(v, 2) == 0 else v
        if mdl in TEXT_MODELS:
            ax.barh(y, v, height=h, facecolor=CARD, edgecolor=MUTED, hatch="///", linewidth=1.2, zorder=3)
        else:
            ax.barh(y, v, height=h, color=MODEL_COLOR[mdl], zorder=3)
        feat = mdl == "xgb_meta_price"
        label(ax, v, y, f"{v:.2f}", dx=6, size=FS_BIG if feat else FS_VAL)
    ax.set_yticks(ypos)
    ax.set_yticklabels([MODEL_KO[k] for k in rows], fontsize=FS_CAT, color=INK, linespacing=1.15)
    ax.get_yticklabels()[n0 - 1].set_fontweight("bold")
    ax.invert_yaxis()
    div = (ypos[n0 - 1] + ypos[n0]) / 2
    ax.axhline(div, color=BORDER, lw=1.0, zorder=1)
    ax.set_xlim(0, 0.92)
    ax.set_ylim(ypos[-1] + 0.7, -0.6)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.1f}"))
    ax.set_xlabel("테스트 데이터 설명력 R² (높을수록 좋음)", labelpad=8)
    style_axes(ax, left_spine=True, xticks=True)
    finish(fig, "fig03_model_ladder.png")


# ================================================================== fig04 SHAP share
def fig04():
    """SHAP shares with country + province read together as one 'origin' group (they are ~the same information,
    see 02b_associations.py / 03e_collinearity_check.py). Shares come from 03e (per-row SHAP summed, then |.|)."""
    cc = json.loads((MODEL_DIR / "collinearity_check.json").read_text(encoding="utf-8"))
    sh = cc["shap_main_origin_combined_pct"]
    s = pd.DataFrame({"feature": list(sh), "share_pct": list(sh.values())}).sort_values("share_pct", ascending=False)
    non_price = 100 - sh["가격"]
    fig, ax = new_fig(left=0.24, right=0.97, top=0.965, bottom=0.215)
    y = np.arange(len(s))
    cols = [WINE if f == "가격" else GRAY_L for f in s["feature"]]
    ax.barh(y, s["share_pct"], height=0.62, color=cols, zorder=3)
    for yi, v, f in zip(y, s["share_pct"], s["feature"]):
        label(ax, v, yi, pct(v), dx=6, size=FS_BIG if f == "가격" else FS_VAL)
    ax.set_yticks(y)
    ax.set_yticklabels([f.replace("(", "\n(") for f in s["feature"]], fontsize=FS_CAT + 1, color=INK)
    ax.invert_yaxis()
    ax.set_xlim(0, 72)
    ax.set_xticks([0, 20, 40, 60])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.set_xlabel("XGBoost 예측에서 차지하는 몫 (%)", labelpad=8)
    style_axes(ax, left_spine=True, xticks=True)
    yb0, yb1 = y[1] - 0.31, y[-1] + 0.31
    xb = 30
    ax.plot([xb - 1, xb, xb, xb - 1], [yb0, yb0, yb1, yb1], color=MUTED, lw=1.4, clip_on=False)
    ax.text(xb + 1.8, (yb0 + yb1) / 2, f"가격 외\n합계 {non_price:.1f}%", fontsize=FS_VAL, fontfamily=SERIF,
            fontweight="bold", va="center", ha="left", color=INK, linespacing=1.25)
    ax.text(0.995, 0.0, "SHAP 평균 절댓값 기준 · 국가와 세부 산지는 합쳐서 계산", transform=ax.transAxes,
            fontsize=FS_NOTE, color=MUTED, ha="right", va="bottom")
    finish(fig, "fig04_shap_share.png")


# ================================================================== fig05 text terms
def fig05():
    t = pd.read_csv(MODEL_DIR / "text_top_terms.csv")
    n = 8
    pos = t[(t["direction"] == "positive") & (t["rank"] <= n)].sort_values("rank")
    neg = t[(t["direction"] == "negative") & (t["rank"] <= n)].sort_values("rank", ascending=False)
    fig, ax = new_fig(left=0.04, right=0.97, top=0.975, bottom=0.185)
    ypos = list(range(len(pos))) + [len(pos) + 1.0 + i for i in range(len(neg))]
    rows = list(pos.itertuples()) + list(neg.itertuples())
    for y, r in zip(ypos, rows):
        ax.barh(y, r.coef, height=0.66, color=WINE if r.coef > 0 else SLATE, zorder=3)
        label(ax, r.coef, y, r.term, dx=6 if r.coef > 0 else -6, ha="left" if r.coef > 0 else "right", size=FS_CAT,
              weight="medium", family=SANS)
    ax.axvline(0, color=MUTED, lw=1.2, zorder=4)
    ax.invert_yaxis()
    lo, hi = min(pos["coef"].min(), neg["coef"].min()), max(pos["coef"].max(), neg["coef"].max())
    ax.set_xlim(np.floor(lo) - 2.4, np.ceil(hi) + 2.2)
    ax.set_ylim(ypos[-1] + 0.65, -0.65)
    ax.set_yticks([])
    ax.set_xticks([-5, 0, 5])
    ax.set_xlabel("평점과의 연관(계수)", labelpad=8)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: sgn(v, 0, plus=False)))
    style_axes(ax, xticks=True)
    ymid_pos, ymid_neg = np.mean(ypos[:len(pos)]), np.mean(ypos[len(pos):])
    ax.annotate("평점을 올리는 단어", (0, ymid_pos), xytext=(-10, 0), textcoords="offset points", ha="right", va="center",
                fontsize=FS_VAL, fontfamily=SERIF, fontweight="bold", color=WINE)
    ax.annotate("평점을 깎는 단어", (0, ymid_neg), xytext=(10, 0), textcoords="offset points", ha="left", va="center",
                fontsize=FS_VAL, fontfamily=SERIF, fontweight="bold", color=SLATE)
    ax.text(0.995, 0.0, "※ 리뷰 작성 후 정보(보조분석)", transform=ax.transAxes, ha="right", va="bottom", fontsize=FS_NOTE,
            color=MUTED)
    finish(fig, "fig05_text_terms.png")


# ================================================================== value bars (fig06, fig08)
def value_bars(ax, d, labels, ypos, xlim):
    h = 0.58
    for (_, r), y in zip(d.iterrows(), ypos):
        sig = (r.ci95_low > 0) or (r.ci95_high < 0)
        c = (WINE if r.mean_residual > 0 else SLATE) if sig else GRAY_L
        ax.barh(y, r.mean_residual, height=h, color=c, zorder=3)
        ax.plot([r.ci95_low, r.ci95_high], [y, y], color=INK, lw=1.7, zorder=5, solid_capstyle="butt")
        for xe in (r.ci95_low, r.ci95_high):
            ax.plot([xe, xe], [y - 0.14, y + 0.14], color=INK, lw=1.7, zorder=5)
        if r.mean_residual >= 0:
            label(ax, r.ci95_high, y, sgn(r.mean_residual), dx=6, size=FS_VAL)
        else:
            label(ax, r.ci95_low, y, sgn(r.mean_residual), dx=-6, ha="right", size=FS_VAL)
    ax.axvline(0, color=MUTED, lw=1.2, zorder=4)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=FS_CAT, color=INK)
    ax.set_xlim(*xlim)
    ax.set_ylim(max(ypos) + 0.6, min(ypos) - 0.6)
    ax.set_xticks(int_ticks(*xlim))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: sgn(v, 0)))
    ax.set_xlabel("가격만으로 예측한 평점 대비 차이 (점)", labelpad=8)
    style_axes(ax, xticks=True)
    tr = ax.get_xaxis_transform()
    ax.annotate("가격에 비해 낮음", (0, 1.02), xycoords=tr, xytext=(-6, 0), textcoords="offset points", ha="right",
                va="bottom", fontsize=FS_NOTE, color=SLATE, fontweight="bold", annotation_clip=False)
    ax.annotate("가격에 비해 높음", (0, 1.02), xycoords=tr, xytext=(6, 0), textcoords="offset points", ha="left",
                va="bottom", fontsize=FS_NOTE, color=WINE, fontweight="bold", annotation_clip=False)
    ax.text(0.995, 0.02, "회색: 신뢰구간이 0 포함", transform=ax.transAxes, ha="right", va="bottom", fontsize=FS_NOTE,
            color=MUTED, bbox=dict(facecolor=CARD, edgecolor="none", pad=2), zorder=8)


def fig06():
    d = pd.read_csv(VALUE_DIR / "value_by_country.csv").sort_values("mean_residual", ascending=False).reset_index(drop=True)
    fig, ax = new_fig(left=0.185, right=0.965, top=0.9, bottom=0.185)
    labels = [COUNTRY_KO.get(c, c) for c in d["country"]]
    xlim = (np.floor(d["ci95_low"].min() * 2) / 2 - 0.5, np.ceil(d["ci95_high"].max() * 2) / 2 + 0.55)
    value_bars(ax, d, labels, list(range(len(d))), xlim)
    finish(fig, "fig06_value_country.png")


def fig08():
    d = pd.read_csv(VALUE_DIR / "value_by_variety.csv").sort_values("mean_residual", ascending=False).reset_index(drop=True)
    d = pd.concat([d.head(6), d.tail(6)]).reset_index(drop=True)
    ypos = list(range(6)) + [6 + 0.7 + i for i in range(6)]
    fig, ax = new_fig(left=0.255, right=0.965, top=0.9, bottom=0.185)
    labels = [VARIETY_KO.get(v, v) for v in d["variety"]]
    xlim = (np.floor(d["ci95_low"].min() * 2) / 2 - 0.5, np.ceil(d["ci95_high"].max() * 2) / 2 + 0.55)
    value_bars(ax, d, labels, ypos, xlim)
    ax.axhline(5 + 0.5 + 0.35, color=GRID, lw=1.0, zorder=1)
    finish(fig, "fig08_value_variety.png")


# ================================================================== fig07 within-critic dumbbell
def fig07():
    w = pd.read_csv(CRITIC_DIR / "within_critic_check.csv")
    v = pd.read_csv(VALUE_DIR / "value_by_country.csv").set_index("country")
    pick = ["Austria", "Germany", "Portugal", "Argentina", "Chile"]
    d = w[w["country"].isin(pick)].copy()
    d["raw"] = d["raw_mean_residual"]
    d["raw_lo"] = d["country"].map(v["ci95_low"])
    d["raw_hi"] = d["country"].map(v["ci95_high"])
    d = d.sort_values("raw", ascending=False).reset_index(drop=True)
    fig, ax = new_fig(left=0.225, right=0.97, top=0.925, bottom=0.185)
    off = 0.22
    for i, r in d.iterrows():
        y_raw, y_in = i - off, i + off
        ax.plot([r.raw, r.within_critic_premium], [y_raw, y_in], color=GRAY_L, lw=2.0, zorder=2)
        ax.plot([r.raw_lo, r.raw_hi], [y_raw, y_raw], color=GRAY_D, lw=2.0, zorder=3)
        ax.plot([r.ci95_low, r.ci95_high], [y_in, y_in], color=WINE, lw=2.0, zorder=3)
        for xe in (r.raw_lo, r.raw_hi):
            ax.plot([xe, xe], [y_raw - 0.07, y_raw + 0.07], color=GRAY_D, lw=2.0, zorder=3)
        for xe in (r.ci95_low, r.ci95_high):
            ax.plot([xe, xe], [y_in - 0.07, y_in + 0.07], color=WINE, lw=2.0, zorder=3)
        ax.scatter([r.raw], [y_raw], s=110, facecolor=CARD, edgecolor=GRAY_D, linewidth=2.4, zorder=6)
        ax.scatter([r.within_critic_premium], [y_in], s=110, facecolor=WINE, edgecolor=CARD, linewidth=1.2, zorder=6)
        hi_raw, hi_in = max(r.raw_hi, r.raw), max(r.ci95_high, r.within_critic_premium)
        a = label(ax, hi_raw, y_raw, sgn(r.raw), dx=6, size=FS_VAL, color=MUTED, path_effects=HALO)
        b = label(ax, hi_in, y_in, sgn(r.within_critic_premium), dx=6, size=FS_VAL, color=WINE, path_effects=HALO)
        if i == 0:   # direct series labels on the first row, small, anchored to the value labels
            ax.annotate("단순 비교", xy=(1, 0.5), xycoords=a, xytext=(8, 0), textcoords="offset points", ha="left",
                        va="center", fontsize=FS_NOTE, color=MUTED, annotation_clip=False)
            ax.annotate("같은 평론가 안에서 비교", xy=(1, 0.5), xycoords=b, xytext=(8, 0), textcoords="offset points",
                        ha="left", va="center", fontsize=FS_NOTE, color=WINE, fontweight="bold", annotation_clip=False)
    # note under Germany: per-critic deviations from key_numbers
    ger = KEY["critic_check"].get("critic_germany_by_critic")
    if ger and "Germany" in set(d["country"]):
        gi = int(d.index[d["country"] == "Germany"][0])
        txt = "평론가별 " + " / ".join(sgn(g["within_critic_deviation_points"]) for g in ger)
        ax.annotate(txt, (0, gi + 0.5), xycoords=ax.get_yaxis_transform(), xytext=(-8, 0), textcoords="offset points",
                    ha="right", va="center", fontsize=FS_NOTE, color=MUTED, annotation_clip=False)
    ax.axvline(0, color=MUTED, lw=1.2, zorder=1)
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels([COUNTRY_KO.get(c, c) for c in d["country"]], fontsize=FS_CAT, color=INK)
    ax.set_ylim(len(d) - 0.4, -0.7)
    lo = np.floor((min(d["raw_lo"].min(), d["ci95_low"].min()) - 0.2) * 2) / 2
    hi_data = max(d["raw_hi"].max(), d["ci95_high"].max())
    ax.set_xlim(lo, hi_data + 1.2)
    ax.set_xticks(int_ticks(lo, hi_data + 1.0))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: sgn(v, 0)))
    ax.set_xlabel("가격만으로 예측한 평점 대비 차이 (점)", labelpad=8)
    style_axes(ax, xticks=True)
    ax.text(0.0, 1.0, "점 = 평균 · 선 = 95% 신뢰구간", transform=ax.transAxes, ha="left", va="bottom", fontsize=FS_NOTE,
            color=MUTED)
    finish(fig, "fig07_value_critic.png")


# ================================================================== fig09 sim all budgets
def fig09():
    s = pd.read_csv(SIM_DIR / "sim_summary.csv")
    ci = pd.read_csv(SIM_DIR / "sim_bootstrap_ci.csv")
    budget = 30
    g = ci[(ci["budget"] == budget) & (ci["comparison"] == "Main - MaxPrice") & (ci["metric"] == "ge90")].iloc[0]
    budgets = sorted(s["budget"].unique())
    fig, ax = new_fig(left=0.105, right=0.97, top=0.84, bottom=0.145)
    ns = len(STRATEGY_ORDER)
    bw, step = 0.155, 0.168
    xpos, val = {}, {}
    for gi, b in enumerate(budgets):
        for si, st in enumerate(STRATEGY_ORDER):
            v = float(s[(s["budget"] == b) & (s["strategy"] == st)]["share_ge90_pct"].iloc[0])
            x = gi + (si - (ns - 1) / 2) * step
            xpos[(b, st)], val[(b, st)] = x, v
            ax.bar(x, v, width=bw, color=STRATEGY_COLOR[st], zorder=3)
            label(ax, x, v, f"{v:.0f}", dx=0, dy=4, ha="center", va="bottom", size=FS_VAL - 3)   # unit in axis title
    xb = delta_bracket(ax, xpos[(budget, "MaxPrice")], val[(budget, "MaxPrice")], xpos[(budget, "Main")],
                       val[(budget, "Main")], bw / 2, gap=0.04)
    y_key = val[(budget, "Main")] + 13
    ax.plot([xb, xb], [val[(budget, "Main")], y_key - 1], color=INK, lw=1.0, zorder=6)
    key_block(ax, xb, y_key, f"{sgn(g['diff'], 1)}%p", ha="center")
    ax.set_xticks(range(len(budgets)))
    ax.set_xticklabels([f"예산 ${b}" for b in budgets], fontsize=FS_CAT, color=INK)
    ax.set_ylim(0, 96)
    ax.set_yticks([0, 30, 60, 90])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.set_xlim(-0.5, len(budgets) - 0.5)
    ylabel_top(ax, "90점 이상 비율 (%)", y=1.1)
    style_axes(ax, grid_axis="y")
    ax.tick_params(axis="x", pad=8)
    handles = [Patch(facecolor=STRATEGY_COLOR[k], label=STRATEGY_KO[k]) for k in STRATEGY_ORDER]
    leg = ax.legend(handles=handles, loc="lower right", bbox_to_anchor=(1.0, 1.01), ncol=ns, frameon=False,
                    fontsize=FS_CAT, handlelength=0.9, handleheight=0.9, columnspacing=1.0, handletextpad=0.4,
                    borderaxespad=0)
    for t in leg.get_texts():
        t.set_color(INK)
    finish(fig, "fig09_sim_all_budgets.png")


# ================================================================== fig10 $30 budget, two panels
def fig10():
    s = pd.read_csv(SIM_DIR / "sim_summary.csv")
    budget = 30
    s = s[s["budget"] == budget].set_index("strategy")
    ci = pd.read_csv(SIM_DIR / "sim_bootstrap_ci.csv")
    ci = ci[(ci["budget"] == budget) & (ci["comparison"] == "Main - MaxPrice")].set_index("metric")
    fig = plt.figure(figsize=FIGSIZE, facecolor=CARD)
    axl = fig.add_axes([0.095, 0.175, 0.385, 0.65], facecolor=CARD)
    axr = fig.add_axes([0.605, 0.175, 0.37, 0.65], facecolor=CARD)
    names = [STRATEGY_KO[k] for k in STRATEGY_ORDER]
    cols = [STRATEGY_COLOR[k] for k in STRATEGY_ORDER]
    x = np.arange(len(STRATEGY_ORDER))
    bw = 0.62

    def panel(ax, col, fmt, ylim, ticks, ylabel, tickfmt, key, note, y_note):
        vals = [float(s.loc[k, col]) for k in STRATEGY_ORDER]
        ax.bar(x, vals, width=bw, color=cols, zorder=3)
        for xi, v, k in zip(x, vals, STRATEGY_ORDER):
            label(ax, xi, v, fmt(v), dx=0, dy=4, ha="center", va="bottom", size=FS_VAL - 3)
        ax.set_xticks(x)
        # 5 bars per panel: stagger the labels on two rows so neighbours never collide
        ax.set_xticklabels([("\n" + n) if i % 2 else n for i, n in enumerate(names)], fontsize=FS_CAT, color=INK,
                           linespacing=1.15)
        ax.set_ylim(*ylim)
        ax.set_yticks(ticks)
        ax.yaxis.set_major_formatter(FuncFormatter(tickfmt))
        ax.set_xlim(-0.6, len(STRATEGY_ORDER) - 0.05)
        ylabel_top(ax, ylabel, y=1.04)
        style_axes(ax, grid_axis="y")
        ax.tick_params(axis="x", pad=8)
        i_max, i_xgb = STRATEGY_ORDER.index("MaxPrice"), STRATEGY_ORDER.index("Main")
        xb = delta_bracket(ax, x[i_max], vals[i_max], x[i_xgb], vals[i_xgb], bw / 2, gap=0.42)
        ax.plot([xb, xb], [max(vals[i_max], vals[i_xgb]), y_note - 0.2 * (ylim[1] - ylim[0]) / 10], color=INK, lw=1.0,
                zorder=6)
        key_block(ax, len(STRATEGY_ORDER) - 0.07, y_note, key, note, ha="right")

    g, p = ci.loc["ge90"], ci.loc["price"]
    panel(axl, "share_ge90_pct", lambda v: f"{v:.1f}", (0, 112), [0, 50, 100], "90점 이상 비율 (%)", lambda v, _: f"{v:.0f}%",
          f"{sgn(g['diff'], 1)}%p", f"95% CI\n{sgn(g['ci95_low'], 1)} ~ {sgn(g['ci95_high'], 1)}", 77)
    panel(axr, "mean_price_usd", lambda v: f"{v:.1f}", (0, 50), [0, 20, 40], "고른 와인의 평균 가격 (USD)",
          lambda v, _: f"${v:.0f}",
          sgn(p["diff"], 1).replace(MINUS, MINUS + "$").replace("+", "+$"),
          f"95% CI\n{MINUS}\\${abs(p['ci95_low']):.1f} ~ {MINUS}\\${abs(p['ci95_high']):.1f}", 34)
    finish(fig, "fig10_sim_30.png")


# ================================================================== fig11 classification ROC-AUC (points >= 90)
def fig11():
    m = pd.read_csv(MODEL_DIR / "classify_metrics.csv")
    m = m[m["split"] == "test"].set_index("model")
    dci = pd.read_csv(MODEL_DIR / "classify_auc_diff_ci.csv").set_index("model")
    fig, ax = new_fig(left=0.30, right=0.97, top=0.9, bottom=0.185)
    n_main = len(CLS_ORDER) - 1
    ypos = list(range(n_main)) + [n_main + 0.5]
    h = 0.6
    vals = [float(m.loc[k, "roc_auc"]) for k in CLS_ORDER]
    for k, y, v in zip(CLS_ORDER, ypos, vals):
        if k == "xgb_reg_score":   # reference: outlined, not a fitted classifier
            ax.barh(y, v, height=h, facecolor=CARD, edgecolor=WINE, linewidth=1.6, zorder=3)
        else:
            ax.barh(y, v, height=h, color=CLS_COLOR[k], zorder=3)
        label(ax, v, y, f"{v:.3f}", dx=6, size=FS_BIG if k == "xgb_clf" else FS_VAL)
    ax.axhline((ypos[n_main - 1] + ypos[n_main]) / 2, color=BORDER, lw=1.0, zorder=1)
    ax.set_yticks(ypos)
    ax.set_yticklabels([CLS_KO[k] for k in CLS_ORDER], fontsize=FS_CAT, color=INK)
    ax.invert_yaxis()
    lo = 0.5
    ax.set_xlim(lo, 0.95)
    ax.set_ylim(ypos[-1] + 0.6, -0.6)
    ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.1f}"))
    ax.set_xlabel("테스트 ROC-AUC (90점 이상 구분력, 0.5 = 무작위)", labelpad=8)
    style_axes(ax, left_spine=True, xticks=True)
    d = dci.loc["xgb_clf"]
    ax.text(0.0, 1.02, f"XGBoost 분류 − 로지스틱(전체 변수): {sgn(d['diff_auc'], 3)}  "
                       f"(95% CI {sgn(d['ci95_low'], 3)} ~ {sgn(d['ci95_high'], 3)})",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=FS_NOTE, color=MUTED)
    finish(fig, "fig11_auc.png")


if __name__ == "__main__":
    print("fonts:", SANS, "/", SERIF)
    for fn in (fig01, fig02, fig03, fig04, fig05, fig06, fig07, fig08, fig09, fig10, fig11):
        fn()
