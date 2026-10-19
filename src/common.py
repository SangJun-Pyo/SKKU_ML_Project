"""Shared helpers for the wine ML project (paths, seed, fonts, text utils)."""
import re
import sys
import warnings
from pathlib import Path

import numpy as np

# Windows consoles (cp949) cannot print names like Grüner / Château -> force UTF-8 output when possible.
for _stream in (sys.stdout, sys.stderr):
    try:
        if _stream is not None and (getattr(_stream, "encoding", "") or "").lower().replace("-", "") != "utf8":
            _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError, OSError):
        pass

ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = ROOT / "data" / "raw" / "winemag-data-130k-v2.csv"
PROC_DIR = ROOT / "data" / "processed"
CLEAN_CSV = PROC_DIR / "wine_clean.csv"
EDA_DIR = ROOT / "outputs" / "eda"
OUT_DIR = ROOT / "outputs"
MODEL_DIR = OUT_DIR / "models"
SIM_DIR = OUT_DIR / "sim"
VALUE_DIR = OUT_DIR / "valuemap"
CRITIC_DIR = OUT_DIR / "critic"
KEY_NUMBERS_JSON = OUT_DIR / "key_numbers.json"
SEED = 42

VINTAGE_MIN, VINTAGE_MAX = 1950, 2017

_WS = re.compile(r"\s+")
_YEAR = re.compile(r"(?<!\d)(\d{4})(?!\d)")


def ensure_dirs():
    for d in (PROC_DIR, EDA_DIR, MODEL_DIR, SIM_DIR, VALUE_DIR, CRITIC_DIR):
        d.mkdir(parents=True, exist_ok=True)


def setup_fonts():
    """Set a Korean-capable matplotlib font (Noto Sans CJK KR -> Malgun Gothic -> AppleGothic)."""
    import matplotlib
    from matplotlib import font_manager

    available = {f.name for f in font_manager.fontManager.ttflist}
    chosen = None
    for name in ("Noto Sans CJK KR", "Malgun Gothic", "AppleGothic"):
        if name in available:
            chosen = name
            break
    if chosen:
        matplotlib.rcParams["font.family"] = chosen
    else:
        msg = ("No Korean font found (Noto Sans CJK KR / Malgun Gothic / AppleGothic). Korean labels will render "
               "as empty boxes. Install one, e.g. Linux: `sudo apt install fonts-noto-cjk`, then delete the "
               "matplotlib font cache (~/.cache/matplotlib) and re-run.")
        warnings.warn(msg, stacklevel=2)
        print("WARNING:", msg, file=sys.stderr, flush=True)
    matplotlib.rcParams["axes.unicode_minus"] = False
    return chosen


def normalize_title(s):
    """Lowercase, strip, collapse whitespace. Vintage digits are kept."""
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return ""
    return _WS.sub(" ", str(s)).strip().lower()


def normalize_text(s):
    """Same normalization, used for descriptions."""
    return normalize_title(s)


def parse_vintage(title):
    """First standalone 4-digit year in [1950, 2017] found in the title, else NaN."""
    if title is None or (isinstance(title, float) and np.isnan(title)):
        return np.nan
    for m in _YEAR.findall(str(title)):
        y = int(m)
        if VINTAGE_MIN <= y <= VINTAGE_MAX:
            return float(y)
    return np.nan
