"""One visual system for every figure in the project.

Colour roles are fixed: legitimate = slot 1 (blue), fraud = slot 2 (orange).
Further series take the next categorical slots in a fixed order and are never
cycled. Grids and axes are hairline and recessive so the data stays loud.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from . import config as C  # noqa: E402

# ---- palette (validated categorical order, light surface) ----------------- #
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
          "#e87ba4", "#008300", "#6250d6", "#e34948"]
LEGIT, FRAUD = SERIES[0], SERIES[1]
CLASS_COLORS = {0: LEGIT, 1: FRAUD}
CLASS_NAMES = {0: "Legitimate", 1: "Fraud"}

BLUE_RAMP = ["#f4f8fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
             "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
SEQUENTIAL = LinearSegmentedColormap.from_list("seq_blue", BLUE_RAMP)
DIVERGING = LinearSegmentedColormap.from_list(
    "div_blue_red", ["#104281", "#3987e5", "#9ec5f4", "#f0efec", "#f3a8a7", "#e34948", "#a32a2a"]
)


def _pick_font() -> str:
    """Prefer the Windows UI sans; fall back to matplotlib's bundled DejaVu Sans."""
    for path, name in (("C:/Windows/Fonts/segoeui.ttf", "Segoe UI"),):
        if Path(path).exists():
            for p in ("C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/segoeuib.ttf",
                      "C:/Windows/Fonts/segoeuii.ttf", "C:/Windows/Fonts/seguisb.ttf"):
                if Path(p).exists():
                    font_manager.fontManager.addfont(p)
            return name
    return "DejaVu Sans"


FONT = _pick_font()


def apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": FONT,
        "font.size": 9.5,
        "text.color": INK,
        "axes.labelcolor": INK_2,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.titlesize": 11,
        "axes.titleweight": "semibold",
        "axes.titlelocation": "left",
        "axes.titlepad": 10,
        "axes.labelsize": 9.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "grid.linestyle": "-",
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "lines.linewidth": 2,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.08,
    })


def titled(ax, title: str, sub: str | None = None) -> None:
    """Panel title in primary ink with an optional secondary-ink subtitle."""
    ax.set_title(title, pad=20 if sub else 8)
    if sub:
        ax.text(0, 1.012, sub, transform=ax.transAxes, ha="left", va="bottom",
                fontsize=8.4, color=INK_2)


def thousands(x, _pos=None) -> str:
    """Axis formatter: full numbers below 10k, compact above."""
    if abs(x) >= 10_000:
        return f"{x / 1000:,.0f}K"
    return f"{x:,.0f}"


def save(fig, name: str) -> Path:
    path = C.FIG_DIR / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    return path


apply_style()
