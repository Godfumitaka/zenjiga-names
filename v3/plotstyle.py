"""図の体裁（日本語の字体と色）。"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

_names = {f.name for f in font_manager.fontManager.ttflist}
for cand in ("Hiragino Sans", "Hiragino Kaku Gothic ProN", "Hiragino Kaku Gothic Pro", "Noto Sans CJK JP"):
    if cand in _names:
        JP = cand
        break
else:
    JP = "DejaVu Sans"

INK, INK2, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#ffffff"
COND_COL = {"N0": "#52514e", "N1": "#d1495b", "N2": "#2a78d6", "M": "#1baf7a", "S": "#e8a33d"}
COND_NAME = {"N0": "N0 命名なし", "N1": "N1 本条件（噂・本人不在）", "N2": "N2 回らない名前",
             "M": "M 記憶の対照", "S": "S 噂を止める"}

plt.rcParams.update({
    "font.family": JP, "font.size": 9, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
    "axes.titlecolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "axes.facecolor": SURF,
    "figure.facecolor": SURF, "axes.spines.top": False, "axes.spines.right": False,
    "axes.unicode_minus": False, "axes.titlesize": 9.5, "axes.titlelocation": "left",
    "legend.frameon": False, "legend.fontsize": 8, "savefig.dpi": 150,
})


def seam(ax, t1, t2, label=True, y=None):
    ax.axvspan(t1, t2, color=GRID, lw=0, zorder=0)
    if label:
        yl = ax.get_ylim()
        ax.text((t1 + t2) / 2, y if y is not None else yl[1], "継ぎ目", ha="center", va="bottom",
                fontsize=7.5, color=INK2)
