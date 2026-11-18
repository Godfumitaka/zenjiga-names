"""v3 段 4 の分析：p × κ の表、κ* と p の関係、R3〜R7 を掃引の上で、Δω・N・輪。

実行：.venv/bin/python -m v3.analyze4
"""
from __future__ import annotations

import json
import os

import numpy as np

from .analysis import describe
from .plotstyle import COND_COL, INK2, MUTED, plt
from .stage3 import evaluate, mwu

OUT = "results/v3/stage4"
PS = [0.02, 0.05, 0.1, 0.2, 0.5]
KS = [0.25, 0.5, 0.75, 1.0]
T, T2 = 1500.0, 260.0


def tg(cond, p=0.1, k=1.0, dw=1.2, N=4, top="all"):
    return f"{cond}|p={p}|k={k}|dw={dw}|N={N}|top={top}"


def load(name):
    fn = f"{OUT}/{name}.json"
    if not os.path.exists(fn):
        return None
    with open(fn) as f:
        return json.load(f)["summaries"]


def n0_ref():
    with open("results/v3/stage3/summary.json") as f:
        return json.load(f)["per_run"]["N0"]


def pk_cell(S, n0, p, k):
    n1 = S[tg("N1", p, k)]
    n2 = S[tg("N2", p, k)]
    m = S[tg("M", p, k)]
    s = S[tg("S", p, k)]
    ren = [x for x in n1 if x["renamed_in_seam"]]
    out = {
        "N1_frac_renamed": float(np.mean([x["renamed_in_seam"] for x in n1])),
        "N1_persist_all": describe([x["persist"] for x in n1]),
        "N1_persist_renamed": describe([x["persist"] for x in ren]) if ren else None,
        "N1_frac_censored_renamed": float(np.mean([x["censored"] for x in ren])) if ren else None,
        "N1_frac_censored_all": float(np.mean([x["censored"] for x in n1])),
        "N1_persist_sustained_renamed": describe([x["persist_sustained"] for x in ren]) if ren else None,
        "R3_mwu_N1renamed_gt_N0": mwu([x["persist"] for x in ren], [x["persist"] for x in n0], "greater"),
        "R3_mwu_sustained": mwu([x["persist_sustained"] for x in ren], [x["persist_sustained"] for x in n0], "greater"),
        "N2_persist": describe([x["persist"] for x in n2]),
        "R4_mwu_N2_lt_N1": mwu([x["persist"] for x in n2], [x["persist"] for x in n1], "less"),
        "R4_mwu_sustained": mwu([x["persist_sustained"] for x in n2], [x["persist_sustained"] for x in n1], "less"),
        "M_persist": describe([x["persist"] for x in m]),
        "R5_mwu_M_lt_N1": mwu([x["persist"] for x in m], [x["persist"] for x in n1], "less"),
        "R5_mwu_sustained": mwu([x["persist_sustained"] for x in m], [x["persist_sustained"] for x in n1], "less"),
        "M_sank": float(np.mean([x["sank"] for x in m])),
        "M_absmem_late": describe([x["absmem_late"] for x in m]),
        "R7_N1_sameName_ne_bin": describe([x["frac_sameName_ne_bin_post"] for x in n1]),
        "R7_N2_sameName_ne_bin": describe([x["frac_sameName_ne_bin_post"] for x in n2]),
        "R7_N1_name_ne_bin": describe([x["frac_name_ne_bin_post"] for x in n1]),
        "R7_N2_name_ne_bin": describe([x["frac_name_ne_bin_post"] for x in n2]),
        "R7_mwu_N1_gt_N2": mwu([x["frac_sameName_ne_bin_post"] for x in n1], [x["frac_sameName_ne_bin_post"] for x in n2], "greater"),
        "N1_name_hold_renamedT2": describe([x["name_hold"] for x in n1 if x["pub_nonzero_t2"]]) if any(x["pub_nonzero_t2"] for x in n1) else None,
        "N2_name_hold_renamedT2": describe([x["name_hold"] for x in n2 if x["pub_nonzero_t2"]]) if any(x["pub_nonzero_t2"] for x in n2) else None,
        "S_standing_before_stop": int(sum(x["standing_before_stop"] for x in s)),
        "S_still_standing_end_of_stop_among_standing": int(sum(x["standing_end_of_stop"] for x in s if x["standing_before_stop"])),
        "S_renamed_during_stop": int(sum(x["renamed_during_stop"] for x in s)),
        "S_nonzero_name_at_stop": int(sum(x["pub_content_stop"] not in (0, -1) for x in s)),
        "S_renamed_during_stop_among_nonzero_at_stop": int(sum(x["renamed_during_stop"] for x in s if x["pub_content_stop"] not in (0, -1))),
        "S_back_to_original_among_renamed": int(sum(x["back_to_original_after_restore"] for x in s if x["renamed_during_stop"])),
        "N1_standing_before_600": int(sum(x["dL_after"] > 0 for x in n1)),
    }
    return out


def kappa_star(cells, p):
    for k in KS:
        c = cells[f"{p}|{k}"]
        fc = c["N1_frac_censored_renamed"]
        if fc is not None and fc >= 0.5:
            return k
    return None


def main():
    n0 = n0_ref()
    ev = {}
    S = load("pk")
    if S:
        cells = {f"{p}|{k}": pk_cell(S, n0, p, k) for p in PS for k in KS}
        ev["pk"] = cells
        ev["kappa_star_by_p"] = {str(p): kappa_star(cells, p) for p in PS}
        figs_pk(cells)
    for name in ("dw", "N", "ring"):
        S2 = load(name)
        if not S2:
            continue
        groups = {}
        for g, runs in S2.items():
            cond, rest = g.split("|", 1)
            groups.setdefault(rest, {})[cond] = runs
        ev[name] = {rest: evaluate(conds, T=T, t2=T2) for rest, conds in groups.items()}
    with open(f"{OUT}/evaluation.json", "w") as f:
        json.dump(ev, f, ensure_ascii=False, indent=1, default=float)
    return ev


def figs_pk(cells):
    def grid(fn):
        return np.array([[np.nan if fn(cells[f"{p}|{k}"]) is None else fn(cells[f"{p}|{k}"]) for p in PS] for k in KS], float)

    panels = [
        ("(a) N1 名前が替わった走りの持続時間（平均）", grid(lambda c: c["N1_persist_renamed"]["mean"] if c["N1_persist_renamed"] else None), "Reds", "{:.0f}"),
        ("(b) N1 名前が替わった走りのうち T まで残った割合", grid(lambda c: c["N1_frac_censored_renamed"]), "Purples", "{:.2f}"),
        ("(c) N1 継ぎ目中に名前が 0 以外へ替わった割合", grid(lambda c: c["N1_frac_renamed"]), "Blues", "{:.2f}"),
        ("(d) M の持続時間（平均）", grid(lambda c: c["M_persist"]["mean"]), "Greens", "{:.0f}"),
        ("(e) 印の時間の割合 N1（同じ名前・刻み違い、平均。e と f は同じ色の尺度）", grid(lambda c: c["R7_N1_sameName_ne_bin"]["mean"]), "Oranges", "{:.3f}"),
        ("(f) 印の時間の割合 N2（同じ名前・刻み違い、平均）", grid(lambda c: c["R7_N2_sameName_ne_bin"]["mean"]), "Oranges", "{:.3f}"),
    ]
    fig, ax = plt.subplots(2, 3, figsize=(12.5, 6.4), gridspec_kw={"hspace": 0.5, "wspace": 0.35})
    r7max = float(np.nanmax(np.concatenate([panels[4][1].ravel(), panels[5][1].ravel()])))
    for pi, (a, (ttl, Z, cm, fmt)) in enumerate(zip(ax.ravel(), panels)):
        vmin, vmax = (0.0, r7max) if pi in (4, 5) else (float(np.nanmin(Z)), float(np.nanmax(Z)))
        im = a.imshow(Z, aspect="auto", origin="lower", cmap=cm, vmin=vmin, vmax=vmax)
        a.set_xticks(range(len(PS)), [str(p) for p in PS])
        a.set_yticks(range(len(KS)), [str(k) for k in KS])
        a.set_xlabel("p")
        a.set_ylabel("κ")
        a.set_title(ttl, fontsize=8.3)
        for i in range(Z.shape[0]):
            for j in range(Z.shape[1]):
                if not np.isnan(Z[i, j]):
                    dark = (Z[i, j] - vmin) / (vmax - vmin + 1e-12) > 0.6
                    a.text(j, i, fmt.format(Z[i, j]), ha="center", va="center", fontsize=7, color="white" if dark else INK2)
        fig.colorbar(im, ax=a, shrink=0.8)
    fig.savefig(f"{OUT}/fig_pk_heatmaps.png", bbox_inches="tight")
    plt.close(fig)

    # κ* と p：T まで残った割合を κ ごとに（p ごとの線）
    fig, ax = plt.subplots(1, 2, figsize=(9.0, 3.2), gridspec_kw={"wspace": 0.3})
    cols = plt.cm.viridis(np.linspace(0.1, 0.85, len(PS)))
    for ci, p in enumerate(PS):
        fc = [cells[f"{p}|{k}"]["N1_frac_censored_renamed"] for k in KS]
        ax[0].plot(KS, [np.nan if v is None else v for v in fc], "o-", ms=3.5, color=cols[ci], label=f"p={p}")
        md = [cells[f"{p}|{k}"]["N1_persist_renamed"]["mean"] if cells[f"{p}|{k}"]["N1_persist_renamed"] else np.nan for k in KS]
        ax[1].plot(KS, md, "o-", ms=3.5, color=cols[ci], label=f"p={p}")
    ax[0].axhline(0.5, color=MUTED, lw=0.7, ls=":")
    ax[0].set_ylim(-0.03, 1.03)
    ax[0].set_xlabel("κ")
    ax[0].set_ylabel("割合")
    ax[0].set_title("(g) N1 名前が替わった走りのうち T まで残った割合\n（κ* = 0.5 以上になる最小の κ。どの p でもない）", fontsize=8)
    ax[0].legend(fontsize=7, loc="upper right")
    ax[1].set_yscale("log")
    ax[1].set_xlabel("κ")
    ax[1].set_ylabel("持続時間の平均")
    ax[1].set_title("(h) N1 名前が替わった走りの持続時間（平均）", fontsize=8.5)
    ax[1].legend(fontsize=7)
    fig.savefig(f"{OUT}/fig_kappa_star.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    ev = main()
    print(json.dumps({k: v for k, v in ev.items() if k == "kappa_star_by_p"}, ensure_ascii=False))
