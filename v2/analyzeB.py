"""v2 段階 B の分析：表、Q1〜Q6、追加の外れの判定、図 (a)〜(e)。

実行：.venv/bin/python -m v2.analyzeB
"""
from __future__ import annotations

import json
import os

import numpy as np
from scipy import stats

from v3.plotstyle import INK2, MUTED, plt, seam

from .model import wrap

OUT = "results/v2/stageB"
CONDS = ["none", "D3", "D0", "C3", "C0", "C3n"]
PS = [0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
SEAMS = [5.0, 10.0, 30.0, 60.0]
GS = [0.5, 1.0, 1.5, 2.5]
OPEN = 0.05
COL = {"none": "#52514e", "D3": "#d1495b", "D0": "#2a78d6", "C3": "#1baf7a", "C0": "#8fd3b6", "C3n": "#e8a33d"}
NAME = {"none": "名前なし", "D3": "D3 離散・噂あり", "D0": "D0 離散・噂なし", "C3": "C3 連続・噂あり",
        "C0": "C0 連続・噂なし", "C3n": "C3n 連続・近いときだけ"}


def key(cond, p=0.1, seam=30.0, g=1.5, N=4, sobs=0.1, T=1000.0):
    return f"{cond}|p={p}|seam={seam}|g={g}|N={N}|sobs={sobs}|T={T}"


def load(name):
    with open(f"{OUT}/{name}.json") as f:
        return json.load(f)["summaries"]


def agent_arrays(s):
    return {k: np.asarray(v, float) for k, v in s.items() if k not in ("cfg",)}


def cell(s):
    a = agent_arrays(s)
    Gj = a["Gj_end"]
    return {"dG_mean": float(np.mean(a["dG"])), "dG_sd": float(np.std(a["dG"])),
            "G_end_mean": float(np.mean(a["G_end"])),
            "frac_agents_open": float((Gj > OPEN).mean()),
            "frac_runs_open": float((np.asarray(a["dG"]) > OPEN).mean()),
            "frac_renamed_t2": float((a["name_t2"] != 0).mean()),
            "frac_open_among_renamed": float((Gj[a["name_t2"] != 0] > OPEN).mean()) if (a["name_t2"] != 0).any() else None,
            "n_renamed": int((a["name_t2"] != 0).sum()),
            "frac_open_among_not_renamed": float((Gj[a["name_t2"] == 0] > OPEN).mean()) if (a["name_t2"] == 0).any() else None}


def pstar(ps, fr):
    """開いたままの割合が 0.5 を横切る p（log p で線形補間）。横切らなければ None。"""
    ps = np.asarray(ps, float)
    fr = np.asarray([np.nan if f is None else f for f in fr], float)
    ok = ~np.isnan(fr)
    ps, fr = ps[ok], fr[ok]
    for i in range(len(ps) - 1):
        if fr[i] >= 0.5 > fr[i + 1]:
            x0, x1 = np.log(ps[i]), np.log(ps[i + 1])
            return float(np.exp(x0 + (fr[i] - 0.5) / (fr[i] - fr[i + 1]) * (x1 - x0)))
    if len(fr) and fr[0] < 0.5:
        return "<=" + str(ps[0])
    if len(fr) and fr[-1] >= 0.5:
        return ">" + str(ps[-1])
    return None


def main():
    main_ = load("main")
    pg = load("pg")
    rob = load("robust")
    lng = load("long")
    ev = {}

    # ---- 表：条件 × p（継ぎ目 30）
    table = {}
    for cond in CONDS:
        if cond == "none":
            table[cond] = {"any": cell(main_[key("none")])}
        else:
            table[cond] = {str(p): cell(main_[key(cond, p)]) for p in PS}
    ev["table_seam30"] = table
    # 継ぎ目の長さごと
    ev["by_seam"] = {str(sm): {cond: ({"any": cell(main_[key("none", seam=sm)])} if cond == "none" else
                                     {str(p): cell(main_[key(cond, p, sm)]) for p in PS}) for cond in CONDS} for sm in SEAMS}

    # ---- Q1：C0・C3・D0 は、どの p でも閉じる（T=1000 で開いていれば T=3000 を見る）
    q1 = {}
    for cond in ("C0", "C3", "D0"):
        rows = {}
        for sm in SEAMS:
            for p in PS:
                c = cell(main_[key(cond, p, sm)])
                rows[f"seam={sm}|p={p}"] = {"frac_agents_open": c["frac_agents_open"], "dG_mean": c["dG_mean"]}
        long_rows = {str(p): cell(lng[key(cond, p, T=3000.0)]) for p in PS}
        q1[cond] = {"max_frac_agents_open_T1000": max(v["frac_agents_open"] for v in rows.values()),
                    "max_dG_mean_T1000": max(v["dG_mean"] for v in rows.values()),
                    "seam30_T3000_frac_open": {p: v["frac_agents_open"] for p, v in long_rows.items()},
                    "seam30_T3000_dG": {p: v["dG_mean"] for p, v in long_rows.items()},
                    "cells": rows}
    q1["holds"] = bool(all(q1[c]["max_frac_agents_open_T1000"] <= 0.05 or max(q1[c]["seam30_T3000_frac_open"].values()) <= 0.05
                           for c in ("C0", "C3", "D0")))
    ev["Q1"] = q1

    # ---- Q2：D3、長い継ぎ目、付け直された主体：p が小さいと開いたまま、大きいと閉じる、境 p* がある
    q2 = {}
    for sm in (30.0, 60.0):
        fr = [cell(main_[key("D3", p, sm)])["frac_open_among_renamed"] for p in PS]
        nr = [cell(main_[key("D3", p, sm)])["n_renamed"] for p in PS]
        frl = [cell(lng[key("D3", p, T=3000.0)])["frac_open_among_renamed"] for p in PS] if sm == 30.0 else None
        q2[str(sm)] = {"frac_open_among_renamed_by_p": dict(zip(map(str, PS), fr)), "n_renamed_by_p": dict(zip(map(str, PS), nr)),
                       "pstar": pstar(PS, fr),
                       "T3000_frac_open_among_renamed_by_p": dict(zip(map(str, PS), frl)) if frl else None,
                       "T3000_pstar": pstar(PS, frl) if frl else None}
    ev["Q2"] = q2

    # ---- Q3：D3、短い継ぎ目、付け直されなかった主体は、名前なしより速く確実に閉じる（同じ走り・同じ主体で対にする）
    #   閉じる時刻は二通り：t_close_final（閉じきった時刻、主）と t_close（最初に < 0.05、走らせる前の定め）
    q3 = {}
    for which in ("t_close_final", "t_close"):
        q3w = {}
        for sm in (5.0, 10.0):
            none = agent_arrays(main_[key("none", seam=sm)])
            rows = {}
            pooled_d, pooled_n = [], []
            for p in PS:
                d = agent_arrays(main_[key("D3", p, sm)])
                sel = d["name_t2"] == 0
                tcd = d[which][sel]
                tcn = none[which][sel]
                big = 1e6
                tcd_f = np.where(np.isnan(tcd), big, tcd)
                tcn_f = np.where(np.isnan(tcn), big, tcn)
                pooled_d += tcd_f.tolist()
                pooled_n += tcn_f.tolist()
                diff = tcd_f - tcn_f
                rows[str(p)] = {"n": int(sel.sum()), "D3_median": float(np.median(tcd_f)), "none_median": float(np.median(tcn_f)),
                                "frac_closed_D3": float(np.mean(~np.isnan(tcd))), "frac_closed_none": float(np.mean(~np.isnan(tcn))),
                                "frac_D3_faster": float((diff < 0).mean()), "frac_tie": float((diff == 0).mean()),
                                "Gj_end_D3_mean": float(d["Gj_end"][sel].mean()), "Gj_end_none_mean": float(none["Gj_end"][sel].mean())}
            pd_, pn_ = np.asarray(pooled_d), np.asarray(pooled_n)
            w = stats.wilcoxon(pd_, pn_, alternative="less", zero_method="zsplit") if np.any(pd_ != pn_) else None
            q3w[str(sm)] = {"by_p": rows, "pooled_wilcoxon_D3_faster_p": float(w.pvalue) if w else None,
                            "pooled_frac_D3_faster": float((pd_ < pn_).mean()), "pooled_frac_D3_slower": float((pd_ > pn_).mean()),
                            "pooled_median_D3": float(np.median(pd_)), "pooled_median_none": float(np.median(pn_))}
        q3[which] = q3w
    ev["Q3"] = q3

    # ---- Q4：D3 の最後のギャップ（分けない）は中くらいの p で最大
    q4 = {}
    for sm in SEAMS:
        dg = [cell(main_[key("D3", p, sm)])["dG_mean"] for p in PS]
        i = int(np.argmax(dg))
        spread = float(np.max(dg) - np.min(dg))
        q4[str(sm)] = {"dG_by_p": dict(zip(map(str, PS), dg)), "argmax_p": PS[i], "range": spread,
                       "interior_max": bool(0 < i < len(PS) - 1 and spread > 0.005)}  # 差が 0.005 rad 以下なら「最大なし」
    ev["Q4"] = q4

    # ---- Q5：p* は g が大きいほど大きい
    q5 = {}
    for g in GS:
        src = main_ if g == 1.5 else pg
        fr = [cell(src[key("D3", p, g=g)])["frac_open_among_renamed"] for p in PS]
        q5[str(g)] = {"frac_open_among_renamed_by_p": dict(zip(map(str, PS), fr)), "pstar": pstar(PS, fr),
                      "dG_by_p": dict(zip(map(str, PS), [cell(src[key("D3", p, g=g)])["dG_mean"] for p in PS]))}
    ev["Q5"] = q5

    # ---- Q6：名前が残った主体（継ぎ目の終わりに 0 以外で、最後も同じ名前）の残り方を g ごとに
    q6 = {}
    for g in GS:
        src = main_ if g == 1.5 else pg
        rec = {"n": 0, "absPsi_pre": [], "absPsi_end": [], "mpsi_end": [], "Gj_end": [], "r_pre": [], "r_end": []}
        for p in PS:
            a = agent_arrays(src[key("D3", p, g=g)])
            sel = (a["name_t2"] != 0) & (a["name_end"] == a["name_t2"])
            rec["n"] += int(sel.sum())
            for k in ("absPsi_pre", "absPsi_end", "mpsi_end", "Gj_end"):
                rec[k] += a[k][sel].tolist()
            runs = np.nonzero(sel.any(1))[0]
            rec["r_pre"] += a["r_pre"][runs].tolist()
            rec["r_end"] += a["r_end"][runs].tolist()
        q6[str(g)] = {k: (float(np.median(v)) if isinstance(v, list) and v else v) for k, v in rec.items()}
        q6[str(g)]["raw"] = {k: v for k, v in rec.items() if isinstance(v, list)}
    ev["Q6"] = q6

    # ---- 追加の外れの判定
    ex = {}
    ex["D3_vs_D0_seam30"] = {str(p): {"D3_dG": cell(main_[key("D3", p)])["dG_mean"], "D0_dG": cell(main_[key("D0", p)])["dG_mean"],
                                      "D3_open": cell(main_[key("D3", p)])["frac_agents_open"], "D0_open": cell(main_[key("D0", p)])["frac_agents_open"]}
                             for p in PS}
    ex["C3_vs_D3_seam30"] = {str(p): {"C3_open": cell(main_[key("C3", p)])["frac_agents_open"], "D3_open": cell(main_[key("D3", p)])["frac_agents_open"]}
                             for p in PS}
    ex["C3n_vs_D3_seam30"] = {str(p): {"C3n_open": cell(main_[key("C3n", p)])["frac_agents_open"], "D3_open": cell(main_[key("D3", p)])["frac_agents_open"],
                                       "C3n_dG": cell(main_[key("C3n", p)])["dG_mean"], "D3_dG": cell(main_[key("D3", p)])["dG_mean"]}
                              for p in PS}
    ex["C3n_vs_D3_seam30_T3000"] = {str(p): {"C3n_open": cell(lng[key("C3n", p, T=3000.0)])["frac_agents_open"],
                                             "D3_open": cell(lng[key("D3", p, T=3000.0)])["frac_agents_open"]} for p in PS}
    ex["pstar_by_g"] = {g: q5[g]["pstar"] for g in q5}
    ev["extra_falsification"] = ex

    # ---- 記述（走らせたあとに足した量。予想の判定には使わない）：開いている時間の長さ
    #   付け直された主体（継ぎ目の終わりの名前 ≠ 0）について、t2 から G_j < 0.05 になるまでの時間。
    #   閉じなければ T − t2 で打ち切り。T=1000 と T=3000 の両方で。
    desc = {}
    for lab, src, T in (("T1000", main_, 1000.0), ("T3000", lng, 3000.0)):
        d = {}
        for cond in CONDS[1:]:
            row = {}
            for p in PS:
                a = agent_arrays(src[key(cond, p, T=T)])
                sel = a["name_t2"] != 0
                tc = a["t_close_final"][sel]
                cens = np.isnan(tc)
                tcf = np.where(cens, T - 90.0, tc)
                row[str(p)] = {"n_renamed": int(sel.sum()), "t_close_median": float(np.median(tcf)) if tcf.size else None,
                               "t_close_mean": float(np.mean(tcf)) if tcf.size else None,
                               "frac_censored": float(cens.mean()) if tcf.size else None}
            d[cond] = row
        none = agent_arrays(src[key("none", T=T)]) if T == 1000.0 else None
        if none is not None:
            d["none_all_agents_t_close_final_median"] = float(np.nanmedian(none["t_close_final"]))
        desc[lab] = d
    ev["descriptive_t_close_renamed"] = desc

    # ---- 頑健さ
    rb = {}
    for lab, kw in (("default", {}), ("N6", dict(N=6)), ("N8", dict(N=8)), ("sobs0", dict(sobs=0.0)), ("sobs0.2", dict(sobs=0.2))):
        src = main_ if lab == "default" else rob
        rb[lab] = {cond: ({"any": cell(src[key("none", **kw)])} if cond == "none" else
                          {str(p): cell(src[key(cond, p, **kw)]) for p in PS}) for cond in CONDS}
    ev["robust"] = rb
    if os.path.exists(f"{OUT}/long_sobs02.json"):
        ls = load("long_sobs02")
        ev["robust_sobs0.2_T3000"] = {cond: ({"any": cell(ls[key("none", sobs=0.2, T=3000.0)])} if cond == "none" else
                                             {str(p): cell(ls[key(cond, p, sobs=0.2, T=3000.0)]) for p in PS}) for cond in CONDS}
    with open(f"{OUT}/evaluation.json", "w") as f:
        json.dump(ev, f, ensure_ascii=False, indent=1, default=str)

    figures(main_, pg, rob, lng, ev)
    return ev


def figures(main_, pg, rob, lng, ev):
    # (a) 既定値での G(t)、r(t)、公の名前
    fig, ax = plt.subplots(3, 1, figsize=(7.4, 7.2), sharex=True, gridspec_kw={"hspace": 0.4})
    for cond in CONDS:
        fn = f"{OUT}/default_full_{cond}.npz"
        if not os.path.exists(fn):
            continue
        o = np.load(fn)
        t = o["t"]
        G = np.abs(wrap(o["hat"].astype(float) - o["psi"].astype(float))).mean(2)
        ax[0].plot(t, np.median(G, 1), color=COL[cond], lw=1.0, label=NAME[cond])
        ax[1].plot(t, np.median(o["r"], 1), color=COL[cond], lw=1.0, label=NAME[cond])
        if cond in ("D3", "D0"):
            nz = (o["name"] != 0).mean((1, 2))
        elif cond == "none":
            continue
        else:
            nz = (np.rint(wrap(o["m"].astype(float)) / (np.pi / 4)).astype(int) % 8 != 0).mean((1, 2))
        ax[2].plot(t, nz, color=COL[cond], lw=1.0, label=NAME[cond])
    ax[0].set_yscale("log")
    ax[0].set_ylabel("G（中央値）")
    ax[0].set_title("(a1) ギャップ G(t)（p=0.1、継ぎ目 30、g=1.5、30 走り）")
    ax[1].set_ylabel("r（中央値）")
    ax[1].set_title("(a2) 同期の度合い r(t)")
    ax[2].set_ylabel("割合")
    ax[2].set_title("(a3) 公の名前（連続では公の記録を刻んだもの）が 0 以外の主体の割合")
    ax[2].set_xlabel("時間")
    for a in ax:
        seam(a, 60, 90, label=False)
        a.legend(fontsize=7, loc="upper right", ncol=2)
    ax[2].set_xlim(0, 1000)
    fig.savefig(f"{OUT}/fig_a_default.png", bbox_inches="tight")
    plt.close(fig)

    # (b) D3 の最後のギャップと、付け直された主体が閉じきるまでの時間：p × 継ぎ目の長さ
    fig, ax = plt.subplots(1, 2, figsize=(8.8, 3.2), gridspec_kw={"wspace": 0.35})
    M = np.array([[cell(main_[key("D3", p, sm)])["dG_mean"] for p in PS] for sm in SEAMS])
    TC = np.full((len(SEAMS), len(PS)), np.nan)
    NR = np.zeros((len(SEAMS), len(PS)), int)
    for i, sm in enumerate(SEAMS):
        for j, p in enumerate(PS):
            a_ = agent_arrays(main_[key("D3", p, sm)])
            sel = a_["name_t2"] != 0
            NR[i, j] = int(sel.sum())
            if sel.any():
                tc = a_["t_close_final"][sel]
                TC[i, j] = np.median(np.where(np.isnan(tc), 1000.0 - 60 - sm, tc))
    for a, Z, ttl, cm, vr, fmt in ((ax[0], M, "(b1) D3 の最後のギャップ ΔG（30 走りの平均、rad）", "Reds", (0, 0.2), "{:.3f}"),
                                   (ax[1], TC, "(b2) D3 で付け直された主体が閉じきるまでの時間（中央値）\n括弧内は付け直された主体の数", "Purples", (0, 300), "{:.0f}")):
        im = a.imshow(Z, aspect="auto", origin="lower", cmap=cm, vmin=vr[0], vmax=vr[1])
        a.set_xticks(range(len(PS)), [str(p) for p in PS])
        a.set_yticks(range(len(SEAMS)), [str(int(s_)) for s_ in SEAMS])
        a.set_xlabel("p（本人を見て確かめる割合）")
        a.set_ylabel("継ぎ目の長さ")
        a.set_title(ttl, fontsize=8.3)
        for i in range(Z.shape[0]):
            for j in range(Z.shape[1]):
                if not np.isnan(Z[i, j]):
                    txt = fmt.format(abs(Z[i, j]) if Z is M else Z[i, j]) + (f"\n({NR[i, j]})" if Z is TC else "")
                    a.text(j, i, txt, ha="center", va="center", fontsize=6.8, color=INK2)
        fig.colorbar(im, ax=a, shrink=0.8)
    fig.savefig(f"{OUT}/fig_b_D3_p_x_seam.png", bbox_inches="tight")
    plt.close(fig)

    # (c) 頑健さ
    labs = ["default", "N6", "N8", "sobs0", "sobs0.2"]
    titles = {"default": "既定（N=4, σ=0.1）", "N6": "N=6", "N8": "N=8", "sobs0": "σ_obs=0", "sobs0.2": "σ_obs=0.2"}
    fig, ax = plt.subplots(1, len(labs), figsize=(12, 2.9), sharey=True, gridspec_kw={"wspace": 0.12})
    for a, lab in zip(ax, labs):
        rb = ev["robust"][lab]
        for cond in CONDS[1:]:
            a.plot(PS, [rb[cond][str(p)]["frac_agents_open"] for p in PS], "o-", ms=3, lw=1, color=COL[cond], label=cond)
        a.axhline(rb["none"]["any"]["frac_agents_open"], color=COL["none"], lw=0.8, ls=":", label="名前なし")
        a.set_xscale("log")
        a.set_title(titles[lab], fontsize=8.5)
        a.set_xlabel("p")
    ax[0].set_ylabel("開いたままの主体の割合")
    ax[0].legend(fontsize=6.5)
    fig.suptitle("(c) 頑健さ：開いたままの主体の割合（継ぎ目 30、g=1.5、T=1000）", fontsize=9, x=0.12, y=1.06, ha="left")
    fig.savefig(f"{OUT}/fig_c_robust.png", bbox_inches="tight")
    plt.close(fig)

    # (d) p* を g ごとに
    fig, ax = plt.subplots(1, 1, figsize=(5.4, 3.2))
    cols = plt.cm.viridis(np.linspace(0.1, 0.85, len(GS)))
    for ci, g in enumerate(GS):
        fr = [ev["Q5"][str(g)]["frac_open_among_renamed_by_p"][str(p)] for p in PS]
        fr = [np.nan if f is None else f for f in fr]
        ax.plot(PS, fr, "o-", ms=3.5, color=cols[ci], label=f"g={g}（p*={ev['Q5'][str(g)]['pstar']}）")
    ax.axhline(0.5, color=MUTED, lw=0.7, ls=":")
    ax.set_xscale("log")
    ax.set_xlabel("p")
    ax.set_ylabel("開いたままの割合")
    ax.set_title("(d) D3：付け直された主体のうち、最後に開いたままの割合（継ぎ目 30）", fontsize=8.5)
    ax.legend(fontsize=7)
    fig.savefig(f"{OUT}/fig_d_pstar_by_g.png", bbox_inches="tight")
    plt.close(fig)

    # (e) 名前が残った主体の |ψ_j| と |m_j − ψ_j|
    fig, ax = plt.subplots(1, 2, figsize=(8.4, 3.0), gridspec_kw={"wspace": 0.3})
    rng = np.random.default_rng(0)
    for i, g in enumerate(GS):
        raw = ev["Q6"][str(g)]["raw"]
        for a, k, ttl in ((ax[0], "absPsi_end", "|ψ_j|（最後）"), (ax[1], "mpsi_end", "|m_j − ψ_j|（最後）")):
            v = np.asarray(raw[k])
            if v.size:
                a.scatter(i + rng.uniform(-0.15, 0.15, v.size), v, s=8, color=INK2, alpha=0.5, lw=0)
                a.hlines(np.median(v), i - 0.25, i + 0.25, color="#d1495b", lw=1.4)
            a.set_title(f"(e) 名前が残った主体の{ttl}", fontsize=8.5)
            a.set_xticks(range(len(GS)), [f"g={g}\n(n={ev['Q6'][str(g)]['n']})" for g in GS])
    pre = np.concatenate([ev["Q6"][str(g)]["raw"]["absPsi_pre"] for g in GS]) if any(ev["Q6"][str(g)]["n"] for g in GS) else np.array([])
    if pre.size:
        ax[0].axhline(np.median(pre), color=MUTED, lw=0.7, ls=":", label="継ぎ目の前の |ψ_j|（中央値）")
        ax[0].legend(fontsize=7)
    fig.savefig(f"{OUT}/fig_e_Q6.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    ev = main()
    short = {k: v for k, v in ev.items() if k in ("Q1", "Q2", "Q4", "Q5", "extra_falsification")}
    for k in ("Q1",):
        for c in ("C0", "C3", "D0"):
            short[k][c].pop("cells", None)
    print(json.dumps(short, ensure_ascii=False, indent=1, default=str)[:6000])
