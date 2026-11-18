"""v3 段 3（名前あり）：条件 N0・N1・N2・M・S を既定値で回す（種 0〜29）。

実行：.venv/bin/python -m v3.stage3
"""
from __future__ import annotations

import json
import os
from dataclasses import replace

import numpy as np
from scipy import stats

from .analysis import describe, summarize_batch, true_bin
from .model import Config
from .plotstyle import COND_COL, COND_NAME, INK2, MUTED, plt, seam
from .runner import run_many, save_batch

OUT = "results/v3/stage3"
SEEDS = list(range(30))
CONDS = ["N0", "N1", "N2", "M", "S"]


def mwu(a, b, alt="two-sided"):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.size == 0 or b.size == 0:
        return {"U": None, "p": None, "n": [int(a.size), int(b.size)]}
    r = stats.mannwhitneyu(a, b, alternative=alt)
    return {"U": float(r.statistic), "p": float(r.pvalue), "n": [int(a.size), int(b.size)], "alt": alt}


def wsr(a, b, alt="two-sided"):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    if np.all(d == 0):
        return {"p": 1.0, "n": int(d.size)}
    r = stats.wilcoxon(a, b, alternative=alt, zero_method="zsplit")
    return {"W": float(r.statistic), "p": float(r.pvalue), "n": int(d.size), "alt": alt}


def run_conditions(base: Config, conds, seeds, outdir, tag=""):
    allst, allsum = {}, {}
    for cond in conds:
        cfgs = [replace(base, cond=cond, seed=s) for s in seeds]
        res = run_many(cfgs)
        st = save_batch(outdir, f"{cond}{tag}", cfgs, res)
        allst[cond] = st
        allsum[cond] = summarize_batch(st, {"configs": [c.to_dict() for c in cfgs]})
    return allst, allsum


def evaluate(allsum, T=1500.0, t2=260.0):
    """R1〜R7 に使う比べ方を一か所で計算する。"""
    ev = {}
    S = allsum

    def col(cond, key, sel=None):
        return [x[key] for x in S[cond] if (sel is None or sel(x))]

    # R1
    ev["R1"] = {c: {"all_reach_0.9_before_100": bool(all(x["t_acc90"] < 100 for x in S[c])),
                    "t_acc90_max": float(max(x["t_acc90"] for x in S[c])),
                    "acc_first100": describe(col(c, "acc_first100")),
                    "acc_after100": describe(col(c, "acc_after100")),
                    "acc_seam": describe(col(c, "acc_seam"))} for c in S}
    # 持続時間の分布
    ev["persist"] = {c: {**describe(col(c, "persist")), "frac_censored": float(np.mean(col(c, "censored")))} for c in S}
    ev["persist_sustained"] = {c: {**describe(col(c, "persist_sustained")),
                                   "frac_censored": float(np.mean(col(c, "censored_sustained")))} for c in S}
    ev["stood"] = {c: float(np.mean(col(c, "stood"))) for c in S}
    ev["sank"] = {c: float(np.mean(col(c, "sank"))) for c in S}
    ev["frac_pos_post"] = {c: describe(col(c, "frac_pos_post")) for c in S}
    ev["absphi_late"] = {c: describe(col(c, "absphi_late")) for c in S}
    # R2
    if "N0" in S:
        ev["R2"] = {"stood": ev["stood"]["N0"], "sank": ev["sank"]["N0"],
                    "holds": bool(ev["stood"]["N0"] == 1.0 and ev["sank"]["N0"] >= 0.9)}
    # R3：N1 で継ぎ目のあいだに公の名前が 0 以外に替わった走り vs N0
    if "N1" in S and "N0" in S:
        ren = [x["renamed_in_seam"] for x in S["N1"]]
        n1r = [x["persist"] for x in S["N1"] if x["renamed_in_seam"]]
        n1n = [x["persist"] for x in S["N1"] if not x["renamed_in_seam"]]
        n0 = col("N0", "persist")
        # 同じ種どうしの対（N1 の名前が替わった走りだけ）
        pairs = [(a["persist"], b["persist"]) for a, b in zip(S["N1"], S["N0"]) if a["renamed_in_seam"]]
        ev["R3"] = {
            "frac_renamed_in_seam": float(np.mean(ren)),
            "frac_pub_nonzero_at_t2": float(np.mean([x["pub_nonzero_t2"] for x in S["N1"]])),
            "persist_N1_renamed": {**describe(n1r), "frac_censored": float(np.mean([p >= T - t2 for p in n1r])) if n1r else None} if n1r else None,
            "persist_N1_not_renamed": describe(n1n) if n1n else None,
            "persist_N0": describe(n0),
            "mwu_N1renamed_gt_N0": mwu(n1r, n0, "greater"),
            "wilcoxon_paired_N1renamed_gt_N0": wsr([a for a, b in pairs], [b for a, b in pairs], "greater") if pairs else None,
            "persist_sustained_N1_renamed": describe([x["persist_sustained"] for x in S["N1"] if x["renamed_in_seam"]]) if n1r else None,
            "persist_sustained_N0": describe(col("N0", "persist_sustained")),
            "mwu_sustained_N1renamed_gt_N0": mwu([x["persist_sustained"] for x in S["N1"] if x["renamed_in_seam"]],
                                                 col("N0", "persist_sustained"), "greater"),
        }
    # R4：N2 < N1、R5：M < N1
    for lab, c in (("R4", "N2"), ("R5", "M")):
        if c in S and "N1" in S:
            ev[lab] = {
                f"persist_{c}": describe(col(c, "persist")),
                "persist_N1": describe(col("N1", "persist")),
                f"mwu_{c}_lt_N1": mwu(col(c, "persist"), col("N1", "persist"), "less"),
                f"wilcoxon_paired_{c}_lt_N1": wsr(col(c, "persist"), col("N1", "persist"), "less"),
                f"mwu_sustained_{c}_lt_N1": mwu(col(c, "persist_sustained"), col("N1", "persist_sustained"), "less"),
                f"sank_{c}": float(np.mean(col(c, "sank"))),
            }
            if c == "N2":
                ren2 = [x["persist"] for x in S["N2"] if x["renamed_in_seam"]]
                ren1 = [x["persist"] for x in S["N1"] if x["renamed_in_seam"]]
                ev[lab]["persist_N2_renamed"] = describe(ren2) if ren2 else None
                ev[lab]["mwu_renamed_N2_lt_N1"] = mwu(ren2, ren1, "less")
                ev[lab]["frac_renamed_in_seam_N2"] = float(np.mean([x["renamed_in_seam"] for x in S["N2"]]))
            if c == "M":
                ev[lab]["absmem_t2"] = describe(col("M", "absmem_t2"))
                ev[lab]["absmem_late"] = describe(col("M", "absmem_late"))
    # R6：S
    if "S" in S:
        s = S["S"]
        ev["R6"] = {
            "frac_standing_before_stop": float(np.mean([x["standing_before_stop"] for x in s])),
            "frac_standing_end_of_stop": float(np.mean([x["standing_end_of_stop"] for x in s])),
            "frac_renamed_during_stop": float(np.mean([x["renamed_during_stop"] for x in s])),
            "among_standing_before_stop": {
                "n": int(sum(x["standing_before_stop"] for x in s)),
                "frac_still_standing_end_of_stop": float(np.mean([x["standing_end_of_stop"] for x in s if x["standing_before_stop"]]))
                if any(x["standing_before_stop"] for x in s) else None,
            },
            "among_renamed_during_stop": {
                "n": int(sum(x["renamed_during_stop"] for x in s)),
                "frac_back_to_original_after_restore": float(np.mean([x["back_to_original_after_restore"] for x in s if x["renamed_during_stop"]]))
                if any(x["renamed_during_stop"] for x in s) else None,
            },
            "pub_content_stop_restore_end": [(x["pub_content_stop"], x["pub_content_restore"], x["pub_content_end"]) for x in s],
        }
    # R7：記述から印へ
    ev["R7"] = {}
    for c in ("N1", "N2", "S", "M"):
        if c in S:
            d = {"frac_name_ne_bin_post": describe(col(c, "frac_name_ne_bin_post"))}
            if c != "M":
                d["frac_sameName_ne_bin_post"] = describe(col(c, "frac_sameName_ne_bin_post"))
                d["name_hold"] = describe(col(c, "name_hold"))
                d["frac_name_hold_censored"] = float(np.mean(col(c, "name_hold_censored")))
                d["n_public_change"] = describe(col(c, "n_public_change"))
            ev["R7"][c] = d
    if "N1" in S and "N2" in S:
        ev["R7"]["mwu_N1_gt_N2_frac_sameName_ne_bin"] = mwu(col("N1", "frac_sameName_ne_bin_post"),
                                                           col("N2", "frac_sameName_ne_bin_post"), "greater")
    return ev


def figures(allst, allsum, outdir, title_suffix=""):
    conds = list(allst)
    t = allst[conds[0]]["t"][0].astype(float)
    fig, ax = plt.subplots(len(conds), 1, figsize=(7.4, 1.7 * len(conds) + 0.6), sharex=True,
                           gridspec_kw={"hspace": 0.45})
    for a, c in zip(np.atleast_1d(ax), conds):
        dl = allst[c]["dL"][:, :, 0].astype(float)
        q = np.percentile(dl, [25, 50, 75], axis=0)
        a.fill_between(t, q[0], q[2], color=COND_COL[c], alpha=0.2, lw=0)
        a.plot(t, q[1], color=COND_COL[c], lw=1.0)
        a.plot(t, dl.mean(0), color=COND_COL[c], lw=0.6, ls="--")
        a.axhline(0, color=MUTED, lw=0.6)
        seam(a, 200, 260, label=False)
        if c == "S":
            a.axvspan(600, 1000, color="#f3e3c3", lw=0, zorder=0)
        a.set_ylim(-3, 12)
        a.set_ylabel("ΔL_self")
        a.set_title(f"{COND_NAME[c]}：中央値（実線）・平均（破線）・四分位の帯{title_suffix}")
    np.atleast_1d(ax)[-1].set_xlabel("時間")
    fig.savefig(f"{outdir}/fig_dL_by_condition{title_suffix and '_'}.png", bbox_inches="tight")
    plt.close(fig)

    # 持続時間
    fig, ax = plt.subplots(1, 2, figsize=(8.4, 3.2), gridspec_kw={"wspace": 0.3})
    rng = np.random.default_rng(0)
    for i, c in enumerate(conds):
        p = np.array([x["persist"] for x in allsum[c]])
        ren = np.array([x.get("renamed_in_seam", False) for x in allsum[c]])
        xj = i + rng.uniform(-0.18, 0.18, p.size)
        ax[0].scatter(xj[~ren], p[~ren] + 1, s=10, color=COND_COL[c], alpha=0.5, lw=0)
        ax[0].scatter(xj[ren], p[ren] + 1, s=14, facecolor="none", edgecolor=COND_COL[c], lw=0.9)
        ax[0].hlines(np.median(p) + 1, i - 0.3, i + 0.3, color=INK2, lw=1.2)
        ps = np.array([x["persist_sustained"] for x in allsum[c]])
        ax[1].scatter(xj, ps + 1, s=10, color=COND_COL[c], alpha=0.6, lw=0)
        ax[1].hlines(np.median(ps) + 1, i - 0.3, i + 0.3, color=INK2, lw=1.2)
    for a, ttl in zip(ax, ("(a) 持続時間＋1（最初に ΔL≦0）\n○＝継ぎ目中に公の名前が 0 以外へ替わった走り",
                           "(b) 持続時間＋1（ΔL≦0 が 20 続くまで）\n横線＝中央値、点線＝打ち切り（T−t2）")):
        a.set_yscale("log")
        a.set_xticks(range(len(conds)), conds)
        a.axhline(1240 + 1, color=MUTED, lw=0.6, ls=":")
        a.set_title(ttl, fontsize=8.5)
    ax[0].set_ylabel("t2 からの時間（対数）")
    fig.savefig(f"{outdir}/fig_persistence.png", bbox_inches="tight")
    plt.close(fig)

    # 公の名前と本人の位置
    namec = [c for c in conds if c in ("N1", "N2", "S")]
    fig, ax = plt.subplots(3, 1, figsize=(7.4, 7.0), sharex=True, gridspec_kw={"hspace": 0.4})
    for c in namec:
        pc = allst[c]["pub_content"][:, :, 0].astype(int)
        ax[0].plot(t, ((pc != 0) & (pc != -1)).mean(0), color=COND_COL[c], lw=1.0, label=COND_NAME[c])
        tb = true_bin(allst[c]["phi"][:, :, 0].astype(float))
        ax[1].plot(t, ((pc != tb) & (pc != -1)).mean(0), color=COND_COL[c], lw=1.0, label=COND_NAME[c])
    if "M" in conds:
        mp = np.abs(allst["M"]["mem_pub"][:, :, 0].astype(float))
        ax[0].plot(t, (true_bin(allst["M"]["mem_pub"][:, :, 0].astype(float)) != 0).mean(0), color=COND_COL["M"], lw=1.0,
                   label="M（記録の刻みが 0 以外）")
    for c in conds:
        ax[2].plot(t, np.median(np.abs(allst[c]["phi"][:, :, 0].astype(float)), 0), color=COND_COL[c], lw=0.9, label=c)
    ax[0].set_ylabel("割合")
    ax[0].set_title("(a) 焦点 j の公の名前の内容が 0 以外である走りの割合（N1 と S は t=600 まで同じ走り）")
    ax[1].set_ylabel("割合")
    ax[1].set_title("(b) 公の名前の内容 ≠ 本人の実際の刻み である走りの割合")
    ax[2].set_yscale("log")
    ax[2].set_ylabel("|φ_j|（中央値）")
    ax[2].set_title("(c) 本人の実際のずれ |φ_j|")
    ax[2].set_xlabel("時間")
    for a in ax:
        seam(a, 200, 260, label=False)
        a.legend(fontsize=7, loc="upper right")
    if "S" in conds:
        for a in ax:
            a.axvspan(600, 1000, color="#f3e3c3", lw=0, zorder=0)
    fig.savefig(f"{outdir}/fig_names.png", bbox_inches="tight")
    plt.close(fig)

    if "M" in conds:
        fig, ax = plt.subplots(1, 1, figsize=(7.4, 2.4))
        mp = allst["M"]["mem_pub"][:, :, 0].astype(float)
        for s in range(min(8, mp.shape[0])):
            ax.plot(t, mp[s], lw=0.5, color=COND_COL["M"], alpha=0.6)
        ax.plot(t, np.median(np.abs(mp), 0), lw=1.2, color=INK2, label="|記録| の中央値")
        seam(ax, 200, 260, label=False)
        ax.set_ylabel("j の公共の記録（角度）")
        ax.set_xlabel("時間")
        ax.set_title("M：j についての連続の記録（他の 3 人の円平均）")
        ax.legend(fontsize=7)
        fig.savefig(f"{outdir}/fig_memory.png", bbox_inches="tight")
        plt.close(fig)


def load_all(outdir, conds, tag=""):
    from .runner import load_batch
    allst = {}
    for c in conds:
        st, _ = load_batch(outdir, f"{c}{tag}")
        allst[c] = st
    with open(f"{outdir}/summary.json") as f:
        allsum = json.load(f)["per_run"]
    return allst, allsum


def main():
    import sys
    if "figs" in sys.argv:
        allst, allsum = load_all(OUT, CONDS)
        figures(allst, allsum, OUT)
        return
    os.makedirs(OUT, exist_ok=True)
    base = Config()
    allst, allsum = run_conditions(base, CONDS, SEEDS, OUT)
    ev = evaluate(allsum)
    with open(f"{OUT}/summary.json", "w") as f:
        json.dump({"evaluation": ev, "per_run": allsum, "base_config": base.to_dict(), "seeds": SEEDS},
                  f, ensure_ascii=False, indent=1, default=float)
    figures(allst, allsum, OUT)
    print(json.dumps(ev, ensure_ascii=False, indent=1, default=float))


if __name__ == "__main__":
    main()
