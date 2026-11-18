"""REPORT.md の表を evaluation.json から作る（手で写さない）。

実行：.venv/bin/python -m v2.report_tables  →  results/v2/stageB/tables.md
"""
from __future__ import annotations

import json

OUT = "results/v2/stageB"
PS = ["0.02", "0.05", "0.1", "0.2", "0.5", "1.0"]
CONDS = ["none", "D3", "D0", "C3", "C0", "C3n"]
NAME = {"none": "名前なし", "D3": "D3 離散・噂あり", "D0": "D0 離散・噂なし", "C3": "C3 連続・噂あり",
        "C0": "C0 連続・噂なし", "C3n": "C3n 連続・近いときだけ"}


def g3(x):
    return "0.000" if abs(x) < 0.0005 else f"{x:+.3f}"


def main():
    with open(f"{OUT}/evaluation.json") as f:
        ev = json.load(f)
    L = []
    t = ev["table_seam30"]
    L.append("**表 1　最後のギャップ ΔG（rad、30 走りの平均）と、開いたままの主体の割合（継ぎ目 30、g=1.5、T=1000）**\n")
    L.append("| 条件 | " + " | ".join(f"p={p}" for p in PS) + " |")
    L.append("|---|" + "---|" * len(PS))
    for c in CONDS:
        if c == "none":
            x = t[c]["any"]
            L.append(f"| {NAME[c]} | " + " | ".join([f"{g3(x['dG_mean'])}（{x['frac_agents_open']:.0%}）"] + ["〃"] * (len(PS) - 1)) + " |")
        else:
            L.append(f"| {NAME[c]} | " + " | ".join(f"{g3(t[c][p]['dG_mean'])}（{t[c][p]['frac_agents_open']:.0%}）" for p in PS) + " |")
    L.append("")
    d = ev["descriptive_t_close_renamed"]["T1000"]
    L.append("**表 2　継ぎ目の終わりに付け直された主体が、閉じきるまでの時間（中央値。括弧内は主体の数。継ぎ目 30）**\n")
    L.append("| 条件 | " + " | ".join(f"p={p}" for p in PS) + " |")
    L.append("|---|" + "---|" * len(PS))
    for c in CONDS[1:]:
        cells = []
        for p in PS:
            r = d[c][p]
            cells.append("—（0）" if r["t_close_median"] is None else f"{r['t_close_median']:.0f}（{r['n_renamed']}）")
        L.append(f"| {NAME[c]} | " + " | ".join(cells) + " |")
    L.append(f"| 名前なし（全主体） | {d['none_all_agents_t_close_final_median']:.0f} | | | | | |")
    cm = ev["descriptive_t_close_renamed"]["D3_vs_D0_t_close_final_seam30"]
    L.append("| D3 と D0 の差（両側 p） | " + " | ".join("—" if cm[p]["p_two_sided"] is None else
                                                    (f"{cm[p]['p_two_sided']:.3f}" if cm[p]["p_two_sided"] >= 0.001 else "<0.001") for p in PS) + " |")
    L.append("")
    q3 = ev["Q3"]["t_close_final"]
    L.append("**表 3　Q3：短い継ぎ目で付け直されなかった主体が閉じきるまでの時間（D3 と、同じ走り・同じ主体の名前なし。p をまとめて）**\n")
    L.append("| 継ぎ目 | D3 中央値 | 名前なし 中央値 | D3 が速い割合 | 遅い割合 | Wilcoxon 片側 p |")
    L.append("|---|---|---|---|---|---|")
    for sm in ("5.0", "10.0"):
        x = q3[sm]
        L.append(f"| {float(sm):.0f} | {x['pooled_median_D3']:.0f} | {x['pooled_median_none']:.0f} | {x['pooled_frac_D3_faster']:.0%} | "
                 f"{x['pooled_frac_D3_slower']:.0%} | {x['pooled_wilcoxon_D3_faster_p']:.1e} |")
    L.append("")
    L.append("**表 4　頑健さ：開いたままの主体の割合（継ぎ目 30、g=1.5）**\n")
    L.append("| 設定 | 条件 | " + " | ".join(f"p={p}" for p in PS) + " |")
    L.append("|---|---|" + "---|" * len(PS))
    labs = [("N6", "N=6"), ("N8", "N=8"), ("sobs0", "σ_obs=0"), ("sobs0.2", "σ_obs=0.2")]
    for lab, nm in labs:
        rb = ev["robust"][lab]
        for c in CONDS[1:]:
            L.append(f"| {nm} | {c} | " + " | ".join(f"{rb[c][p]['frac_agents_open']:.0%}" for p in PS) + " |")
    if "robust_sobs0.2_T3000" in ev:
        rb = ev["robust_sobs0.2_T3000"]
        for c in CONDS[1:]:
            L.append(f"| σ_obs=0.2、T=3000 | {c} | " + " | ".join(f"{rb[c][p]['frac_agents_open']:.0%}" for p in PS) + " |")
    L.append("")
    q5 = ev["Q5"]
    L.append("**表 5　Q5：付け直された主体のうち、最後に開いたままの割合（D3、継ぎ目 30）**\n")
    L.append("| g | " + " | ".join(f"p={p}" for p in PS) + " | p* |")
    L.append("|---|" + "---|" * len(PS) + "---|")
    for g in ("0.5", "1.0", "1.5", "2.5"):
        fr = q5[g]["frac_open_among_renamed_by_p"]
        L.append(f"| {g} | " + " | ".join("—" if fr[p] is None else f"{fr[p]:.2f}" for p in PS) + f" | {q5[g]['pstar']} |")
    L.append("")
    q6 = ev["Q6"]
    L.append("**表 6　Q6：名前が残った主体（t2 で 0 以外、最後まで同じ名前）の数（D3、全 p をまとめて）**：" +
             "、".join(f"g={g}：{q6[g]['n']}" for g in ("0.5", "1.0", "1.5", "2.5")))
    with open(f"{OUT}/tables.md", "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
