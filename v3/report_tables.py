"""REPORT_v3.md の表を results から作る（手で写さない）。

実行：.venv/bin/python -m v3.report_tables  →  results/v3/tables.md
"""
from __future__ import annotations

import json

PS = ["0.02", "0.05", "0.1", "0.2", "0.5"]
KS = ["0.25", "0.5", "0.75", "1.0"]


def fp(x):
    return "—" if x is None else (f"{x:.3f}" if x >= 0.001 else f"{x:.0e}")


def main():
    L = []
    s3 = json.load(open("results/v3/stage3/summary.json"))["evaluation"]
    L.append("**表 1　段 3（既定値：κ=1、p=0.1、Δω=1.2、N=4、全員がつながる、種 30）**\n")
    L.append("| 条件 | 立つ | 沈む | 持続時間 中央値／平均／最大 | 持続版 中央値 | T まで残った |")
    L.append("|---|---|---|---|---|---|")
    for c in ("N0", "N1", "N2", "M", "S"):
        p = s3["persist"][c]
        ps = s3["persist_sustained"][c]
        L.append(f"| {c} | {s3['stood'][c]:.0%} | {s3['sank'][c]:.0%} | {p['median']:.1f}／{p['mean']:.1f}／{p['max']:.0f} | "
                 f"{ps['median']:.1f} | {p['frac_censored']:.0%} |")
    r3 = s3["R3"]
    pr = r3["persist_N1_renamed"]
    L.append(f"| N1 のうち継ぎ目中に名前が 0 以外へ（{pr['n']}/30） | | | {pr['median']:.1f}／{pr['mean']:.1f}／{pr['max']:.0f} | "
             f"{r3['persist_sustained_N1_renamed']['median']:.1f} | {pr['frac_censored']:.0%} |")
    L.append("")
    ev = json.load(open("results/v3/stage4/evaluation.json"))
    pk = ev["pk"]
    L.append("**表 2　N1：継ぎ目中に名前が 0 以外へ替わった走りの割合／その走りの持続時間の平均（中央値）／T まで残った割合／N0 より長いかの片側 p**\n")
    L.append("| κ ＼ p | " + " | ".join(PS) + " |")
    L.append("|---|" + "---|" * len(PS))
    for k in KS:
        cells = []
        for p in PS:
            c = pk[f"{p}|{k}"]
            pr = c["N1_persist_renamed"]
            cells.append(f"{c['N1_frac_renamed']:.0%}／{pr['mean']:.0f}（{pr['median']:.0f}）／{c['N1_frac_censored_renamed']:.0%}／{fp(c['R3_mwu_N1renamed_gt_N0']['p'])}"
                         if pr else f"{c['N1_frac_renamed']:.0%}／—")
        L.append(f"| {k} | " + " | ".join(cells) + " |")
    L.append(f"\nκ*（名前が替わった走りの半分以上が T まで残る最小の κ）：" +
             "、".join(f"p={p}：{'なし' if v is None else v}" for p, v in ev["kappa_star_by_p"].items()))
    L.append("")
    L.append("**表 3　持続時間の平均：N1 ／ N2 ／ M（括弧内は片側 p：N2<N1、M<N1）**\n")
    L.append("| κ ＼ p | " + " | ".join(PS) + " |")
    L.append("|---|" + "---|" * len(PS))
    for k in KS:
        cells = []
        for p in PS:
            c = pk[f"{p}|{k}"]
            cells.append(f"{c['N1_persist_all']['mean']:.0f}／{c['N2_persist']['mean']:.0f}／{c['M_persist']['mean']:.0f}"
                         f"（{fp(c['R4_mwu_N2_lt_N1']['p'])}, {fp(c['R5_mwu_M_lt_N1']['p'])}）")
        L.append(f"| {k} | " + " | ".join(cells) + " |")
    L.append("")
    L.append("**表 4　R7：t2 以後、同じ公の名前のまま内容が本人の刻みと違う時間の割合（平均）：N1 ／ N2**\n")
    L.append("| κ ＼ p | " + " | ".join(PS) + " |")
    L.append("|---|" + "---|" * len(PS))
    for k in KS:
        L.append(f"| {k} | " + " | ".join(f"{pk[f'{p}|{k}']['R7_N1_sameName_ne_bin']['mean']:.3f}／{pk[f'{p}|{k}']['R7_N2_sameName_ne_bin']['mean']:.3f}"
                                          for p in PS) + " |")
    L.append("")
    L.append("**表 5　R6（S）：t=600 に立っていた走り → 止めた終わりにまだ立っていた走り ／ t=600 に名前が 0 以外だった走り → 止めているあいだに付け直された走り ／ 噂を戻して元の名前に戻った走り（各 30 走り中）**\n")
    L.append("| κ ＼ p | " + " | ".join(PS) + " |")
    L.append("|---|" + "---|" * len(PS))
    for k in KS:
        cells = []
        for p in PS:
            c = pk[f"{p}|{k}"]
            cells.append(f"{c['S_standing_before_stop']}→{c['S_still_standing_end_of_stop_among_standing']} ／ "
                         f"{c['S_nonzero_name_at_stop']}→{c['S_renamed_during_stop_among_nonzero_at_stop']} ／ {c['S_back_to_original_among_renamed']}")
        L.append(f"| {k} | " + " | ".join(cells) + " |")
    L.append("")
    L.append("**表 6　Δω・N・輪（p=0.1、κ=1）：持続時間の平均（中央値）と、N1 の名前が替わった走りが N0 より長いかの片側 p**\n")
    L.append("| 設定 | N0 | N1 | N2 | M | N1 名前替わり（割合） | p（N1 替わり > N0） | 立つ（N0） | R1 |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    rows = [("dw", "Δω=0.5", "p=0.1|k=1.0|dw=0.5|N=4|top=all"), ("dw", "Δω=1.6", "p=0.1|k=1.0|dw=1.6|N=4|top=all"),
            ("N", "N=3", "p=0.1|k=1.0|dw=1.2|N=3|top=all"), ("N", "N=6", "p=0.1|k=1.0|dw=1.2|N=6|top=all"),
            ("ring", "輪 N=4", "p=0.1|k=1.0|dw=1.2|N=4|top=ring"), ("ring", "輪 N=6", "p=0.1|k=1.0|dw=1.2|N=6|top=ring")]
    for sw, lab, key in rows:
        if sw not in ev or key not in ev[sw]:
            continue
        e = ev[sw][key]
        pe = e["persist"]
        r3 = e["R3"]
        r1 = all(v["all_reach_0.9_before_100"] for v in e["R1"].values())
        L.append(f"| {lab} | " + " | ".join(f"{pe[c]['mean']:.0f}（{pe[c]['median']:.0f}）" for c in ("N0", "N1", "N2", "M")) +
                 f" | {r3['frac_renamed_in_seam']:.0%} | {fp(r3['mwu_N1renamed_gt_N0']['p'])} | {e['stood']['N0']:.0%} | {'当たり' if r1 else '外れ'} |")
    with open("results/v3/tables.md", "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
