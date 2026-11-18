"""v2 段階 A（短い確かめ）：v1 の条件を既定値 g=1.5 だけで回し、gap_check.py の値を再現する。

仕様 §3 に挙がっている四条件（名前なし・連続の評判・丸めだけ・付け直しにくい名前 δ=0.6）。
gap_check.py と同じ設定（固定の行き先 a=(−1.4, −0.5, 0.4, 1.5)、種 1、T=500、継ぎ目 [60, 90)）。
gap_check.py そのものも同じ場で走らせ、数字が一致するかを見る。

実行：.venv/bin/python -m v2.stageA
"""
from __future__ import annotations

import json
import os

import numpy as np

from .model import Config, simulate, wrap

OUT = "results/v2/stageA"
CONDS = [("none", "名前なし"), ("cont", "連続の評判"), ("round", "丸めだけ"), ("delta", "付け直しにくい名前 δ=0.6")]


def main():
    os.makedirs(OUT, exist_ok=True)
    src = open("gap_check.py").read().split("for g in [1.0, 1.5, 2.5]:")[0]
    ns = {}
    exec(src, ns)
    rows, lines = [], []
    gc_args = {"none": ("A", 0.6), "cont": ("Bc", 0.6), "round": ("B", np.pi / 8 + 1e-9), "delta": ("B", 0.6)}
    for g in (1.5,):
        for mc, lab in CONDS:
            cfg = Config(cond=mc, g=g, R=1, seed=1, T=500.0, seam=30.0, a_fixed=(-1.4, -0.5, 0.4, 1.5))
            o = simulate(cfg)
            gap = np.abs(wrap(o["final_hat"] - o["final_psi"]))[0]
            r0, psi0, gap0 = ns["run"](*gc_args[mc][:1], g, delta=gc_args[mc][1])
            row = {"g": g, "cond": mc, "label": lab, "gap_per_agent": gap.round(4).tolist(),
                   "gap_mean": float(gap.mean()), "gap_max": float(gap.max()),
                   "r_end": float(o["r"][-1, 0]), "absPsi_max_end": float(np.abs(o["final_psi"]).max()),
                   "n_rename": o["n_rename"][0].tolist(),
                   "maxdiff_vs_gap_check": float(np.abs(gap - np.abs(gap0)).max()),
                   "config": cfg.to_dict()}
            rows.append(row)
            lines.append(f"g={g} {lab}: ギャップ（主体ごと）={np.round(gap, 3).tolist()}  最大 {gap.max():.3f}  "
                         f"r={row['r_end']:.4f}  gap_check との差 {row['maxdiff_vs_gap_check']:.1e}")
    with open(f"{OUT}/summary.json", "w") as f:
        json.dump({"rows": rows, "target": {"none": 0.0, "round": 0.02, "delta0.6": 0.20},
                   "note": "名前の付いた主体（押しやられた外側の 2 人）のギャップを、仕様 §3 の目安と比べる"},
                  f, ensure_ascii=False, indent=1)
    with open(f"{OUT}/stageA.txt", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
