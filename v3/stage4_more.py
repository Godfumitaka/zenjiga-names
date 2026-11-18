"""v3 段 4 の追加（時間が残った分）：小さい p の標本を増やす。

段 4 の p × κ では、p=0.02 で名前が替わる走りが 30 本中 6 本しかなく、κ* がないという結論の標本が小さかった。
そこで p ∈ {0.02, 0.05} × κ ∈ {0.25, 1} の N1・N2・M と、比べる相手の N0 を、種 30〜129（100 本）足す。
種 0〜29（段 3・段 4）と合わせて 130 本で見る。模型も予想も変えていない。

実行：.venv/bin/python -m v3.stage4_more  →  results/v3/stage4_more/
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import replace
from multiprocessing import Pool

import numpy as np

from .analysis import describe
from .model import Config
from .runner import NPROC, git_rev
from .stage3 import mwu
from .stage4 import tag, work

OUT = "results/v3/stage4_more"
NEW = list(range(30, 130))
PS = [0.02, 0.05]
KS = [0.25, 1.0]


def main():
    os.makedirs(OUT, exist_ok=True)
    base = Config()
    cfgs = [replace(base, cond="N0", seed=s) for s in NEW]
    for p in PS:
        for k in KS:
            for cond in ("N1", "N2", "M"):
                cfgs += [replace(base, cond=cond, p=p, kappa=k, seed=s) for s in NEW]
    t0 = time.time()
    with Pool(NPROC) as pool:
        res = pool.map(work, cfgs, chunksize=2)
    groups = {}
    for c, (s, _) in zip(cfgs, res):
        groups.setdefault(tag(c), []).append(s)
    with open(f"{OUT}/runs.json", "w") as f:
        json.dump({"git_rev": git_rev(), "seeds": NEW, "base_config": base.to_dict(), "summaries": groups}, f, default=float)
    print(f"{len(cfgs)} 走り, {time.time() - t0:.0f} 秒", flush=True)

    # 種 0〜29 と合わせる
    with open("results/v3/stage3/summary.json") as f:
        n0_old = json.load(f)["per_run"]["N0"]
    with open("results/v3/stage4/pk.json") as f:
        pk_old = json.load(f)["summaries"]
    n0 = n0_old + groups[tag(replace(base, cond="N0"))]
    ev = {"n_seeds": len(n0)}
    for p in PS:
        for k in KS:
            g = {cond: pk_old[tag(replace(base, cond=cond, p=p, kappa=k))] + groups[tag(replace(base, cond=cond, p=p, kappa=k))]
                 for cond in ("N1", "N2", "M")}
            ren = [x for x in g["N1"] if x["renamed_in_seam"]]
            cell = {
                "N1_frac_renamed": float(np.mean([x["renamed_in_seam"] for x in g["N1"]])),
                "N1_n_renamed": len(ren),
                "N1_persist_renamed": describe([x["persist"] for x in ren]) if ren else None,
                "N1_n_censored_renamed": int(sum(x["censored"] for x in ren)),
                "N1_frac_censored_renamed": float(np.mean([x["censored"] for x in ren])) if ren else None,
                "N1_censored_seeds": [x["seed"] for x in g["N1"] if x["censored"]],
                "N0_persist": describe([x["persist"] for x in n0]),
                "R3_mwu_N1renamed_gt_N0": mwu([x["persist"] for x in ren], [x["persist"] for x in n0], "greater"),
                "R3_mwu_sustained": mwu([x["persist_sustained"] for x in ren], [x["persist_sustained"] for x in n0], "greater"),
                "N1_persist_all": describe([x["persist"] for x in g["N1"]]),
                "N2_persist": describe([x["persist"] for x in g["N2"]]),
                "M_persist": describe([x["persist"] for x in g["M"]]),
                "R4_mwu_N2_lt_N1": mwu([x["persist"] for x in g["N2"]], [x["persist"] for x in g["N1"]], "less"),
                "R5_mwu_M_lt_N1": mwu([x["persist"] for x in g["M"]], [x["persist"] for x in g["N1"]], "less"),
                "R7_N1_sameName_ne_bin": describe([x["frac_sameName_ne_bin_post"] for x in g["N1"]]),
                "R7_N2_sameName_ne_bin": describe([x["frac_sameName_ne_bin_post"] for x in g["N2"]]),
                "R7_mwu_N1_gt_N2": mwu([x["frac_sameName_ne_bin_post"] for x in g["N1"]], [x["frac_sameName_ne_bin_post"] for x in g["N2"]], "greater"),
            }
            ev[f"{p}|{k}"] = cell
    with open(f"{OUT}/evaluation.json", "w") as f:
        json.dump(ev, f, ensure_ascii=False, indent=1, default=float)
    for key, c in ev.items():
        if key == "n_seeds":
            continue
        pr = c["N1_persist_renamed"]
        print(key, f"renamed {c['N1_n_renamed']}/130 ({c['N1_frac_renamed']:.0%})",
              f"persist renamed mean {pr['mean']:.0f} median {pr['median']:.0f}" if pr else "",
              f"censored {c['N1_n_censored_renamed']} seeds {c['N1_censored_seeds']}",
              f"R3 p={c['R3_mwu_N1renamed_gt_N0']['p']:.3g} R4 p={c['R4_mwu_N2_lt_N1']['p']:.3g} R5 p={c['R5_mwu_M_lt_N1']['p']:.3g}",
              f"N1/N2/M mean {c['N1_persist_all']['mean']:.0f}/{c['N2_persist']['mean']:.0f}/{c['M_persist']['mean']:.0f}",
              f"R7 {c['R7_N1_sameName_ne_bin']['mean']:.3f}/{c['R7_N2_sameName_ne_bin']['mean']:.3f} p={c['R7_mwu_N1_gt_N2']['p']:.3g}")


if __name__ == "__main__":
    main()
