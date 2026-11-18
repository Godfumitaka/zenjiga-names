"""v3 段 4（掃引）。p と κ を優先する（TASK_claude_code.md の 6）。

掃引（どれも種 0〜29、T=1500）
  pk    ：p ∈ {0.02,0.05,0.1,0.2,0.5} × κ ∈ {0.25,0.5,0.75,1} × 条件 {N1, N2, M, S}
          （N0 は p・κ によらないので段 3 の N0 を使う）
  dw    ：Δω ∈ {0.5, 1.6} × 五条件（1.2 は段 3）
  N     ：N ∈ {3, 6} × 五条件（4 は段 3）
  ring  ：輪（各主体は隣の二人とだけ引き合う）× N ∈ {4, 6} × 五条件

保存：走りごとの量（analysis.run_summary）を JSON に、焦点 j の ΔL_self と公の名前の内容の時系列を npz に。
実行：.venv/bin/python -m v3.stage4 [pk dw N ring]
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import replace
from multiprocessing import Pool

import numpy as np

from .analysis import run_summary
from .model import Config, simulate
from .runner import NPROC, git_rev

OUT = "results/v3/stage4"
SEEDS = list(range(30))
PS = [0.02, 0.05, 0.1, 0.2, 0.5]
KS = [0.25, 0.5, 0.75, 1.0]
CONDS = ["N0", "N1", "N2", "M", "S"]


def work(cfg: Config):
    r = simulate(cfg)
    st = {k: np.asarray(v)[None] for k, v in r.items()}
    s = run_summary(st, 0, cfg.to_dict())
    j = cfg.focal
    tr = {"dL": r["dL"][:, j].astype(np.float16), "pub": r["pub_content"][:, j].astype(np.int8),
          "absphi": np.abs(r["phi"][:, j]).astype(np.float16), "x_to": r["x_to"][:, j].astype(np.float16),
          "mem": r["mem_pub"][:, j].astype(np.float16)}
    return s, tr


def tag(c: Config):
    return f"{c.cond}|p={c.p}|k={c.kappa}|dw={c.dw}|N={c.N}|top={c.topology}"


def build():
    base = Config()
    sw = {"pk": [], "dw": [], "N": [], "ring": []}
    for p in PS:
        for k in KS:
            for cond in ("N1", "N2", "M", "S"):
                sw["pk"] += [replace(base, cond=cond, p=p, kappa=k, seed=s) for s in SEEDS]
    for dw in (0.5, 1.6):
        for cond in CONDS:
            sw["dw"] += [replace(base, cond=cond, dw=dw, seed=s) for s in SEEDS]
    for N in (3, 6):
        for cond in CONDS:
            sw["N"] += [replace(base, cond=cond, N=N, seed=s) for s in SEEDS]
    for N in (4, 6):
        for cond in CONDS:
            sw["ring"] += [replace(base, cond=cond, N=N, topology="ring", seed=s) for s in SEEDS]
    return sw


def main():
    os.makedirs(OUT, exist_ok=True)
    sw = build()
    which = [a for a in sys.argv[1:] if a in sw] or list(sw)
    for name in which:
        cfgs = sw[name]
        t0 = time.time()
        with Pool(NPROC) as pool:
            res = pool.map(work, cfgs, chunksize=2)
        groups = {}
        for c, (s, tr) in zip(cfgs, res):
            groups.setdefault(tag(c), []).append((c, s, tr))
        summ = {g: [s for _, s, _ in v] for g, v in groups.items()}
        cfg_json = {g: {"base_config": v[0][0].to_dict(), "seeds": [c.seed for c, _, _ in v]} for g, v in groups.items()}
        with open(f"{OUT}/{name}.json", "w") as f:
            json.dump({"sweep": name, "git_rev": git_rev(), "configs": cfg_json, "summaries": summ}, f,
                      ensure_ascii=False, default=float)
        arrs = {}
        for g, v in groups.items():
            for key in v[0][2]:
                arrs[f"{g}|{key}"] = np.stack([tr[key] for _, _, tr in v])
        arrs["t"] = np.arange(10.0, cfgs[0].T, 1.0, dtype=np.float32)[: res[0][1]["dL"].shape[0]]
        np.savez_compressed(f"{OUT}/{name}_traces.npz", **arrs)
        print(f"{name}: {len(cfgs)} 走り, {time.time() - t0:.0f} 秒", flush=True)


if __name__ == "__main__":
    main()
