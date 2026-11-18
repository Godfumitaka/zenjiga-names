"""v2 段階 B（主な探索）：名前を他の人たちのあいだの命名ゲームから生ませる。

一回の設定 = 種 0 で R=30 本の走り（継ぎ目の行き先 a と初期位相は走りごとに違い、
同じ種を使うので、条件・p・継ぎ目の長さのあいだで a と初期位相は共通）。

実験
  main    ：六条件（none, D3, D0, C3, C0, C3n）× p ∈ {0.02,0.05,0.1,0.2,0.5,1} × 継ぎ目 ∈ {5,10,30,60}、g=1.5
  pg      ：D3 × p × g ∈ {0.5,1,1.5,2.5}（継ぎ目 30）と、比べるための none × g
  robust  ：六条件 × p × {N=6, N=8, σ_obs=0, σ_obs=0.2}（継ぎ目 30、g=1.5）
  long    ：none 以外の五条件 × p（継ぎ目 30）を T=3000 で（閉じないのが遅いだけでないかを見る）

量（§6）：ギャップ G_j = |wrap(ψ̂_j − ψ_j)|。主な量は、継ぎ目の前 [40,60) と最後 [T−50,T) の
G = (1/N)Σ_j G_j の差。主体ごとに、継ぎ目の終わりの公の名前（連続の条件では公の記録を刻んだもの）で分ける。
「開いたまま」は G_j(最後) > 0.05 rad（段階 B を走らせる前に決めた。丸めの残り 0.02 の 2.5 倍）。

実行：.venv/bin/python -m v2.stageB
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import replace
from multiprocessing import Pool

import numpy as np

from .model import ANG, Config, nearest, simulate, wrap

OUT = "results/v2/stageB"
CONDS = ["none", "D3", "D0", "C3", "C0", "C3n"]
PS = [0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
SEAMS = [5.0, 10.0, 30.0, 60.0]
GS = [0.5, 1.0, 1.5, 2.5]
OPEN = 0.05
CLOSE = 0.05


def summarize(c: Config, o):
    t = o["t"]
    G = np.abs(wrap(o["hat"] - o["psi"]))  # (時刻, 走り, 主体)
    t2 = c.t1 + c.seam
    pre = (t >= 40) & (t < 60)
    end = t >= c.T - 50
    Gj_pre = G[pre].mean(0)
    Gj_end = G[end].mean(0)
    k2 = int(np.nonzero(t >= t2)[0][0])
    if c.cond in ("D3", "D0"):
        name_t2 = o["name"][k2].astype(int)
        name_end = o["name"][-1].astype(int)
    else:
        name_t2 = nearest(o["m"][k2])
        name_end = nearest(o["m"][-1])
    absPsi_pre = np.abs(o["psi"][pre]).mean(0)
    absPsi_end = np.abs(o["psi"][end]).mean(0)
    mpsi_end = np.abs(wrap(o["m"][end] - o["psi"][end])).mean(0)
    # 継ぎ目のあと、G_j が CLOSE を下回った最初の時刻（下回らなければ nan）
    post = t >= t2
    tc = np.full(Gj_end.shape, np.nan)
    below = (G < CLOSE) & post[:, None, None]
    anyb = below.any(0)
    first = below.argmax(0)
    tc[anyb] = t[first[anyb]] - t2
    # 閉じきった時刻：t2 以後で G_j ≥ CLOSE だった最後の記録の次（ずっと < CLOSE なら 0、最後も ≥ CLOSE なら nan）
    above = (G >= CLOSE) & post[:, None, None]
    anya = above.any(0)
    last = above.shape[0] - 1 - above[::-1].argmax(0)
    tcf = np.zeros(Gj_end.shape)
    tcf[anya] = t[np.minimum(last[anya] + 1, len(t) - 1)] - t2
    tcf[anya & (last == len(t) - 1)] = np.nan
    r = o["r"]
    out = {
        "cfg": c.to_dict(),
        "G_pre": G[pre].mean((0, 2)).tolist(),
        "G_end": G[end].mean((0, 2)).tolist(),
        "dG": (G[end].mean((0, 2)) - G[pre].mean((0, 2))).tolist(),
        "r_pre": r[pre].mean(0).tolist(),
        "r_end": r[end].mean(0).tolist(),
        "Gj_pre": Gj_pre.tolist(),
        "Gj_end": Gj_end.tolist(),
        "name_t2": name_t2.tolist(),
        "name_end": name_end.tolist(),
        "absPsi_pre": absPsi_pre.tolist(),
        "absPsi_end": absPsi_end.tolist(),
        "mpsi_end": mpsi_end.tolist(),
        "t_close": tc.tolist(),  # 最初に G_j < CLOSE になった時刻（走らせる前の定め）
        "t_close_final": tcf.tolist(),  # 閉じきった時刻（段階 B を走らせたあとに足した。DECISIONS v2-7）
        "n_rename": o["n_rename"].tolist(),
        "a": o["a"].tolist(),
        "final_m": o["final_m"].tolist(),
        "final_psi": o["final_psi"].tolist(),
    }
    sub = slice(None, None, 2)  # 記録（1 の時間ごと）を 2 の時間ごとに間引いて保存
    traces = {"G_t": G.mean(2)[sub].astype(np.float16), "r_t": r[sub].astype(np.float16), "t": t[sub].astype(np.float32)}
    return out, traces


def work(c: Config):
    o = simulate(c)
    s, tr = summarize(c, o)
    full = None
    if c.cond in CONDS and c.p == 0.1 and c.seam == 30.0 and c.g == 1.5 and c.N == 4 and c.sobs == 0.1 and c.T == 1000.0:
        full = {k: (v.astype(np.float32) if isinstance(v, np.ndarray) and v.dtype == np.float64 else v) for k, v in o.items()}
    return s, tr, full


def build_configs():
    base = Config(R=30, seed=0, T=1000.0)
    exps = {"main": [], "pg": [], "robust": [], "long": [], "long_sobs02": []}
    for seam in SEAMS:
        exps["main"].append(replace(base, cond="none", p=0.1, seam=seam))
        for cond in CONDS[1:]:
            for p in PS:
                exps["main"].append(replace(base, cond=cond, p=p, seam=seam))
    for g in GS:
        exps["pg"].append(replace(base, cond="none", p=0.1, g=g))
        if g == 1.5:
            continue  # main と同じ
        for p in PS:
            exps["pg"].append(replace(base, cond="D3", p=p, g=g))
    for lab, kw in (("N6", dict(N=6)), ("N8", dict(N=8)), ("sobs0", dict(sobs=0.0)), ("sobs0.2", dict(sobs=0.2))):
        exps["robust"].append(replace(base, cond="none", p=0.1, **kw))
        for cond in CONDS[1:]:
            for p in PS:
                exps["robust"].append(replace(base, cond=cond, p=p, **kw))
    for cond in CONDS[1:]:
        for p in PS:
            exps["long"].append(replace(base, cond=cond, p=p, T=3000.0))
    # 走らせたあとに足した：σ_obs=0.2 で開いたままの主体が出たので、T=3000 で遅いだけかを見る
    exps["long_sobs02"].append(replace(base, cond="none", p=0.1, sobs=0.2, T=3000.0))
    for cond in CONDS[1:]:
        for p in PS:
            exps["long_sobs02"].append(replace(base, cond=cond, p=p, sobs=0.2, T=3000.0))
    return exps


def key_of(c):
    d = c.to_dict() if hasattr(c, "to_dict") else c
    return f"{d['cond']}|p={d['p']}|seam={d['seam']}|g={d['g']}|N={d['N']}|sobs={d['sobs']}|T={d['T']}"


def main():
    os.makedirs(OUT, exist_ok=True)
    exps = build_configs()
    which = [a for a in sys.argv[1:] if a in exps] or [e for e in exps if e != "long_sobs02"]
    for name in which:
        cfgs = exps[name]
        t0 = time.time()
        with Pool(max(1, (os.cpu_count() or 2) - 1)) as pool:
            res = pool.map(work, cfgs, chunksize=1)
        summ = {key_of(c): s for c, (s, _, _) in zip(cfgs, res)}
        with open(f"{OUT}/{name}.json", "w") as f:
            json.dump({"experiment": name, "seed": 0, "R": 30, "open_threshold": OPEN, "close_threshold": CLOSE,
                       "configs": [c.to_dict() for c in cfgs], "summaries": summ}, f, ensure_ascii=False)
        tr = {}
        for c, (_, t_, _) in zip(cfgs, res):
            k = key_of(c)
            tr[k + "|G_t"] = t_["G_t"]
            tr[k + "|r_t"] = t_["r_t"]
        tr["t"] = res[0][1]["t"]
        np.savez_compressed(f"{OUT}/{name}_traces.npz", **tr)
        for c, (_, _, full) in zip(cfgs, res):
            if full is not None:
                np.savez_compressed(f"{OUT}/default_full_{c.cond}.npz", **full)
                with open(f"{OUT}/default_full_{c.cond}.json", "w") as f:
                    json.dump(c.to_dict(), f)
        print(f"{name}: {len(cfgs)} 設定, {time.time() - t0:.0f} 秒", flush=True)


if __name__ == "__main__":
    main()
