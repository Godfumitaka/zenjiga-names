"""v3 段 2（較正）：最小の命名ゲームを単独で再現する。

一様混合（毎回、話し手と聞き手を全員から無作為に選ぶ）で、
  - 合意までのやりとりの回数 t_conv
  - 語の総数（全員の語彙の大きさの和）の最大 N_w^max
が人数 N の 3/2 乗に比例するかを確かめる（Baronchelli ら 2006）。
やりとりの規則は v3 の模型の噂と同じ関数（naming.ng_interaction）。

実行：.venv/bin/python -m v3.stage2
"""
from __future__ import annotations

import json
import os
from multiprocessing import Pool

import numpy as np

from .naming import ng_interaction
from .plotstyle import INK2, MUTED, plt
from .runner import NPROC, git_rev

OUT = "results/v3/stage2"
NS = [50, 100, 200, 400, 800, 1600, 3200, 6400, 12800]
RUNS = 20


def run_ng(args):
    N, seed = args
    rng = np.random.default_rng([N, seed])
    V = [set() for _ in range(N)]
    count = {}  # 語 → 持っている人数
    nxt = [0]

    def invent():
        nxt[0] += 1
        return nxt[0]

    Nw = 0
    Nw_max, t_max = 0, 0
    t = 0
    trace_every = max(1, int(N ** 1.5 / 50))
    trace = []
    while True:
        s = int(rng.integers(N))
        l = int(rng.integers(N - 1))
        if l >= s:
            l += 1
        old_s, old_l = set(V[s]), set(V[l])
        w, ok, inv = ng_interaction(V[s], V[l], rng, invent)
        t += 1
        for x in old_s - V[s]:
            count[x] -= 1
        for x in V[s] - old_s:
            count[x] = count.get(x, 0) + 1
        for x in old_l - V[l]:
            count[x] -= 1
        for x in V[l] - old_l:
            count[x] = count.get(x, 0) + 1
        Nw += len(V[s]) - len(old_s) + len(V[l]) - len(old_l)
        if Nw > Nw_max:
            Nw_max, t_max = Nw, t
        if t % trace_every == 0:
            trace.append((t, Nw))
        if ok and Nw == N and count[w] == N:
            break
    return {"N": N, "seed": seed, "t_conv": t, "Nw_max": Nw_max, "t_max": t_max, "trace": trace}


def main():
    os.makedirs(OUT, exist_ok=True)
    jobs = [(N, s) for N in NS for s in range(RUNS)]
    with Pool(NPROC) as pool:
        res = pool.map(run_ng, jobs, chunksize=1)
    rows = [{k: v for k, v in r.items() if k != "trace"} for r in res]
    lnN = np.log([r["N"] for r in rows])
    fits = {}
    rng = np.random.default_rng(0)
    fits_large, fits_xl = {}, {}
    for key in ("t_conv", "Nw_max", "t_max"):
        for dst, sel in ((fits, np.ones(len(rows), bool)), (fits_large, np.array([r["N"] >= 400 for r in rows])),
                         (fits_xl, np.array([r["N"] >= 1600 for r in rows]))):
            y = np.log([r[key] for r in rows])[sel]
            x = lnN[sel]
            slope, icpt = np.polyfit(x, y, 1)
            bs = []
            idx = np.arange(x.size)
            for _ in range(1000):
                b = rng.choice(idx, idx.size)
                bs.append(np.polyfit(x[b], y[b], 1)[0])
            dst[key] = {"slope": float(slope), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                        "prefactor": float(np.exp(icpt))}
    by_N = {}
    for N in NS:
        rr = [r for r in rows if r["N"] == N]
        by_N[N] = {k: {"mean": float(np.mean([r[k] for r in rr])), "sd": float(np.std([r[k] for r in rr]))}
                   for k in ("t_conv", "Nw_max", "t_max")}
        by_N[N].update({f"{k}/N^1.5": float(np.mean([r[k] for r in rr]) / N ** 1.5) for k in ("t_conv", "Nw_max", "t_max")})
    summary = {"Ns": NS, "runs_per_N": RUNS, "seeds": "rng = default_rng([N, seed]), seed = 0..19",
               "fits_loglog": fits, "fits_loglog_N>=400": fits_large, "fits_loglog_N>=1600": fits_xl, "by_N": by_N, "per_run": rows, "git_rev": git_rev(),
               "rule": "naming.ng_interaction（v3 の噂と同じ関数）、一様混合"}
    with open(f"{OUT}/summary.json", "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    with open(f"{OUT}/config.json", "w") as f:
        json.dump({"Ns": NS, "runs_per_N": RUNS, "seed_rule": "default_rng([N, seed])"}, f, indent=1)
    print(json.dumps({"fits": fits, "fits_N>=400": fits_large, "fits_N>=1600": fits_xl}, ensure_ascii=False, indent=1))

    fig, ax = plt.subplots(1, 2, figsize=(8.2, 3.3), gridspec_kw={"wspace": 0.32})
    cols = plt.cm.viridis(np.linspace(0.1, 0.9, len(NS)))
    for ci, N in enumerate(NS):
        rr = [r for r in res if r["N"] == N][:3]
        for r in rr:
            tr = np.array(r["trace"])
            ax[0].plot(tr[:, 0] / N ** 1.5, tr[:, 1] / N ** 1.5, color=cols[ci], lw=0.8,
                       label=f"N={N}" if r is rr[0] else None)
    ax[0].set_xlabel("やりとりの回数 / N^1.5")
    ax[0].set_ylabel("語の総数 / N^1.5")
    ax[0].set_title("(a) 語の総数の推移（N^1.5 でそろえる）")
    ax[0].legend(fontsize=7)
    Ns = np.array(NS, float)
    for key, mk, lab in (("t_conv", "o", "合意までの回数 t_conv"), ("Nw_max", "s", "語の総数の最大 N_w^max"),
                         ("t_max", "^", "最大の時刻 t_max")):
        m = np.array([by_N[N][key]["mean"] for N in NS])
        ax[1].loglog(Ns, m, mk, color=INK2 if key != "Nw_max" else "#d1495b", ms=4,
                     label=f"{lab}：傾き {fits[key]['slope']:.2f}（N≥1600 で {fits_xl[key]['slope']:.2f}）")
        ax[1].loglog(Ns, fits[key]["prefactor"] * Ns ** fits[key]["slope"], color=MUTED, lw=0.7)
    ax[1].loglog(Ns, Ns ** 1.5, ":", color=MUTED, lw=0.8, label="N^1.5")
    ax[1].set_xlabel("人数 N")
    ax[1].set_title("(b) 比例の確かめ（各 N で 20 走りの平均）")
    ax[1].legend(fontsize=7)
    fig.savefig(f"{OUT}/fig_stage2.png", bbox_inches="tight")
    print("saved")


if __name__ == "__main__":
    main()
