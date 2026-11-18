"""v3 段 1（名前なし）：世界・学習者・継ぎ目。

1. v3_stage1_smoke2.py の値（§7）を再現する（同じ乱数列で、数字が一致するかを見る）。
2. 学習者がどの通路が自分かを知らないこと、探索の揺らぎが指令の中にあることを確かめる。
3. 間引きの幅の較正（継ぎ目の前の ΔL_self だけを見る）。
4. R1・R2 を、種 30 で確かめる（N0、T=1500）。

実行：.venv/bin/python -m v3.stage1
"""
from __future__ import annotations

import json
import os
from dataclasses import replace

import numpy as np

from .analysis import describe, summarize_batch
from .model import Config, simulate
from .plotstyle import INK2, MUTED, plt, seam
from .runner import run_many, save_batch

OUT = "results/v3/stage1"
SEEDS = list(range(30))


def smoke_repro():
    src = open("v3_stage1_smoke2.py").read().split("for seed in range(3):")[0]
    ns = {}
    exec(src, ns)
    lines = []
    rows = []
    for seed in range(3):
        o = ns["run"](seed)
        r = simulate(Config(seed=seed, T=600.0))
        mine = np.column_stack([r["t"], r["dL"][:, 0], r["chat_ok"][:, 0], r["chat_ok"].mean(1),
                                r["ebar"][:, 0], np.abs(r["phi"][:, 0])])
        diff = np.abs(o - mine).max(0)

        def avg(a, b, col, arr=mine):
            m = (arr[:, 0] >= a) & (arr[:, 0] < b)
            return arr[m, col].mean()

        row = dict(seed=seed, acc_lt100=avg(0, 100, 3), acc_gt100=avg(100, 600, 3),
                   dL_pre=avg(150, 200, 1), dL_seam=avg(220, 260, 1), dL_after=avg(260, 300, 1),
                   dL_later=avg(350, 600, 1), ebar_pre=avg(150, 200, 4), ebar_seam=avg(220, 260, 4),
                   absphi_seam=avg(220, 260, 5), absphi_later=avg(350, 600, 5),
                   maxdiff_vs_smoke2=[float(x) for x in diff])
        rows.append(row)
        lines.append(
            f"種 {seed}: 帰属の正しさ t<100={row['acc_lt100']:.2f} t>100={row['acc_gt100']:.2f} | "
            f"ΔL 前 {row['dL_pre']:+.1f} 中 {row['dL_seam']:+.1f} 直後 {row['dL_after']:+.1f} 後 {row['dL_later']:+.1f} | "
            f"ē 前 {row['ebar_pre']:.4f} 中 {row['ebar_seam']:.4f} | |φ_j| 中 {row['absphi_seam']:.2f} 後 {row['absphi_later']:.2f} | "
            f"smoke2 との差の最大 {max(diff):.1e}")
    return rows, lines


def main():
    os.makedirs(OUT, exist_ok=True)
    report = {}

    # ---- 1. 事前の確かめの再現
    rows, lines = smoke_repro()
    report["smoke_repro"] = rows
    spec = ("§7 の値：ΔL 前 −1.1/−0.7/+0.7、中 +5.5/+9.5/+7.5、直後 +3.3/+1.5/+0.5、後 −1.0/−0.7/−0.4；"
            "ē 前 0.0004 中 0.37；|φ_j| 中 1.3 後 0.01；帰属 1.00")
    with open(f"{OUT}/smoke_repro.txt", "w") as f:
        f.write(spec + "\n" + "\n".join(lines) + "\n")
    print("\n".join(lines))

    # ---- 2+4. 本番（N0、種 30、T=1500）
    base = Config(cond="N0")
    cfgs = [replace(base, seed=s) for s in SEEDS]
    res = run_many(cfgs)
    st = save_batch(OUT, "N0_thin20", cfgs, res)
    meta = {"configs": [c.to_dict() for c in cfgs]}
    summ = summarize_batch(st, meta)

    # ---- 2. 確かめ：通路の並べ替え・はじめの帰属・写しを入れ替えたとき・ζ を写しから外したとき
    perms = st["perm"]  # (種, k, c)
    own_channel = np.array([[int(np.nonzero(perms[s, k] == k)[0][0]) for k in range(4)] for s in range(len(SEEDS))])
    distinct_perm = float(np.mean([len({tuple(p) for p in perms[s]}) for s in range(len(SEEDS))]))
    acc0 = float(st["acc_ok"][:, 0, :].mean())  # 最初の歩（相関の更新 1 回のあと）
    checks = {
        "own_channel_index_hist": np.bincount(own_channel.ravel(), minlength=4).tolist(),
        "mean_distinct_perms_per_run_(of_4)": distinct_perm,
        "acc_at_t0_after_one_update": acc0,
    }
    diag = {}
    for lab, kw in (("copy_shift1", dict(copy_shift=1)), ("zeta_not_in_copy", dict(zeta_in_copy=False)),
                    ("no_zeta_s0_0", dict(s0=0.0))):
        dc = [replace(base, seed=s, T=400.0, **kw) for s in SEEDS[:10]]
        dr = run_many(dc)
        acc = np.stack([r["acc_ok"] for r in dr]).astype(float)  # (種, 時刻, 主体)
        at = dr[0]["acc_t"]
        d = {"acc_mean_t>=50": float(acc[:, at >= 50].mean())}
        if lab == "copy_shift1":
            # 帰属した通路が、写しを渡した主体（k+1）の通路になっている割合
            who = np.stack([r["acc_who"] for r in dr])  # (種, 時刻, 主体)
            target = (np.arange(4) + 1) % 4
            d["frac_attributed_to_copy_source_t>=50"] = float((who[:, at >= 50] == target).mean())
            d["note"] = "写しを主体 k+1 の指令にすると、帰属は k+1 の通路に移る（帰属は写しとの相関だけで決まる）"
        diag[lab] = d
        print(lab, d)
    checks["diagnostics"] = diag

    # 探索の揺らぎ：記録した ζ の実際の大きさと、σ0√(1+c_e ē) の比較（全主体）
    t = st["t"][0].astype(float)
    zeta = st["zeta"].astype(float)
    eb = st["ebar"].astype(float)
    sd_pred = 0.1 * np.sqrt(1 + 5 * eb)
    zs = {}
    for lab, (a, b) in (("pre", (150, 200)), ("seam", (220, 260)), ("late", (1000, 1500))):
        m = (t >= a) & (t < b)
        zs[lab] = {"realized_sd": float(zeta[:, m, :].std()), "predicted_sd_rms": float(np.sqrt((sd_pred[:, m, :] ** 2).mean()))}
    checks["zeta_sd"] = zs
    report["checks"] = checks
    print(json.dumps(checks, ensure_ascii=False, indent=1))

    # ---- 3. 間引きの幅の較正（継ぎ目の前だけを見る）
    calib = {}
    for thin in (10, 20, 40):
        if thin == 20:
            s_ = summ
        else:
            cc = [replace(base, seed=s, thin=thin, T=260.0) for s in SEEDS]
            rr = run_many(cc)
            stt = save_batch(OUT, f"N0_thin{thin}_pre_only", cc, rr)
            tt = stt["t"][0].astype(float)
            dl = stt["dL"][:, :, 0].astype(float)
            m = (tt >= 10) & (tt < 200)
            m2 = (tt >= 150) & (tt < 200)
            s_ = [{"dL_pre_all": float(dl[i, m].mean()), "dL_pre": float(dl[i, m2].mean()),
                   "frac_pos_pre": float((dl[i, m] > 0).mean())} for i in range(len(SEEDS))]
        calib[thin] = {"dL_pre_all": describe([x["dL_pre_all"] for x in s_]),
                       "dL_pre_[150,200)": describe([x["dL_pre"] for x in s_]),
                       "frac_pos_pre": describe([x["frac_pos_pre"] for x in s_])}
    report["thin_calibration"] = calib
    print(json.dumps(calib, ensure_ascii=False, indent=1))

    # ---- 4. R1・R2
    r1 = {
        "t_acc90": describe([x["t_acc90"] for x in summ]),
        "acc_first100": describe([x["acc_first100"] for x in summ]),
        "acc_after100": describe([x["acc_after100"] for x in summ]),
        "acc_seam": describe([x["acc_seam"] for x in summ]),
        "all_runs_reach_0.9_before_100": bool(all(x["t_acc90"] < 100 for x in summ)),
    }
    stood = [x["stood"] for x in summ]
    sank = [x["sank"] for x in summ]
    r2 = {
        "frac_stood_(mean_dL_seam>0)": float(np.mean(stood)),
        "frac_sank_(dL<=0_after_t2)": float(np.mean(sank)),
        "persist": describe([x["persist"] for x in summ]),
        "persist_sustained": describe([x["persist_sustained"] for x in summ]),
        "dL_pre": describe([x["dL_pre"] for x in summ]),
        "dL_seam": describe([x["dL_seam"] for x in summ]),
        "dL_just_after": describe([x["dL_just_after"] for x in summ]),
        "dL_after": describe([x["dL_after"] for x in summ]),
        "dL_late": describe([x["dL_late"] for x in summ]),
        "ebar_pre": describe([x["ebar_pre"] for x in summ]),
        "ebar_seam": describe([x["ebar_seam"] for x in summ]),
        "absphi_seam": describe([x["absphi_seam"] for x in summ]),
        "absphi_late": describe([x["absphi_late"] for x in summ]),
    }
    r2["R2_holds"] = bool(r2["frac_stood_(mean_dL_seam>0)"] == 1.0 and r2["frac_sank_(dL<=0_after_t2)"] >= 0.9)
    report["R1"] = r1
    report["R2"] = r2
    report["per_run"] = summ
    with open(f"{OUT}/summary.json", "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=1, default=float)
    print(json.dumps({"R1": r1, "R2": r2}, ensure_ascii=False, indent=1))

    # ---- 図
    fig, ax = plt.subplots(4, 1, figsize=(7.2, 9.0), sharex=True, gridspec_kw={"hspace": 0.35})
    at = st["acc_t"][0].astype(float)
    acc = st["acc_ok"].astype(float).mean(axis=(0, 2))
    ax[0].plot(at, acc, color=INK2, lw=1.2)
    ax[0].axhline(0.9, color=MUTED, lw=0.7, ls=":")
    ax[0].set_ylim(0, 1.05)
    ax[0].set_ylabel("帰属の正しさ")
    ax[0].set_title("(a) 帰属の正しさ（30 種 × 4 主体の平均）")
    dl = st["dL"][:, :, 0].astype(float)
    q = np.percentile(dl, [25, 50, 75], axis=0)
    ax[1].fill_between(t, q[0], q[2], color="#d1495b", alpha=0.2, lw=0)
    ax[1].plot(t, q[1], color="#d1495b", lw=1.2, label="中央値（四分位の帯）")
    for s in range(3):
        ax[1].plot(t, dl[s], color=INK2, lw=0.5, alpha=0.6)
    ax[1].axhline(0, color=MUTED, lw=0.7)
    ax[1].set_ylabel("ΔL_self（焦点 j）")
    ax[1].set_title("(b) 自己の項 ΔL_self(t)：正なら立っている（細線：種 0〜2）")
    ax[1].legend(loc="upper right")
    ebm = eb[:, :, 0]
    ax[2].semilogy(t, np.median(ebm, 0), color="#2a78d6", lw=1.2, label="焦点 j")
    ax[2].semilogy(t, np.median(eb[:, :, 1:].mean(2), 0), color=MUTED, lw=1.0, label="他の 3 人の平均")
    ax[2].set_ylabel("不確かさ ē（中央値）")
    ax[2].set_title("(c) 不確かさ ē(t)")
    ax[2].legend(loc="upper right")
    ax[3].plot(t, np.median(sd_pred[:, :, 0], 0), color="#1baf7a", lw=1.2, label="σ0√(1+c_e ē)（焦点 j）")
    ax[3].plot(t, np.abs(zeta[:, :, 0]).mean(0) * np.sqrt(np.pi / 2), color=INK2, lw=0.6, alpha=0.7,
               label="実際の |ζ| の平均 ×√(π/2)")
    ax[3].set_ylabel("探索の揺らぎ")
    ax[3].set_title("(d) 指令の中の探索の揺らぎ ζ の大きさ")
    ax[3].legend(loc="upper right")
    ax[3].set_xlabel("時間")
    for a in ax:
        seam(a, 200, 260, label=False)
        a.set_xlim(0, 600)
    fig.savefig(f"{OUT}/fig_stage1.png", bbox_inches="tight")

    fig, ax = plt.subplots(1, 1, figsize=(7.2, 2.6))
    ax.fill_between(t, q[0], q[2], color="#d1495b", alpha=0.2, lw=0)
    ax.plot(t, q[1], color="#d1495b", lw=1.0)
    ax.axhline(0, color=MUTED, lw=0.7)
    seam(ax, 200, 260, label=False)
    ax.set_xlim(0, 1500)
    ax.set_xlabel("時間")
    ax.set_ylabel("ΔL_self")
    ax.set_title("N0：ΔL_self(t) の全体（中央値と四分位、30 種）")
    fig.savefig(f"{OUT}/fig_stage1_full.png", bbox_inches="tight")
    print("saved")


if __name__ == "__main__":
    main()
