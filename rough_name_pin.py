#!/usr/bin/env python3
"""
雑な版：名前は自己像を通してだけ位相に効く（四体の位相模型に名前の層を足す試作）

記号
  θ_j   : j 番目の主体の位相（j = 1..N, N = 4）
  ω_j   : j の固有の速さ
  K     : 群れにそろう引き（結合の強さ）
  r, Θ  : 同期の度合いと群れの平均位相。 r e^{iΘ} = (1/N) Σ_k e^{iθ_k}
  ψ_j   : 群れの中での j の位置 = wrap(θ_j − Θ)（関係のなかの自分）
  h_j   : j の自己像「自分は群れのどこにいるか」（式では ψ̂_j）
  g     : 自己像どおりに動く強さ
  η     : 自己像を更新する速さ
  s     : 自己像が「他者がもつ印 m_j」に引かれる重み
  m_j   : 他者がもつ j の印。条件で中身だけが違う
            A  名前なし     : m_j = ψ_j（他者の印はなく、いまの自分がそのまま映る）
            Bc 連続の評判   : m_j = ρ_j、ρ_j は ψ_j を速さ μ で連続に追う
            B  名前あり     : m_j = L_j、離散の印。|ψ_j − L_j| > δ のときだけ、
                              幅 b の目盛りのいちばん近いものへ付け直す
  継ぎ目 : t1 ≤ t < t2 のあいだ、外から各自を群れの中の別々の位置 a_j へ押しやる（強さ F）

式
  dθ_j/dt = ω_j + K r sin(Θ − θ_j) + g sin(h_j − ψ_j) + F·[継ぎ目中]·sin(a_j − ψ_j)
  dh_j/dt = η [ (1 − s)(ψ_j − h_j) + s (m_j − h_j) ]
名前は θ の式に直接は入らず、h（自己像）を通してだけ効く。
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

N = 4
OMEGA = 1.0 + np.array([-0.06, -0.02, 0.02, 0.06])
P = dict(K=1.0, eta=0.3, s=0.5, mu=0.3, delta=0.6, binw=np.pi / 4,
         F=4.0, t1=60.0, t2=90.0, T=500.0, dt=0.01)


def wrap(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


def simulate(cond, g, a, rec_every=10, seed=1, **kw):
    p = dict(P, **kw)
    R = a.shape[0]
    rng = np.random.default_rng(seed)
    theta = rng.uniform(-0.2, 0.2, (R, N))
    hat = np.zeros((R, N))
    mark = np.zeros((R, N))  # Bc: ρ, B: L
    g = np.asarray(g, float).reshape(R, 1)
    steps = int(round(p["T"] / p["dt"]))
    nrec = steps // rec_every
    r_rec = np.empty((R, nrec))
    L_rec = np.empty((R, N, nrec)) if cond == "B" else None
    n_rename = np.zeros(R, int)
    dt = p["dt"]
    for n in range(steps):
        t = n * dt
        z = np.exp(1j * theta).mean(axis=1)
        r = np.abs(z)
        Th = np.angle(z)
        psi = wrap(theta - Th[:, None])
        if n % rec_every == 0:
            k = n // rec_every
            r_rec[:, k] = r
            if L_rec is not None:
                L_rec[:, :, k] = mark
        dtheta = (OMEGA + p["K"] * r[:, None] * np.sin(Th[:, None] - theta)
                  + g * np.sin(hat - psi))
        if p["t1"] <= t < p["t2"]:
            dtheta = dtheta + p["F"] * np.sin(a - psi)
        m = psi if cond == "A" else mark
        dhat = p["eta"] * ((1 - p["s"]) * wrap(psi - hat) + p["s"] * wrap(m - hat))
        if cond == "Bc":
            mark = wrap(mark + dt * p["mu"] * wrap(psi - mark))
        elif cond == "B":
            jump = np.abs(wrap(psi - mark)) > p["delta"]
            if jump.any():
                mark = np.where(jump, wrap(np.round(psi / p["binw"]) * p["binw"]), mark)
                n_rename += jump.sum(axis=1)
        theta = theta + dt * dtheta
        hat = wrap(hat + dt * dhat)
    tt = np.arange(nrec) * rec_every * dt
    return tt, r_rec, L_rec, n_rename, wrap(theta - np.angle(np.exp(1j * theta).mean(1))[:, None])


def deficit(tt, r_rec, T):
    pre = r_rec[:, (tt >= 40) & (tt < 60)].mean(axis=1)
    post = r_rec[:, tt >= T - 50].mean(axis=1)
    return pre, post, pre - post


if __name__ == "__main__":
    T = P["T"]
    # ---- 一回の例（同じ継ぎ目で三条件を比べる）
    a_main = np.array([[-1.4, -0.5, 0.4, 1.5]])
    g_main = 1.5
    main = {}
    for c in ["A", "Bc", "B"]:
        tt, rr, LL, nr, psi_end = simulate(c, [g_main], a_main)
        pre, post, d = deficit(tt, rr, T)
        main[c] = (tt, rr[0], LL, nr, psi_end)
        rec = tt[(tt > P["t2"]) & (np.abs(rr[0] - pre[0]) < 0.005)]
        print(f"[main g={g_main}] {c:2s} r_pre={pre[0]:.4f} r_post={post[0]:.4f} "
              f"deficit={d[0]:.4f} renames={nr[0]} first-recovered-t="
              f"{rec[0] if rec.size else None} psi_end={np.round(psi_end[0], 3)}")
    LL = main["B"][2][0]
    tt = main["B"][0]
    for j in range(N):
        changes = np.where(np.diff(LL[j]) != 0)[0]
        print(f"  B name of agent {j}: ",
              [(round(tt[i + 1], 1), round(LL[j, i + 1] / (np.pi / 4))) for i in changes],
              "(units of pi/4)")

    # ---- g を動かす（30 通りの継ぎ目の平均）
    g_vals = np.linspace(0, 3, 13)
    S = 30
    rng = np.random.default_rng(7)
    a_s = rng.uniform(-1.6, 1.6, (S, N))
    a_s -= a_s.mean(axis=1, keepdims=True)
    A_rows = np.tile(a_s, (len(g_vals), 1))
    G_rows = np.repeat(g_vals, S)
    sweep = {}
    for c in ["A", "Bc", "B"]:
        tt2, rr2, _, nr2, _ = simulate(c, G_rows, A_rows, rec_every=50)
        _, _, d2 = deficit(tt2, rr2, T)
        sweep[c] = d2.reshape(len(g_vals), S)
        print(f"[sweep] {c:2s} mean deficit by g:", np.round(sweep[c].mean(1), 4))
        print(f"        {c:2s} max |deficit| :", np.round(np.abs(sweep[c]).max(), 4))
    # 純粋な丸め（付け直しの幅を目盛りの半分にする）でも粘るか
    tt3, rr3, _, _, _ = simulate("B", G_rows, A_rows, rec_every=50, delta=np.pi / 8 + 1e-9)
    _, _, d3 = deficit(tt3, rr3, T)
    print("[sweep] B (delta = half bin) mean deficit by g:",
          np.round(d3.reshape(len(g_vals), S).mean(1), 4))
    np.savez("/home/claude/rough_name_pin_results.npz", g_vals=g_vals,
             **{f"sweep_{k}": v for k, v in sweep.items()},
             sweep_B_halfbin=d3.reshape(len(g_vals), S),
             **{f"main_r_{k}": v[1] for k, v in main.items()}, main_t=main["A"][0])

    # ---- 図
    SURF, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
    GRID, AXIS = "#e1e0d9", "#c3c2b7"
    COL = {"A": "#2a78d6", "B": "#eb6834", "Bc": "#1baf7a"}
    LS = {"A": "-", "B": (0, (5, 2.5)), "Bc": (0, (1.2, 2.0))}
    NAME = {"A": "名前なし", "Bc": "連続の評判", "B": "名前あり（離散の印）"}
    plt.rcParams.update({
        "font.family": "Noto Sans CJK JP", "font.size": 9.5,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.facecolor": SURF,
        "figure.facecolor": SURF, "axes.spines.top": False, "axes.spines.right": False,
    })
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.2, 7.8), dpi=200,
                                   gridspec_kw={"height_ratios": [1.15, 1], "hspace": 0.55})
    xmax = 300
    ax1.axvspan(P["t1"], P["t2"], color=GRID, lw=0, zorder=0)
    ax1.text((P["t1"] + P["t2"]) / 2, 0.90, "継ぎ目", ha="center", va="center",
             color=INK2, fontsize=9)
    for c in ["A", "Bc", "B"]:
        tt, rr = main[c][0], main[c][1]
        m = tt <= xmax
        ax1.plot(tt[m], rr[m], color=COL[c], ls=LS[c], lw=1.8, label=NAME[c], zorder=3)
    rB = main["B"][1][main["B"][0] <= xmax][-1]
    ax1.text(xmax - 4, 1.004, "名前なし・連続の評判（元まで戻る）", color=INK2,
             ha="right", va="bottom", fontsize=8.8)
    ax1.text(xmax - 4, rB - 0.006, "名前あり（戻りきらない）", color=INK2,
             ha="right", va="top", fontsize=8.8)
    ax1.set_xlim(0, xmax)
    ax1.set_ylim(0.55, 1.035)
    ax1.set_xlabel("時間")
    ax1.set_ylabel("同期の度合い r")
    ax1.set_title(f"一回の例：継ぎ目の前後の r（g = {g_main}）", loc="left", fontsize=10.5)
    ax1.grid(axis="y", color=GRID, lw=0.6)
    ax1.legend(loc="lower right", frameon=False, fontsize=8.5, labelcolor=INK2)

    for c in ["A", "Bc", "B"]:
        ax2.plot(g_vals, sweep[c].mean(1), color=COL[c], ls=LS[c], lw=1.8,
                 marker="o", ms=4.5, mec=SURF, mew=1.0, label=NAME[c], zorder=3)
    ax2.axvline(g_main, color=AXIS, lw=0.8, zorder=1)
    ax2.text(3.08, sweep["B"].mean(1)[-1], "名前あり", color=INK2, va="center", fontsize=9)
    ax2.text(3.08, 0.0, "名前なし・\n連続の評判", color=INK2, va="center", fontsize=9)
    ax2.set_xlim(0, 3)
    ax2.set_xlabel("自己像どおりに動く強さ g")
    ax2.set_ylabel("戻りきらなさ（前の r − 終わりの r）")
    ax2.set_title("g を変えたとき（継ぎ目 30 通りの平均）", loc="left", fontsize=10.5)
    ax2.grid(axis="y", color=GRID, lw=0.6)
    fig.savefig("/home/claude/rough_name_pin.png", bbox_inches="tight", facecolor=SURF)
    print("saved")
