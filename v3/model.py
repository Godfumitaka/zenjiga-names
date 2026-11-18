"""v3 の模型（SPEC_v3_full_model.md §2〜§5）。

世界（§2）
  dθ_k = (ω_k + u_k) dt + σ_phys dW_k。世界の側に結合はない。
  継ぎ目：t1 ≤ t < t2 のあいだ、焦点の主体 j の ω を Δω ずらす。

学習者（§3。各主体 k）
  受け取るもの：全員の位相を、主体ごとに固定した無作為な並べ替え perm[k] の順の通路で見る
  （perm[k][c] = 通路 c に映る主体）。k は perm を知らない。学習者の計算に入るのは
  「通路の順に並んだ位置」と「自分の指令の写し」だけで、perm は世界の側（どの通路に誰が
  映るか）を作るのにだけ使う。
  出どころの区別（§3.2）：通路ごとの増分と指令の写しの相関 ρ_c（時定数 τ_b）。ĉ_self = argmax。
  指令（§3.3）：u_k = K/N Σ_{c≠ĉ} sin(θ_c − θ_ĉ − κ x_k(c)) + ζ_k、ζ_k ~ N(0, σ0²(1+c_e ē_k))。
  記述長（§3.4）：直近 W 歩の位置を thin 歩おきに間引き、2^N 個の型 M_S を比べる。

名前（§4）と条件（§5）
  N0：x ≡ 0。N1：噂（本人不在）と観測。N2：噂なし。M：連続の記録。S：N1 で噂を止めて戻す。
  名前が効くのは、他の人の指令の x_k(j) を通してだけ。本人 j は自分についての語彙を持たない
  （x_j(j) ≡ 0）。

乱数：世界と学習者は rng = default_rng(seed)（v3_stage1_smoke2.py と同じ順で引く）。
  名前の出来事は rng_n = default_rng([seed, 1])、分析用の同点の割り振りは rng_a = default_rng([seed, 2])。
  そのため N0 では v3_stage1_smoke2.py と同じ乱数列になる。
"""
from __future__ import annotations

import itertools
from collections import Counter
from dataclasses import dataclass, asdict

import numpy as np

from .naming import ng_interaction


def wrap(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


@dataclass
class Config:
    seed: int = 0
    # 世界（§2）
    N: int = 4
    K: float = 1.0
    dt: float = 0.05
    T: float = 1500.0
    t1: float = 200.0
    t2: float = 260.0
    dw: float = 1.2
    focal: int = 0
    spread: float = 0.0
    sphys: float = 0.02
    topology: str = "all"  # "all"（全員がつながる）| "ring"（隣の二人とだけ引き合う）
    # 学習者（§3）
    s0: float = 0.1
    ce: float = 5.0
    tau_e: float = 5.0
    tau_b: float = 20.0
    W: int = 200
    thin: int = 20
    every: int = 20
    # 名前（§4）と条件（§5）
    cond: str = "N0"  # N0 | N1 | N2 | M | S
    kappa: float = 1.0
    rate: float = 2.0  # ν + λ（各話題 j について単位時間あたり）
    p: float = 0.1  # λ / (ν + λ)
    sobs: float = 0.1
    binw: float = np.pi / 4
    m_alpha: float = 0.5  # M の観測での更新の重み
    s_stop: float = 600.0  # S：噂を止める時刻
    s_restore: float = 1000.0  # S：噂を戻す時刻
    name_all_topics: bool = True  # 【Claude の選択】全主体が話題になる（焦点の j だけではない）
    # 診断用（模型の一部ではない。段 1 の確かめでだけ False にする）
    zeta_in_copy: bool = True  # False：ζ を世界には入れるが、指令の写しには入れない
    copy_shift: int = 0  # ≠0：主体 k に、主体 (k+shift) の指令を写しとして渡す（帰属が写しだけで決まることの確かめ）

    def to_dict(self):
        return asdict(self)


def subsets_of(N):
    return [S for r in range(N + 1) for S in itertools.combinations(range(N), r)]


def simulate(cfg: Config):
    c = cfg
    N = c.N
    rng = np.random.default_rng(c.seed)
    rng_n = np.random.default_rng([c.seed, 1])
    rng_a = np.random.default_rng([c.seed, 2])

    # ---- 世界と学習者の初期状態（v3_stage1_smoke2.py と同じ順で乱数を引く）
    om = 1.0 + c.spread * rng.standard_normal(N)
    perm = np.array([rng.permutation(N) for _ in range(N)])  # perm[k][c] = 通路 c に映る主体（世界の側）
    th = rng.uniform(-0.1, 0.1, N)
    m_u = np.zeros(N)
    m_d = np.zeros((N, N))
    m_uu = np.full(N, 1e-6)
    m_dd = np.full((N, N), 1e-6)
    m_ud = np.zeros((N, N))
    chat = rng.integers(0, N, N)  # はじめの帰属は無作為
    ebar = np.zeros(N)
    W = c.W
    hist = np.zeros((W, N))
    hp = 0
    filled = 0
    aprev = np.zeros((N, N))  # 直前の記述（各主体・各通路のずれ）
    subsets = subsets_of(N)
    Mm = np.array([[1.0 if cc in S else 0.0 for cc in range(N)] for S in subsets])
    sizes = Mm.sum(axis=1)
    a_b = c.dt / c.tau_b
    nb = int(round(2 * np.pi / c.binw))

    # 引き合う相手（世界の側のつながり）。全員：自分以外の全員。輪：隣の二人。
    if c.topology == "all":
        nbr = [set(range(N)) for _ in range(N)]  # 帰属した通路を除く全通路（smoke2 と同じ）
    elif c.topology == "ring":
        nbr = [{(k - 1) % N, (k + 1) % N} for k in range(N)]
    else:
        raise ValueError(c.topology)
    # 通路 c が引き合いの相手かどうか（k から見て）。世界の側の構造で、自他の区別は含まない。
    coupled = np.array([[perm[k][cc] in nbr[k] for cc in range(N)] for k in range(N)])

    # ---- 名前
    words = c.cond in ("N1", "N2", "S")
    memory = c.cond == "M"
    topics = list(range(N)) if c.name_all_topics else [c.focal]
    vocab = [[set() for _ in range(N)] for _ in range(N)]  # vocab[k][j]：k が j について持つ語
    cur = -np.ones((N, N), dtype=np.int64)  # k が j について最後に使った／受け入れた語
    wcontent: list[int] = []  # 語の内容（刻みの番号 0..nb-1）
    xmat = np.zeros((N, N))  # x_k(j)：k が j に期待するずれ（角度）
    rmem = np.zeros((N, N))  # M：連続の見積もり
    public = -np.ones(N, dtype=np.int64)  # 公の名前（分析用）
    n_public_change = np.zeros(N, dtype=np.int64)
    renamed_in_seam = np.zeros(N, dtype=bool)  # 継ぎ目のあいだに公の名前の内容が 0 以外になったか
    n_events = {"gossip": 0, "obs": 0}

    def ang(b):
        return wrap(b * c.binw)

    def obs_bin(j, phi):
        return int(np.rint(wrap(phi[j] + c.sobs * rng_n.standard_normal()) / c.binw)) % nb

    def new_word(b):
        wcontent.append(b)
        return len(wcontent) - 1

    def set_x(k, j):
        xmat[k, j] = ang(wcontent[cur[k, j]]) if cur[k, j] >= 0 else 0.0

    def update_public(j, t):
        if memory:
            return
        cnt = Counter()
        for k in range(N):
            if k != j:
                cnt.update(vocab[k][j])
        if not cnt:
            return
        mx = max(cnt.values())
        tied = sorted(w for w, v in cnt.items() if v == mx)
        if public[j] in tied:
            return
        new = tied[int(rng_a.integers(len(tied)))] if len(tied) > 1 else tied[0]
        if public[j] >= 0 and wcontent[new] != wcontent[public[j]]:
            n_public_change[j] += 1
        public[j] = new
        if c.t1 <= t < c.t2 and wcontent[new] != 0:
            renamed_in_seam[j] = True

    def gossip(j, phi):
        others = [k for k in range(N) if k != j]
        k, l = rng_n.choice(others, 2, replace=False)
        if memory:  # 公共の記録：二人が円平均を取り合う
            m = np.angle(np.exp(1j * rmem[k, j]) + np.exp(1j * rmem[l, j]))
            rmem[k, j] = rmem[l, j] = m
            xmat[k, j] = xmat[l, j] = m
            return
        # 最小の命名ゲーム（naming.py。段 2 の較正と同じ関数）。語彙が空の話し手は、自分の観測した刻みで語を作る。
        w, ok, _ = ng_interaction(vocab[k][j], vocab[l][j], rng_n, lambda: new_word(obs_bin(j, phi)))
        cur[k, j] = w  # 話し手：言った語
        set_x(k, j)
        if ok:  # 聞き手：成功したときだけ受け入れる
            cur[l, j] = w
            set_x(l, j)

    def observe(j, phi):
        others = [k for k in range(N) if k != j]
        k = others[int(rng_n.integers(len(others)))]
        if memory:
            xi = c.sobs * rng_n.standard_normal()
            rmem[k, j] = wrap(rmem[k, j] + c.m_alpha * wrap(phi[j] + xi - rmem[k, j]))
            xmat[k, j] = rmem[k, j]
            return
        b = obs_bin(j, phi)
        if cur[k, j] >= 0 and wcontent[cur[k, j]] == b:  # 確認
            vocab[k][j] = {cur[k, j]}
        else:  # 刻みが違う：内容 b の新しい語を作って足す
            w = new_word(b)
            vocab[k][j].add(w)
            cur[k, j] = w
            set_x(k, j)

    # ---- 記録
    steps = int(round(c.T / c.dt))
    acc_t, acc_ok, acc_who = [], [], []
    recs = {key: [] for key in ("t", "dL", "Sstar", "chat_ok", "ebar", "phi", "zeta", "udet",
                                "pub_word", "pub_content", "modal_content", "x_to", "mem_pub",
                                "vocab_focal", "r")}

    for n in range(steps):
        t = n * c.dt
        z = np.exp(1j * th).mean()
        Th = np.angle(z)
        phi = wrap(th - Th)
        hist[hp] = phi
        hp = (hp + 1) % W
        filled = min(filled + 1, W)
        mis = np.mean(wrap(phi[perm] - aprev) ** 2, axis=1)  # 直前の記述との食い違い
        ebar += (c.dt / c.tau_e) * (mis - ebar)

        # 名前の出来事（世界の側。本人には何も届かない）
        if words or memory:
            gossip_on = not (c.cond == "N2" or (c.cond == "S" and c.s_stop <= t < c.s_restore))
            nu = (1 - c.p) * c.rate if gossip_on else 0.0
            lam = c.p * c.rate
            tot = nu + lam
            if tot > 0:
                nev = rng_n.poisson(tot * c.dt, size=len(topics))
                for ti, j in enumerate(topics):
                    if nev[ti] == 0:
                        continue
                    for _ in range(nev[ti]):
                        if rng_n.random() < lam / tot:
                            observe(j, phi)
                            n_events["obs"] += 1
                        else:
                            gossip(j, phi)
                            n_events["gossip"] += 1
                    update_public(j, t)

        # 指令（§3.3）
        kx = c.kappa * xmat  # kx[k, 主体]
        np.fill_diagonal(kx, 0.0)  # 本人についての語彙は持たない（念のため）
        u = np.zeros(N)
        for k in range(N):
            me = perm[k][chat[k]]
            ch = [cc for cc in range(N) if cc != chat[k] and coupled[k][cc]]
            others = perm[k][ch]
            u[k] = c.K / N * np.sum(np.sin(th[others] - th[me] - kx[k, others]))
        udet = u.copy()
        zeta = c.s0 * np.sqrt(1 + c.ce * ebar) * rng.standard_normal(N)
        if c.zeta_in_copy:
            u += zeta
            u_world = u
        else:
            u_world = u + zeta
        w = om.copy()
        if c.t1 <= t < c.t2:
            w[c.focal] += c.dw
        dth = (w + u_world) * c.dt + c.sphys * np.sqrt(c.dt) * rng.standard_normal(N)
        th = th + dth

        # 出どころの区別（§3.2）：各通路の増分と、自分の指令の写しの相関
        if c.copy_shift:
            u = np.roll(u, -c.copy_shift)  # 診断用：他人の指令を写しとして渡す
        d = dth[perm]  # d[k][c]
        m_u += a_b * (u - m_u)
        m_d += a_b * (d - m_d)
        m_uu += a_b * (u * u - m_uu)
        m_dd += a_b * (d * d - m_dd)
        m_ud += a_b * (u[:, None] * d - m_ud)
        cov = m_ud - m_u[:, None] * m_d
        var = (m_uu - m_u ** 2)[:, None] * (m_dd - m_d ** 2)
        corr = cov / np.sqrt(np.maximum(var, 1e-12))
        chat = np.argmax(corr, axis=1)
        if n % c.every == 0:  # 帰属の正しさは最初から記録する（R1 用）
            acc_t.append(t)
            acc_ok.append([perm[q][chat[q]] == q for q in range(N)])
            acc_who.append([perm[q][chat[q]] for q in range(N)])  # 帰属した通路に映っている主体（分析用）

        # 記述長（§3.4）
        if filled == W and n % c.every == 0:
            nn = (W // c.thin) * N
            order = (np.arange(W) + hp) % W
            Xall = hist[order][:: c.thin]
            dL = np.zeros(N)
            sstar_focal = 0
            for k in range(N):
                X = Xall[:, perm[k]]  # 間引いて通路の順に（k に見えるのはこれだけ）
                a = np.array([np.angle(np.exp(1j * X[:, cc]).mean()) for cc in range(N)])
                rin = np.mean(wrap(X - a) ** 2, axis=0)
                rout = np.mean(X ** 2, axis=0)
                s2 = (Mm @ rin + (1 - Mm) @ rout) / N + 1e-12
                L = nn / 2 * np.log(s2) + (sizes + 1) / 2 * np.log(nn)
                si = int(np.argmin(L))
                has = Mm[:, chat[k]] > 0  # 帰属した通路 ĉ_self を含む型
                dL[k] = L[~has].min() - L[has].min()
                aprev[k] = a * Mm[si]
                if k == c.focal:
                    # 記録は主体の番号で（分析用）：S* に入った通路を主体に戻す
                    sstar_focal = int(sum(1 << int(perm[k][cc]) for cc in subsets[si]))
            j = c.focal
            recs["t"].append(t)
            recs["dL"].append(dL)
            recs["Sstar"].append(sstar_focal)
            recs["chat_ok"].append(np.array([perm[q][chat[q]] == q for q in range(N)]))
            recs["ebar"].append(ebar.copy())
            recs["phi"].append(phi.copy())
            recs["zeta"].append(zeta.copy())
            recs["udet"].append(udet.copy())
            recs["pub_word"].append(public.copy())
            recs["pub_content"].append(np.array([wcontent[w_] if w_ >= 0 else -1 for w_ in public]))
            mc = []
            for jj in range(N):
                cs = [wcontent[cur[k, jj]] for k in range(N) if k != jj and cur[k, jj] >= 0]
                mc.append(Counter(cs).most_common(1)[0][0] if cs else -1)
            recs["modal_content"].append(np.array(mc))
            xt = np.array([np.angle(np.mean(np.exp(1j * xmat[[k for k in range(N) if k != jj], jj])))
                           for jj in range(N)])
            recs["x_to"].append(xt)
            recs["mem_pub"].append(np.array([np.angle(np.mean(np.exp(1j * rmem[[k for k in range(N) if k != jj], jj])))
                                             for jj in range(N)]))
            recs["vocab_focal"].append(sum(len(vocab[k][j]) for k in range(N) if k != j))
            recs["r"].append(abs(z))

    out = {k: np.asarray(v) for k, v in recs.items()}
    out["acc_t"] = np.asarray(acc_t)
    out["acc_ok"] = np.asarray(acc_ok)
    out["acc_who"] = np.asarray(acc_who)
    out["perm"] = perm
    out["n_public_change"] = n_public_change
    out["renamed_in_seam"] = renamed_in_seam
    out["n_gossip"] = np.array(n_events["gossip"])
    out["n_obs"] = np.array(n_events["obs"])
    out["n_words"] = np.array(len(wcontent))
    return out
