"""v2 の模型（SPEC_name_layer.md §2〜§4）。R 本の走りを並べて一度に回す。

  dθ_j/dt = ω_j + K r sin(Θ − θ_j) + g sin(ψ̂_j − ψ_j) + F·1[t1≤t<t2] sin(a_j − ψ_j)
  dψ̂_j/dt = η [ (1 − s)(ψ_j − ψ̂_j) + s (m_j − ψ̂_j) ]
角度の差はすべて wrap。名前 m_j は θ の式に直接入らず、自己像の式にだけ入る。
本人の自己像 ψ̂_j は名前の更新に使わない（名前の更新に入るのは他の人の観測した ψ_j だけ）。

条件
  段階 A（v1 の条件。gap_check.py と同じ）：
    none   名前なし        m_j = ψ_j
    cont   連続の評判      dρ_j/dt = μ (ψ_j − ρ_j)、m_j = ρ_j
    round  丸めだけ        m_j = ψ_j に最寄りの刻み（δ = 刻みの半分 + 1e-9 の付け直しと同じ）
    delta  付け直しにくい  |ψ_j − m_j| > δ のときだけ最寄りの刻みへ
  段階 B（§4。名前を命名ゲームから生ませる。δ は使わない）：
    D3  離散・噂あり   D0  離散・噂なし（ν=0）
    C3  連続・噂あり   C0  連続・噂なし   C3n 連続・噂あり・近いときだけ合わせる

乱数：段階 B の D3 では calib_seam.py と同じ順に引く（継ぎ目の行き先 a、初期位相、各歩の出来事）。
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

B = np.pi / 4


def wrap(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


ANG = wrap(np.arange(8) * B)


def nearest(x):
    return np.rint(wrap(x) / B).astype(int) % 8


@dataclass
class Config:
    cond: str = "D3"
    seed: int = 0
    R: int = 30  # 並べて回す走りの数（種一つで R 本）
    N: int = 4
    K: float = 1.0
    eta: float = 0.3
    s: float = 0.5
    g: float = 1.5
    F: float = 4.0
    t1: float = 60.0
    seam: float = 30.0
    T: float = 1000.0
    dt: float = 0.01
    omega_spread: float = 0.06  # ω = 1 + linspace(−0.06, 0.06, N)（N=4 で既定の (−0.06, −0.02, 0.02, 0.06)）
    rate: float = 2.0  # ν + λ
    p: float = 0.1
    sobs: float = 0.1
    alpha: float = 0.5  # 連続の観測の重み
    eps: float = np.pi / 4  # C3n の幅
    mu: float = 0.3  # 段階 A の連続の評判
    delta: float = 0.6  # 段階 A の付け直しの幅
    a_fixed: tuple | None = None  # 段階 A：gap_check.py の固定の行き先
    init_spread: float = 0.2
    rec_every: int = 100  # 記録の間隔（歩）

    def to_dict(self):
        d = asdict(self)
        return d


def simulate(c: Config):
    R, N = c.R, c.N
    rng = np.random.default_rng(c.seed)
    om = 1.0 + np.linspace(-c.omega_spread, c.omega_spread, N)
    if c.a_fixed is not None:
        a = np.tile(np.asarray(c.a_fixed, float), (R, 1))
    else:
        a = rng.uniform(-1.6, 1.6, (R, N))
        a -= a.mean(1, keepdims=True)
    th = rng.uniform(-c.init_spread, c.init_spread, (R, N))
    hat = np.zeros((R, N))
    t1, t2 = c.t1, c.t1 + c.seam
    cond = c.cond
    discrete = cond in ("D3", "D0")
    contin = cond in ("C3", "C0", "C3n")
    gossip_on = cond in ("D3", "C3", "C3n")
    # 名前の状態
    inv = np.zeros((R, N, N, 8), bool)  # inv[走り, j(名前の主), k(持ち手), 語]
    inv[:, :, :, 0] = True
    name = np.zeros((R, N), int)  # 公の名前（刻みの番号）
    n_rename = np.zeros((R, N), int)
    rho = np.zeros((R, N, N))  # 連続：rho[走り, j, k]
    mark = np.zeros((R, N))  # 段階 A の cont / delta / round
    m_cont = np.zeros((R, N))  # 連続：公の記録（円平均）
    steps = int(round(c.T / c.dt))
    nrec = steps // c.rec_every
    rec = {"t": np.zeros(nrec), "r": np.zeros((nrec, R)), "psi": np.zeros((nrec, R, N)),
           "hat": np.zeros((nrec, R, N)), "m": np.zeros((nrec, R, N)), "name": np.zeros((nrec, R, N), np.int8)}
    others_of = [[k for k in range(N) if k != j] for j in range(N)]
    lam_frac = c.p
    ev_rate = c.rate if gossip_on else c.rate * c.p  # 噂なしの条件では観測（割合 λ）だけ
    if not gossip_on:
        lam_frac = 1.0

    for n in range(steps):
        t = n * c.dt
        z = np.exp(1j * th).mean(1)
        r = np.abs(z)
        Th = np.angle(z)
        psi = wrap(th - Th[:, None])
        if cond == "none":
            m = psi
        elif cond in ("cont", "delta", "round"):
            m = mark
        elif discrete:
            m = ANG[name]
        else:
            m = m_cont
        if n % c.rec_every == 0:
            q = n // c.rec_every
            rec["t"][q] = t
            rec["r"][q] = r
            rec["psi"][q] = psi
            rec["hat"][q] = hat
            rec["m"][q] = m
            rec["name"][q] = name
        d = om + c.K * r[:, None] * np.sin(Th[:, None] - th) + c.g * np.sin(hat - psi)
        if t1 <= t < t2:
            d = d + c.F * np.sin(a - psi)
        dh = c.eta * ((1 - c.s) * wrap(psi - hat) + c.s * wrap(m - hat))
        # 段階 A の印の更新（ψ は歩の初めの値）
        if cond == "cont":
            mark = wrap(mark + c.dt * c.mu * wrap(psi - mark))
        elif cond in ("delta", "round"):
            dl = c.delta if cond == "delta" else B / 2 + 1e-9
            jump = np.abs(wrap(psi - mark)) > dl
            if jump.any():
                mark = np.where(jump, wrap(np.round(psi / B) * B), mark)
                n_rename += jump
        hat = wrap(hat + c.dt * dh)
        th = th + c.dt * d
        # 段階 B の出来事（calib_seam.py と同じ順で乱数を引く）
        if discrete or contin:
            ev = rng.random((R, N)) < ev_rate * c.dt
            if ev.any():
                for ri, j in zip(*np.nonzero(ev)):
                    others = others_of[j]
                    if rng.random() < lam_frac:  # 観測：本人のふるまいを見る
                        k = rng.choice(others)
                        obs = psi[ri, j] + rng.normal(0, c.sobs)
                        if discrete:
                            w = nearest(obs)
                            if inv[ri, j, k, w]:
                                inv[ri, j, k, :] = False
                            inv[ri, j, k, w] = True
                        else:
                            diff = wrap(obs - rho[ri, j, k])
                            if cond != "C3n" or abs(diff) < c.eps:
                                rho[ri, j, k] = wrap(rho[ri, j, k] + c.alpha * diff)
                    else:  # 噂：本人のいないところで
                        k, l = rng.choice(others, 2, replace=False)
                        if discrete:
                            words = np.nonzero(inv[ri, j, k])[0]
                            w = rng.choice(words) if words.size else nearest(psi[ri, j] + rng.normal(0, c.sobs))
                            if inv[ri, j, l, w]:
                                inv[ri, j, k, :] = False
                                inv[ri, j, l, :] = False
                                inv[ri, j, k, w] = True
                                inv[ri, j, l, w] = True
                            else:
                                inv[ri, j, l, w] = True
                        else:
                            if cond != "C3n" or abs(wrap(rho[ri, j, k] - rho[ri, j, l])) < c.eps:
                                mm = np.angle(np.exp(1j * rho[ri, j, k]) + np.exp(1j * rho[ri, j, l]))
                                rho[ri, j, k] = rho[ri, j, l] = mm
                    if discrete:
                        cnt = inv[ri, j][others].sum(0)
                        best = np.flatnonzero(cnt == cnt.max())
                        if name[ri, j] not in best:
                            name[ri, j] = best[0]
                            n_rename[ri, j] += 1
                    else:
                        m_cont[ri, j] = np.angle(np.mean(np.exp(1j * rho[ri, j, others])))
    out = {k: v for k, v in rec.items()}
    out["a"] = a
    out["n_rename"] = n_rename
    out["final_psi"] = psi
    out["final_hat"] = hat
    out["final_m"] = m
    out["final_name"] = name
    return out
