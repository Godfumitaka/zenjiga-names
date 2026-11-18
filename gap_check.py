# 雑な版の最後の状態で、自己像のずれ |ψ̂_j − ψ_j|（ギャップ）を確かめる
import numpy as np
from rough_name_pin import OMEGA, P, wrap, N

def run(cond, g, delta=0.6, T=500.0, dt=0.01, seed=1):
    p = dict(P); a = np.array([-1.4, -0.5, 0.4, 1.5])
    rng = np.random.default_rng(seed)
    th = rng.uniform(-0.2, 0.2, N); hat = np.zeros(N); mark = np.zeros(N)
    for n in range(int(T / dt)):
        t = n * dt
        z = np.exp(1j * th).mean(); r = abs(z); Th = np.angle(z)
        psi = wrap(th - Th)
        d = OMEGA + p["K"] * r * np.sin(Th - th) + g * np.sin(hat - psi)
        if p["t1"] <= t < p["t2"]:
            d = d + p["F"] * np.sin(a - psi)
        m = psi if cond == "A" else mark
        dh = p["eta"] * ((1 - p["s"]) * wrap(psi - hat) + p["s"] * wrap(m - hat))
        if cond == "Bc":
            mark = wrap(mark + dt * p["mu"] * wrap(psi - mark))
        elif cond == "B":
            j = np.abs(wrap(psi - mark)) > delta
            mark = np.where(j, wrap(np.round(psi / p["binw"]) * p["binw"]), mark)
        th = th + dt * d; hat = wrap(hat + dt * dh)
    return r, psi, wrap(hat - psi)

for g in [1.0, 1.5, 2.5]:
    for cond, dl, name in [("A", 0.6, "名前なし"), ("Bc", 0.6, "連続の評判"),
                           ("B", np.pi/8 + 1e-9, "離散・丸めだけ"), ("B", 0.6, "離散・δ=0.6")]:
        r, psi, gap = run(cond, g, delta=dl)
        print(f"g={g} {name:8s} r={r:.4f} |psi|max={np.abs(psi).max():.3f} gap(rad)={np.round(np.abs(gap),3)}")
