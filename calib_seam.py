# 段階 B の前提だけを確かめる：継ぎ目のあいだに公の名前が付け直されるか（継ぎ目の終わりで止め、その後は見ない）
import numpy as np
def wrap(x): return (x + np.pi) % (2*np.pi) - np.pi
B = np.pi/4; ANG = wrap(np.arange(8)*B)
def nearest(x): return np.rint(wrap(x)/B).astype(int) % 8
def run(p, seam=30.0, R=60, N=4, rate=2.0, sig=0.1, g=1.5, s=0.5, K=1.0, eta=0.3, F=4.0, dt=0.01, seed=0):
    rng = np.random.default_rng(seed)
    om = 1.0 + np.array([-0.06, -0.02, 0.02, 0.06])
    a = rng.uniform(-1.6, 1.6, (R, N)); a -= a.mean(1, keepdims=True)
    th = rng.uniform(-0.2, 0.2, (R, N)); hat = np.zeros((R, N))
    inv = np.zeros((R, N, N, 8), bool); inv[:, :, :, 0] = True        # inv[run, j(名前の主), k(持ち手), 語]
    name = np.zeros((R, N), int)
    t1, t2 = 60.0, 60.0 + seam
    for n in range(int(t2/dt)):
        t = n*dt
        z = np.exp(1j*th).mean(1); r = np.abs(z); Th = np.angle(z); psi = wrap(th - Th[:, None])
        m = ANG[name]
        d = om + K*r[:, None]*np.sin(Th[:, None]-th) + g*np.sin(hat-psi)
        if t1 <= t < t2: d = d + F*np.sin(a-psi)
        hat = wrap(hat + dt*eta*((1-s)*wrap(psi-hat) + s*wrap(m-hat)))
        th = th + dt*d
        ev = rng.random((R, N)) < rate*dt
        for ri, j in zip(*np.nonzero(ev)):
            others = [k for k in range(N) if k != j]
            if rng.random() < p:                                    # 観測：本人のふるまいが言う
                k = rng.choice(others); w = nearest(psi[ri, j] + rng.normal(0, sig))
                if inv[ri, j, k, w]: inv[ri, j, k, :] = False
                inv[ri, j, k, w] = True
            else:                                                   # 噂：本人のいないところで
                k, l = rng.choice(others, 2, replace=False)
                words = np.nonzero(inv[ri, j, k])[0]
                w = rng.choice(words) if words.size else nearest(psi[ri, j] + rng.normal(0, sig))
                if inv[ri, j, l, w]:
                    inv[ri, j, k, :] = False; inv[ri, j, l, :] = False
                    inv[ri, j, k, w] = True; inv[ri, j, l, w] = True
                else:
                    inv[ri, j, l, w] = True
            cnt = inv[ri, j][[k for k in range(N) if k != j]].sum(0)
            best = np.flatnonzero(cnt == cnt.max())
            if name[ri, j] not in best: name[ri, j] = best[0]
    target = nearest(a)                                           # 継ぎ目で押しやられた先の刻み
    moved = target != 0
    flipped = (name != 0) & moved
    return flipped.sum()/moved.sum(), moved.sum()
for p in [0.02, 0.1, 0.5]:
    for seam in [10.0, 30.0]:
        f, n = run(p, seam)
        print(f"p={p:<4} 継ぎ目 {seam:>4.0f}: 押しやられた主体のうち、継ぎ目の終わりまでに公の名前が付け直された割合 {f:.2f}（{n} 人中）")
