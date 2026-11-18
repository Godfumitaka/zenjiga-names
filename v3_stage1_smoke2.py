# v3 段 1 の動作確認（名前なし）。基準線 R1・R2 が設計どおりに出るかだけを見る。
import numpy as np, itertools
def wrap(x): return (x + np.pi) % (2*np.pi) - np.pi
def run(seed=0, N=4, K=1.0, dt=0.05, T=600.0, t1=200.0, t2=260.0, dw=1.2, spread=0.0, thin=20,
        sphys=0.02, s0=0.1, ce=5.0, tau_e=5.0, tau_b=20.0, W=200, every=20):
    rng = np.random.default_rng(seed)
    om = 1.0 + spread*rng.standard_normal(N)
    perm = np.array([rng.permutation(N) for _ in range(N)])        # perm[k][c] = 通路 c に映る主体
    th = rng.uniform(-0.1, 0.1, N)
    m_u = np.zeros(N); m_d = np.zeros((N, N)); m_uu = np.full(N, 1e-6); m_dd = np.full((N, N), 1e-6); m_ud = np.zeros((N, N))
    chat = rng.integers(0, N, N)                                     # はじめの帰属は無作為
    ebar = np.zeros(N); hist = np.zeros((W, N)); hp = 0; filled = 0
    aprev = np.zeros((N, N))   # 直前の記述（各主体・各通路のずれ）
    subsets = [S for r in range(N+1) for S in itertools.combinations(range(N), r)]
    out = []
    a_b = dt/tau_b
    for n in range(int(T/dt)):
        t = n*dt
        z = np.exp(1j*th).mean(); Th = np.angle(z); phi = wrap(th - Th)
        hist[hp] = phi; hp = (hp+1) % W; filled = min(filled+1, W)
        mis = np.mean(wrap(phi[perm] - aprev)**2, axis=1)       # 直前の記述との食い違い
        ebar += (dt/tau_e)*(mis - ebar)
        u = np.zeros(N)
        for k in range(N):
            me = perm[k][chat[k]]
            others = [perm[k][c] for c in range(N) if c != chat[k]]
            u[k] = K/N*np.sum(np.sin(th[others] - th[me]))
        u += s0*np.sqrt(1 + ce*ebar)*rng.standard_normal(N)
        w = om.copy()
        if t1 <= t < t2: w[0] += dw
        dth = (w + u)*dt + sphys*np.sqrt(dt)*rng.standard_normal(N)
        th = th + dth
        # 出どころの区別：各主体 k が、各通路の増分と自分の指令の相関を追う
        d = dth[perm]                                                # d[k][c]
        m_u += a_b*(u - m_u); m_d += a_b*(d - m_d)
        m_uu += a_b*(u*u - m_uu); m_dd += a_b*(d*d - m_dd); m_ud += a_b*(u[:, None]*d - m_ud)
        cov = m_ud - m_u[:, None]*m_d; var = (m_uu - m_u**2)[:, None]*(m_dd - m_d**2)
        corr = cov/np.sqrt(np.maximum(var, 1e-12)); chat = np.argmax(corr, axis=1)
        if filled == W and n % every == 0:
            nn = (W//thin)*N
            for k in range(N):
                order = (np.arange(W) + hp) % W
                X = hist[order][::thin][:, perm[k]]                 # 間引いて通路の順に
                best = {}
                for S in subsets:
                    R = X.copy()
                    for c in S:
                        a = np.angle(np.exp(1j*X[:, c]).mean()); R[:, c] = wrap(X[:, c] - a)
                    s2 = np.mean(R**2) + 1e-12
                    best[S] = nn/2*np.log(s2) + (len(S)+1)/2*np.log(nn)
                Sstar = min(best, key=best.get)
                with_self = min(v for S, v in best.items() if chat[k] in S)
                without = min(v for S, v in best.items() if chat[k] not in S)
                R = X.copy()
                for c in Sstar:
                    a = np.angle(np.exp(1j*X[:, c]).mean()); R[:, c] = wrap(X[:, c] - a)
                aprev[k] = np.array([np.angle(np.exp(1j*X[:, c]).mean()) if c in Sstar else 0.0 for c in range(N)])
                if k == 0:
                    out.append((t, without - with_self, int(perm[0][chat[0]] == 0),
                                np.mean([perm[q][chat[q]] == q for q in range(N)]), ebar[0], abs(phi[0])))
    return np.array(out)
for seed in range(3):
    o = run(seed)
    def avg(a, b, col): m = (o[:, 0] >= a) & (o[:, 0] < b); return o[m, col].mean()
    print(f"種 {seed}: 帰属の正しさ（全員, t<100）={avg(0,100,3):.2f}  (t>100)={avg(100,600,3):.2f}")
    print(f"   ΔL_self  継ぎ目前[150,200)={avg(150,200,1):+.1f}  継ぎ目中[220,260)={avg(220,260,1):+.1f}  "
          f"直後[260,300)={avg(260,300,1):+.1f}  後[350,600)={avg(350,600,1):+.1f}")
    print(f"   不確かさ ē  前={avg(150,200,4):.4f} 中={avg(220,260,4):.4f} 後={avg(350,600,4):.4f}   |φ_j| 中={avg(220,260,5):.2f} 後={avg(350,600,5):.2f}")
