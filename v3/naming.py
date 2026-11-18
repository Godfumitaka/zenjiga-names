"""最小の命名ゲームの一回のやりとり（Baronchelli ら 2006）。

v3 の模型の「噂」（model.py）と、段 2 の単独の較正（stage2.py）が、同じこの関数を使う。
"""
from __future__ import annotations


def ng_interaction(Vs: set, Vl: set, rng, invent):
    """話し手の語彙 Vs、聞き手の語彙 Vl（その場で書き換える）。

    - 話し手は Vs から語を一つ無作為に選ぶ。Vs が空なら invent() で新しい語を作り、Vs に足して言う。
    - 聞き手が その語を持っていれば成功：二人とも語彙をその語だけにする。
    - 持っていなければ失敗：聞き手がその語を足す。
    返り値：(言った語, 成功したか, 話し手が新しく作ったか)
    """
    if not Vs:
        w = invent()
        Vs.add(w)
        invented = True
    else:
        vs = sorted(Vs)
        w = vs[int(rng.integers(len(vs)))]
        invented = False
    if w in Vl:
        Vs.clear()
        Vs.add(w)
        Vl.clear()
        Vl.add(w)
        return w, True, invented
    Vl.add(w)
    return w, False, invented
