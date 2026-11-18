"""走りごとの量（§3.4 の立つ・沈む・残る、§4 の名前、§6 の R1〜R7 に使う量）。

すべて焦点の主体 j（cfg.focal）の学習者から見た量。ΔL_self は j 自身の記述で、
帰属した通路 ĉ_self を使って計算されている（model.py）。
"""
from __future__ import annotations

import numpy as np


def true_bin(phi, binw=np.pi / 4):
    nb = int(round(2 * np.pi / binw))
    return np.rint(((phi + np.pi) % (2 * np.pi) - np.pi) / binw).astype(int) % nb


def run_summary(st, i, cfg):
    """st：stack された結果、i：走りの番号、cfg：その走りの設定（dict）。"""
    j = cfg["focal"]
    t1, t2, T = cfg["t1"], cfg["t2"], cfg["T"]
    t = st["t"][i].astype(float)
    dL = st["dL"][i][:, j].astype(float)
    out = {"seed": int(cfg["seed"]), "cond": cfg["cond"]}

    def win(a, b):
        return (t >= a) & (t < b)

    out["dL_pre"] = float(dL[win(150, t1)].mean())
    out["dL_pre_all"] = float(dL[win(10, t1)].mean())
    out["frac_pos_pre"] = float((dL[win(10, t1)] > 0).mean())
    out["dL_seam"] = float(dL[win(t1, t2)].mean())
    out["dL_seam_late"] = float(dL[win(t1 + 20, t2)].mean())
    out["frac_pos_seam"] = float((dL[win(t1, t2)] > 0).mean())
    out["stood"] = bool(out["dL_seam"] > 0)
    out["dL_just_after"] = float(dL[win(t2, t2 + 40)].mean())
    out["dL_after"] = float(dL[win(t2 + 90, min(600, T))].mean()) if T > t2 + 90 else float("nan")
    out["dL_late"] = float(dL[win(T - 500, T)].mean())
    # 持続時間：t2 から ΔL_self ≤ 0 に戻るまで（文字どおり、最初に ≤0 になった評価の時刻）
    post = t >= t2
    idx = np.nonzero(post & (dL <= 0))[0]
    if idx.size:
        out["persist"] = float(t[idx[0]] - t2)
        out["censored"] = False
    else:
        out["persist"] = float(T - t2)
        out["censored"] = True
    out["sank"] = not out["censored"]
    # 頑健さ用：≤0 が 20 の時間続いた最初の時点まで
    ev = cfg["every"] * cfg["dt"]
    need = int(round(20 / ev))
    neg = (dL <= 0) & post
    run_len = 0
    out["persist_sustained"] = float(T - t2)
    out["censored_sustained"] = True
    for q in np.nonzero(post)[0]:
        run_len = run_len + 1 if neg[q] else 0
        if run_len >= need:
            out["persist_sustained"] = float(t[q - need + 1] - t2)
            out["censored_sustained"] = False
            break
    out["frac_pos_post"] = float((dL[win(t2 + 20, T)] > 0).mean())
    # 本人のずれ・不確かさ
    phi_j = st["phi"][i][:, j].astype(float)
    eb = st["ebar"][i][:, j].astype(float)
    out["absphi_seam"] = float(np.abs(phi_j[win(220, t2)]).mean())
    out["absphi_late"] = float(np.abs(phi_j[win(T - 500, T)]).mean())
    out["ebar_pre"] = float(eb[win(150, t1)].mean())
    out["ebar_seam"] = float(eb[win(220, t2)].mean())
    out["ebar_late"] = float(eb[win(T - 500, T)].mean())
    # 帰属の正しさ（R1）
    at = st["acc_t"][i].astype(float)
    ok = st["acc_ok"][i].astype(float)  # (時刻, 主体)
    allok = ok.mean(1)
    hit = np.nonzero(allok > 0.9)[0]
    out["t_acc90"] = float(at[hit[0]]) if hit.size else float("nan")
    out["acc_first100"] = float(allok[at < 100].mean())
    out["acc_after100"] = float(allok[at >= 100].mean())
    out["acc_seam"] = float(allok[(at >= t1) & (at < t2)].mean())
    # 名前（j について）
    if cfg["cond"] in ("N1", "N2", "S"):
        pc = st["pub_content"][i][:, j].astype(int)
        pw = st["pub_word"][i][:, j].astype(int)
        tb = true_bin(phi_j, cfg["binw"])
        k2 = np.nonzero(t >= t2)[0][0]
        out["renamed_in_seam"] = bool(st["renamed_in_seam"][i][j])
        out["pub_content_t2"] = int(pc[k2])
        out["pub_nonzero_t2"] = bool(pc[k2] not in (0, -1))
        out["pub_content_end"] = int(pc[-1])
        out["n_public_change"] = int(st["n_public_change"][i][j])
        pm = t >= t2
        out["frac_name_ne_bin_post"] = float((pc[pm] != tb[pm]).mean())
        same = pw[pm] == pw[k2]
        out["frac_sameName_ne_bin_post"] = float((same & (pc[pm] != tb[pm])).mean())
        # 公の名前の内容が t2 の内容から最初に替わるまでの時間（替わらなければ打ち切り）
        chg = np.nonzero(pm & (pc != pc[k2]))[0]
        out["name_hold"] = float(t[chg[0]] - t2) if chg.size else float(T - t2)
        out["name_hold_censored"] = not bool(chg.size)
        # S 用：噂を止めた時刻・戻した時刻の名前
        if cfg["cond"] == "S":
            for lab, tt in (("stop", cfg["s_stop"]), ("restore", cfg["s_restore"])):
                q = np.nonzero(t >= tt)[0][0]
                out[f"pub_content_{lab}"] = int(pc[q])
                out[f"pub_word_{lab}"] = int(pw[q])
                out[f"dL_at_{lab}"] = float(dL[max(0, q - 10):q].mean())
            out["pub_content_end"] = int(pc[-1])
            out["pub_word_end"] = int(pw[-1])
            q_stop = np.nonzero(t >= cfg["s_stop"])[0][0]
            q_res = np.nonzero(t >= cfg["s_restore"])[0][0]
            out["dL_before_stop"] = float(dL[win(cfg["s_stop"] - 50, cfg["s_stop"])].mean())
            out["dL_end_of_stop"] = float(dL[win(cfg["s_restore"] - 50, cfg["s_restore"])].mean())
            out["dL_after_restore"] = float(dL[win(T - 100, T)].mean())
            out["standing_before_stop"] = bool(out["dL_before_stop"] > 0)
            out["standing_end_of_stop"] = bool(out["dL_end_of_stop"] > 0)
            out["renamed_during_stop"] = bool(pc[q_res] != pc[q_stop])
            out["back_to_original_after_restore"] = bool(pc[-1] == pc[q_stop] and pc[q_res] != pc[q_stop])
    if cfg["cond"] == "M":
        mp = st["mem_pub"][i][:, j].astype(float)
        k2 = np.nonzero(t >= t2)[0][0]
        out["mem_t2"] = float(mp[k2])
        out["absmem_t2"] = float(abs(mp[k2]))
        out["absmem_late"] = float(np.abs(mp[win(T - 500, T)]).mean())
        tb = true_bin(phi_j, cfg["binw"])
        mb = true_bin(mp, cfg["binw"])
        pm = t >= t2
        out["frac_name_ne_bin_post"] = float((mb[pm] != tb[pm]).mean())
    return out


def summarize_batch(st, meta):
    return [run_summary(st, i, c) for i, c in enumerate(meta["configs"])]


def describe(vals):
    v = np.asarray(vals, float)
    return {"n": int(v.size), "mean": float(np.nanmean(v)), "median": float(np.nanmedian(v)),
            "q25": float(np.nanpercentile(v, 25)), "q75": float(np.nanpercentile(v, 75)),
            "min": float(np.nanmin(v)), "max": float(np.nanmax(v))}
