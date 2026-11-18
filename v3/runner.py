"""走らせて保存する道具。設定（JSON）と乱数の種を結果と一緒に置く。"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import time
from dataclasses import replace
from multiprocessing import Pool

import numpy as np

from .model import Config, simulate

NPROC = max(1, (os.cpu_count() or 2) - 1)


def git_rev():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def run_many(cfgs, nproc=NPROC):
    with Pool(nproc) as pool:
        return pool.map(simulate, cfgs, chunksize=1)


def stack(results):
    out = {}
    for key in results[0]:
        arrs = [np.asarray(r[key]) for r in results]
        try:
            a = np.stack(arrs)
        except ValueError:  # 長さが違う（ありえないはずだが念のため）
            a = np.array(arrs, dtype=object)
        if a.dtype == np.float64:
            a = a.astype(np.float32)
        out[key] = a
    return out


def save_batch(dirpath, name, cfgs, results, extra=None):
    os.makedirs(dirpath, exist_ok=True)
    st = stack(results)
    np.savez_compressed(os.path.join(dirpath, f"{name}.npz"), **st)
    meta = {
        "name": name,
        "base_config": cfgs[0].to_dict(),
        "seeds": [c.seed for c in cfgs],
        "configs": [c.to_dict() for c in cfgs],
        "git_rev": git_rev(),
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python": platform.python_version(),
        "numpy": np.__version__,
    }
    if extra:
        meta.update(extra)
    with open(os.path.join(dirpath, f"{name}.json"), "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1, default=float)
    return st


def load_batch(dirpath, name):
    st = dict(np.load(os.path.join(dirpath, f"{name}.npz"), allow_pickle=True))
    with open(os.path.join(dirpath, f"{name}.json")) as f:
        meta = json.load(f)
    return st, meta


def seeds_cfgs(base: Config, seeds, **kw):
    return [replace(base, seed=int(s), **kw) for s in seeds]
