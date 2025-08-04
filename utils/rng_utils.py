from .backend import np, is_cupy

def get_rng(seed=None):
    """Backendに応じて乱数生成器 (numpy.Generator or cupy.RandomState) を返す"""
    if is_cupy():
        import cupy as cp
        return cp.random.RandomState(seed)
    else:
        import numpy as _np
        return _np.random.default_rng(seed)

def normal(rng, mean=0.0, var=1.0, size=None, dtype=np().complex64):
    """正規分布乱数をBackendに依存して生成"""
    if dtype in (np().float32, np().float64):
        return (rng.normal(loc=mean, scale=var**0.5, size=size)).astype(dtype)
    elif dtype == np().complex64:
        return (
            rng.normal(loc=mean, scale=(var/2)**0.5, size=size).astype(np().float32)
            + 1j * rng.normal(loc=mean, scale=(var/2)**0.5, size=size).astype(np().float32)
        )
    elif dtype == np().complex128:
        return (
            rng.normal(loc=mean, scale=(var/2)**0.5, size=size).astype(np().float64)
            + 1j * rng.normal(loc=mean, scale=(var/2)**0.5, size=size).astype(np().float64)
        )
    else:
        raise ValueError("Unsupported dtype.")

def uniform(rng, low=0.0, high=1.0, size=None, dtype=np().float32):
    """一様分布乱数をBackendに依存して生成"""
    vals = rng.uniform(low=low, high=high, size=size)
    return vals.astype(dtype)

def randint(rng, low: int, high: int, size=None):
    """
    整数一様分布乱数をBackendに依存して生成。
    numpy/cupyのrng.integers/rng.randintをラップし、Python intにキャスト。

    Args:
        rng: get_rng()で得た乱数生成器
        low: 最小値（含む）
        high: 最大値（含まない）
        size: 生成サイズ（Noneならスカラ）

    Returns:
        intまたはintのリスト/配列
    """
    if is_cupy():
        vals = rng.randint(low, high, size=size)
    else:
        vals = rng.integers(low, high, size=size)
    if size is None:
        return int(vals)  # スカラはPython intにキャスト
    else:
        # 配列の場合は要素をPython intに変換（list comprehension）
        return [int(v) for v in vals]

def poisson(rng, lam, size=None):
    """
    ポアソン分布乱数をBackendに依存して生成。
    lam: 期待値（float または array）
    size: 形状
    """
    return rng.poisson(lam=lam, size=size)

