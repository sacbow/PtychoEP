# utils/rng_utils.py
from .backend import np, is_cupy

def get_rng(seed=None):
    """Backendに応じて乱数生成器 (numpy.Generator or cupy.RandomState) を返す"""
    if is_cupy():
        import cupy as cp
        return cp.random.RandomState(seed)
    else:
        import numpy as _np
        return _np.random.default_rng(seed)

def normal(rng, mean=0.0, var=1.0, size = None, dtype = np().complex64):
    """正規分布乱数をBackendに依存して生成"""
    if dtype in (np().float32 , np().float64):
        return (rng.normal(loc=mean, scale=var**0.5, size = size)).astype(dtype)
    elif dtype in (np().complex64 , np().complex128):
        return (rng.normal(loc=mean, scale=(var/2)**0.5, size = size) + 1j * rng.normal(loc=mean, scale=(var/2)**0.5, size = size)).astype(dtype)
    else:
        raise ValueError("Unsupported dtype.")
