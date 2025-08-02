# utils/backend.py
import numpy as _np
try:
    import cupy as _cp
except ImportError:
    _cp = None

_backend = _np  # デフォルトはnumpy

def set_backend(name: str):
    global _backend
    if name == "numpy":
        _backend = _np
    elif name == "cupy":
        if _cp is None:
            raise ImportError("Cupy is not installed.")
        _backend = _cp
    else:
        raise ValueError(f"Unknown backend: {name}")

def np():
    """現在のbackendモジュール（numpy or cupy）を返す"""
    return _backend

def is_cupy():
    return _backend.__name__ == "cupy"
