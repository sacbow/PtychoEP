from __future__ import annotations
from .uncertain_array import UncertainArray
from ...backend import np

class AccumulativeUncertainArray:
    """
    ガウス事後分布の積型表現を保持するクラス。
    mean × precision と precision の2つを内部的に保持する。
    """

    def __init__(self, shape, dtype=np().complex64):
        self.shape = shape
        self.dtype = dtype
        self._numerator = np().zeros(shape, dtype=dtype)  # mean * precision
        self._precision = np().ones(shape, dtype=np().float32)

    def _normalize_indices(self, indices):
        """
        None → 全域、(slice, slice) → そのまま、
        それ以外はエラー。
        """
        if indices is None:
            return slice(None), slice(None)
        if (isinstance(indices, tuple) and len(indices) == 2
                and isinstance(indices[0], slice) and isinstance(indices[1], slice)):
            return indices
        raise TypeError("indices must be None or a tuple of two slice objects")

    def add(self, ua: UncertainArray, indices: tuple[slice, slice] = None):
        sl_y, sl_x = self._normalize_indices(indices)
        self._numerator[sl_y, sl_x] += ua.mean * ua.precision
        self._precision[sl_y, sl_x] += ua.precision
    
    def subtract(self, ua: UncertainArray, indices: tuple[slice, slice] = None):
        sl_y, sl_x = self._normalize_indices(indices)
        self._numerator[sl_y, sl_x] -= ua.mean * ua.precision
        self._precision[sl_y, sl_x] -= ua.precision

    def get_mean(self, indices: tuple[slice, slice] = None) -> np().ndarray:
        sl_y, sl_x = self._normalize_indices(indices)
        return self._numerator[sl_y, sl_x] / self._precision[sl_y, sl_x]

    def get_precision(self, indices: tuple[slice, slice] = None) -> np().ndarray:
        sl_y, sl_x = self._normalize_indices(indices)
        return self._precision[sl_y, sl_x]

    def get_ua(self, indices: tuple[slice, slice] = None) -> UncertainArray:
        sl_y, sl_x = self._normalize_indices(indices)
        mean = self._numerator[sl_y, sl_x] / self._precision[sl_y, sl_x]
        precision = self._precision[sl_y, sl_x]
        return UncertainArray(mean=mean, precision=precision, dtype=self.dtype)

    def to_ua(self) -> UncertainArray:
        """全域を UA として返す"""
        mean = self._numerator / self._precision
        return UncertainArray(mean=mean, precision=self._precision, dtype=self.dtype)

    def clear(self):
        self._numerator[...] = 0
        self._precision[...] = 1
