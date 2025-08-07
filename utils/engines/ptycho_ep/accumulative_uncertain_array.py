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

    def add(self, ua: UncertainArray, indices: tuple[slice, slice]):
        """
        UncertainArrayの値を指定されたindicesに加算。
        indices: 2次元スライス (row_slice, col_slice)
        """
        sl_y, sl_x = indices
        self._numerator[sl_y, sl_x] += ua.mean * ua.precision
        self._precision[sl_y, sl_x] += ua.precision
    
    def subtract(self, ua: UncertainArray, indices: tuple[slice, slice]):
        """
        指定領域から UncertainArray の値を減算。
        """
        sl_y, sl_x = indices
        self._numerator[sl_y, sl_x] -= ua.mean * ua.precision
        self._precision[sl_y, sl_x] -= ua.precision

    def get_mean(self, indices: tuple[slice, slice]) -> np().ndarray:
        """指定領域の mean をオンデマンドで取得"""
        sl_y, sl_x = indices
        numerator = self._numerator[sl_y, sl_x]
        precision = self._precision[sl_y, sl_x]
        return numerator / precision

    def get_precision(self, indices: tuple[slice, slice]) -> np().ndarray:
        return self._precision[indices[0], indices[1]]

    def get_ua(self, indices: tuple[slice, slice]) -> UncertainArray:
        """指定領域の UncertainArray を取得"""
        mean = self.get_mean(indices)
        precision = self.get_precision(indices)
        return UncertainArray(mean=mean, precision=precision, dtype=self.dtype)

    def to_ua(self) -> UncertainArray:
        """全体を UncertainArray として返す"""
        mean = self._numerator / self._precision
        return UncertainArray(mean=mean, precision=self._precision, dtype=self.dtype)

    def clear(self):
        """内部状態をリセット"""
        self._numerator[...] = 0
        self._precision[...] = 0
