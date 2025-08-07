from __future__ import annotations
from ...backend import np
from ...rng_utils import normal

class UncertainArray:
    """
    ガウス変数 (複素/実) の期待値と逆分散(precision)をまとめて管理するクラス。
    """
    def __init__(self, mean: np().ndarray, precision: np().ndarray = 1.0, dtype = np().complex64):
        self.mean = np().asarray(mean, dtype = dtype)
        self.dtype = dtype
        if np().isscalar(precision) or precision.ndim == 0:
            self.scalar_precision = True
            self.precision = np().asarray(precision, dtype = np().float32)
        elif mean.shape == precision.shape:
            self.scalar_precision = False
            self.precision = np().asarray(precision, dtype = np().float32)
        else:
            raise ValueError("precision shape mismatch.")

    @classmethod
    def zeros(cls, shape, dtype=np().complex64):
        return cls(np().zeros(shape, dtype=dtype), np().ones(shape, dtype=np().float32))
    
    @classmethod
    def normal(cls, shape, rng, dtype=np().complex64):
        return cls(normal(rng=rng, size=shape, dtype=dtype), np().ones(shape, dtype=np().float32), dtype=dtype)

    def copy(self):
        return UncertainArray(self.mean.copy(), self.precision.copy(), dtype=self.dtype)

    def to_tuple(self):
        """(mean, precision) タプルで返す（既存のMessagePassing互換）"""
        return self.mean, self.precision
    
    def __mul__(self, other : UncertainArray):
        if self.scalar_precision != other.scalar_precision:
            raise ValueError("both of the UAs should have scalar/array-type precision")
        precision_mul = self.precision + other.precision
        product_mul = self.precision * self.mean + other.precision * other.mean
        mean_mul = product_mul/precision_mul
        return UncertainArray(mean = mean_mul, precision = precision_mul)
    
    def __truediv__(self, other : UncertainArray):
        if self.scalar_precision != other.scalar_precision:
            raise ValueError("both of the UAs should have scalar/array-type precision")
        precision_div = np().maximum(self.precision - other.precision, 1.0)
        product_div = self.precision * self.mean - other.precision * other.mean
        mean_div = product_div/precision_div
        return UncertainArray(mean = mean_div, precision = precision_div)
    
    def to_scalar_precision(self) -> UncertainArray:
        """
        precisionをスカラー化する。
        precision (逆分散) を分散に変換し平均を取り、その逆数を返すことで調和平均を実現。
        """
        if self.scalar_precision:
            return self
        # 分散 = 1 / precision
        variance = 1.0 / self.precision
        mean_variance = np().mean(variance)
        scalar_precision = 1.0 / mean_variance
        return UncertainArray(self.mean.copy(), precision=scalar_precision, dtype=self.dtype)

    def to_array_precision(self) -> UncertainArray:
        """
        precisionを配列化する。
        scalar_precision=Trueのとき、mean.shapeと同じ配列precisionにブロードキャスト。
        """
        if not self.scalar_precision:
            return self
        array_precision = np().ones_like(self.mean.real, dtype=np().float32) * self.precision
        return UncertainArray(self.mean.copy(), precision=array_precision, dtype=self.dtype)
    
    def damp_with(self, other: UncertainArray, damping: float) -> UncertainArray:
        """
        自身(self)をraw、otherをoldとしてdampingを適用する。
        r_new = damping * r_raw + (1-damping) * r_old
        gamma_new = 1 / (damping/sqrt(gamma_raw) + (1-damping)/sqrt(gamma_old))^2
        """
        if self.scalar_precision != other.scalar_precision:
            raise ValueError("dampingには同じprecisionタイプ（scalar/array）のUAが必要です")

        mean_damped = damping * self.mean + (1 - damping) * other.mean
        gamma_damped = 1.0 / (
            damping / np().sqrt(self.precision) + (1 - damping) / np().sqrt(other.precision)
        ) ** 2

        return UncertainArray(mean=mean_damped, precision=gamma_damped, dtype=self.dtype)

# --- fft utils ---
from .uncertain_array import UncertainArray as UA

def fft_ua(uarray: UA, norm="ortho") -> UA:
    """
    UA.mean に fft2 を適用し、精度はスカラーに変換（調和平均）。
    """
    fft_mean = np().fft.fft2(uarray.mean, norm=norm)
    if uarray.scalar_precision:
        scalar_precision = uarray.precision
    else:
        variance = 1.0 / uarray.precision
        scalar_precision = 1.0 / np().mean(variance)

    return UA(mean=fft_mean, precision=scalar_precision, dtype=uarray.dtype)

def ifft_ua(uarray: UA, norm="ortho") -> UA:
    """
    UA.mean に ifft2 を適用し、精度はスカラーに変換（調和平均）。
    """
    ifft_mean = np().fft.ifft2(uarray.mean, norm=norm)
    if uarray.scalar_precision:
        scalar_precision = uarray.precision
    else:
        variance = 1.0 / uarray.precision
        scalar_precision = 1.0 / np().mean(variance)

    return UA(mean=ifft_mean, precision=scalar_precision, dtype=uarray.dtype)

