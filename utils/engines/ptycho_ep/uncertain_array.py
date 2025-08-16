from __future__ import annotations
from ...backend import np
from ...rng_utils import normal

class UncertainArray:
    """
    A container class representing a (possibly complex) Gaussian variable
    using its mean and precision (inverse variance).
    """
    def __init__(self, mean: np().ndarray, precision: np().ndarray = 1.0, dtype = np().complex64):
        self.mean = np().asarray(mean, dtype = dtype)
        self.shape = mean.shape
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
    def zeros(cls, shape, dtype=np().complex64, scalar_precision = True):
        if scalar_precision:
            return cls(np().zeros(shape, dtype=dtype), 1.0)
        else:
            return cls(np().zeros(shape, dtype=dtype), np().ones(shape, dtype=np().float32))

    
    @classmethod
    def normal(cls, shape, rng, dtype=np().complex64, scalar_precision = True):
        if scalar_precision:
            return cls(normal(rng=rng, size=shape, dtype=dtype), 1.0)
        else:
            return cls(normal(rng=rng, size=shape, dtype=dtype), np().ones(shape, dtype=np().float32))

    def copy(self):
        return UncertainArray(self.mean.copy(), self.precision.copy(), dtype=self.dtype)

    def to_tuple(self):
        """Return the (mean, precision) pair as a tuple (compatible with legacy message passing)."""
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
        Convert the current precision into a scalar precision.

        The new precision is computed as the harmonic mean of variances
        (i.e., inverse of precision), then inverted to get scalar precision.
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
        Convert the current precision into array precision.

        If scalar_precision=True, broadcast the scalar precision into an array
        matching the shape of mean.
        """
        if not self.scalar_precision:
            return self
        array_precision = np().ones_like(self.mean.real, dtype=np().float32) * self.precision
        return UncertainArray(self.mean.copy(), precision=array_precision, dtype=self.dtype)
    
    def slice(self, indices: tuple[slice, slice]) -> "UncertainArray":
        """Extract a patch UA using a (slice, slice) index."""
        y, x = indices
        mean_sub = self.mean[y, x]
        if self.scalar_precision:
            prec_sub = self.precision           
        else:
            prec_sub = self.precision[y, x]
        return UncertainArray(mean=mean_sub, precision=prec_sub, dtype=self.dtype)


    def __getitem__(self, key) -> "UncertainArray":
        if isinstance(key, tuple) and len(key) == 2 \
           and isinstance(key[0], slice) and isinstance(key[1], slice):
            return self.slice(key)
        raise TypeError("UncertainArray.__getitem__ expects (slice, slice)")
    
    def damp_with(self, other: UncertainArray, damping: float) -> UncertainArray:
        """
        Apply damping between the current UA (raw) and another UA (previous).

        r_new = damping * r_raw + (1 - damping) * r_old
        gamma_new = 1 / (damping/sqrt(gamma_raw) + (1-damping)/sqrt(gamma_old))^2

        Requires both UAs to have the same precision type (scalar/array).
        """
        if self.scalar_precision != other.scalar_precision:
            raise ValueError("UA.damp_with : uncompatible precision type")

        mean_damped = damping * self.mean + (1 - damping) * other.mean
        gamma_damped = 1.0 / (
            damping / np().sqrt(self.precision) + (1 - damping) / np().sqrt(other.precision)
        ) ** 2

        return UncertainArray(mean=mean_damped, precision=gamma_damped, dtype=self.dtype)
    
    def scaled(self, gain, to_array_when_nonuniform: bool = True, precision_floor: float = 0.0):
        """
        Scale the UA by a complex gain elementwise.

        Updates mean and precision as:
            mean'      = gain * mean
            precision' = |gain|^2 * precision

        Parameters
        ----------
        gain : scalar or ndarray
            Broadcastable gain factor.
        to_array_when_nonuniform : bool
            If True and gain is non-uniform, scalar precision is promoted to array precision.
        precision_floor : float
            Minimum precision threshold to avoid numerical instability.
        """
        xp = np()
        g = xp.asarray(gain)
        g_abs2 = xp.abs(g)**2

        new_mean = g * self.mean

        if xp.isscalar(g) or g.shape == () or (g_abs2 == g_abs2.flat[0]).all():
            # scalar gain
            new_prec = self.precision / xp.maximum(g_abs2, precision_floor)
            return UncertainArray(mean=new_mean, precision=new_prec, dtype=self.dtype)

        # array gain
        if self.scalar_precision and to_array_when_nonuniform:
            # スカラー精度 → 配列精度に昇格して各画素で重み付け
            new_prec = self.precision / xp.maximum(g_abs2.astype(xp.float32) , precision_floor)
            return UncertainArray(mean=new_mean, precision=new_prec, dtype=self.dtype)
        else:
            raise ValueError("Cannot produce scalar-precision UA from array-gain")

# --- fft utils ---
from .uncertain_array import UncertainArray as UA

def fft_ua(uarray: UA, norm="ortho") -> UA:
    """
    Apply 2D FFT to UA.mean. Converts precision to scalar using harmonic mean of variances.
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
    Apply 2D IFFT to UA.mean. Converts precision to scalar using harmonic mean of variances.
    """
    ifft_mean = np().fft.ifft2(uarray.mean, norm=norm)
    if uarray.scalar_precision:
        scalar_precision = uarray.precision
    else:
        variance = 1.0 / uarray.precision
        scalar_precision = 1.0 / np().mean(variance)

    return UA(mean=ifft_mean, precision=scalar_precision, dtype=uarray.dtype)

