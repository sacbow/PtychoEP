from __future__ import annotations
from .uncertain_array import UncertainArray
from ptychoep.backend.backend import np

class AccumulativeUncertainArray:
    """
    A class that represents a Gaussian posterior in a product form.
    Internally maintains mean × precision and precision separately.
    """

    def __init__(self, shape, dtype=np().complex64):
        self.shape = shape
        self.dtype = dtype
        self._numerator = np().zeros(shape, dtype=dtype)  # mean * precision
        self._precision = np().ones(shape, dtype=np().float32)

    def _normalize_indices(self, indices):
        """
        Normalize indexing input.
        - None → full region
        - (slice, slice) → return as is
        - otherwise → raise error
        """
        if indices is None:
            return slice(None), slice(None)
        if (isinstance(indices, tuple) and len(indices) == 2
                and isinstance(indices[0], slice) and isinstance(indices[1], slice)):
            return indices
        raise TypeError("indices must be None or a tuple of two slice objects")

    def add(self, ua: UncertainArray, indices: tuple[slice, slice] = None):
        """Add a UA to the accumulator at specified region."""
        sl_y, sl_x = self._normalize_indices(indices)
        self._numerator[sl_y, sl_x] += ua.mean * ua.precision
        self._precision[sl_y, sl_x] += ua.precision
    
    def subtract(self, ua: UncertainArray, indices: tuple[slice, slice] = None):
        """Subtract a UA from the accumulator at specified region."""
        sl_y, sl_x = self._normalize_indices(indices)
        self._numerator[sl_y, sl_x] -= ua.mean * ua.precision
        self._precision[sl_y, sl_x] -= ua.precision

    def get_mean(self, indices: tuple[slice, slice] = None) -> np().ndarray:
        """Return the mean of the accumulated belief at specified region."""
        sl_y, sl_x = self._normalize_indices(indices)
        return self._numerator[sl_y, sl_x] / self._precision[sl_y, sl_x]

    def get_precision(self, indices: tuple[slice, slice] = None) -> np().ndarray:
        """Return the precision of the accumulated belief at specified region."""
        sl_y, sl_x = self._normalize_indices(indices)
        return self._precision[sl_y, sl_x]

    def get_ua(self, indices: tuple[slice, slice] = None) -> UncertainArray:
        """Return an UncertainArray representing the belief at specified region."""
        sl_y, sl_x = self._normalize_indices(indices)
        mean = self._numerator[sl_y, sl_x] / self._precision[sl_y, sl_x]
        precision = self._precision[sl_y, sl_x]
        return UncertainArray(mean=mean, precision=precision, dtype=self.dtype)

    def to_ua(self) -> UncertainArray:
        """Return the full accumulated belief as a single UncertainArray."""
        mean = self._numerator / self._precision
        return UncertainArray(mean=mean, precision=self._precision, dtype=self.dtype)

    def clear(self):
        """Reset the accumulator to default values (zero mean, unit precision)."""
        self._numerator[...] = 0
        self._precision[...] = 1
