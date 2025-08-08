from __future__ import annotations
from .uncertain_array import UncertainArray as UA
from typing import Optional
from ...backend import np

class Probe:
    """
    Hadamard積で作用するプローブ。
    data: 複素振幅 (H, W)
    forward:  UA_x -> UA_y  (y = P ⊙ x)
    backward: UA_y -> UA_x  (x = conj(P) ⊙ y)
    """
    def __init__(self, data: Optional[np().ndarray] = None , eps: float = 1e-10, dtype=np().complex64):
        self.dtype = dtype
        self.eps = float(eps)
        if data is not None:
            self.set_data(data)
        else:
            self.data = None
       

    def set_data(self, data: np().ndarray):
        xp = np()
        arr = xp.asarray(data, dtype = self.dtype)
        if arr.ndim != 2:
            raise ValueError("Probe.data must be 2D.")

        self.data = arr
        self.abs2 = xp.maximum(xp.abs(arr) ** 2, self.eps)
        self.data_inv = self.data.conj()/self.abs2

    def forward(self, msg: UA) -> UA:
        """
        x ~ UA(mean, precision) -> y = P ⊙ x
        mean_y = P ⊙ mean_x
        precision_y = precision_x / |P|^2   (要素ごと or スカラー)
        """
        xp = np()
        if msg.mean.shape != self.data.shape:
            raise ValueError("Probe.forward: shape mismatch.")
        if self.data is None:
            raise RuntimeError("Probe.forward: No data in probe")
        mean = self.data * msg.mean
        precision = msg.precision / self.abs2
        return UA(mean=mean, precision=precision.astype(xp.float32), dtype=mean.dtype)

    def backward(self, msg: UA) -> UA:
        """
        逆伝搬（Object側へ戻す）
        mean_x = conj(P) ⊙ mean_y
        precision_x = precision_y * |P|^2
        """
        xp = np()
        if msg.mean.shape != self.data.shape:
            raise ValueError("Probe.backward: shape mismatch.")
        if self.data is None:
            raise RuntimeError("Probe.forward: No data in probe")
        mean = self.data_inv * msg.mean
        precision = self.abs2 * msg.precision
        return UA(mean=mean, precision=precision.astype(xp.float32), dtype=mean.dtype)
