from __future__ import annotations
from .uncertain_array import UncertainArray as UA
from ...backend import np

# --- BaseDenoiser ---
class BaseDenoiser:
    def __init__(self, shape, damping: float = 1.0):
        self.shape = shape
        self.damping = damping
        self.input_msg: UA = None
        self.old_output_msg: UA = None
        self.belief: UA = None

    def receive_msg(self, msg: UA):
        self.input_msg = msg

    def forward_msg(self) -> UA:
        raw_output_msg = self.compute_forward_msg()
        if self.damping == 1.0 or self.old_output_msg is None:
            self.old_output_msg = raw_output_msg
            return raw_output_msg
        else:
            damped_output_msg = raw_output_msg.damp_with(self.old_output_msg, self.damping)
            self.old_output_msg = damped_output_msg
            return damped_output_msg

    def compute_forward_msg(self) -> UA:
        raise NotImplementedError()

    def compute_belief(self):
        raise NotImplementedError()

# --- GaussianPriorDenoiser ---
class GaussianPriorDenoiser(BaseDenoiser):
    def compute_belief(self):
        if self.input_msg is None:
            raise RuntimeError("No input message")
        self.belief = UA.zeros(shape=self.shape, dtype=self.input_msg.dtype) * self.input_msg

    def compute_forward_msg(self) -> UA:
        return UA.zeros(shape=self.shape, dtype=self.input_msg.dtype)

# --- Spike-and-Slab ---
from .uncertain_array import UncertainArray as UA
from .denoiser import BaseDenoiser
from ...backend import np

class SparsePriorDenoiser(BaseDenoiser):
    """
    Bernoulli-Gaussianスパース事前分布に基づくdenoiser。
    
    事前分布:  X ~ (1 - rho) * δ0 + rho * N(0, 1)
    """

    def __init__(self, shape, rho: float = 0.1, damping: float = 1.0):
        super().__init__(shape=shape, damping=damping)
        self.rho = rho

    def compute_belief(self):
        if self.input_msg is None:
            raise RuntimeError("No input message")

        r, gamma = self.input_msg.mean, self.input_msg.precision
        abs_r2 = np().abs(r) ** 2
        gamma_ratio = gamma / (gamma + 1)

        A = np().maximum(
            self.rho * np().exp(-gamma_ratio * abs_r2) / (1 + gamma),
            1e-12
        )
        B = (1 - self.rho) * np().exp(-gamma * abs_r2)
        pi = A / (A + B)
        one_minus_pi = B / (A + B)

        mean = pi * gamma_ratio * r
        d_mean = gamma_ratio * (1 + gamma * one_minus_pi * gamma_ratio * abs_r2)
        precision = gamma / d_mean

        self.belief = UA(mean=mean, precision=precision, dtype=self.input_msg.dtype)

    def compute_forward_msg(self) -> UA:
        self.compute_belief()
        return self.belief / self.input_msg

# --- Output denosier of Phase Retrieval ---
from .uncertain_array import UncertainArray as UA
from .denoiser import BaseDenoiser
from ...backend import np
from ...ptycho.data import DiffractionData


class PROutputDenoiser(BaseDenoiser):
    """
    |z| + N(0, 1/gamma_w) モデルに基づく観測ノードdenoiser。
    観測データと推定exit waveの誤差もforwardメッセージとともに返す。
    """

    def __init__(self, shape, data: DiffractionData, gamma_w: float, damping: float = 1.0):
        super().__init__(shape=shape, damping=damping)
        self.y = data.intensity()  # 強度データ
        self.gamma_w = gamma_w

    def compute_belief(self):
        raise NotImplementedError("PROutputDenoiser does not compute a belief (only sends messages).")

    def compute_forward_msg(self):
        if self.input_msg is None:
            raise RuntimeError("No input message provided.")

        # --- 入力から必要な情報を取り出す
        p = self.input_msg.mean
        tau = self.input_msg.precision

        # 振幅と位相
        p_abs = np().abs(p)
        p_ang = p / np().maximum(p_abs, 1e-12)  # 単位複素数で angle を代替（高速）

        # 観測値と振幅推定値のMSE
        mse = float(np().mean((p_abs - np().sqrt(self.y))**2))

        # PR denoising step（スカラー近似）
        y_p = np().mean(self.y / np().maximum(p_abs, 1e-12))
        numer = tau * p_abs + (self.gamma_w / 2) * self.y
        denom = tau + self.gamma_w / 2
        z = p_ang * numer / denom

        # 精度推定
        lam = 2 * (self.gamma_w + 2 * tau) / (self.gamma_w * y_p / tau + 4)

        # 出力メッセージ
        output_msg = UA(mean=z, precision=lam.astype(np().float32), dtype=p.dtype)

        # 誤差情報を付加（属性に保存するか、outputに含めてもよい）
        output_msg.meta = {"mse": mse}

        return output_msg


