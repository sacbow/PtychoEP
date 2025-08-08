from abc import ABC, abstractmethod
from ..backend import np
from ..rng_utils import get_rng, normal, poisson

class Noise(ABC):
    """ノイズモデル基底クラス（強度ベース dB単位のSN比計算を内蔵）"""
 
    def __matmul__(self, ptycho):
        ptycho.noise_stats = self._apply_noise_and_compute_snr(ptycho)
        return ptycho

    @abstractmethod
    def _apply_noise_and_compute_snr(self, ptycho):
        pass

    def _compute_snr_db(self, clean_amp, noisy_amp):
        """強度ベース (Intensity) でのSN比[dB]を計算"""
        noise_intensity = (clean_amp - noisy_amp)**2

        signal_power = np().sum(clean_amp**2)
        noise_power = np().sum(noise_intensity)
        snr_db = 10 * np().log10(signal_power / (noise_power + 1e-12))
        return snr_db

class GaussianNoise(Noise):
    def __init__(self, var: float = 1e-3):
        self.var = var

    def _apply_noise_and_compute_snr(self, ptycho):
        rng = get_rng()
        snr_values = []
        for d in ptycho._diff_data:
            clean = d.diffraction.copy()
            noise = normal(rng, mean=0.0, var=self.var, size=d.diffraction.shape, dtype=d.diffraction.dtype)
            d.diffraction = d.diffraction + noise
            d.gamma_w = 1.0 / self.var
            snr_values.append(self._compute_snr_db(clean, d.diffraction))

        snr_mean = sum(snr_values) / len(snr_values) if snr_values else 0.0
        return {
                    "type": "Gaussian",
                    "var": self.var,
                    "snr_mean_db": float(snr_mean)
                }


class PoissonNoise(Noise):
    def __init__(self, scale: float = 1e5):
        """
        Args:
            scale: 光子数スケーリング因子（例: 1e5）
        """
        self.scale = scale

    def _apply_noise_and_compute_snr(self, ptycho):
        rng = get_rng()
        snr_values = []
        for d in ptycho._diff_data:
            clean = d.diffraction.copy()
            intensity = np().abs(clean) ** 2
            expected_counts = intensity * self.scale
            sampled_counts = poisson(rng = rng, lam = expected_counts).astype(np().float32)
            noisy_intensity = sampled_counts / self.scale
            d.diffraction = np().sqrt(noisy_intensity).astype(clean.dtype)
            d.gamma_w = 4.0 * self.scale # approx: Var[sqrt(Poisson(λ)/scale)] ≈ 1/(4*scale)
            snr_values.append(self._compute_snr_db(clean, d.diffraction))

        snr_mean = sum(snr_values) / len(snr_values) if snr_values else 0.0
        return {
                    "type": "Poisson",
                    "scale": self.scale,
                    "snr_mean_db": float(snr_mean)
                }
