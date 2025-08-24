from abc import ABC, abstractmethod
from ptychoep.backend.backend import np
from ptychoep.rng.rng_utils import get_rng, normal, poisson


class Noise(ABC):
    """
    Abstract base class for noise models.

    Each subclass must implement `_apply_noise_and_compute_snr`, which injects
    noise into the ptychographic diffraction data and returns a dictionary
    summarizing noise statistics such as average SNR.

    This class also overloads the `@` operator so that noise injection can be
    done elegantly as: `GaussianNoise(...) @ ptycho`.

    Returns:
        The input Ptycho object with noise added to its diffraction data.
        Also attaches a `.noise_stats` dictionary to the object.
    """

    def __matmul__(self, ptycho):
        ptycho.noise_stats = self._apply_noise_and_compute_snr(ptycho)
        return ptycho

    @abstractmethod
    def _apply_noise_and_compute_snr(self, ptycho):
        pass

    def _compute_snr_db(self, clean_amp, noisy_amp):
        """
        Compute intensity-based SNR in decibels.

        Args:
            clean_amp: Clean amplitude (ground truth)
            noisy_amp: Noisy amplitude (after noise injection)

        Returns:
            SNR in dB
        """
        noise_intensity = (clean_amp - noisy_amp)**2
        signal_power = np().sum(clean_amp**2)
        noise_power = np().sum(noise_intensity)
        snr_db = 10 * np().log10(signal_power / (noise_power + 1e-12))
        return snr_db


class GaussianNoise(Noise):
    """
    Additive Gaussian noise model for diffraction amplitude.

    Args:
        var: Variance of the Gaussian noise
        seed: RNG seed for reproducibility
    """

    def __init__(self, var: float = 1e-3, seed: int = None):
        self.var = var
        self.seed = seed

    def _apply_noise_and_compute_snr(self, ptycho):
        rng = get_rng(self.seed)
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
    """
    Poisson noise model with Anscombe transformation.

    This simulates photon count statistics and maps the result back to amplitude
    via square root after Anscombe transform.

    Args:
        scale: Photon count scaling factor (e.g., 1e5)
        seed: RNG seed for reproducibility
    """

    def __init__(self, scale: float = 1e5, seed: int = None):
        self.scale = scale
        self.seed = seed

    def _apply_noise_and_compute_snr(self, ptycho):
        rng = get_rng(self.seed)
        snr_values = []

        for d in ptycho._diff_data:
            clean = d.diffraction.copy()
            intensity = np().abs(clean) ** 2
            expected_counts = intensity * self.scale
            sampled_counts = poisson(rng=rng, lam=expected_counts).astype(np().float32)
            noisy_intensity = (sampled_counts + 3.0 / 8.0) / self.scale  # Anscombe transform
            d.diffraction = np().sqrt(noisy_intensity).astype(clean.dtype)
            d.gamma_w = 4.0 * self.scale  # Variance approx for sqrt(Poisson) with Anscombe
            snr_values.append(self._compute_snr_db(clean, d.diffraction))

        snr_mean = sum(snr_values) / len(snr_values) if snr_values else 0.0
        return {
            "type": "Poisson",
            "scale": self.scale,
            "snr_mean_db": float(snr_mean)
        }
