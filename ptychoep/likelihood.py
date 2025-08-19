from __future__ import annotations
from .uncertain_array import UncertainArray as UA
from backend.backend import np
from ptycho.data import DiffractionData
from typing import Optional


class Likelihood:
    """
    Output Likelihood node for EP-based ptychography.

    This node models the observation of noisy diffraction data |z| + noise
    and performs a local approximation of the posterior over the complex field z.
    """

    def __init__(self, diff: DiffractionData, parent: "FFTChannel"):
        """
        Initialize the Likelihood node.

        Parameters
        ----------
        diff : DiffractionData
            The observed diffraction data node (contains intensity, etc.).
        parent : FFTChannel
            The parent FFTChannel this Likelihood is linked to.
        """
        self.diff = diff
        self.damping = 1.0
        self.parent = parent

        self.y = diff.diffraction  # observed amplitude (not intensity)
        self.gamma_w = diff.gamma_w if diff.gamma_w is not None else 1.0

        self.msg_from_fft: Optional[UA] = None  # Forward message from FFTChannel
        self.belief: Optional[UA] = None        # Posterior over z
        self.error: float = 0.0                 # Optional amplitude MSE for logging

    def compute_belief(self):
        """
        Compute the approximate posterior z_hat using Laplace approximation.
        This implementation follows the Laplace approximation approach described in:

        S. K. Shastri and P. Schniter,
        "Fast and Robust Phase Retrieval via Deep Expectation-Consistent Approximation",
        IEEE Trans. Signal Process., 2024.
        See Eq. (38)-(39), Appendix A and B.

        Uses:
            y       = observed amplitude (sqrt of intensity)
            z0      = mean of incoming message
            v0      = variance of incoming message (1 / precision)
            gamma_w = measurement precision

        Sets self.belief with mean and precision.
        """
        if self.msg_from_fft is None:
            raise RuntimeError("Likelihood.compute_belief: msg_from_fft not set")

        xp = np()
        z0 = self.msg_from_fft.mean
        tau = self.msg_from_fft.precision
        v0 = 1.0 / tau
        v = 1.0/self.gamma_w

        abs_z0 = xp.abs(z0)
        abs_z0_safe = abs_z0
        unit_phase = z0 / abs_z0_safe

        # Posterior mean (amplitude-domain Laplace approx)
        z_hat_amp = (v0 * self.y + 2 * v * abs_z0_safe) / (v0 + 2 * v)
        z_hat = unit_phase * z_hat_amp

        # Posterior precision
        v_hat = (v0 * (v0 * self.y + 4 * v * abs_z0_safe)) / (2.0 * abs_z0_safe * (v0 + 2 * v))
        #v_hat = xp.maximum(v_hat, 1e-8)
        precision = 1.0 / v_hat

        self.belief = UA(mean=z_hat, precision=precision, dtype=z0.dtype)
        self.error = float(xp.mean((abs_z0 - self.y) ** 2))

    def backward(self) -> None:
        """
        Backward message passing: update FFTChannel.msg_from_likelihood.

        This performs the following steps:
        1. Computes the belief (posterior over z) via Laplace approximation.
        2. Calculates the backward message as:
            msg_back = belief / msg_from_fft
        3. Applies damping:
            msg_new = damp_with(prev_msg, damping=self.damping)

        This damped message is then sent back to the FFTChannel.
        """

        self.compute_belief()
        msg_back_raw = self.belief.to_scalar_precision() / self.msg_from_fft
        msg_back_new = msg_back_raw.damp_with(self.parent.msg_from_likelihood, damping = self.damping)
        self.parent.msg_from_likelihood = msg_back_new