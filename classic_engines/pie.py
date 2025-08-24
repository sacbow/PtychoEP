# utils/engines/pie.py
from .base_pie import BasePIE
from ptychoep.backend.backend import np

class PIE(BasePIE):
    """
    Implementation of the classic PIE (Ptychographical Iterative Engine) algorithm
    with a fixed probe.

    This class extends BasePIE and implements the object update rule for PIE. 
    Since the probe is fixed throughout the reconstruction process, certain 
    quantities derived from the probe (e.g., its magnitude, conjugate, and max) 
    are precomputed and cached for efficiency.

    Attributes:
        prb_conj (ndarray): Complex conjugate of the probe.
        prb_abs (ndarray): Magnitude of the probe.
        prb_max (float): Maximum magnitude of the probe.

    Args:
        ptycho (Ptycho): Ptycho object containing object size, probe, and scan data.
        alpha (float): Regularization parameter for object update.
        obj_init (ndarray or None): Optional initial guess for the object. 
                                    If None, initialized with complex Gaussian noise.
        callback (callable or None): Optional function to log or monitor progress at each iteration.
        dtype (dtype): Data type for internal arrays (default: complex64).
    """


    def __init__(self, ptycho, alpha=0.1, obj_init=None, callback=None, dtype = np().complex64):
        super().__init__(ptycho, alpha, obj_init, dtype, callback)
        self.prb_conj = self.prb.conj()
        self.prb_abs = self.xp.abs(self.prb)
        self.prb_max = self.xp.max(self.prb_abs)

    def _update_object(self, proj_wave, exit_wave, indices):
        yy, xx = indices
        delta = (
            self.prb_abs * self.prb_conj * (proj_wave - exit_wave)
            / (self.prb_max * (self.prb_abs**2 + self.alpha * self.prb_max**2))
        )
        self.obj[yy, xx] += delta


