from ptychoep.ptycho.projector import Fourier_projector
from ptychoep.backend.backend import np
from .base_pie import BasePIE

class ePIE(BasePIE):
    """
    Implementation of the ePIE (extended Ptychographical Iterative Engine) algorithm.

    Unlike the classic PIE, ePIE simultaneously updates both the object and the probe 
    during iterative reconstruction. This is particularly useful when the probe is unknown 
    or partially inaccurate.

    Attributes:
        prb (ndarray): Probe array (can be initialized externally or taken from the Ptycho object).
        beta (float): Step size for the probe update.

    Args:
        ptycho (Ptycho): Ptycho object containing object and scan information.
        alpha (float): Step size for the object update.
        beta (float): Step size for the probe update.
        obj_init (ndarray or None): Optional initial guess for the object. If None, random complex.
        prb_init (ndarray or None): Optional initial guess for the probe. If None, taken from `ptycho.prb`.
        callback (callable or None): Optional function to monitor or log progress per iteration.
        dtype (dtype): Data type for internal arrays.
        seed (int or None): Optional random seed for reproducibility.

    Returns:
        A tuple of (reconstructed object, reconstructed probe).
    """

    def __init__(self, ptycho, alpha=0.1, beta=0.1, obj_init=None, prb_init = None, callback=None, dtype = np().complex64, seed : int = None):
        super().__init__(ptycho, alpha, obj_init, dtype, callback, seed)
        self.prb = prb_init if prb_init is not None else ptycho.prb
        self.beta = beta
    
    # The update_object step tends to be more time-consuming than update_probe.
    # This is likely due to the fact that it performs an in-place write to self.obj[yy, xx], which is more expensive than read-only access.

    def _update_object(self, proj_wave, exit_wave, indices):
        yy, xx = indices
        prb_abs = self.xp.abs(self.prb)
        prb_conj = self.prb.conj()
        prb_max = self.xp.max(prb_abs)
        delta = self.alpha * prb_conj * (proj_wave - exit_wave) / prb_max**2
        self.obj[yy, xx] += delta

    def _update_probe(self, proj_wave, exit_wave, indices):
        yy, xx = indices
        obj_patch = self.obj[yy, xx]
        obj_abs = self.xp.abs(obj_patch)
        obj_conj = obj_patch.conj()
        obj_max = self.xp.max(obj_abs)
        delta_prb = self.beta * obj_conj *  (proj_wave - exit_wave) / obj_max**2
        self.prb += delta_prb

    def run(self, n_iter=100):
        for it in range(n_iter):
            err = 0.0
            for d in self.ptycho._diff_data:
                yy, xx = d.indices
                obj_patch = self.obj[yy, xx]
                exit_wave = self.prb * obj_patch
                proj_wave, err_val = Fourier_projector(exit_wave, d.diffraction)
                err += err_val

                self._update_object(proj_wave, exit_wave, (yy, xx))
                self._update_probe(proj_wave, exit_wave, (yy, xx))

            if self.callback:
                self.callback(it, err / len(self.ptycho._diff_data), self.obj)

        return self.obj, self.prb

