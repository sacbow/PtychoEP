from ptychoep.ptycho.projector import Fourier_projector
from ptychoep.backend.backend import np
from .base_pie import BasePIE

class rPIE(BasePIE):
    """
    Relaxed Ptychographical Iterative Engine (rPIE).

    This algorithm is an extension of the conventional PIE algorithm that 
    introduces a relaxation parameter for more stable updates of the object 
    and probe. It performs iterative updates of both object and probe 
    estimates based on the measured diffraction data.

    Attributes:
        alpha (float): Relaxation parameter for the object update.
        beta (float): Relaxation parameter for the probe update.
        obj (ndarray): Current estimate of the object.
        prb (ndarray): Current estimate of the probe.
        callback (callable): Optional callback function to monitor progress.
        dtype (np.dtype): Data type for internal arrays.
        seed (int): Random seed for initialization.

    Notes:
        - The probe is updated in each iteration using the same principle as the object.
        - The computational cost is nearly the same as ePIE, as both update 
          object and probe with similar operations and FFT projections.
    """

    def __init__(self, ptycho, alpha=0.1, beta=0.1, obj_init=None, prb_init = None, callback=None, dtype = np().complex64, seed : int = None):
        super().__init__(ptycho, alpha, obj_init, dtype, callback, seed)
        self.prb = prb_init if prb_init is not None else ptycho.prb
        self.beta = beta

    def _update_object(self, old_probe, proj_wave, exit_wave, indices):
        yy, xx = indices
        prb_abs = self.xp.abs(old_probe)
        prb_conj = old_probe.conj()
        prb_max = self.xp.max(old_probe)
        denom = (1 - self.alpha) * (prb_abs**2) + self.alpha * (prb_max**2)
        delta = prb_conj * (proj_wave - exit_wave) / denom
        self.obj[yy, xx] += self.alpha * delta

    def _update_probe(self, old_object_patch, proj_wave, exit_wave, indices):
        yy, xx = indices
        obj_abs = self.xp.abs(old_object_patch)
        obj_conj = old_object_patch.conj()
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

                old_object_patch, old_probe = self.obj[yy, xx].copy(), self.prb.copy()

                self._update_object(old_probe, proj_wave, exit_wave, (yy, xx))
                self._update_probe(old_object_patch, proj_wave, exit_wave, (yy, xx))

            if self.callback:
                self.callback(it, err / len(self.ptycho._diff_data), self.obj)

        return self.obj, self.prb