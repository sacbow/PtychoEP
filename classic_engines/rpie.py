from ptycho.projector import Fourier_projector
from .base_pie import BasePIE

class rPIE(BasePIE):
    def __init__(self, ptycho, alpha=0.1, beta=0.1, obj_init=None, callback=None):
        super().__init__(ptycho, alpha, obj_init, callback)
        self.beta = beta

    def _update_object(self, proj_wave, exit_wave, indices):
        yy, xx = indices
        prb_abs = self.xp.abs(self.prb)
        prb_conj = self.prb.conj()
        prb_max = self.xp.max(prb_abs)
        denom = self.beta * (prb_abs**2) + (1 - self.beta) * (prb_max**2)
        delta = prb_conj * (proj_wave - exit_wave) / denom
        self.obj[yy, xx] += self.alpha * delta

    def _update_probe(self, proj_wave, exit_wave, indices):
        yy, xx = indices
        obj_patch = self.obj[yy, xx]
        obj_abs = self.xp.abs(obj_patch)
        obj_conj = obj_patch.conj()
        obj_max = self.xp.max(obj_abs)
        denom_prb = self.beta * (obj_abs**2) + (1 - self.beta) * (obj_max**2)
        delta_prb = obj_conj * (proj_wave - exit_wave) / denom_prb
        self.prb += self.beta * delta_prb

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
