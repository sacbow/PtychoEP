# utils/engines/pie.py
from .base_pie import BasePIE

class PIE(BasePIE):
    def __init__(self, ptycho, alpha=0.1, obj_init=None, callback=None):
        super().__init__(ptycho, alpha, obj_init, callback)
        # PIEはプローブ固定なのでキャッシュ可能
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


