from backend.backend import np
from rng.rng_utils import get_rng, normal
from ptycho.core import Ptycho
from ptycho.projector import Fourier_projector

class BasePIE:
    def __init__(self, ptycho: Ptycho, alpha: float = 0.1, obj_init=None, callback=None):
        self.xp = np()  # backend抽象化
        self.ptycho = ptycho
        self.alpha = self.xp.asarray(alpha)
        self.callback = callback

        # オブジェクト初期化: ランダム複素分布
        if obj_init is None:
            rng = get_rng()
            self.obj = normal(rng, mean=0.0, var=1.0, size=(ptycho.obj_len, ptycho.obj_len), dtype=self.xp.complex64)
        else:
            self.obj = self.xp.array(obj_init)

        # プローブ設定
        self.prb = self.xp.array(ptycho.prb.copy())
        # FFT関数
        self.fft2 = self.xp.fft.fft2
        self.ifft2 = self.xp.fft.ifft2

    def run(self, n_iter=100):
        for it in range(n_iter):
            err = 0.0
            for d in self.ptycho._diff_data:
                yy, xx = d.indices
                obj_patch = self.obj[yy, xx]
                exit_wave = self.prb * obj_patch

                proj_wave, error_val = Fourier_projector(exit_wave, d.diffraction)
                err += error_val

                # オブジェクト更新
                self._update_object(proj_wave, exit_wave, (yy, xx))

            avg_err = float(err / len(self.ptycho._diff_data))

            # コールバック呼び出し
            if self.callback:
                self.callback(it, avg_err, self.obj)

        return self.obj


    def _update_object(self, proj_wave, exit_wave, indices):
        raise NotImplementedError("派生クラスで実装してください")
