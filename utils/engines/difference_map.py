from ..backend import np
from ..rng_utils import get_rng, normal
from ..ptycho.projector import Fourier_projector

class DifferenceMap:
    def __init__(self, ptycho, beta=1.0, obj_init=None, prb_init=None, callback=None):
        self.xp = np()
        self.ptycho = ptycho
        self.beta = beta
        self.callback = callback

        # オブジェクト初期化
        if obj_init is None:
            rng = get_rng()
            self.obj = normal(rng, mean=0.0, var=1.0,
                              size=(ptycho.obj_len, ptycho.obj_len),
                              dtype=self.xp.complex64)
        else:
            self.obj = self.xp.array(obj_init)

        # プローブ初期化
        self.prb = self.xp.array(prb_init) if prb_init is not None else self.xp.array(ptycho.prb.copy())

        # Diffractionデータ
        self.diffs = self.xp.stack([d.diffraction for d in ptycho._diff_data])  # (n_scan, H, W)
        self.indices = [d.indices for d in ptycho._diff_data]  # 各スキャン位置のインデックス
        self.prb_len = ptycho.prb_len
        self.n_scan = len(self.diffs)

        # FFT関数
        self.fft2 = self.xp.fft.fft2
        self.ifft2 = self.xp.fft.ifft2

        self.all_yy = self.xp.concatenate([yy.flatten() for yy, _ in self.indices])
        self.all_xx = self.xp.concatenate([xx.flatten() for _, xx in self.indices])

    def run(self, n_iter=100):
        xp = self.xp
        exit_waves = self._compute_exit_waves()
        Phi, _ = Fourier_projector(exit_waves, self.diffs)

        for it in range(n_iter):
            _, err = Fourier_projector(exit_waves, self.diffs, return_per_scan=False)
            if self.callback:
                self.callback(it, float(err), self.obj)

            self._update_object_probe(Phi)
            exit_waves = self._compute_exit_waves()
            Phi = Phi + Fourier_projector(2 * exit_waves - Phi, self.diffs)[0] - exit_waves

        return self.obj, self.prb

    def _compute_exit_waves(self):
        xp = self.xp
        patches = xp.stack([self.obj[yy, xx] for yy, xx in self.indices])  # (n_scan, H, W)
        return self.prb[None, :, :] * patches

    def _update_object_probe(self, Phi):
        xp = self.xp

        # --- オブジェクト更新 (ベクトル化) ---
        num_obj_real = xp.zeros_like(self.obj.real)
        num_obj_imag = xp.zeros_like(self.obj.imag)
        den_obj = xp.zeros_like(self.obj.real) + 1e-10

        prb_conj = self.prb.conj()
        prb_abs2 = xp.abs(self.prb) ** 2

        # 全スキャンのインデックスと値を一括展開
        
        all_val = xp.concatenate([(prb_conj * phi).flatten() for phi in Phi])
        all_prb_abs2 = xp.concatenate([prb_abs2.flatten() for _ in Phi])

        # scatter加算を一括実行
        xp.add.at(num_obj_real, (self.all_yy, self.all_xx), all_val.real)
        xp.add.at(num_obj_imag, (self.all_yy, self.all_xx), all_val.imag)
        xp.add.at(den_obj, (self.all_yy, self.all_xx), all_prb_abs2)

        self.obj = (num_obj_real + 1j * num_obj_imag) / den_obj

        # --- プローブ更新 (従来通り) ---
        num_prb = xp.zeros_like(self.prb)
        den_prb = xp.zeros_like(self.prb) + 1e-10
        for (yy, xx), phi in zip(self.indices, Phi):
            obj_patch = self.obj[yy, xx]
            num_prb += obj_patch.conj() * phi
            den_prb += xp.abs(obj_patch) ** 2
        self.prb = num_prb / den_prb
