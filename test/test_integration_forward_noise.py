import pytest
from PtychoEP.utils.backend import set_backend, is_cupy, np as backend_np
from PtychoEP.utils.ptycho.core import Ptycho
from PtychoEP.utils.ptycho.forward import generate_diffraction
from PtychoEP.utils.ptycho.scan_utils import generate_spiral_scan_positions
from PtychoEP.utils.ptycho.noise import GaussianNoise, PoissonNoise

@pytest.mark.parametrize("backend", ["numpy", "cupy"])
def test_forward_noise_integration(backend):
    """forwardとnoiseの統合フローをnumpy/cupy両バックエンドで検証"""
    set_backend(backend)
    # --- Ptycho準備 ---
    p = Ptycho()
    obj = backend_np().ones((64, 64), dtype=backend_np().complex64)
    prb = backend_np().ones((16, 16), dtype=backend_np().complex64)
    p.set_object(obj)
    p.set_probe(prb)

    # --- Forward: 回折像生成 ---
    positions = [(32, 32), (40, 40)]  # 固定位置
    p.forward_and_set_diffraction(positions)

    assert len(p.diffs) == len(positions)
    assert all(d.shape == (16, 16) for d in p.diffs)

    # --- GaussianNoise適用 ---
    GaussianNoise(var=1e-3) @ p
    assert "snr_mean_db" in p.noise_stats
    snr_gauss = p.noise_stats["snr_mean_db"]
    assert backend_np().isfinite(snr_gauss)

    # --- PoissonNoise適用 ---
    PoissonNoise(scale=1e4) @ p
    assert "snr_mean_db" in p.noise_stats
    snr_poisson = p.noise_stats["snr_mean_db"]
    assert backend_np().isfinite(snr_poisson)

    # --- SNRはノイズ種によって異なる（同値でないはず）
    assert snr_gauss != snr_poisson
