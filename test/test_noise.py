import pytest
from ptychoep.ptycho.core import Ptycho
from ptychoep.ptycho.noise import GaussianNoise, PoissonNoise
from ptychoep.backend.backend import set_backend, np as backend_np

@pytest.fixture(autouse=True)
def setup_backend():
    """numpyバックエンドで固定"""
    set_backend("numpy")

def create_simple_ptycho():
    """シンプルなPtychoオブジェクトを作成（64x64のobj, 16x16のprobe, 1点diffraction）"""
    p = Ptycho()
    obj = backend_np().ones((64, 64), dtype=backend_np().complex64)
    prb = backend_np().ones((16, 16), dtype=backend_np().complex64)
    p.set_object(obj)
    p.set_probe(prb)
    # 中心位置に1つだけdiffractionデータを生成
    from ptychoep.ptycho.forward import generate_diffraction
    positions = [(32, 32)]
    diffs = generate_diffraction(p, positions)
    p.set_diffraction_from_forward(diffs)
    return p

def test_gaussian_noise_addition_and_snr():
    """GaussianNoiseが回折像を変更し、SN比が計算されること"""
    p = create_simple_ptycho()
    clean_diff = p.diffs[0].copy()

    GaussianNoise(var=1e-3) @ p

    noisy_diff = p.diffs[0]
    # ノイズ付加によりデータが変化している
    assert not backend_np().allclose(clean_diff, noisy_diff)

    # SN比がptycho.noise_statsに記録され、dB値として妥当（正の有限値）である
    stats = p.noise_stats
    assert stats["type"] == "Gaussian"
    assert stats["var"] == pytest.approx(1e-3)
    assert "snr_mean_db" in stats
    assert backend_np().isfinite(stats["snr_mean_db"])
    assert stats["snr_mean_db"] > 0  # 強度がゼロでない限り正の値になる

def test_poisson_noise_addition_and_snr():
    """PoissonNoiseが回折像を変更し、SN比が計算されること"""
    p = create_simple_ptycho()
    clean_diff = p.diffs[0].copy()

    PoissonNoise(scale=1e4) @ p

    noisy_diff = p.diffs[0]
    # ノイズ付加によりデータが変化している
    assert not backend_np().allclose(clean_diff, noisy_diff)

    # SN比がptycho.noise_statsに記録され、dB値として妥当（正の有限値）である
    stats = p.noise_stats
    assert stats["type"] == "Poisson"
    assert stats["scale"] == pytest.approx(1e4)
    assert "snr_mean_db" in stats
    assert backend_np().isfinite(stats["snr_mean_db"])
    assert stats["snr_mean_db"] > 0

def test_noise_snr_dependency():
    """ノイズ強度やスケールによってSN比が変動することを確認"""
    p1 = create_simple_ptycho()
    GaussianNoise(var=1e-4) @ p1
    snr_low_noise = p1.noise_stats["snr_mean_db"]

    p2 = create_simple_ptycho()
    GaussianNoise(var=1e-2) @ p2
    snr_high_noise = p2.noise_stats["snr_mean_db"]

    # ノイズ分散が大きいほどSN比が低下するはず
    assert snr_high_noise < snr_low_noise
