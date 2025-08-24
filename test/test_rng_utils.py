import pytest
import numpy as np
from ptychoep.backend.backend import set_backend, np as backend_np, is_cupy
from ptychoep.rng.rng_utils import get_rng, normal

@pytest.mark.parametrize("backend", ["numpy", "cupy"])
def test_normal_real_and_complex(backend):
    set_backend(backend)
    # cupyがインストールされていない場合はskip
    if backend == "cupy" and not is_cupy():
        pytest.skip("cupy is not available")

    rng = get_rng(seed=123)

    # --- 実数乱数 ---
    x_real = normal(rng, mean=0.0, var=1.0, size=(8, 8), dtype=backend_np().float32)
    assert x_real.shape == (8, 8)
    assert x_real.dtype == backend_np().float32

    # 統計的に平均/分散が概ね正しいことを確認 (ゆるい閾値)
    mean_real = backend_np().asnumpy(x_real).mean() if backend == "cupy" else x_real.mean()
    var_real = backend_np().asnumpy(x_real).var() if backend == "cupy" else x_real.var()
    assert abs(mean_real) < 0.5
    assert 0.5 < var_real < 1.5

    # --- 複素数乱数 ---
    x_complex = normal(rng, mean=0.0, var=1.0, size=(8, 8), dtype=backend_np().complex64)
    assert x_complex.shape == (8, 8)
    assert x_complex.dtype == backend_np().complex64

    # 実部と虚部の統計特性を確認
    x_complex_np = backend_np().asnumpy(x_complex) if backend == "cupy" else x_complex
    assert abs(x_complex_np.real.mean()) < 0.5
    assert abs(x_complex_np.imag.mean()) < 0.5

def test_seed_reproducibility():
    """同じシードで同じ結果になるかを確認"""
    set_backend("numpy")
    rng1 = get_rng(seed=42)
    rng2 = get_rng(seed=42)

    x1 = normal(rng1, var=1.0, size=(4, 4), dtype=backend_np().float32)
    x2 = normal(rng2, var=1.0, size=(4, 4), dtype=backend_np().float32)
    assert np.allclose(x1, x2)

def test_invalid_dtype():
    """サポート外dtypeで例外が発生するかを確認"""
    set_backend("numpy")
    rng = get_rng(seed=0)
    with pytest.raises(ValueError):
        _ = normal(rng, size=(2, 2), dtype="int32")
