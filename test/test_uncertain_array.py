import pytest
from PtychoEP.utils.backend import set_backend, np as backend_np
from PtychoEP.utils.rng_utils import get_rng
from PtychoEP.utils.engines.ptycho_ep.uncertain_array import UncertainArray

@pytest.mark.parametrize("backend", ["numpy", "cupy"])
def test_init_and_properties(backend):
    set_backend(backend)
    mean = backend_np().ones((4, 4), dtype=backend_np().complex64)
    ua = UncertainArray(mean)
    assert ua.mean.shape == (4, 4)
    assert ua.scalar_precision is True
    assert ua.precision.shape == ()

@pytest.mark.parametrize("backend", ["numpy", "cupy"])
def test_array_precision_init(backend):
    set_backend(backend)
    mean = backend_np().zeros((2, 2), dtype=backend_np().complex64)
    precision = backend_np().ones((2, 2), dtype=backend_np().float32) * 2.0
    ua = UncertainArray(mean, precision)
    assert ua.scalar_precision is False
    assert backend_np().allclose(ua.precision, 2.0)

@pytest.mark.parametrize("backend", ["numpy", "cupy"])
def test_zeros_and_normal(backend):
    set_backend(backend)
    ua_zero = UncertainArray.zeros((3, 3))
    assert ua_zero.mean.shape == (3, 3)
    assert ua_zero.scalar_precision is False

    rng = get_rng(42)
    ua_norm = UncertainArray.normal((3, 3), rng)
    assert ua_norm.mean.shape == (3, 3)
    assert backend_np().iscomplexobj(ua_norm.mean)

@pytest.mark.parametrize("backend", ["numpy", "cupy"])
def test_mul_and_div(backend):
    set_backend(backend)
    mean = backend_np().ones((2, 2), dtype=backend_np().complex64)
    ua1 = UncertainArray(mean, backend_np().ones((2, 2)))
    ua2 = UncertainArray(mean * 2, backend_np().ones((2, 2)) * 3)
    prod = ua1 * ua2
    assert backend_np().all(prod.precision == 4)  # 1+3
    div = prod / ua1
    assert div.mean.shape == (2, 2)

@pytest.mark.parametrize("backend", ["numpy", "cupy"])
def test_precision_conversions(backend):
    set_backend(backend)
    mean = backend_np().zeros((2, 2), dtype=backend_np().complex64)
    precision = backend_np().ones((2, 2), dtype=backend_np().float32) * 2.0
    ua_array = UncertainArray(mean, precision)
    ua_scalar = ua_array.to_scalar_precision()
    assert ua_scalar.scalar_precision is True
    ua_back = ua_scalar.to_array_precision()
    assert ua_back.scalar_precision is False
    assert ua_back.precision.shape == mean.shape

@pytest.mark.parametrize("backend", ["numpy", "cupy"])
def test_damp_with(backend):
    set_backend(backend)
    mean1 = backend_np().ones((2, 2), dtype=backend_np().complex64)
    mean2 = backend_np().zeros((2, 2), dtype=backend_np().complex64)
    ua1 = UncertainArray(mean1, backend_np().ones((2, 2)))
    ua2 = UncertainArray(mean2, backend_np().ones((2, 2)) * 4.0)
    damped = ua1.damp_with(ua2, damping=0.5)
    assert backend_np().allclose(damped.mean, 0.5)

@pytest.mark.parametrize("backend", ["numpy", "cupy"])
def test_invalid_precision_shapes(backend):
    set_backend(backend)
    mean = backend_np().zeros((2, 2), dtype=backend_np().complex64)
    wrong_precision = backend_np().ones((3, 3))
    with pytest.raises(ValueError):
        UncertainArray(mean, wrong_precision)

@pytest.mark.parametrize("backend", ["numpy", "cupy"])
def test_mul_type_mismatch(backend):
    set_backend(backend)
    mean = backend_np().zeros((2, 2), dtype=backend_np().complex64)
    ua_scalar = UncertainArray(mean, 1.0)
    ua_array = UncertainArray(mean, backend_np().ones((2, 2)))
    with pytest.raises(ValueError):
        _ = ua_scalar * ua_array
    with pytest.raises(ValueError):
        _ = ua_array / ua_scalar

@pytest.mark.parametrize("backend", ["numpy", "cupy"])
def test_copy_independence(backend):
    set_backend(backend)
    ua = UncertainArray(backend_np().zeros((2, 2), dtype=backend_np().complex64))
    ua_copy = ua.copy()
    ua.mean[0, 0] = 10
    assert ua_copy.mean[0, 0] != 10
