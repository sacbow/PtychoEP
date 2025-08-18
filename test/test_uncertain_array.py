import pytest
from backend.backend import set_backend, np as backend_np
from rng.rng_utils import get_rng
from ptychoep.uncertain_array import UncertainArray

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
    ua_zero = UncertainArray.zeros((3, 3), scalar_precision = True)
    assert ua_zero.mean.shape == (3, 3)
    assert ua_zero.scalar_precision is True

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

@pytest.mark.parametrize("backend", ["numpy", "cupy"])
def test_scaled(backend):
    set_backend(backend)
    xp = backend_np()
    mean = xp.ones((2, 2), dtype=xp.complex64)
    ua = UncertainArray(mean, 1.0)  # scalar precision

    # Uniform gain
    ua_scaled = ua.scaled(2.0)
    assert ua_scaled.scalar_precision is True
    assert xp.allclose(ua_scaled.mean, mean * 2)
    
    # Precision comparison (avoid cupy-to-numpy error)
    if hasattr(ua_scaled.precision, "get"):
        scalar_prec = ua_scaled.precision.get().item()
    else:
        scalar_prec = ua_scaled.precision.item()
    assert scalar_prec == pytest.approx(0.25)

    # Non-uniform gain should raise error in scalar mode
    gain = xp.array([[1.0, 2.0], [3.0, 4.0]], dtype=xp.float32)
    # Non-uniform gain promoted to array precision
    ua_arr = ua.scaled(gain, to_array_when_nonuniform=True)
    assert ua_arr.scalar_precision is False
    expected_prec = 1/gain**2 * 1.0
    assert xp.allclose(ua_arr.precision, expected_prec)

@pytest.mark.parametrize("backend", ["numpy", "cupy"])
def test_fft_and_ifft(backend):
    set_backend(backend)
    xp = backend_np()
    mean = xp.ones((4, 4), dtype=xp.complex64)
    ua = UncertainArray(mean, 1.0)  # scalar precision
    from ptychoep.uncertain_array import fft_ua, ifft_ua

    # Forward FFT
    ua_fft = fft_ua(ua)
    assert ua_fft.mean.shape == ua.mean.shape
    assert ua_fft.scalar_precision is True

    # Inverse FFT
    ua_ifft = ifft_ua(ua_fft)
    assert ua_ifft.mean.shape == ua.mean.shape
    assert ua_ifft.scalar_precision is True
    assert xp.allclose(xp.abs(ua_ifft.mean), xp.abs(ua.mean), atol=1e-4)
