import pytest
from backend.backend import set_backend, np as backend_np
from ptychoep.uncertain_array import UncertainArray
from ptychoep.accumulative_uncertain_array import AccumulativeUncertainArray

@pytest.mark.parametrize("backend", ["numpy", "cupy"])
def test_add_and_get_ua(backend):
    set_backend(backend)
    xp = backend_np()
    shape = (4, 4)
    aua = AccumulativeUncertainArray(shape)
    mean = xp.ones(shape, dtype=xp.complex64)
    prec = xp.ones(shape, dtype=xp.float32)
    ua = UncertainArray(mean, prec)

    aua.add(ua)
    belief = aua.to_ua()
    assert xp.allclose(belief.mean, mean * 0.5)
    assert xp.allclose(belief.precision, prec + 1.0)  # initial precision is 1

@pytest.mark.parametrize("backend", ["numpy", "cupy"])
def test_subtract(backend):
    set_backend(backend)
    xp = backend_np()
    shape = (4, 4)
    aua = AccumulativeUncertainArray(shape)
    mean = xp.ones(shape, dtype=xp.complex64)
    prec = xp.ones(shape, dtype=xp.float32)
    ua = UncertainArray(mean, prec)

    aua.add(ua)
    aua.subtract(ua)
    belief = aua.to_ua()
    assert xp.allclose(belief.mean, xp.zeros_like(mean))
    assert xp.allclose(belief.precision, xp.ones_like(prec))  # back to initial

@pytest.mark.parametrize("backend", ["numpy", "cupy"])
def test_patch_get_ua(backend):
    set_backend(backend)
    xp = backend_np()
    shape = (6, 6)
    patch = (slice(1, 3), slice(1, 3))
    aua = AccumulativeUncertainArray(shape)

    patch_mean = xp.ones((2, 2), dtype=xp.complex64) * 2.0
    patch_prec = xp.ones((2, 2), dtype=xp.float32) * 3.0
    ua = UncertainArray(patch_mean, patch_prec)

    aua.add(ua, indices=patch)
    ua_patch = aua.get_ua(indices=patch)

    assert xp.allclose(ua_patch.mean, patch_mean * 0.75)
    assert xp.allclose(ua_patch.precision, patch_prec + 1.0)

@pytest.mark.parametrize("backend", ["numpy", "cupy"])
def test_clear(backend):
    set_backend(backend)
    xp = backend_np()
    shape = (3, 3)
    aua = AccumulativeUncertainArray(shape)
    ua = UncertainArray(xp.ones(shape, dtype=xp.complex64), 2.0)
    aua.add(ua)
    aua.clear()
    belief = aua.to_ua()
    assert xp.allclose(belief.mean, 0)
    assert xp.allclose(belief.precision, 1.0)

@pytest.mark.parametrize("backend", ["numpy", "cupy"])
def test_invalid_index_type(backend):
    set_backend(backend)
    aua = AccumulativeUncertainArray((4, 4))
    ua = UncertainArray.zeros((4, 4))
    with pytest.raises(TypeError):
        aua.add(ua, indices=5)
