import pytest
from backend.backend import set_backend, np as backend_np
from ptychoep.fft_channel import FFTChannel
from ptychoep.uncertain_array import UncertainArray
from ptychoep.probe import Probe
from ptycho.data import DiffractionData


@pytest.mark.parametrize("backend", ["numpy", "cupy"])
def test_fft_channel_initialization(backend):
    set_backend(backend)
    xp = backend_np()

    obj_init = xp.ones((8, 8), dtype=xp.complex64)
    probe_data = xp.ones((4, 4), dtype=xp.complex64)
    patch = (slice(2, 6), slice(2, 6))

    class DummyObject:
        def __init__(self):
            self.object_init = obj_init

    diff = DiffractionData(diffraction=xp.ones((4, 4)), position=(4, 4))
    diff.indices = patch

    probe = Probe(data=probe_data, parent=DummyObject(), diffraction=diff)
    fft_ch = FFTChannel(parent_probe=probe, diff=diff)

    # Check that msg_from_likelihood is initialized
    assert isinstance(fft_ch.msg_from_likelihood, UncertainArray)
    assert fft_ch.msg_from_likelihood.scalar_precision


@pytest.mark.parametrize("backend", ["numpy", "cupy"])
def test_fft_channel_forward(backend):
    set_backend(backend)
    xp = backend_np()

    shape = (4, 4)
    probe_data = xp.ones(shape, dtype=xp.complex64)

    class DummyObject:
        def __init__(self):
            self.object_init = xp.ones((8, 8), dtype=xp.complex64)

    class DummyLikelihood:
        def __init__(self):
            self.msg_from_fft = None

    diff = DiffractionData(diffraction=xp.ones(shape), position=(4, 4))
    diff.indices = (slice(2, 6), slice(2, 6))

    probe = Probe(data=probe_data, parent=DummyObject(), diffraction=diff)
    fft_ch = FFTChannel(parent_probe=probe, diff=diff)
    fft_ch.likelihood = DummyLikelihood()

    ua = UncertainArray(xp.ones(shape, dtype=xp.complex64), precision=1.0)
    fft_ch.input_belief = ua

    fft_ch.forward()
    assert isinstance(fft_ch.likelihood.msg_from_fft, UncertainArray)


@pytest.mark.parametrize("backend", ["numpy", "cupy"])
def test_fft_channel_backward(backend):
    set_backend(backend)
    xp = backend_np()

    shape = (4, 4)

    class DummyObject:
        def __init__(self):
            self.object_init = xp.ones((8, 8), dtype=xp.complex64)

    diff = DiffractionData(diffraction=xp.ones(shape), position=(4, 4))
    diff.indices = (slice(2, 6), slice(2, 6))

    probe = Probe(data=xp.ones(shape, dtype=xp.complex64), parent=DummyObject(), diffraction=diff)
    fft_ch = FFTChannel(parent_probe=probe, diff=diff)

    fft_ch.msg_from_likelihood = UncertainArray(xp.ones(shape, dtype=xp.complex64), precision=1.0)
    fft_ch.backward()

    assert isinstance(fft_ch.msg_to_probe, UncertainArray)


@pytest.mark.parametrize("backend", ["numpy", "cupy"])
def test_fft_channel_missing_inputs(backend):
    set_backend(backend)
    xp = backend_np()

    class DummyObject:
        def __init__(self):
            self.object_init = xp.ones((8, 8), dtype=xp.complex64)

    diff = DiffractionData(diffraction=xp.ones((4, 4)), position=(4, 4))
    diff.indices = (slice(2, 6), slice(2, 6))

    probe = Probe(data=xp.ones((4, 4), dtype=xp.complex64), parent=DummyObject(), diffraction=diff)
    fft_ch = FFTChannel(parent_probe=probe, diff=diff)

    # input_belief not set → forward should fail
    with pytest.raises(RuntimeError):
        fft_ch.forward()

    # msg_from_likelihood not set → backward should fail
    fft_ch.msg_from_likelihood = None
    with pytest.raises(RuntimeError):
        fft_ch.backward()
